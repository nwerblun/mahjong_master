import os.path
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential, Model
from keras.layers import BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Reshape, Input, Concatenate, Dropout, Lambda
from input_output_utils import get_datasets
import yolo_globals as yg
import cv2 as cv
import numpy as np


def yolo_loss(y_true, y_pred):
    # y true bbox x,y,w,h are in grid cell units. E.g., ranging from 0->13
    y_true_bboxes = y_true[..., yg.LABEL_BBOX_INDEX_START:yg.LABEL_BBOX_INDEX_END]  # Bx13x13x6x4
    y_true_xy = y_true_bboxes[..., 0:2]  # Bx13x13x6x2
    y_true_wh = y_true_bboxes[..., 2:]  # Bx13x13x6x2
    
    y_true_confs = y_true[..., yg.LABEL_CONFIDENCE_INDEX]  # Bx13x13x6
    y_true_classes = y_true[..., yg.LABEL_CLASS_INDEX_START:yg.LABEL_CLASS_INDEX_END]  # Bx13x13x6x36

    # Get a matrix of all objects in the image regardless of which cell they're assigned to.
    y_all_bboxes = tf.identity(y_true_bboxes)
    y_all_bboxes_xywh = tf.reshape(y_all_bboxes, [-1, 1, 1, 1, (yg.GRID_W * yg.GRID_H * yg.NUM_ANCHOR_BOXES), 4])
    y_all_bboxes_xy = y_all_bboxes_xywh[..., 0:2]
    y_all_bboxes_wh = y_all_bboxes_xywh[..., 2:]

    # The model should adjust the output so x,y -> 0 to 13 and the pred/conf is sigmoided. w,h ideally 0->13
    # Create a matrix where each element is just its grid cell location to shift x,y into grid cell units
    # Note that x,y are swapped because x -> col. instead of row in the image grid
    # Should contain 0,1,2,3,4...13 repeated 13 times, then reshaped
    # After reshape it should be a single value in each cell corresponding to its column location
    cell_indicies_x = tf.tile(tf.range(yg.GRID_W), [yg.GRID_H])
    cell_indicies_x = tf.reshape(cell_indicies_x, [-1, yg.GRID_H, yg.GRID_W, 1, 1])
    cell_indicies_x = tf.cast(cell_indicies_x, tf.float32)
    
    # Switches axes 2/1 but keeps all other axes the same. AKA, just transpose the grid
    cell_indicies_y = tf.transpose(cell_indicies_x, [0, 2, 1, 3, 4])
    # tf.shape(tensor) gets the shape but tensor.shape doesn't work for some reason. Anyway this just gets the current batch size.
    # I don't use yg.BATCH_SIZE because during validation it changes to a different number and is not reflected in that variable.
    cell_grid = tf.tile(tf.concat([cell_indicies_x, cell_indicies_y], -1), [tf.shape(y_pred)[0], 1, 1, yg.NUM_ANCHOR_BOXES, 1])
    
    # Start by grabbing just xy to sigmoid, so it ranges 0->1. Add cell grid to adjust units. Assumes order is xywh
    y_pred_bboxes = y_pred[..., yg.PRED_BBOX_INDEX_START:yg.PRED_BBOX_INDEX_END]
    y_pred_xy = y_pred_bboxes[..., 0:2]
    y_pred_wh = y_pred_bboxes[..., 2:]
    
    y_pred_xy = tf.sigmoid(y_pred_xy) + cell_grid

    # Now grab just the wh and exp it to make it positive, then scale it by the anchor box sizes
    #  effectively gets a 'scaled' bbox of the same aspect ratio where the network is predicting the 'scale'
    # It will be in units of grids (0->13)
    y_pred_wh = tf.exp(y_pred_wh) * tf.reshape(yg.ANCHOR_BOXES_GRID_UNITS, [1, 1, 1, yg.NUM_ANCHOR_BOXES, 2])

    # Set confidence range 0->1
    y_pred_confs = y_pred[..., yg.PRED_CONFIDENCE_INDEX]
    y_pred_confs = keras.activations.sigmoid(y_pred_confs)
    
    # Don't softmax classes because it's done during the loss calc.
    y_pred_classes = y_pred[..., yg.PRED_CLASS_INDEX_START:yg.PRED_CLASS_INDEX_END]

    def calc_xywh_loss(y_true_xy, y_true_wh, y_pred_xy, y_pred_wh, y_true_conf):
        # Use y_true_confs to make a mask since it's 1 or 0 where it should be
        mask = tf.expand_dims(y_true_conf, axis=-1) * yg.LAMBDA_COORD
        num_objs = tf.reduce_sum(tf.cast(mask > 0.0, tf.float32))
        loss_xy = tf.reduce_sum(tf.square(y_true_xy - y_pred_xy) * mask) / (num_objs + 1e-6) / 2.
        loss_wh = tf.reduce_sum(tf.square(tf.sqrt(y_true_wh) - tf.sqrt(y_pred_wh)) * mask) / (num_objs + 1e-6) / 2.
        if yg.DEBUG_PRINT > 1:
            tf.print("\n")
            tf.print("Predicted xy of batch 1, box 0, 12x7", y_pred_xy[0, 11, 6, 0, :])
            tf.print("Predicted wh of batch 1, box 0, 12x7", y_pred_wh[0, 11, 6, 0, :])
        if yg.DEBUG_PRINT > 0:
            tf.print("\n")
            tf.print("XY Loss: ", loss_xy)
            tf.print("WH Loss: ", loss_wh)
        return loss_wh + loss_xy

    def calc_class_loss(y_true_cls, y_pred_cls, y_true_conf):
        class_mask = y_true_conf * yg.LAMBDA_CLASS
        num_objs = tf.reduce_sum(tf.cast(class_mask > 0.0, tf.float32))
        # assumes non-softmaxed outputs and assumes true class labels are 0, 1 only
        true_box_class = tf.argmax(y_true_cls, -1)
        if yg.DEBUG_PRINT > 1:
            tf.print("Example true box class on batch 1, 12x7\n", true_box_class[0, 11, 6, :])
            tf.print("Example masked pred box class on batch 1, 12x7\n", tf.argmax((y_pred_cls * tf.expand_dims(class_mask, axis=-1))[0, 11, 6, :, :], axis=-1))
            tf.print("Example softmaxed masked pred box class prob. on batch 1, 12x7\n", tf.reduce_max((tf.nn.softmax(y_pred_cls) * tf.expand_dims(class_mask, axis=-1))[0, 11, 6, :, :], axis=-1))
        cls_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_box_class,
                                                                  logits=y_pred_cls)
        cls_loss = tf.reduce_sum(cls_loss * class_mask) / (num_objs + 1e-6)
        if yg.DEBUG_PRINT > 1:
            tf.print("Example masked class loss on batch 1, 12x7\n", (cls_loss * class_mask)[0, 11, 6, :])
        return cls_loss

    def get_iou_scores(y_true_xy, y_true_wh, y_pred_xy, y_pred_wh):
        true_mins = y_true_xy - (y_true_wh / 2.)
        true_maxes = y_true_xy + (y_true_wh / 2.)

        pred_mins = y_pred_xy - (y_pred_wh / 2.)
        pred_maxes = y_pred_xy + (y_pred_wh / 2.)

        intersect_mins = tf.maximum(pred_mins, true_mins)
        intersect_maxes = tf.minimum(pred_maxes, true_maxes)
        intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

        true_areas = y_true_wh[..., 0] * y_true_wh[..., 1]
        pred_areas = y_pred_wh[..., 0] * y_pred_wh[..., 1]

        union_areas = pred_areas + true_areas - intersect_areas
        iou_scores = tf.truediv(intersect_areas, union_areas)
        return iou_scores

    def calc_conf_loss(y_true_xy, y_true_wh, y_pred_xy, y_pred_wh, y_all_bbox_xy, y_all_bbox_wh, y_true_conf, y_pred_conf):
        # Get IOU scores between predicted bboxes and ONLY the object they are assigned to guess. 0 out everything else with y_true_conf
        pred_vs_assigned_masked_ious = get_iou_scores(y_true_xy, y_true_wh, y_pred_xy, y_pred_wh) * y_true_conf
        # Get IOU scores between predicted bboxes and ALL objects in the image. It's based on xy/wh so if there is a non-0 IOU
        # It means that there's an object near a prediction that it may not have been assigned to.
        # y_pred has dims Bx13x13x6x4
        # ious has dims   Bx1x1x1x1014x4
        # To broadcast they must have same dims or one of them =1.
        # Expand y_pred to Bx13x13x6x1   x4
        # ious matches w/  Bx1 x1 x1x1024x4
        pred_vs_global_objs_ious = get_iou_scores(y_all_bbox_xy, y_all_bbox_wh, tf.expand_dims(y_pred_xy, axis=4), tf.expand_dims(y_pred_wh, axis=4))
        pred_vs_global_best_ious = tf.reduce_max(pred_vs_global_objs_ious, axis=4)  # Bx13x13x6
        # Create a mask where it is 1 if the bbox and IOU between any nearby objects are <0.6 AND there's not supposed to be an object there
        conf_mask = tf.cast((pred_vs_global_best_ious < 0.6), tf.float32) * (1 - y_true_conf) * yg.LAMBDA_NO_OBJECT
        # Overall mask is >0 if there's an object in the label and >0 if there's no object and no bbox with IOU >0.6 with nearby objects
        conf_mask = conf_mask + pred_vs_assigned_masked_ious * yg.LAMBDA_OBJECT
        num_objs = tf.reduce_sum(tf.cast(conf_mask > 0.0, tf.float32))
        loss_conf = tf.reduce_sum(tf.square(pred_vs_assigned_masked_ious - y_pred_conf) * conf_mask) / (num_objs + 1e-6) / 2.
        return loss_conf

    xywh_loss = calc_xywh_loss(y_true_xy, y_true_wh, y_pred_xy, y_pred_wh, y_true_confs)
    class_loss = calc_class_loss(y_true_classes, y_pred_classes, y_true_confs)
    conf_loss = calc_conf_loss(y_true_xy, y_true_wh, y_pred_xy, y_pred_wh, y_all_bboxes_xy, y_all_bboxes_wh, y_true_confs, y_pred_confs)

    # tf.print("class loss:", class_loss)
    # tf.print("conf loss:", conf_loss)
    if yg.DEBUG_PRINT > 0:
        tf.print("Class Loss: ", class_loss)
        tf.print("Confidence Loss: ", conf_loss)
    loss = xywh_loss + class_loss + conf_loss
    # tf.print("Loss is ", loss)
    # tf.print("Loss final shape is: ", tf.shape(loss))
    return loss


def get_learning_schedule():
    schedule = [
        (0, 0.001),
        (28, 0.0001)
    ]

    def update(epoch, lr):
        if epoch < schedule[0][0] or epoch > schedule[-1][0]:
            return lr
        for e, r in schedule:
            if epoch == e:
                return r
        return lr
    return keras.callbacks.LearningRateScheduler(update)


def make_model():
    # Based on yolov2
    leaky_relu = keras.layers.LeakyReLU(alpha=0.1)
    # 3 for RGB channels
    inp = Input(shape=(yg.IMG_W, yg.IMG_H, 3))

    # Input conv
    x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding="same")(inp)
    x = BatchNormalization()(x)
    x = leaky_relu(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Second input conv, only needed when img size gets large and it needs to be cut down more
    # x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same")(x)
    # x = BatchNormalization()(x)
    # x = leaky_relu(x)
    # x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Dropout(0.15)(x)

    # First conv stack
    x = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding="same")(x)
    x = BatchNormalization()(x)
    x = leaky_relu(x)

    x = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding="same")(x)
    x = BatchNormalization()(x)
    x = leaky_relu(x)
    
    x = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding="same")(x)
    x = BatchNormalization()(x)
    x = leaky_relu(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Second conv stack
    x = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding="same")(x)
    x = BatchNormalization()(x)
    x = leaky_relu(x)

    x = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding="same")(x)
    x = BatchNormalization()(x)
    x = leaky_relu(x)
    
    x = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding="same")(x)
    x = BatchNormalization()(x)
    x = leaky_relu(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Third conv stack
    x = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding="same")(x)
    x = BatchNormalization()(x)
    x = leaky_relu(x)

    x = Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding="same")(x)
    x = BatchNormalization()(x)
    x = leaky_relu(x)
    
    x = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding="same")(x)
    x = BatchNormalization()(x)
    x = leaky_relu(x)
    
    x = Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding="same")(x)
    x = BatchNormalization()(x)
    x = leaky_relu(x)
    
    x = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding="same")(x)
    x = BatchNormalization()(x)
    x = leaky_relu(x)

    # Feed forward reorg connection
    skip_conn = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding="same")(x)
    skip_conn = BatchNormalization()(skip_conn)
    skip_conn = leaky_relu(skip_conn)

    def space_to_depthx2(lyr):
        return tf.nn.space_to_depth(lyr, block_size=2)
    skip_conn = Lambda(space_to_depthx2)(skip_conn)

    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Final conv stack
    x = Conv2D(filters=1024, kernel_size=(3, 3), strides=(1, 1), padding="same")(x)
    x = BatchNormalization()(x)
    x = leaky_relu(x)

    x = Conv2D(filters=512, kernel_size=(1, 1), strides=(1, 1), padding="same")(x)
    x = BatchNormalization()(x)
    x = leaky_relu(x)

    x = Conv2D(filters=1024, kernel_size=(3, 3), strides=(1, 1), padding="same")(x)
    x = BatchNormalization()(x)
    x = leaky_relu(x)

    x = Conv2D(filters=512, kernel_size=(1, 1), strides=(1, 1), padding="same")(x)
    x = BatchNormalization()(x)
    x = leaky_relu(x)

    x = Dropout(0.05)(x)

    x = Conv2D(filters=1024, kernel_size=(3, 3), strides=(1, 1), padding="same")(x)
    x = BatchNormalization()(x)
    x = leaky_relu(x)

    x = Conv2D(filters=1024, kernel_size=(3, 3), strides=(1, 1), padding="same")(x)
    x = BatchNormalization()(x)
    x = leaky_relu(x)

    x = Conv2D(filters=1024, kernel_size=(3, 3), strides=(1, 1), padding="same")(x)
    x = BatchNormalization()(x)
    x = leaky_relu(x)

    x = Concatenate()([skip_conn, x])

    x = Dropout(0.15)(x)

    # Last layer + reshape
    x = Conv2D(filters=1024, kernel_size=(3, 3), strides=(1, 1), padding="same")(x)
    x = BatchNormalization()(x)
    x = leaky_relu(x)

    x = Conv2D(filters=(5 + yg.NUM_CLASSES)*yg.NUM_ANCHOR_BOXES, kernel_size=(1, 1), strides=(1, 1), padding="same")(x)
    x = BatchNormalization()(x)
    x = leaky_relu(x)

    out = Reshape(target_shape=yg.YOLO_OUTPUT_SHAPE)(x)
    model = Model(inp, out)
    model.summary()
    return model


def train_model(test_after=True, output_json=False):
    train_ds, valid_ds, test_ds, n_train, n_valid, n_test = get_datasets()
    yolo_model = make_model()
    yolo_model.compile(loss=yolo_loss, optimizer="adam")

    model_save_filename = "model.h5"
    early_cb = keras.callbacks.EarlyStopping(patience=25, restore_best_weights=True)
    mid_cb = keras.callbacks.ModelCheckpoint(
        model_save_filename, monitor="loss", save_best_only=True
    )
    backup_cb = keras.callbacks.BackupAndRestore(backup_dir=".\\tmp\\backup")

    history = yolo_model.fit(
        train_ds,
        epochs=yg.NUM_EPOCHS,
        validation_data=valid_ds,
        callbacks=[get_learning_schedule(), early_cb, mid_cb, backup_cb],
        steps_per_epoch=int(n_train//max(yg.BATCH_SIZE, 1)),
        validation_steps=int(n_valid//max(yg.BATCH_SIZE * yg.TRAIN_VAL_TEST_SPLIT_RATIO_TUPLE[1], 1))
    )

    print("Evaluation", yolo_model.evaluate(valid_ds,
                                            steps=int(n_valid //
                                                      max(yg.BATCH_SIZE * yg.TRAIN_VAL_TEST_SPLIT_RATIO_TUPLE[1], 1))))
    yolo_model.save(yg.ROOT_ML_PATH + "model.h5", save_format="h5")
    if output_json:
        model_h5_to_json_weights(yolo_model)

    if test_after:
        _test_model(test_ds, n_train, from_h5=True)


def load_model(from_json=False):
    try:
        if from_json:
            if not os.path.exists(yg.ROOT_ML_PATH + "model.json") or not os.path.exists(yg.ROOT_ML_PATH + "model_weights.h5"):
                yolo_model = keras.models.load_model(yg.ROOT_ML_PATH + "model.h5",
                                                     custom_objects={
                                                         "yolo_loss": yolo_loss
                                                     },
                                                     compile=True)
                model_h5_to_json_weights(yolo_model)

            f = open(yg.ROOT_ML_PATH + "model.json", "r")
            model_text = f.read()
            f.close()
            yolo_model = keras.models.model_from_json(model_text)
            yolo_model.load_weights(yg.ROOT_ML_PATH + "model_weights.h5")
            yolo_model.compile(loss=yolo_loss, optimizer="adam")
        else:
            yolo_model = keras.models.load_model(yg.ROOT_ML_PATH + "model.h5",
                                                 custom_objects={
                                                     "yolo_loss": yolo_loss
                                                 },
                                                 compile=True)
    except FileNotFoundError:
        print("Could not find model files.")
        return None
    return yolo_model


def model_h5_to_json_weights(model):
    with open(yg.ROOT_ML_PATH + "model.json", "w") as outfile:
        arch = model.to_json()
        outfile.write(arch)
        outfile.close()
    model.save_weights(yg.ROOT_ML_PATH + "model_weights.h5")


def _test_model(ds, n_examples, from_h5=True):
    print("Testing. Loading model...")
    yolo_model = load_model(from_json=(not from_h5))
    print("Evaluation",
          yolo_model.evaluate(ds,
                              steps=int(n_examples // max(yg.BATCH_SIZE * yg.TRAIN_VAL_TEST_SPLIT_RATIO_TUPLE[2], 1))
                              ))


def _prediction_steps(model_out):
    # Assumes batch size of 1, e.g. only for prediction on a single image
    cell_indicies_x = tf.tile(tf.range(yg.GRID_W), [yg.GRID_H])
    cell_indicies_x = tf.reshape(cell_indicies_x, [1, yg.GRID_H, yg.GRID_W, 1, 1])
    cell_indicies_x = tf.cast(cell_indicies_x, tf.float32)

    # Switches axes 2/1 but keeps all other axes the same. AKA, just transpose the grid
    cell_indicies_y = tf.transpose(cell_indicies_x, [0, 2, 1, 3, 4])
    cell_grid = tf.tile(tf.concat([cell_indicies_x, cell_indicies_y], -1), [1, 1, 1, yg.NUM_ANCHOR_BOXES, 1])

    # Start by grabbing just xy to sigmoid, so it ranges 0->1. Add cell grid to adjust units. Assumes order is xywh
    y_pred_bboxes = model_out[..., yg.PRED_BBOX_INDEX_START:yg.PRED_BBOX_INDEX_END]
    y_pred_xy = y_pred_bboxes[..., 0:2]
    y_pred_wh = y_pred_bboxes[..., 2:]

    # Convert to % img
    y_pred_xy = (tf.sigmoid(y_pred_xy) + cell_grid) / yg.GRID_W

    # Now grab just the wh and exp it to make it positive, then scale it by the anchor box sizes
    #  effectively gets a 'scaled' bbox of the same aspect ratio where the network is predicting the 'scale'
    y_pred_wh = tf.exp(y_pred_wh) * tf.reshape(yg.ANCHOR_BOXES_GRID_UNITS, [1, 1, 1, yg.NUM_ANCHOR_BOXES, 2])

    # Convert back to % img
    y_pred_wh = y_pred_wh / yg.GRID_W

    # Set confidence range 0->1
    y_pred_confs = model_out[..., yg.PRED_CONFIDENCE_INDEX]
    y_pred_confs = keras.activations.sigmoid(y_pred_confs)

    # Softmax class outputs
    y_pred_classes = tf.nn.softmax(model_out[..., yg.PRED_CLASS_INDEX_START:yg.PRED_CLASS_INDEX_END])
    return np.squeeze(y_pred_xy.numpy()), np.squeeze(y_pred_wh.numpy()), np.squeeze(y_pred_confs.numpy()), np.squeeze(y_pred_classes.numpy())


def predict_on_img_file(img_path, from_json=False, specific_model=None):
    try:
        img = cv.imread(img_path)
    except FileNotFoundError:
        print("Could not find image file", img_path)
        return None

    # CV loads in BGR order, want RGB
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = cv.resize(img, (yg.IMG_H, yg.IMG_W))
    img = img / 255.0
    img = img.reshape((1,)+img.shape)
    if specific_model is None:
        m = load_model(from_json)
    else:
        m = specific_model
    out = m(img, training=False)
    
    return _prediction_steps(out)


def predict_on_img_obj(img, from_json=False, specific_model=None):
    # Assumes the image comes from something like pyautogui where it's in RGB order
    img = cv.resize(img, (yg.IMG_H, yg.IMG_W))
    img = img / 255.0
    img = img.reshape((1,) + img.shape)
    if specific_model is None:
        m = load_model(from_json)
    else:
        m = specific_model
    out = m(img, training=False)

    return _prediction_steps(out)




