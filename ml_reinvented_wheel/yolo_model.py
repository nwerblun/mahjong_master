import os.path
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.regularizers import l2
from input_output_utils import get_datasets
import yolo_globals as yg
import cv2 as cv
import numpy as np


class YoloReshape(keras.layers.Layer):
    def __init__(self, target_shape, **kwargs):
        super().__init__()
        self.target_shape = tuple(target_shape)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'target_shape': self.target_shape
        })
        return config

    def call(self, inputs):
        # Don't specify batch size, TF takes care of it
        batch_size = tf.shape(inputs)[0]
        reshaped_inps = tf.reshape(inputs, [-1, yg.GRID_W, yg.GRID_H, yg.NUM_ANCHOR_BOXES, 5+yg.NUM_CLASSES])

        # Reshape class probabilities and DON'T softmax. Should be size (B x GridW x GridH x NumBoxes x NumClasses)
        class_probs = reshaped_inps[..., yg.PRED_CLASS_INDEX_START:yg.PRED_CLASS_INDEX_END]
        # Commented out because the cross entr. function in loss assumes non-softmaxed logits
        # class_probs = keras.activations.softmax(class_probs)
        # tf.print("Class probs shape", tf.shape(class_probs))

        # Each grid cell's confidence. A single value. Size (B x GridW x GridH x NumBBoxes x 1)
        cell_conf = reshaped_inps[..., yg.PRED_CONFIDENCE_INDEX]
        # Confidence should range 0->1 so sigmoid it
        cell_conf = keras.activations.sigmoid(cell_conf)
        # B x 13 x 13 x 6 -> B x 13 x 13 x 6 x 1
        cell_conf = tf.expand_dims(cell_conf, axis=-1)
        # tf.print("Obj confs shape", tf.shape(cell_conf))

        # Create a matrix where each element is just its grid cell location to shift x,y into grid cell units
        # Note that x,y are swapped because x -> col. instead of row in the image grid
        # Should contain 0,1,2,3,4...13 repeated 13 times, then reshaped
        # After reshape it should be a single value in each cell corresponding to its column location
        cell_indicies_x = tf.tile(tf.range(yg.GRID_W), [yg.GRID_H])
        cell_indicies_x = tf.reshape(cell_indicies_x, [-1, yg.GRID_H, yg.GRID_W, 1, 1])
        cell_indicies_x = tf.cast(cell_indicies_x, tf.float32)

        # Switches axes 2/1 but keeps all other axes the same. AKA, just transpose the grid
        cell_indicies_y = tf.transpose(cell_indicies_x, [0, 2, 1, 3, 4])
        cell_grid = tf.tile(tf.concat([cell_indicies_x, cell_indicies_y], -1),
                            [batch_size, 1, 1, yg.NUM_ANCHOR_BOXES, 1])
        # tf.print("Cell grid shape", tf.shape(cell_grid))
        # tf.print("Cell grid 5, 3", cell_grid[..., 5, 3, :, :])

        # Bounding box lists
        # Size should be (B x GridW x GridH x NumBBoxes x 4)
        # * 4 since there's x,y,w,h for each BBox
        # Start by grabbing just xy to sigmoid, so it ranges 0->1. Add cell grid to adjust units. Assumes order is xywh
        bboxes_xy = reshaped_inps[..., yg.PRED_BBOX_INDEX_START:yg.PRED_BBOX_INDEX_START+2]
        bboxes_xy = tf.sigmoid(bboxes_xy) + cell_grid
        # tf.print("BBox xy size", tf.shape(bboxes_xy))

        # Now grab just the wh and exp it to make it positive, then scale it by the anchor box sizes
        # This effectively gets a 'scaled' bbox of the same aspect ratio where the network is predicting the 'scale'
        bboxes_wh = reshaped_inps[..., yg.PRED_BBOX_INDEX_START+2:yg.PRED_BBOX_INDEX_END]
        # tf.print("BBox wh size", tf.shape(bboxes_wh))
        bboxes_wh = tf.exp(bboxes_wh) * \
            tf.cast(tf.reshape(yg.ANCHOR_BOXES_GRID_UNITS, [1, 1, 1, yg.NUM_ANCHOR_BOXES, 2]), tf.float32)
        bboxes = tf.concat([bboxes_xy, bboxes_wh], axis=-1)

        # tf.print("BBox shape", tf.shape(bboxes))
        bboxes = tf.reshape(bboxes, [-1, yg.GRID_W, yg.GRID_H, yg.NUM_ANCHOR_BOXES, 4])
        # The axes should be B x GridW x GridH x NumBBoxes, ?, so concatenation will happen on the last axis
        outputs = keras.layers.concatenate([bboxes, cell_conf, class_probs])
        # tf.print("Output shape", tf.shape(outputs))
        return outputs


def yolo_loss(y_true, y_pred):
    # y true bbox x,y,w,h are in grid cell units. E.g., ranging from 0->13
    y_true_bboxes = y_true[..., yg.LABEL_BBOX_INDEX_START:yg.LABEL_BBOX_INDEX_END]  # Bx13x13x6x4
    y_true_confs = y_true[..., yg.LABEL_CONFIDENCE_INDEX]  # Bx13x13x6
    y_true_confs = tf.expand_dims(y_true_confs, axis=-1)  # Bx13x13x6x1
    y_true_classes = y_true[..., yg.LABEL_CLASS_INDEX_START:yg.LABEL_CLASS_INDEX_END]  # Bx13x13x6x36

    # The model should adjust the output so x,y -> 0 to 13 and the pred/conf is sigmoided. w,h ideally 0->13
    y_pred_bboxes = y_pred[..., yg.PRED_BBOX_INDEX_START:yg.PRED_BBOX_INDEX_END]
    y_pred_confs = y_pred[..., yg.PRED_CONFIDENCE_INDEX]
    y_pred_confs = tf.expand_dims(y_pred_confs, axis=-1)
    y_pred_classes = y_pred[..., yg.PRED_CLASS_INDEX_START:yg.PRED_CLASS_INDEX_END]

    def calc_xywh_loss(y_true_xywh, y_pred_xywh, y_true_conf):
        # Use y_true_confs to make a mask since it's 1 or 0 where it should be
        mask = y_true_conf * yg.LAMBDA_COORD
        num_objs = tf.reduce_sum(tf.cast(mask > 0.0, tf.float32))
        y_true_xy = y_true_xywh[..., :2]
        y_true_wh = y_true_xywh[..., 2:]
        y_pred_xy = y_pred_xywh[..., :2]
        y_pred_wh = y_pred_xywh[..., 2:]
        # tf.print("\nnum_objs xywh", num_objs)
        loss_xy = tf.reduce_sum(tf.square(y_true_xy - y_pred_xy) * mask) / (num_objs + 1e-6) / 2.
        # loss_wh = tf.reduce_sum(tf.square(tf.sqrt(y_true_wh) - tf.sqrt(y_pred_wh)) * mask) / (num_objs + 1e-6) / 2.
        loss_wh = tf.reduce_sum(tf.square(y_true_wh - y_pred_wh) * mask) / (num_objs + 1e-6) / 2.
        # tf.print("loss wh", loss_wh)
        # tf.print("loss wh square term", tf.reduce_sum(tf.square(y_true_wh - y_pred_wh) * mask))
        return loss_wh + loss_xy

    def calc_class_loss(y_true_cls, y_pred_cls, y_true_conf):
        class_mask = y_true_conf * yg.LAMBDA_CLASS
        num_objs = tf.reduce_sum(tf.cast(class_mask > 0.0, tf.float32))
        # assumes non-softmaxed outputs and assumes true class labels are 0, 1 only
        true_box_class = tf.argmax(y_true_cls, -1)
        cls_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.cast(true_box_class, tf.int32),
                                                                  logits=y_pred_cls)
        cls_loss = tf.expand_dims(cls_loss, axis=-1)
        cls_loss = tf.reduce_sum(cls_loss * class_mask) / (num_objs + 1e-6)
        return cls_loss

    def get_masked_iou_scores(y_true_xywh, y_pred_xywh, y_true_conf):
        y_true_xy = y_true_xywh[..., :2]
        y_true_wh = y_true_xywh[..., 2:]
        y_pred_xy = y_pred_xywh[..., :2]
        y_pred_wh = y_pred_xywh[..., 2:]

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
        iou_scores = tf.expand_dims(iou_scores, axis=-1)
        return iou_scores * y_true_conf

    def calc_conf_loss(y_true_xywh, y_pred_xywh, y_true_conf, y_pred_conf):
        # I didn't do this correctly. I don't have the best ious ignoring assignments. Might not be good.
        masked_ious = get_masked_iou_scores(y_true_xywh, y_pred_xywh, y_true_conf)
        conf_mask = tf.cast((masked_ious < 0.6), tf.float32) * (1 - y_true_conf) * yg.LAMBDA_NO_OBJECT
        conf_mask = conf_mask + masked_ious * yg.LAMBDA_OBJECT
        num_objs = tf.reduce_sum(tf.cast(conf_mask > 0.0, tf.float32))
        loss_conf = tf.reduce_sum(tf.square(masked_ious - y_pred_conf) * conf_mask) / (num_objs + 1e-6) / 2.
        return loss_conf

    xywh_loss = calc_xywh_loss(y_true_bboxes, y_pred_bboxes, y_true_confs)
    class_loss = calc_class_loss(y_true_classes, y_pred_classes, y_true_confs)
    conf_loss = calc_conf_loss(y_true_bboxes, y_pred_bboxes, y_true_confs, y_pred_confs)

    # tf.print("class loss:", class_loss)
    # tf.print("conf loss:", conf_loss)

    loss = xywh_loss + class_loss + conf_loss
    # tf.print("Loss is ", loss)
    # tf.print("Loss final shape is: ", tf.shape(loss))
    return loss


def get_learning_schedule():
    schedule = [
        (0, 0.0015),
        (5, 0.001),
        (20, 0.0008)
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
    # Based on yolov2 tiny
    leaky_relu = keras.layers.LeakyReLU(alpha=0.1)
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1),
                     input_shape=(yg.IMG_W, yg.IMG_H, 3), padding='same',
                     activation=leaky_relu, kernel_regularizer=l2(5e-4)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    model.add(Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1),
                     input_shape=(yg.IMG_W, yg.IMG_H, 3), padding='same',
                     activation=leaky_relu, kernel_regularizer=l2(5e-4)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    model.add(Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same',
                     activation=leaky_relu, kernel_regularizer=l2(5e-4)))
    model.add(Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same',
                     activation=leaky_relu, kernel_regularizer=l2(5e-4)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same',
                     activation=leaky_relu, kernel_regularizer=l2(5e-4)))
    model.add(Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same',
                     activation=leaky_relu, kernel_regularizer=l2(5e-4)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same',
                     activation=leaky_relu, kernel_regularizer=l2(5e-4)))
    model.add(Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same',
                     activation=leaky_relu, kernel_regularizer=l2(5e-4)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same',
                     activation=leaky_relu, kernel_regularizer=l2(5e-4)))
    model.add(Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding='same',
                     activation=leaky_relu, kernel_regularizer=l2(5e-4)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same',
                     activation=leaky_relu, kernel_regularizer=l2(5e-4)))
    model.add(Conv2D(filters=512, kernel_size=(1, 1), strides=(1, 1), padding='same',
                     activation=leaky_relu, kernel_regularizer=l2(5e-4)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same'))

    model.add(Conv2D(filters=((5 + yg.NUM_CLASSES)*yg.NUM_ANCHOR_BOXES),
                     kernel_size=(1, 1), activation=leaky_relu, kernel_regularizer=l2(5e-4)))

    model.add(YoloReshape(target_shape=yg.YOLO_OUTPUT_SHAPE))
    model.summary()
    return model


def train_model(test_after=True, output_json=False):
    train_ds, valid_ds, test_ds, n_train, n_valid, n_test = get_datasets()
    yolo_model = make_model()
    yolo_model.compile(loss=yolo_loss, optimizer="adam")

    model_save_filename = "model.h5"
    early_cb = keras.callbacks.EarlyStopping(patience=12, restore_best_weights=True)
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
    yolo_model.save("model.h5", save_format="h5")
    if output_json:
        model_h5_to_json_weights(yolo_model)

    if test_after:
        _test_model(test_ds, n_train, from_h5=True)


def load_model(from_json=False):
    try:
        if from_json:
            if not os.path.exists("model.json") or not os.path.exists("model_weights.h5"):
                yolo_model = keras.models.load_model("model.h5",
                                                     custom_objects={
                                                         "yolo_loss": yolo_loss,
                                                         "YoloReshape": YoloReshape
                                                     },
                                                     compile=True)
                model_h5_to_json_weights(yolo_model)

            f = open("model.json", "r")
            model_text = f.read()
            f.close()
            yolo_model = keras.models.model_from_json(model_text)
            yolo_model.load_weights("model_weights.h5")
            yolo_model.compile(loss=yolo_loss, optimizer="adam")
        else:
            yolo_model = keras.models.load_model("model.h5",
                                                 custom_objects={
                                                     "yolo_loss": yolo_loss,
                                                     "YoloReshape": YoloReshape
                                                 },
                                                 compile=True)
    except FileNotFoundError:
        print("Could not find model files.")
        return None
    return yolo_model


def model_h5_to_json_weights(model):
    with open("model.json", "w") as outfile:
        arch = model.to_json()
        outfile.write(arch)
        outfile.close()
    model.save_weights("model_weights.h5")


def _test_model(ds, n_examples, from_h5=True):
    print("Testing. Loading model...")
    yolo_model = load_model(from_json=(not from_h5))
    print("Evaluation",
          yolo_model.evaluate(ds,
                              steps=int(n_examples // max(yg.BATCH_SIZE * yg.TRAIN_VAL_TEST_SPLIT_RATIO_TUPLE[2], 1))
                              ))


def predict_on_img(img_path):
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
    m = load_model()
    out = m(img, training=False)
    return np.squeeze(out.numpy())
