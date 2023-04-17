import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.regularizers import l2
from input_output_utils import get_datasets
import yolo_globals as yg


class YoloReshape(keras.layers.Layer):
    def __init__(self, target_shape):
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
        grid_dims = [self.target_shape[0], self.target_shape[1]]

        # Input will be one large flat array with Batch x grid_w x grid_h x (bboxes * 5 + num_classes) elements
        start_to_num_classes = grid_dims[0] * grid_dims[1] * yg.NUM_CLASSES
        num_classes_to_bboxes = start_to_num_classes + grid_dims[0] * grid_dims[1] * yg.NUM_BOUNDING_BOXES

        # Reshape class probabilities and softmax. Should be size (B x GridW x GridH x NumClasses)
        class_probs = tf.reshape(inputs[..., :start_to_num_classes],
                                 [-1, grid_dims[0], grid_dims[1], yg.NUM_CLASSES])
        # Convert to prob. distribution
        class_probs = keras.activations.softmax(class_probs)

        # Each grid cell's confidence. A single value. Size (B x GridW x GridH x NumBBoxes)
        cell_conf = tf.reshape(inputs[..., start_to_num_classes:num_classes_to_bboxes],
                               [-1, grid_dims[0], grid_dims[1], yg.NUM_BOUNDING_BOXES])
        # Confidence should range 0->1 so sigmoid it
        cell_conf = keras.activations.sigmoid(cell_conf)

        # Bounding box lists
        # Size should be (B x GridW x GridH x NumBBoxes * 4)
        # * 4 since there's x,y,w,h for each BBox
        bboxes = tf.reshape(inputs[..., num_classes_to_bboxes:],
                            [-1, grid_dims[0], grid_dims[1], yg.NUM_BOUNDING_BOXES * 4])
        # It's in terms of % of size of grid cell and should range from 0->1. Sigmoid it
        bboxes = keras.activations.sigmoid(bboxes)

        # The axes should be B x GridW x GridH x ?, so concatenation will happen on the last axis
        outputs = keras.layers.concatenate([class_probs, cell_conf, bboxes])
        return outputs


def yolo_loss(y_true, y_pred):
    # Shamelessly copied from JY-112553's github implementation
    def xy_center_to_xy_min_max(xy, wh):
        xy_min = xy - wh / 2
        xy_max = xy + wh / 2
        return xy_min, xy_max

    def iou(pred_tl, pred_br, true_tl, true_br):
        intersect_top_lefts = tf.math.maximum(pred_tl, true_tl)
        intersect_bottom_rights = tf.math.minimum(pred_br, true_br)
        intersect_wh = tf.math.maximum(intersect_bottom_rights - intersect_top_lefts, 0.)
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

        # Same logic as above, but for each box individually
        iou_pred_wh = pred_br - pred_tl
        true_wh = true_br - true_tl
        pred_areas = iou_pred_wh[..., 0] * iou_pred_wh[..., 1]
        true_areas = true_wh[..., 0] * true_wh[..., 1]

        # pred area + true area will double count the area that is common to them. Remove that extra area
        union_areas = pred_areas + true_areas - intersect_areas
        return intersect_areas / union_areas

    def xywh_to_absolute_coords(xywhc):
        # Input is the x, y, w, h, c pair for a single bbox. x,y is in terms of grid size, w,h is % of img
        conv_row_index = tf.range(start=0, stop=yg.GRID_H)
        conv_col_index = tf.range(start=0, stop=yg.GRID_W)
        # Create list [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 0, 1, 2, ...] 13 times if grid_w = 13
        conv_row_index = tf.tile(conv_row_index, [yg.GRID_W])

        # Create list of
        # [[0 - 13],
        #  [0 - 13],
        #  ...] 13 times if grid_w = 13
        conv_col_index = tf.tile(tf.expand_dims(conv_col_index, 0), [yg.GRID_H, 1])
        # Transpose and flatten
        # This generates first
        # [[0, 0, 0, ...],
        #  [1, 1, 1, ...],
        #  [2, 2, 2, ...]
        #  ...]
        # then flattens it into [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, ...]
        conv_col_index = tf.transpose(conv_col_index).reshape([-1])

        # [[0, 0], [1, 0], [2, 0], ... [0, 1], [1, 1], [2, 1], ... [13, 13]] if grid_w = 13
        conv_index = tf.transpose(tf.stack([conv_row_index, conv_col_index]))
        conv_index = tf.reshape(conv_index, [-1, yg.GRID_H, yg.GRID_W, 1, 2])
        # Add dummy dimensions so we can broadcast
        conv_dims = tf.reshape(tf.convert_to_tensor([yg.GRID_H, yg.GRID_W], dtype=tf.float32), [1, 1, 1, 1, 2])
        # Adds xy center loc + grid cell index = absolute loc. in terms of grid cells
        # -> divide by grid_w to get location as % of image
        # Multiply by image width to get absolute img coord
        bbox_xy = (xywhc[..., :2] + conv_index) / conv_dims * yg.IMG_W  # Assumes img_w = img_h
        # wh is already in terms of image width, get absolute width by multiplying
        bbox_wh = xywhc[..., 2:4] * yg.IMG_W
        return bbox_xy, bbox_wh

    # All dims below assume grid is 13x13 and there are 35 classes
    # The comment specifies the dimensions after slicing
    # B is batch size
    label_classes = y_true[..., yg.LABEL_CLASS_INDEX_START:yg.LABEL_CLASS_INDEX_END]  # B x 13 x 13 x 35
    label_bboxes = y_true[..., yg.LABEL_BBOX_INDEX_START:yg.LABEL_BBOX_INDEX_END]  # B x 13 x 13 x 4
    label_confidence = y_true[..., yg.LABEL_CONFIDENCE_INDEX]  # B x 13 x 13
    result_mask = tf.expand_dims(label_confidence, -1)  # B x 13 x 13 x 1

    # Predictions are formatted different. After classes are 4 confidences, then 16 xywh pairs
    # First 35 is still the class predictions. Dims assume 4 bounding box guesses.
    pred_classes = y_pred[..., :yg.PRED_CLASS_INDEX_END:yg.PRED_CLASS_INDEX_END]  # B x 13 x 13 x 35
    pred_confs = y_pred[..., yg.PRED_CONFIDENCE_INDEX_START:yg.PRED_CONFIDENCE_INDEX_END]  # B x 13 x 13 x 4
    pred_bboxes = y_pred[..., yg.PRED_CONFIDENCE_INDEX_END:]  # B x 13 x 13 x 16

    # Using a -1 means 'make the dimensions 13x13x1x4 and then figure out the -1 to make it work'
    # Essentially the -1 will just be the batch size
    reshaped_label_bboxes = tf.reshape(label_bboxes, [-1, yg.GRID_H, yg.GRID_W, 1, 4])

    # We have 4 predictions of a bounding box per grid cell, each with xywh
    reshaped_pred_bboxes = tf.reshape(pred_bboxes, [-1, yg.GRID_H, yg.GRID_W, yg.NUM_BOUNDING_BOXES, 4])

    label_xy_absolute, label_wh_absolute = xywh_to_absolute_coords(reshaped_label_bboxes)  # each B x 13 x 13 x 1 x 2

    # Insert an empty dimension at position 3 for whatever reason. I think 3 or 4 doesn't matter for the label
    # Essentially, each 13x13 cell has a single length-2 vector containing x, y
    label_xy_absolute = tf.expand_dims(label_xy_absolute, 3)  # B x 13 x 13 x 1 x 1 x 2
    label_wh_absolute = tf.expand_dims(label_wh_absolute, 3)  # B x 13 x 13 x 1 x 1 x 2

    # Convert center xy + wh to top left xy and bottom right xy
    label_xy_tl, label_xy_br = xy_center_to_xy_min_max(label_xy_absolute, label_wh_absolute)  # B x 13 x 13 x 1 x 1 x 2

    pred_xy_absolute, pred_wh_absolute = xywh_to_absolute_coords(reshaped_pred_bboxes)  # each B x 13 x 13 x 4 x 2
    pred_xy_absolute = tf.expand_dims(pred_xy_absolute, 4)  # B x 13 x 13 x 4 x 1 x 2
    pred_wh_absolute = tf.expand_dims(pred_wh_absolute, 4)  # B x 13 x 13 x 4 x 1 x 2
    pred_xy_tl, predict_xy_br = xy_center_to_xy_min_max(pred_xy_absolute, pred_wh_absolute)

    # Compute IOU scores. 4 scores, 1 per box. Score is a single number
    iou_scores = iou(pred_xy_tl, predict_xy_br, label_xy_tl, label_xy_br)  # B x 13 x 13 x 4 x 1
    # I think this just reduces dimensions by 1. Not sure why this is done
    all_grid_bboxes_ious = tf.math.maximum(iou_scores, axis=4)  # B x 13 x 13 x 4
    # This actually computes maxes. Axis 3 has 4 elements, one IOU per box
    best_bbox_iou_per_cell = tf.math.maximum(all_grid_bboxes_ious, axis=3, keepdims=True)  # B x 13 x 13 x 1

    # Values are floats. Create a mask of which boxes in each cell have the highest IOU to 0 out the rest.
    # This is all booleans, so convert it to a float to get 1 or 0
    bbox_mask = tf.cast(all_grid_bboxes_ious >= best_bbox_iou_per_cell, tf.float32)  # B x 13 x 13 x 4

    # Compute loss.
    # If there should be no object, then the error is 0 - predicted.
    # bbox_mask * result mask should only = 1 if both the label exists there, and the grid properly predicted that
    # If that is so, then 1 - 1 = 0, and we ignore it. If either the label or the grid cell says no object, then
    # 1 - 0 = 1. We then add loss = how far away from 0 the prediction is. Predictions close to 0 = low loss.
    no_object_loss = yg.NO_OBJ_SCALE * (1 - bbox_mask * result_mask) * tf.math.square(0 - pred_confs)
    # Opposite case. Don't want to penalize the network for trying to find objects.
    # We add no loss if it predicts no object present. If it does, then just add the squared error off of 1
    object_loss = bbox_mask * result_mask * tf.math.square(1 - pred_confs)
    confidence_loss = no_object_loss + object_loss
    # Reduce to a single number across all grid cells and their boxes
    confidence_loss = tf.math.reduce_sum(confidence_loss)

    # How wrong of a class did it pick? If the label exists here, we should predict a class matching the label.
    class_loss = result_mask * tf.math.square(label_classes - pred_classes)
    class_loss = tf.math.reduce_sum(class_loss)

    # Re-compute all xy pairs to compute how wrong the box coordinates were
    label_xy, label_wh = xywh_to_absolute_coords(label_bboxes)
    pred_xy, pred_wh = xywh_to_absolute_coords(pred_bboxes)

    bbox_mask = tf.expand_dims(bbox_mask, -1)
    result_mask = tf.expand_dims(result_mask, -1)

    # Add no loss for cells without an object. If an object is present, we take the square of the difference of the
    # bounding box center coordinates. yolo_head produces absolute coordinates, convert it to % of image size
    box_loss = yg.BBOX_XY_LOSS_SCALE * bbox_mask * result_mask * tf.math.square((label_xy - pred_xy) / yg.IMG_W)
    # The above gets the loss from the location of the box. This calculates the loss from width/height mismatch
    # From the original paper:
    """
    Our error metric should reflect that small deviations in large boxes matter less than in small boxes.
    To partially address this we predict the square root of the bounding box width and height
    instead of the width and height directly.
    """
    box_loss += yg.BBOX_WH_LOSS_SCALE * bbox_mask * result_mask * \
        tf.math.square((tf.math.sqrt(label_wh) - tf.math.sqrt(pred_wh)) / yg.IMG_W)
    box_loss = tf.math.reduce_sum(box_loss)

    # Total loss is the sum of all loss terms
    loss = confidence_loss + class_loss + box_loss
    return loss


def get_learning_schedule():
    schedule = [
        (0, 0.01),
        (75, 0.001),
        (105, 0.0001)
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
    # Based on yolov2 tiny with some weird combo of yolov1
    leaky_relu = keras.layers.LeakyReLU(alpha=0.1)
    model = Sequential()
    model.add(Conv2D(filters=yg.IMG_W//yg.GRID_W, kernel_size=(yg.GRID_H, yg.GRID_W), strides=(1, 1),
                     input_shape=(yg.IMG_W, yg.IMG_H, 3), padding='same',
                     activation=leaky_relu, kernel_regularizer=l2(5e-4)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    model.add(Conv2D(filters=16, kernel_size=(3, 3), padding='same',
                     activation=leaky_relu, kernel_regularizer=l2(5e-4)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same',
                     activation=leaky_relu, kernel_regularizer=l2(5e-4)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same',
                     activation=leaky_relu, kernel_regularizer=l2(5e-4)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same',
                     activation=leaky_relu, kernel_regularizer=l2(5e-4)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same',
                     activation=leaky_relu, kernel_regularizer=l2(5e-4)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same',
                     activation=leaky_relu, kernel_regularizer=l2(5e-4)))
    model.add(Conv2D(filters=1024, kernel_size=(1, 1), padding='same',
                     activation=leaky_relu, kernel_regularizer=l2(5e-4)))
    model.add(Conv2D(filters=55, kernel_size=(1, 1), activation=leaky_relu, kernel_regularizer=l2(5e-4)))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Dense(1024))
    model.add(Dropout(0.5))
    model.add(Dense(yg.YOLO_TOTAL_DIMENSIONS, activation='sigmoid'))
    model.add(YoloReshape(target_shape=yg.YOLO_OUTPUT_SHAPE))
    model.summary()
    return model


def train_model(test_after=True, output_json=False):
    train_ds, valid_ds, test_ds = get_datasets()
    yolo_model = make_model()
    yolo_model.compile(loss=yolo_loss, optimizer="adam")

    model_save_filename = "model.h5"
    early_cb = keras.callbacks.EarlyStopping(patience=12, restore_best_weights=True)
    mid_cb = keras.callbacks.ModelCheckpoint(
        model_save_filename, monitor="val_accuracy", save_best_only=True
    )
    backup_cb = keras.callbacks.BackupAndRestore(backup_dir=".\\tmp\\backup")

    history = yolo_model.fit(
        train_ds,
        epochs=yg.NUM_EPOCHS,
        validation_data=valid_ds,
        callbacks=[get_learning_schedule(), early_cb, mid_cb, backup_cb],
        steps_per_epoch=int(len(train_ds)//yg.BATCH_SIZE),
        validation_steps=int(len(valid_ds)//(yg.BATCH_SIZE * yg.TRAIN_VAL_TEST_SPLIT_RATIO_TUPLE[1]))
    )

    print("Evaluation", yolo_model.evaluate(valid_ds,
                                            steps=int(len(valid_ds) //
                                                      (yg.BATCH_SIZE * yg.TRAIN_VAL_TEST_SPLIT_RATIO_TUPLE[1]))))
    if output_json:
        model_h5_to_json_weights(yolo_model)

    if test_after:
        _test_model(test_ds, from_h5=True)


def model_h5_to_json_weights(model=None):
    if model is None:
        yolo_model = keras.models.load_model("model.h5")
        yolo_model.compile(loss=yolo_loss, optimizer="adam")
    else:
        yolo_model = model
    with open("model.json", "w") as outfile:
        arch = yolo_model.to_json()
        outfile.write(arch)
        outfile.close()

    yolo_model.save_weights("model_weights.h5")


def _test_model(ds, from_h5=True):
    print("Testing. Loading model...")
    if from_h5:
        yolo_model = keras.models.load_model(".\\model.h5")
    else:
        f = open("model.json", "r")
        model_text = f.read()
        f.close()
        yolo_model = keras.models.model_from_json(model_text)
        yolo_model.load_weights("model_weights.h5")

    yolo_model.compile(loss=yolo_loss, optimizer="adam")
    print("Evaluation",
          yolo_model.evaluate(ds,
                              steps=int(len(ds) // (yg.BATCH_SIZE * yg.TRAIN_VAL_TEST_SPLIT_RATIO_TUPLE[2]))
                              ))
