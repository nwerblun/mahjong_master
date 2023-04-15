import numpy as np
#import tensorflow as tf

# say img is at (x,y) top left of 285, 112 and bottom right of 360, 227 of a 448x448 img
# 7x7 grid
# center of img is at 322.5, 169.5
# width of img is 75
# height of img is 115
# convert to % of img
# x,y of center = 322.5/448, 169.5/448, 75/448, 115/448
# x,y = 0.719, 0.378
# w,h = 0.167, 0.257
# 0.719 * 7 = 5.033 -> x falls into grid cell col 5
# 0.378 * 7 = 2.646 -> y falls into grid cell row 2
# how far past top left of grid cell 5,2 are we? Grid cell 5,2 should be at 320,128
# x = 5.033 - 5 = 0.033
# y = 0.646
# This is % of grid cell size beyond the boundary at the top left 320, 128
# w,h is relative to the overall image. Knowing which cell, how far from the top left, and how wide/tall the box is
# is enough to reconstruct it.
# Into our label matrix at [5][2] we add [0.033, 0.646, 0.167, 0.257]

y_true_classes = [0.]*5 + [1.] + [0.]*14
y_true_box = [0.033, 0.646, 0.167, 0.257]
y_true_obj_detected = [1.]
y_true_single_bbox_pred = y_true_classes + y_true_box + y_true_obj_detected
y_true_np = np.array(y_true_single_bbox_pred)
y_true = np.zeros((1, 7, 7, 30))
y_true[0][2][5][:25] = y_true_np


y_pred_classes = [0.]*8 + [1.] + [0.]*11
y_pred_box = [0.1, 0.47, 0.08, 0.220]  # center: 326,158.1 tl: 308,108 br:344,207
y_pred_box2 = [0.4, 0.15, 0.167, 0.094]  # center: 346,138.1 tl: 308,117 br:383,159
y_pred_obj_detected = [0.87, 0.34]
y_pred_single_bbox_pred = y_pred_classes + y_pred_obj_detected + y_pred_box + y_pred_box2
y_pred_np = np.array(y_pred_single_bbox_pred)
y_pred = np.zeros((1, 7, 7, 30))
y_pred[0][2][5] = y_pred_np

# For some reason, yolo addresses it as [batch, col, row, ...] so grid order is Y,X
# But within the predictions and labels, the order is [bb_x, bb_y, bb_w, bb_h]


def xywh2minmax(xy, wh):
    # Convert from center xy to top left xy and bottom right xy
    xy_min = xy - wh / 2
    xy_max = xy + wh / 2

    return xy_min, xy_max


def iou(pred_mins, pred_maxes, true_mins, true_maxes):
    # pred_mins, true_mins are the collection of top left corners of the label / prediction boxes
    # the maximum X of the top left corners and the maximum Y of the top left corners is the top left corner
    # of the overall intersection
    # Dims Bx7x7x2x1x2 for all of them
    intersect_mins = np.maximum(pred_mins, true_mins)
    # Same logic as above, but this is for bottom right cornhers
    intersect_maxes = np.minimum(pred_maxes, true_maxes)
    # the width of the intersection box is the bottom right x - top left x and bottom right y - top right y
    # If that goes beyond the image, clamp it to 0.
    intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
    # intersect_wh after going through all the dimensions is basically a 2x2 array with
    # row 0 = x_diff, y_diff of the bounding box 1 with the label
    # row 1 = x_diff, y_diff of the bounding box 2 with the label
    # row_0[0] * row_0[1] = area of bounding box 1 with the label
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

    # Same logic as above, but for each box individually
    pred_wh = pred_maxes - pred_mins
    true_wh = true_maxes - true_mins
    pred_areas = pred_wh[..., 0] * pred_wh[..., 1]
    true_areas = true_wh[..., 0] * true_wh[..., 1]

    # pred area + true area will double count the area that is common to them. Remove that extra area
    union_areas = pred_areas + true_areas - intersect_areas
    iou_scores = intersect_areas / union_areas

    return iou_scores


def yolo_head(feats):
    # Feats = [x, y, w, h]
    # x, y are in terms of % of grid cell. w, h are fraction of total image size
    # Gets the grid size (7x7 most of the time)
    conv_dims = np.shape(feats)[1:3]
    # Generates [0, 1, 2, 4, 5, 6] if the grid is 7x7.
    conv_height_index = np.arange(0, stop=conv_dims[0])
    # Same as above. Technically could be different if you use a non-square grid.
    conv_width_index = np.arange(0, stop=conv_dims[1])
    # Tile repeats something many times. We repeat [0, 1, ..., 6] x 7 in the same dimension.
    # [0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2...]
    # Dimensions are (49,)
    conv_height_index = np.tile(conv_height_index, [conv_dims[1]])

    # expand_dims, 0 is the same as conv_width_index[0]. This gets the 0-6 list from above. Not the repeated one.
    # We repeat it -> 7 times down then 1 time ->
    # Dimensions are (7, 7)
    # looks like
    # [[0 - 6],
    #  [0 - 6],
    #  ...]
    # 7 times total
    conv_width_index = np.tile(
        np.expand_dims(conv_width_index, 0), [conv_dims[0], 1])
    # Transpose first, then flatten.
    # This generates first
    # [[0, 0, 0, ...],
    #  [1, 1, 1, ...],
    #  [2, 2, 2, ...]
    #  ...]
    # then flattens it into [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, ...] dimensions (49,)
    conv_width_index = np.transpose(conv_width_index).flatten()
    # (49,2). Basically identifiers for every grid cell's location.
    # [[0, 0], [1, 0], [2, 0], ... [0, 1], [1, 1], [2, 1], ... [6, 6]]
    conv_index = np.transpose(np.stack([conv_height_index, conv_width_index]))
    # Reshape into Batch_sizex7x7x1x2
    conv_index = np.reshape(conv_index, [-1, conv_dims[0], conv_dims[1], 1, 2])

    # conv_dims = [7, 7]. Add dummy dimensions so it can broadcast with conv_index from above.
    conv_dims = np.reshape(conv_dims, [1, 1, 1, 1, 2])
    # feats[..., :2] grabs the first two (xy) values from each of the Bx7x7x30 pairs. There is an extra dim that was
    # added earlier so it's Bx7x7x1x30 but it doesn't matter.
    # feats[..., :2] + conv_index adds the grid cell coordinate to each location in the label matrix.
    # So if grid cell row 2, col 5 has a label 10% of the way through the cell and 30% down the cell,
    # feats[batch][2][5][:][:2] will be [0.1, 0.5] and conv_index[batch][2][5][:][:] = [2, 5]
    # The sum will be 2.1, 5.5. Divide that by the number of grid cells (7x7, which is stored in conv_dims)
    # and then you get 2.1/7, 5.5/7. This is equivalent to what % of the total image to the right and down it is
    # Multiply by the width/height of the image (both 448) to get the absolute xy coordinates of the center of the box
    # wh is similar, but it's already in terms of % of image size, so just multiply by image size to convert.
    box_xy = (feats[..., :2] + conv_index) / conv_dims * 448
    box_wh = feats[..., 2:4] * 448

    return box_xy, box_wh


def yolo_loss(y_true, y_pred):
    # We are guessing 2 bounding boxes, each with x,y,w,h,confidence = 5 terms * 2 = 10.
    # There's 20 classes in this implementation. A bounding box only predicts if an object is present
    # The grid cell picks the class, of which there are 20.
    # In total we have 5 terms * 2 and 20 classes = 30 items per grid cell.
    # Bx7x7x30 (B = batch size). The last 5 are all 0's for the label since we don't guess 2 bounding boxes
    # First 20 are the classes. 20-23 are the xywh
    # 24 is the confidence
    label_class = y_true[..., :20]  # ? * 7 * 7 * 20
    label_box = y_true[..., 20:24]  # ? * 7 * 7 * 4
    response_mask = y_true[..., 24]  # ? * 7 * 7
    # This is Bx7x7 where each element is 1 or 0 since it's a label. Expand dims -1 adds 1 empty dimension at the end
    # This is to make the dimensions align with other stuff, but it's still just 1 or 0 on each of the 7x7 cells
    response_mask = np.expand_dims(response_mask, -1)  # ? * 7 * 7 * 1

    # Predictions are formatted different. First 20 is still the class predictions
    predict_class = y_pred[..., :20]  # ? * 7 * 7 * 20
    # Now, 20 and 21 are the confidence scores for each box
    predict_trust = y_pred[..., 20:22]  # ? * 7 * 7 * 2
    # 22 - 30 are the xywh for bounding box 1 and bounding box 2
    predict_box = y_pred[..., 22:]  # ? * 7 * 7 * 8

    # Using a -1 means 'make the dimensions 7x7x1x4 and then figure out the -1 to make it work'
    # Essentially the -1 will just be the batch size
    _label_box = np.reshape(label_box, [-1, 7, 7, 1, 4])
    # We have 2 predictions of a bounding box per grid cell, hence the extra dimension.
    _predict_box = np.reshape(predict_box, [-1, 7, 7, 2, 4])

    # Terrible name, but essentially transforms the xywh into absolute coordinates
    # Initially, the location in the matrix determines the grid cell it's in, and xy is what fraction of the grid cell
    # it is off from the top-left corner. So 0.1, 0.5 in label_box[2][5] means from the top left of row 2, col 5
    # we go 10% of the grid cell -> and then 50% of the grid cell down to reach the center of the box
    # yolo_head undoes this and returns xy coordinates relative to the top left of the image and w, h as a fraction
    # of the entire image
    label_xy, label_wh = yolo_head(_label_box)  # ? * 7 * 7 * 1 * 2, ? * 7 * 7 * 1 * 2
    # Insert an empty dimension at position 3 for whatever reason. I think 3 or 4 doesn't matter for the label
    # Essentially, each 7x7 cell has a single length-2 vector containing x, y
    label_xy = np.expand_dims(label_xy, 3)  # ? * 7 * 7 * 1 * 1 * 2
    label_wh = np.expand_dims(label_wh, 3)  # ? * 7 * 7 * 1 * 1 * 2
    # Convert center xy + wh to top left xy and bottom right xy
    label_xy_min, label_xy_max = xywh2minmax(label_xy, label_wh)  # ? * 7 * 7 * 1 * 1 * 2, ? * 7 * 7 * 1 * 1 * 2

    predict_xy, predict_wh = yolo_head(_predict_box)  # ? * 7 * 7 * 2 * 2, ? * 7 * 7 * 2 * 2
    predict_xy = np.expand_dims(predict_xy, 4)  # ? * 7 * 7 * 2 * 1 * 2
    predict_wh = np.expand_dims(predict_wh, 4)  # ? * 7 * 7 * 2 * 1 * 2
    predict_xy_min, predict_xy_max = xywh2minmax(predict_xy, predict_wh)  # ? * 7 * 7 * 2 * 1 * 2, ? * 7 * 7 * 2 * 1 * 2

    # Compute IOU scores. 2 scores, 1 per box. Score is a single number
    iou_scores = iou(predict_xy_min, predict_xy_max, label_xy_min, label_xy_max)  # ? * 7 * 7 * 2 * 1
    # I think this just reduces dimensions by 1. Not sure why this is done
    best_ious = np.max(iou_scores, axis=4)  # ? * 7 * 7 * 2
    # This actually computes maxes. Axis 3 has 2 elements, one IOU per box
    best_box = np.max(best_ious, axis=3, keepdims=True)  # ? * 7 * 7 * 1

    # Values are floats. Create a mask of which boxes in each cell have the highest IOU so we can effectively
    # ignore the second box. This is all booleans, so convert it to a float to get 1 or 0
    #box_mask = np.cast(best_ious >= best_box, np.float32)  # ? * 7 * 7 * 2
    box_mask = (best_ious >= best_box).astype(np.float32)  # ? * 7 * 7 * 2

    # Compute loss.
    # If there should be no object, then the error is 0 - predicted.
    # box_mask * response_mask should only = 1 if both the label exists there, and the grid properly predicted that
    # If that is so, then 1 - 1 = 0, and we ignore it. If either the label or the grid cell says no object, then
    # 1 - 0 = 1. We then add loss = how far away from 0 the prediction is. Predictions close to 0 = low loss.
    no_object_loss = 0.5 * (1 - box_mask * response_mask) * np.square(0 - predict_trust)
    # Opposite case. Don't want to penalize the network for trying to find objects.
    # We add no loss if it predicts no object present. If it does, then just add the squared error off of 1
    object_loss = box_mask * response_mask * np.square(1 - predict_trust)
    confidence_loss = no_object_loss + object_loss
    # Reduce to a single number across all grid cells and their boxes
    confidence_loss = np.sum(confidence_loss)

    # How wrong of a class did it pick? If the label exists here, we should predict a class matching the label.
    class_loss = response_mask * np.square(label_class - predict_class)
    class_loss = np.sum(class_loss)

    # Re-compute all xy pairs so we can compute how wrong the box coordinates were
    _label_box = np.reshape(label_box, [-1, 7, 7, 1, 4])
    _predict_box = np.reshape(predict_box, [-1, 7, 7, 2, 4])

    label_xy, label_wh = yolo_head(_label_box)  # ? * 7 * 7 * 1 * 2, ? * 7 * 7 * 1 * 2
    predict_xy, predict_wh = yolo_head(_predict_box)  # ? * 7 * 7 * 2 * 2, ? * 7 * 7 * 2 * 2

    box_mask = np.expand_dims(box_mask, -1)
    response_mask = np.expand_dims(response_mask, -1)

    # Add no loss for cells without an object. If an object is present, we take the square of the difference of the
    # bounding box center coordinates. yolo_head produces absolute coordinates, convert it to % of image size
    box_loss = 5 * box_mask * response_mask * np.square((label_xy - predict_xy) / 448)
    # The above gets the loss from the location of the box. This calculates the loss from width/height mismatch
    # From the original paper:
    """
    Our error metric should reflect that small deviations in large boxes matter less than in small boxes.
    To partially address this we predict the square root of the bounding box width and height
    instead of the width and height directly.
    """
    box_loss += 5 * box_mask * response_mask * np.square((np.sqrt(label_wh) - np.sqrt(predict_wh)) / 448)
    box_loss = np.sum(box_loss)

    # Total loss is the sum of all loss terms
    loss = confidence_loss + class_loss + box_loss

    return loss


yolo_loss(y_true, y_pred)

