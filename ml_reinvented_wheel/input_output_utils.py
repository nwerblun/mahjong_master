import cv2 as cv
import numpy as np
import os
import yolo_globals as yg
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import get_named_colors_mapping
from random import choice


# colors = list(get_named_colors_mapping().keys())
colors = [
    "red",
    "orangered",
    "magenta",
    "blue",
    "crimson",
    "lime",
    "yellow",
    "lawngreen",
    "tomato",
    "cyan"
]


# TODO: this
def check_if_grid_size_and_bbox_num_large_enough():
    # Go through the dataset and 'assign' each object to a grid cell.
    # If we run out of bboxes, then raise a warning
    pass


def file_to_img_label(example_tuple):
    img_path, lbl_path = example_tuple
    try:
        f = open(lbl_path)
        label_txt = f.readlines()
        f.close()
    except FileNotFoundError:
        print("Could not find label file", lbl_path, "skipping.")
        return None, None

    try:
        img = cv.imread(img_path)
    except FileNotFoundError:
        print("Could not find image file", img_path, "skipping.")
        return None, None

    # CV loads in BGR order, want RGB
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = cv.resize(img, (yg.IMG_H, yg.IMG_W))
    # Max is 255 (img are stored in 8-bits / channel) rescale to 0->1 max
    img = img / 255.0

    label_matrix = np.zeros(yg.YOLO_OUTPUT_SHAPE)
    for line in label_txt:
        # Formatted as class, x_center, y_center, box_w, box_h delimited by spaces
        # Class is a number based on yolo_globals.class_map
        # X, Y, W, H is already in units of % of image width/height -> range from 0->1
        split_line = [float(el) for el in line.strip().split(" ")]
        cls = split_line[0]
        x_center = split_line[1]
        y_center = split_line[2]
        w = split_line[3]
        h = split_line[4]
        # Will be a number between 0 -> grid size with a fraction representing how far into the cell.
        grid_loc = [yg.GRID_W * x_center, yg.GRID_H * y_center]
        # Y value determines row. Top left of image is 0,0 and y is downwards, x is to the right.
        grid_row = int(grid_loc[1])
        grid_col = int(grid_loc[0])
        y_offset_from_grid_row = grid_loc[1] - grid_row
        x_offset_from_grid_col = grid_loc[0] - grid_col
        # Just to ensure we only assign the label to 1 cell
        if label_matrix[grid_row, grid_col, yg.LABEL_CONFIDENCE_INDEX] == 0:
            label_matrix[grid_row, grid_col, yg.LABEL_BBOX_INDEX_START:yg.LABEL_BBOX_INDEX_END] = \
                np.array([x_offset_from_grid_col, y_offset_from_grid_row, w, h])
            label_matrix[grid_row, grid_col, yg.LABEL_CONFIDENCE_INDEX] = 1
            label_matrix[grid_row, grid_col, yg.LABEL_CLASS_INDEX_START+cls] = 1
        return img, label_matrix


# TODO: this
def augment_ds():
    return


def get_datasets():
    root = yg.ROOT_DATASET_PATH
    all_files = os.listdir(root)
    all_img = [i for i in all_files if os.path.splitext(i)[1] == yg.IMG_FILETYPE]
    all_labels = [i for i in all_files if os.path.splitext(i)[1] == yg.LABEL_FILETYPE]
    assert len(all_img) == len(all_labels) and\
           all([os.path.splitext(all_img[i])[0] == os.path.splitext(all_labels[i])[0] for i in range(len(all_img))])

    train_split, val_split, test_split = yg.TRAIN_VAL_TEST_SPLIT_RATIO_TUPLE

    all_examples = zip(all_img, all_labels)
    rng = np.random.RandomState(yg.GLOBAL_RNG_SEED)
    rng.shuffle(all_examples)

    all_ds = tf.data.Dataset.from_tensor_slices(all_examples)
    all_ds.map(file_to_img_label)
    total_len = len(all_ds)
    train_end_ind = int(total_len*train_split)
    valid_end_ind = train_end_ind + int(total_len*val_split)

    train_ds = all_ds[:train_end_ind]
    valid_ds = all_ds[train_end_ind:valid_end_ind]
    test_ds = all_ds[valid_end_ind:]

    train_ds = train_ds.shuffle(buffer_size=yg.BATCH_SIZE * 4, seed=yg.GLOBAL_RNG_SEED).repeat().batch(yg.BATCH_SIZE)
    valid_ds = valid_ds.shuffle(buffer_size=int(yg.BATCH_SIZE * val_split) * 4, seed=yg.GLOBAL_RNG_SEED).repeat()\
        .batch(int(yg.BATCH_SIZE * val_split))

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    valid_ds = valid_ds.prefetch(tf.data.AUTOTUNE)
    return train_ds, valid_ds, test_ds


def img_to_pred_input(img_path):
    try:
        img = cv.imread(img_path)
    except FileNotFoundError:
        print("Could not find image file", img_path)
        return None

    # CV loads in BGR order, want RGB
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img_h, img_w = img.shape[0:2]
    img = cv.resize(img, (yg.IMG_H, yg.IMG_W))
    # Max is 255 (img are stored in 8-bits / channel) rescale to 0->1 max
    return tf.convert_to_tensor(img / 255., dtype=tf.float32)


def draw_pred_output_and_plot(img_path, output_arr, class_thresh=0.7, conf_thresh=0.6, unsquish=True):
    try:
        img = cv.imread(img_path)
    except FileNotFoundError:
        print("Could not find image file", img_path)
        return None

    # CV loads in BGR order, want RGB
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img_h, img_w = img.shape[0:2]
    if not unsquish:
        img = cv.resize(img, (yg.IMG_H, yg.IMG_W))
        img_h = yg.IMG_H
        img_w = yg.IMG_W

    plt.imshow(img)
    y_spacing = img_h / yg.GRID_H
    x_spacing = img_w / yg.GRID_W
    for i in range(1, yg.GRID_H):
        plt.axhline(y=i * y_spacing, color="k", linestyle="--", alpha=0.2)
    for i in range(1, yg.GRID_W):
        plt.axvline(x=i * x_spacing, color="k", linestyle="--", alpha=0.2)

    for row in range(yg.GRID_H):
        for col in range(yg.GRID_W):
            pred_classes = output_arr[row, col, yg.PRED_CLASS_INDEX_START:yg.PRED_CLASS_INDEX_END]
            pred_confidences = output_arr[row, col, yg.PRED_CONFIDENCE_INDEX_START:yg.PRED_CONFIDENCE_INDEX_END]

            highest_prob_class_name = yg.CLASS_MAP[np.argmax(pred_classes)]
            highest_prob_class_amt = pred_classes[np.argmax(pred_classes)]
            if highest_prob_class_amt < class_thresh:
                continue

            pred_bboxes = output_arr[row, col, yg.PRED_CONFIDENCE_INDEX_END:].reshape((yg.NUM_BOUNDING_BOXES, 4))
            for bbox_ind in range(yg.NUM_BOUNDING_BOXES):
                if pred_confidences[bbox_ind] < conf_thresh:
                    continue
                x_rel = (col+pred_bboxes[bbox_ind][0]) / yg.GRID_W
                y_rel = (row+pred_bboxes[bbox_ind][1]) / yg.GRID_H
                w_rel = pred_bboxes[bbox_ind][2]
                h_rel = pred_bboxes[bbox_ind][3]

                color = choice(colors)
                plt.plot([x_rel * img_w], [y_rel * img_h], marker="x", markersize=4, color=color)
                anchor_xy = ((x_rel - w_rel / 2) * img_w, (y_rel - h_rel / 2) * img_h)
                # Anchor point is the bottom left of the rect
                rect = Rectangle(anchor_xy, w_rel * img_w, h_rel * img_h, linewidth=2.5, edgecolor=color,
                                 facecolor='none')
                plt.gca().add_patch(rect)
                # Anchor point seems to be assuming 0,0 is the top left
                text_anchor_xy = ((x_rel - w_rel / 2) * img_w, ((y_rel - h_rel / 2) * img_h) + 5)
                annotation = highest_prob_class_name + ": " + str(highest_prob_class_amt) + \
                    "\nObj conf: " + str(pred_confidences[bbox_ind])
                plt.annotate(annotation, text_anchor_xy)
    plt.show()


def img_and_label_plot(img_path, squish=False):
    try:
        img = cv.imread(img_path)
    except FileNotFoundError:
        print("Could not find image file", img_path)
        return None

    # CV loads in BGR order, want RGB
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img_h, img_w = img.shape[0:2]
    if squish:
        img = cv.resize(img, (yg.IMG_H, yg.IMG_W))
        img_h = yg.IMG_H
        img_w = yg.IMG_W

    annotations_path = os.path.splitext(img_path)[0] + ".txt"
    try:
        f = open(annotations_path)
        label_txt = f.readlines()
        f.close()
    except FileNotFoundError:
        print("Oops, no text")
        return

    plt.imshow(img)
    y_spacing = img_h / yg.GRID_H
    x_spacing = img_w / yg.GRID_W
    for i in range(1, yg.GRID_H):
        plt.axhline(y=i*y_spacing, color="k", linestyle="--", alpha=0.2)
    for i in range(1, yg.GRID_W):
        plt.axvline(x=i*x_spacing, color="k", linestyle="--", alpha=0.2)

    for an in label_txt:
        info = [float(el) for el in an.strip().split(" ")]
        cls = info[0]
        x_rel, y_rel = info[1], info[2]
        w_rel, h_rel = info[3], info[4]
        color = choice(colors)
        plt.plot([x_rel*img_w], [y_rel*img_h], marker="x", markersize=4, color=color)
        anchor_xy = ((x_rel - w_rel/2)*img_w,  (y_rel - h_rel/2)*img_h)
        # Anchor point is the bottom left of the rect
        rect = Rectangle(anchor_xy, w_rel*img_w, h_rel*img_h, linewidth=2.5, edgecolor=color, facecolor='none')
        plt.gca().add_patch(rect)
        # Anchor point seems to be assuming 0,0 is the top left
        text_anchor_xy = ((x_rel - w_rel/2)*img_w,  ((y_rel - h_rel/2)*img_h)+5)
        plt.annotate(yg.CLASS_MAP[cls], text_anchor_xy)
    plt.show()


img_and_label_plot(".\\img\\mcr_mahjong_trainer_297.png", squish=True)







