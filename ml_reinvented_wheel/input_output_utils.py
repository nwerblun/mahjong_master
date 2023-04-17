import cv2 as cv
import numpy as np
import os
import yolo_globals as yg
import tensorflow as tf


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
        split_line = line.split(" ")
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
    return tf.Tensor(img / 255.)


def draw_pred_output_and_plot(output_tensor):
    pass







