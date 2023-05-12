import cv2 as cv
import numpy as np
import os
import yolo_globals as yg
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from kmeans_for_anchor_boxes import _iou
# from matplotlib.colors import get_named_colors_mapping
from random import choice, randint, uniform


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


def check_if_grid_size_and_bbox_num_large_enough():
    # Go through the dataset and 'assign' each object to a grid cell.
    # If we run out of bboxes, then raise a warning
    all_files = list(os.listdir(yg.ROOT_DATASET_PATH))
    annotations = [yg.ROOT_DATASET_PATH+f for f in all_files if os.path.splitext(f)[1] == yg.LABEL_FILETYPE]
    fails = []
    fails_objs = []
    fails_loc = []
    for ann in annotations:
        f = open(ann, "r")
        lines = f.readlines()
        lines = [e.strip().split(" ") for e in lines]
        f.close()
        grid_assignments = np.zeros((yg.GRID_H, yg.GRID_W))
        for line in lines:
            # X, Y, W, H is already in units of % of image width/height -> range from 0->1
            split_line = [float(el) for el in line]
            x_center = split_line[1]
            y_center = split_line[2]
            # Will be a number between 0 -> grid size with a fraction representing how far into the cell.
            grid_loc = [yg.GRID_W * x_center, yg.GRID_H * y_center]
            # Y value determines row. Top left of image is 0,0 and y is downwards, x is to the right.
            grid_row = int(grid_loc[1])
            grid_col = int(grid_loc[0])
            grid_assignments[grid_row, grid_col] += 1
            if grid_assignments[grid_row, grid_col] > yg.NUM_ANCHOR_BOXES and (ann not in fails):
                fails += [ann]
                fails_objs += [grid_assignments[grid_row, grid_col]]
                fails_loc += [(grid_row, grid_col)]
    return len(fails) == 0, fails, fails_objs, fails_loc


def _find_best_unused_anchor_box(box_w, box_h, label_entry):
    best_iou = -1
    best_anchor_ind = -1
    for i in range(len(yg.ANCHOR_BOXES)):
        box = np.array([box_w, box_h])
        clust = np.array([yg.ANCHOR_BOXES[i]])
        iou = _iou(box, clust)
        # print("Box of w/h", str(box_w), "x", str(box_h), "iou with anchor", str(i), "=", str(iou))
        if iou > best_iou and not any(label_entry[i]):
            best_iou = iou
            best_anchor_ind = i
    return best_anchor_ind, best_iou


def file_to_img_label(example_tuple):
    img_path, lbl_path = example_tuple[0], example_tuple[1]
    img_path = yg.ROOT_DATASET_PATH + img_path.numpy().decode('utf-8')
    lbl_path = yg.ROOT_DATASET_PATH + lbl_path.numpy().decode('utf-8')
    if yg.DEBUG_PRINT > 2:
        tf.print("Getting file: ", img_path)
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
        cls = int(split_line[0])
        x_center = split_line[1]
        y_center = split_line[2]
        w = split_line[3]
        h = split_line[4]

        # Will be a number between 0 -> grid size with a fraction representing how far into the cell.
        grid_x = yg.GRID_W * x_center
        grid_y = yg.GRID_H * y_center

        # Y value determines row. Top left of image is 0,0 and y is downwards, x is to the right.
        grid_row = int(grid_y)
        grid_col = int(grid_x)

        best_anchor_ind, iou = _find_best_unused_anchor_box(w, h, label_matrix[grid_row, grid_col])
        if best_anchor_ind == -1:
            raise ValueError("Not enough anchor boxes to assign to")
        # print("Best anchor: ", str(best_anchor_ind), " with iou:", str(iou))

        # Convert from % image to % grid
        w *= yg.GRID_W
        h *= yg.GRID_H

        label_matrix[grid_row, grid_col, best_anchor_ind, yg.LABEL_BBOX_INDEX_START:yg.LABEL_BBOX_INDEX_END] = \
            np.array([grid_x, grid_y, w, h])
        label_matrix[grid_row, grid_col, best_anchor_ind, yg.LABEL_CONFIDENCE_INDEX] = 1
        class_lst = [0]*yg.NUM_CLASSES
        class_lst[cls] = 1
        label_matrix[grid_row, grid_col, best_anchor_ind, yg.LABEL_CLASS_INDEX_START:yg.LABEL_CLASS_INDEX_END] = \
            np.array(class_lst)
    return img, label_matrix


def clean_aug_files():
    root = yg.ROOT_DATASET_PATH
    all_files = os.listdir(root)
    all_aug_img = [i for i in all_files if os.path.splitext(i)[1] == yg.IMG_FILETYPE
                   and "_aug_" in os.path.splitext(i)[0]]
    all_aug_labels = [i for i in all_files if os.path.splitext(i)[1] == yg.LABEL_FILETYPE
                      and "_aug_" in os.path.splitext(i)[0]]

    if len(all_aug_img) == 0 and len(all_aug_labels) == 0:
        print("No files to clean up.")
        return

    print("These files will be deleted")
    for f in all_aug_labels:
        print(yg.ROOT_DATASET_PATH + f)
    for f in all_aug_img:
        print(yg.ROOT_DATASET_PATH + f)

    prompt = input("If this is ok, type Y/y. Enter N/n or any other character to cancel.")
    if prompt.lower() != "y":
        return

    for f in all_aug_labels:
        os.remove(yg.ROOT_DATASET_PATH + f)
    for f in all_aug_img:
        os.remove(yg.ROOT_DATASET_PATH + f)


def augment_ds_zoom(passes=1, zoom_override=None, deadzone=None):
    root = yg.ROOT_DATASET_PATH
    all_files = os.listdir(root)
    all_img = [i for i in all_files if os.path.splitext(i)[1] == yg.IMG_FILETYPE
               and "_aug_" not in os.path.splitext(i)[0]]
    all_labels = [i for i in all_files if os.path.splitext(i)[1] == yg.LABEL_FILETYPE
                  and "_aug_" not in os.path.splitext(i)[0]]
    assert len(all_img) == len(all_labels)
    assert not any([("_aug_" in i) for i in all_img])
    assert not any([("_aug_" in i) for i in all_labels])

    for iter in range(passes):
        print("Zoom augmentations pass #", str(iter), ". Augmenting", str(len(all_img)), "images...")
        for img_path, lbl_path in zip(all_img, all_labels):
            f = open(yg.ROOT_DATASET_PATH + lbl_path, "r")
            label_txt = f.readlines()
            f.close()

            img = cv.imread(yg.ROOT_DATASET_PATH + img_path)
            img_h, img_w = img.shape[0:2]
            
            zoom_factor = 1.0
            if deadzone is None:
                deadzone = (1.0, 1.0)
                
            while (deadzone[0] <= zoom_factor <= deadzone[1]):
                if zoom_override is not None:
                    zoom_factor = uniform(zoom_override[0], zoom_override[1])
                else:
                    zoom_factor = uniform(0.92, 1.08)

            translate = np.eye(3)
            translate[0:2, 2] = [-img_w/2, -img_h/2]  # Zoom to this pixel by shifting it to 0,0

            zoom = np.diag([zoom_factor, zoom_factor, 1])  # Apply scaling around 0,0

            inverse_translate = np.eye(3)
            inverse_translate[0:2, 2] = [(img_w-1)/2, (img_h-1)/2]  # shift this point back to its original position

            H = inverse_translate @ zoom @ translate  # Overall transformation matrix
            M = H[0:2]  # CV2 uses 2x3 matrix of the form M = [T  B] where T is 2x2 and B is 2x1 to compute y = Tx + B

            # Warp img using the same matrix
            new_img = cv.warpAffine(img, dsize=(img_w, img_h), M=M, flags=cv.INTER_NEAREST)

            new_anns = []
            for ann in label_txt:
                split_line = ann.strip().split(" ")
                old_lbl_x, old_lbl_y = float(split_line[1]), float(split_line[2])
                old_lbl_w, old_lbl_h = float(split_line[3]), float(split_line[4])
                # Convert label positions to new coords
                # Convert to absolute coords instead of % and vectorize. The 1 is so we have matching dims.
                old_pos_vec = np.array([old_lbl_x * img_w, old_lbl_y * img_h, 1]).reshape((3, 1))
                # Apply transformation
                new_pos_vec = M.dot(old_pos_vec).reshape((2,))
                # W/H is just scaled by the zoom factor
                new_lbl_w, new_lbl_h = old_lbl_w * zoom_factor, old_lbl_h * zoom_factor
                # Rescale coords back to % of image before writing to file.
                new_ann = " ".join([split_line[0],
                                    str(new_pos_vec[0]/img_w),
                                    str(new_pos_vec[1]/img_h),
                                    str(new_lbl_w),
                                    str(new_lbl_h)])
                new_anns += [new_ann]

            aug = "_aug_zoom" + "{:.2f}".format(zoom_factor).replace(".", "_")
            cv.imwrite(yg.ROOT_DATASET_PATH + os.path.splitext(img_path)[0] + aug + yg.IMG_FILETYPE, new_img)
            f = open(yg.ROOT_DATASET_PATH + os.path.splitext(lbl_path)[0] + aug + yg.LABEL_FILETYPE, "w")
            for ann in new_anns:
                f.write(ann + "\n")
            f.close()


def augment_ds_translate(passes=1, override_shift_range=None, deadzone=None):
    root = yg.ROOT_DATASET_PATH
    all_files = os.listdir(root)
    all_img = [i for i in all_files if os.path.splitext(i)[1] == yg.IMG_FILETYPE
               and "_aug_" not in os.path.splitext(i)[0]]
    all_labels = [i for i in all_files if os.path.splitext(i)[1] == yg.LABEL_FILETYPE
                  and "_aug_" not in os.path.splitext(i)[0]]
    assert len(all_img) == len(all_labels)
    assert not any([("_aug_" in i) for i in all_img])
    assert not any([("_aug_" in i) for i in all_labels])

    for iter in range(passes):
        print("Translation augmentations pass #", str(iter), ". Augmenting", str(len(all_img)), "images...")
        for img_path, lbl_path in zip(all_img, all_labels):
            f = open(yg.ROOT_DATASET_PATH + lbl_path, "r")
            label_txt = f.readlines()
            f.close()

            img = cv.imread(yg.ROOT_DATASET_PATH + img_path)
            img_h, img_w = img.shape[0:2]

            if override_shift_range is not None:
                shift_right_left_min = override_shift_range[0]
                shift_right_left_max = override_shift_range[1]
                shift_up_down_min = override_shift_range[2]
                shift_up_down_max = override_shift_range[3]
            else:
                shift_right_left_min = -350
                shift_right_left_max = 350
                shift_up_down_min = -120
                shift_up_down_max = 50
                
            if deadzone is None:
                deadzone = (0, 0, 0, 0)
                
            while deadzone[0] <= shift_rl_amt <= deadzone[1]:
                shift_rl_amt = randint(shift_right_left_min, shift_right_left_max)
            
            while deadzone[2] <= shift_ud_amt <= deadzone[3]:
                shift_ud_amt = randint(shift_up_down_min, shift_up_down_max)
                
            translation_matrix = np.float32([[1, 0, shift_rl_amt], [0, 1, shift_ud_amt]])
            new_img = cv.warpAffine(img, translation_matrix, (img_w, img_h))
            new_anns = []
            for ann in label_txt:
                split_line = ann.strip().split(" ")
                old_x, old_y = float(split_line[1]), float(split_line[2])
                new_x, new_y = (old_x + shift_rl_amt/img_w), (old_y + shift_ud_amt/img_h)
                new_line = " ".join([split_line[0], str(new_x), str(new_y), split_line[3], split_line[4]])
                new_anns += [new_line]

            srl_aug = "sr" + str(shift_rl_amt) if shift_rl_amt >= 0 else "sl" + str(shift_rl_amt)[1:]
            sud_aug = "sd" + str(shift_ud_amt) if shift_ud_amt >= 0 else "su" + str(shift_ud_amt)[1:]
            aug = "_aug_" + srl_aug + "_" + sud_aug
            f = open(yg.ROOT_DATASET_PATH + os.path.splitext(lbl_path)[0] + aug + yg.LABEL_FILETYPE, "w")
            for ann in new_anns:
                f.write(ann + "\n")
            f.close()
            cv.imwrite(yg.ROOT_DATASET_PATH + os.path.splitext(img_path)[0] + aug + yg.IMG_FILETYPE, new_img)


def get_datasets():
    root = yg.ROOT_DATASET_PATH
    all_files = os.listdir(root)
    all_img = [i for i in all_files if os.path.splitext(i)[1] == yg.IMG_FILETYPE]
    all_labels = [i for i in all_files if os.path.splitext(i)[1] == yg.LABEL_FILETYPE]
    assert len(all_img) == len(all_labels) and\
           all([os.path.splitext(all_img[i])[0] == os.path.splitext(all_labels[i])[0] for i in range(len(all_img))])

    train_split, val_split, test_split = yg.TRAIN_VAL_TEST_SPLIT_RATIO_TUPLE

    all_examples = list(zip(all_img, all_labels))
    rng = np.random.RandomState(yg.GLOBAL_RNG_SEED)
    rng.shuffle(all_examples)
    total_len = len(all_examples)
    train_end_ind = int(total_len * train_split)
    valid_end_ind = train_end_ind + int(total_len * val_split)

    train_examples = all_examples[:train_end_ind]
    valid_examples = all_examples[train_end_ind:valid_end_ind]
    test_examples = all_examples[valid_end_ind:]

    num_train_examples = len(train_examples)
    num_valid_examples = len(valid_examples)
    num_test_examples = len(test_examples)

    train_ds = tf.data.Dataset.from_tensor_slices(train_examples)
    train_ds = train_ds.map(lambda x: tf.py_function(file_to_img_label, inp=[x], Tout=[tf.float32, tf.float32]))

    valid_ds = tf.data.Dataset.from_tensor_slices(valid_examples)
    valid_ds = valid_ds.map(lambda x: tf.py_function(file_to_img_label, inp=[x], Tout=[tf.float32, tf.float32]))

    test_ds = tf.data.Dataset.from_tensor_slices(test_examples)
    test_ds = test_ds.map(lambda x: tf.py_function(file_to_img_label, inp=[x], Tout=[tf.float32, tf.float32]))

    # Batching takes too much memory I think. files are already shuffled, so I guess we are just going in.
    # Mem of params = # trainable params * 4 bytes / param (assuming float32)
    #   for 31 million params, this is ~124MB
    # Training mem = mem of params * 3 (once for forward/back prop + other stuff)
    #   Total ~370MB
    # For each layer, 4 bytes * (shape[0]*shape[1]*...)
    #   For a conv layer of 416x416x64 this is 1.2MB
    # Take the total of all layer output sizes and * batch size to get memory of a forward pass
    if yg.DS_BUFFER_SIZE > 0:
        train_ds = train_ds.shuffle(buffer_size=yg.DS_BUFFER_SIZE,
                                    seed=yg.GLOBAL_RNG_SEED,
                                    reshuffle_each_iteration=True).repeat().batch(yg.BATCH_SIZE)
        valid_ds = valid_ds.shuffle(buffer_size=yg.DS_BUFFER_SIZE,
                                    seed=yg.GLOBAL_RNG_SEED*4,
                                    reshuffle_each_iteration=True).repeat().batch(
            max(int(yg.BATCH_SIZE * val_split), 1)
        )
        test_ds = test_ds.shuffle(buffer_size=yg.DS_BUFFER_SIZE,
                                  seed=yg.GLOBAL_RNG_SEED*6,
                                  reshuffle_each_iteration=True).repeat().batch(
            max(int(yg.BATCH_SIZE * test_split), 1)
    )
    else:
        train_ds = train_ds.repeat().batch(yg.BATCH_SIZE)
        valid_ds = valid_ds.repeat().batch(max(int(yg.BATCH_SIZE * val_split), 1))
        test_ds = test_ds.repeat().batch(max(int(yg.BATCH_SIZE * test_split), 1))

    # train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    # valid_ds = valid_ds.prefetch(tf.data.AUTOTUNE)
    return train_ds, valid_ds, test_ds, num_train_examples, num_valid_examples, num_test_examples


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


def draw_pred_output_and_plot(img_path, y_pred_xy, y_pred_wh, y_pred_confs, y_pred_classes, class_thresh=0.7, conf_thresh=0.6, unsquish=True):
    # TODO: Implement NMS
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

    bboxes = []
    class_probs = []
    class_names = []
    confs = []
    num_over_conf_but_not_class = 0
    num_over_class_but_not_conf = 0
    for row in range(yg.GRID_H):
        for col in range(yg.GRID_W):
            for bbox in range(yg.NUM_ANCHOR_BOXES):
                curr_conf = y_pred_confs[row, col, bbox]
                print("\nBBox", str(bbox), "in row x col", str(row), "x", str(col), "Confidence:", str(curr_conf))
                class_argmax_ind = np.argmax(y_pred_classes[row, col, bbox])
                highest_prob_class_name = yg.CLASS_MAP[class_argmax_ind]
                highest_prob_class_amt = y_pred_classes[row, col, bbox, class_argmax_ind]
                print("Most likely class is", str(highest_prob_class_name), "with prob.", str(highest_prob_class_amt))
                if highest_prob_class_amt < class_thresh and curr_conf >= conf_thresh:
                    num_over_conf_but_not_class += 1
                elif highest_prob_class_amt >= class_thresh and curr_conf < conf_thresh:
                    num_over_class_but_not_conf += 1
                if highest_prob_class_amt < class_thresh or curr_conf < conf_thresh:
                    continue

                pred_bbox = np.hstack((y_pred_xy[row, col, bbox], y_pred_wh[row, col, bbox]))
                bboxes += [pred_bbox]
                class_probs += [highest_prob_class_amt]
                class_names += [highest_prob_class_name]
                confs += [curr_conf]
    print("Found ", str(len(bboxes)), " bboxes over both thresholds")
    print("Found ", str(num_over_class_but_not_conf), " bboxes over class threshold but not confidence threshold")
    print("Found ", str(num_over_conf_but_not_class), " bboxes over confidence threshold but not class threshold")

    for ind, bbox in enumerate(bboxes):
        x_rel = bbox[0]
        y_rel = bbox[1]
        w_rel = bbox[2]
        h_rel = bbox[3]

        color = choice(colors)
        plt.plot([x_rel * img_w], [y_rel * img_h], marker="x", markersize=4, color=color)
        anchor_xy = ((x_rel - w_rel / 2) * img_w, (y_rel - h_rel / 2) * img_h)
        # Anchor point is the bottom left of the rect
        rect = Rectangle(anchor_xy, w_rel * img_w, h_rel * img_h, linewidth=2.5, edgecolor=color,
                         facecolor='none')
        plt.gca().add_patch(rect)
        # Anchor point seems to be assuming 0,0 is the top left
        text_anchor_xy = ((x_rel - w_rel / 2) * img_w, ((y_rel - h_rel / 2) * img_h) + 5)
        annotation = class_names[ind] + ": " + str(class_probs[ind]) + "\nObj conf: " + str(confs[ind])
        plt.annotate(annotation, text_anchor_xy)
    plt.show()


def img_and_label_plot(img_path, squish=False, highlight_cell=None):
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
    if highlight_cell is not None:
        grid_anchor_x = highlight_cell[1] * x_spacing
        grid_anchor_y = highlight_cell[0] * y_spacing
        rect = Rectangle((grid_anchor_x, grid_anchor_y), x_spacing, y_spacing, linewidth=5,
                         edgecolor="pink", facecolor='none')
        plt.gca().add_patch(rect)
        plt.annotate("Highlighted Cell", (grid_anchor_x, grid_anchor_y), xytext=(0, 0), arrowprops=
            {
                "headwidth": 10,
                "headlength": 10
            })
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
