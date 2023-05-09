from yolo_model import *
from input_output_utils import *
import os
from random import randint

aug = False
check = True
train = True
test_pred = False
plot_test = False
if __name__ == "__main__":
    if aug:
        clean_aug_files()
        augment_ds_zoom()
        augment_ds_translate(override_shift_range=(-270, 270, -100, 40))

    if check:
        good, max_objs, max_objs_file, grid_row, grid_col = check_if_grid_size_and_bbox_num_large_enough()
        if not good:
            print("Not enough bboxes to hold", str(max_objs), "objects in cell", str(grid_row),
                  "x", str(grid_col), "in file", max_objs_file)
            img_and_label_plot(os.path.splitext(max_objs_file)[0]+yg.IMG_FILETYPE, highlight_cell=(grid_row, grid_col))

    if train:
        train_model(test_after=True, output_json=True)

    if test_pred:
        pred = predict_on_img(".\\img\\mcr_mahjong_trainer_53.png")
        draw_pred_output_and_plot(".\\img\\mcr_mahjong_trainer_53.png", pred,
                                  class_thresh=0.65, conf_thresh=0.2, unsquish=True)

    if plot_test:
        root = yg.ROOT_DATASET_PATH
        all_files = os.listdir(root)
        all_img = [i for i in all_files if os.path.splitext(i)[1] == yg.IMG_FILETYPE]
        img_path = all_img[randint(0, len(all_img))]
        print("Displaying", img_path)
        img_and_label_plot(yg.ROOT_DATASET_PATH + img_path)
