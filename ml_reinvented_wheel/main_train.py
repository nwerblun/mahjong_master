from yolo_model import *
from input_output_utils import *
import os
from random import randint
from kmeans_for_anchor_boxes import *

analyze_data = False
clean = False
aug = False
check = False
train = False
test_pred = True
plot_random_img_with_labels = False
show_specific_img = False
specific_img = "mcr_mahjong_trainer_567_aug_sl52_su45_aug_zoom0_95_aug_alpha1_04_beta58" + yg.IMG_FILETYPE
if __name__ == "__main__":

    if analyze_data:
        clust = kmeans_with_visual()
        print(clust)
    
    if clean:
        clean_aug_files()
        
    if aug:
        augment_ds_translate(1, override_shift_range=(-215, 215, -75, 30), deadzone=(-50, 50, -15, 8))
        augment_ds_zoom(zoom_override=(0.95, 1.05), ignore_aug=False)
        augment_ds_translate(1, override_shift_range=(-215, 215, -75, 30), deadzone=(-50, 50, -15, 8))
        augment_brightness()

    if check:
        good, fail_files, fail_amts, fail_locs = check_if_grid_size_and_bbox_num_large_enough()
        if not good:
            for i in range(len(fail_files)):
                print("File failed with", ">=", str(fail_amts[i]), "objects in cell", str(fail_locs[i][0]), "x", str(fail_locs[i][1]))
                print(os.path.splitext(fail_files[i])[0])
            print(str(len(fail_files)), "failed files.")
            prompt = input("Delete all?")
            if prompt.lower() == "y":
                for f in fail_files:
                    img_path = os.path.splitext(f)[0] + yg.IMG_FILETYPE
                    os.remove(f)
                    os.remove(img_path)

    if train:
        train_model(test_after=True, output_json=True)

    if test_pred:
        xy, wh, cnf, cls = predict_on_img(".\\img\\mcr_mahjong_trainer_125.png")
        draw_pred_output_and_plot(".\\img\\mcr_mahjong_trainer_125.png", xy, wh, cnf, cls,
                                  class_thresh=0.95, conf_thresh=0.75, unsquish=True)

    if plot_random_img_with_labels:
        root = yg.ROOT_DATASET_PATH
        all_files = os.listdir(root)
        all_img = [i for i in all_files if os.path.splitext(i)[1] == yg.IMG_FILETYPE]
        img_path = all_img[randint(0, len(all_img))]
        print("Displaying", img_path)
        img_and_label_plot(yg.ROOT_DATASET_PATH + img_path)

    if show_specific_img:
        img_and_label_plot(yg.ROOT_DATASET_PATH + specific_img, squish=True)
        img_and_label_plot(yg.ROOT_DATASET_PATH + specific_img, squish=False)


