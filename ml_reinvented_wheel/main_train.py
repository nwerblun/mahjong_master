from yolo_model import *
from input_output_utils import *
import os
from kmeans_for_anchor_boxes import *


def analyze_data():
    clust = kmeans_with_visual()
    print(clust)


def clean():
    clean_aug_files()


def aug():
    augment_ds_translate(1, override_shift_range=(-215, 215, -75, 30), deadzone=(-50, 50, -15, 8))
    augment_ds_zoom(zoom_override=(0.95, 1.05), ignore_aug=False)
    augment_ds_translate(1, override_shift_range=(-215, 215, -75, 30), deadzone=(-50, 50, -15, 8))
    augment_brightness()


def check():
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


def train():
    train_model(test_after=True, output_json=True)


def test_pred(pred_img, from_json):
    xy, wh, cnf, cls = predict_on_img(pred_img, from_json)
    draw_pred_output_and_plot(pred_img, xy, wh, cnf, cls,
                              class_thresh=0.95, conf_thresh=0.75, unsquish=True)


def plot_specific_img_with_labels(path, squish, highlight_cell):
    img_and_label_plot(path, squish=squish, highlight_cell=highlight_cell)


if __name__ == "__main__":

    # Plot w/h k-means clusters
    # analyze_data()

    # Remove augmented files
    # clean()

    # Create augmented files
    # aug()

    # Verify grid cell size / num bboxes are appropriate and count max assignments per cell
    # check()

    # Create model and run fit
    # train()

    # Plot network output prediction on an image
    test_pred(".\\img\\mcr_mahjong_trainer_146.png", True)

    # Take an image and plot its labels (true labels, not predicted)
    specific_img = r"C:\Users\NWerblun\Desktop\Projects and old school stuff\mahjong_master\ml_reinvented_wheel\img\mcr_mahjong_trainer_54.png"
    # plot_specific_img_with_labels(specific_img, False, None)

        












