from yolo_model import *
from input_output_utils import *
import os
from random import randint

aug = False
check = True
train = True
test_pred = False
plot_test = False
view_files = False
if __name__ == "__main__":
    if aug:
        clean_aug_files()
        augment_ds_zoom(zoom_override=(0.98, 1.02))
        #augment_ds_translate(override_shift_range=(-150, 150, -80, 25))

    if check:
        good, fail_files, fail_amts, fail_locs = check_if_grid_size_and_bbox_num_large_enough()
        if not good:
            for i in range(len(fail_files)):
                print("File failed with", str(fail_amts[i]), ">= objects in cell", str(fail_locs[i][0]), "x", str(fail_locs[i][1]))
                print(os.path.splitext(fail_files[i])[0])
            print(str(len(fail_files)), "failed files.")
            prompt = input("Delete all?")
            if prompt.lower() == "y":
                for f in fail_files:
                    img_path = os.path.splitext(f)[0] + yg.IMG_FILETYPE
                    os.remove(f)
                    os.remove(img_path)
            # img_and_label_plot(os.path.splitext(max_objs_file)[0]+yg.IMG_FILETYPE, highlight_cell=(grid_row, grid_col))

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

    if view_files:
        files_to_view = [
             r"C:\Users\NWerblun\Desktop\Projects and old school stuff\mahjong_master\ml_reinvented_wheel\img\mcr_mahjong_trainer_113_aug_sl12_su14.png",
             r"C:\Users\NWerblun\Desktop\Projects and old school stuff\mahjong_master\ml_reinvented_wheel\img\mcr_mahjong_trainer_271.png",
             r"C:\Users\NWerblun\Desktop\Projects and old school stuff\mahjong_master\ml_reinvented_wheel\img\mcr_mahjong_trainer_296.png",
             r"C:\Users\NWerblun\Desktop\Projects and old school stuff\mahjong_master\ml_reinvented_wheel\img\mcr_mahjong_trainer_290_aug_sr94_sd2.png",
             r"C:\Users\NWerblun\Desktop\Projects and old school stuff\mahjong_master\ml_reinvented_wheel\img\mcr_mahjong_trainer_246_aug_sl110_su56.png",
             r"C:\Users\NWerblun\Desktop\Projects and old school stuff\mahjong_master\ml_reinvented_wheel\img\mcr_mahjong_trainer_224_aug_zoom1_00.png",
             r"C:\Users\NWerblun\Desktop\Projects and old school stuff\mahjong_master\ml_reinvented_wheel\img\mcr_mahjong_trainer_130_aug_zoom1_01.png",
             r"C:\Users\NWerblun\Desktop\Projects and old school stuff\mahjong_master\ml_reinvented_wheel\img\mcr_mahjong_trainer_156_aug_sr124_su43.png",
             r"C:\Users\NWerblun\Desktop\Projects and old school stuff\mahjong_master\ml_reinvented_wheel\img\mahjongtime_lite_23.png",
             r"C:\Users\NWerblun\Desktop\Projects and old school stuff\mahjong_master\ml_reinvented_wheel\img\mcr_mahjong_trainer_256_aug_sl24_su45.png",
             r"C:\Users\NWerblun\Desktop\Projects and old school stuff\mahjong_master\ml_reinvented_wheel\img\mcr_mahjong_trainer_122.png",
             r"C:\Users\NWerblun\Desktop\Projects and old school stuff\mahjong_master\ml_reinvented_wheel\img\mcr_mahjong_trainer_230.png",
         ]
        for f in files_to_view:
            img_and_label_plot(f, squish=True)
