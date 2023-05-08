from yolo_model import *
from input_output_utils import *
import os

if __name__ == "__main__":
    good, max_objs, max_objs_file, grid_row, grid_col = check_if_grid_size_and_bbox_num_large_enough()
    if not good:
        print("Not enough bboxes to hold", str(max_objs), "objects in cell", str(grid_row),
              "x", str(grid_col), "in file", max_objs_file)
        img_and_label_plot(os.path.splitext(max_objs_file)[0]+yg.IMG_FILETYPE, highlight_cell=(grid_row, grid_col))
    else:
        train_model(test_after=True, output_json=True)
        #pred = predict_on_img(".\\img\\mcr_mahjong_trainer_53.png")
        #draw_pred_output_and_plot(".\\img\\mcr_mahjong_trainer_53.png", pred, class_thresh=0.03, conf_thresh=0)
