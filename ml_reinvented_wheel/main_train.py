from yolo_model import *
from input_output_utils import *

if __name__ == "__main__":
    good, max_objs, max_objs_file = check_if_grid_size_and_bbox_num_large_enough()
    if not good:
        print("Not enough bboxes to hold", str(max_objs), "in file", max_objs_file)
    else:
        train_model(test_after=True, output_json=True)
