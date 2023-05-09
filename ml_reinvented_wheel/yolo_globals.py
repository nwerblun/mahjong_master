import numpy as np

IMG_W = 416*2
IMG_H = 416*2
GRID_W = 13
GRID_H = 13
# Class 36 is Z1. It is not currently used. Maybe remove?
NUM_CLASSES = 36
# I chose 6 because that's how many objects can appear in one 13x13 grid cell
NUM_ANCHOR_BOXES = 6
# Obtained from kmeans in kmeans_for_anchor_boxes.py
# Shape is (NBOXES, 2)
# Format is Width, Height in % of total image size
ANCHOR_BOXES = np.array([
 [0.03242392, 0.04467245],
 [0.02949143, 0.07183909],
 [0.02436721, 0.06027983],
 [0.03933732, 0.05368105],
 [0.02738487, 0.03401295],
 [0.03639444, 0.09813662]
])
ANCHOR_BOXES_GRID_UNITS = np.array([[x*GRID_W, y*GRID_H] for x, y in ANCHOR_BOXES])

# Each bounding box has [x,y,w,h,confidence]
YOLO_OUTPUT_SHAPE = (13, 13, NUM_ANCHOR_BOXES, 5 + NUM_CLASSES)
YOLO_TOTAL_DIMENSIONS = YOLO_OUTPUT_SHAPE[0] * YOLO_OUTPUT_SHAPE[1] * YOLO_OUTPUT_SHAPE[2] * YOLO_OUTPUT_SHAPE[3]

LABEL_BBOX_INDEX_START = 0
LABEL_BBOX_INDEX_END = 4  # Assuming it will be used in slicing, so it will be exclusive on the end
LABEL_CONFIDENCE_INDEX = LABEL_BBOX_INDEX_END
LABEL_CLASS_INDEX_START = LABEL_CONFIDENCE_INDEX + 1
LABEL_CLASS_INDEX_END = LABEL_CLASS_INDEX_START + NUM_CLASSES

PRED_BBOX_INDEX_START = 0
PRED_BBOX_INDEX_END = 4  # Assuming it will be used in slicing, so it will be exclusive on the end
PRED_CONFIDENCE_INDEX = PRED_BBOX_INDEX_END
PRED_CLASS_INDEX_START = PRED_CONFIDENCE_INDEX + 1
PRED_CLASS_INDEX_END = PRED_CLASS_INDEX_START + NUM_CLASSES


# No reason for being 0+9, it was just easier to copy-paste
CLASS_MAP = {
    0: "b1",
    1: "b2",
    2: "b3",
    3: "b4",
    4: "b5",
    5: "b6",
    6: "b7",
    7: "b8",
    8: "b9",
    0+9: "c1",
    1+9: "c2",
    2+9: "c3",
    3+9: "c4",
    4+9: "c5",
    5+9: "c6",
    6+9: "c7",
    7+9: "c8",
    8+9: "c9",
    0+18: "d1",
    1+18: "d2",
    2+18: "d3",
    3+18: "d4",
    4+18: "d5",
    5+18: "d6",
    6+18: "d7",
    7+18: "d8",
    8+18: "d9",
    0+27: "drg",
    1+27: "drr",
    2+27: "drw",
    0+30: "we",
    1+30: "wn",
    2+30: "ws",
    3+30: "ww",
    0+34: "f1",
    0+35: "z1"
}
INVERSE_CLASS_MAP = {
    "b1": 0,
    "b2": 1,
    "b3": 2,
    "b4": 3,
    "b5": 4,
    "b6": 5,
    "b7": 6,
    "b8": 7,
    "b9": 8,
    "c1": 0+9,
    "c2": 1+9,
    "c3": 2+9,
    "c4": 3+9,
    "c5": 4+9,
    "c6": 5+9,
    "c7": 6+9,
    "c8": 7+9,
    "c9": 8+9,
    "d1": 0+18,
    "d2": 1+18,
    "d3": 2+18,
    "d4": 3+18,
    "d5": 4+18,
    "d6": 5+18,
    "d7": 6+18,
    "d8": 7+18,
    "d9": 8+18,
    "drg": 0+27,
    "drr": 1+27,
    "drw": 2+27,
    "we": 0+30,
    "wn": 1+30,
    "ws": 2+30,
    "ww": 3+30,
    "f1": 0+34,
    "z1": 0+35
}
BATCH_SIZE = 8
DS_BUFFER_SIZE = 0
TRAIN_VAL_TEST_SPLIT_RATIO_TUPLE = 0.8, 0.1, 0.1
GLOBAL_RNG_SEED = 12039
ROOT_DATASET_PATH = "C:\\Users\\NWerblun\\Desktop\\Projects and old school stuff\\" \
                    "mahjong_master\\ml_reinvented_wheel\\img\\"
IMG_FILETYPE = ".png"
LABEL_FILETYPE = ".txt"

LAMBDA_NO_OBJECT = 1.0
LAMBDA_OBJECT = 5.0
LAMBDA_COORD = 1.0
LAMBDA_CLASS = 1.0

NUM_EPOCHS = 100


