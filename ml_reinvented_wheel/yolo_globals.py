import numpy as np

# Choose a grid such that max of 2 objects appears in any grid cell (33 for this version)
# Keep doubling that until it reaches an acceptable image size (33 * 16 = 528)
IMG_W = 528
IMG_H = 528
GRID_W = 33
GRID_H = 33
NUM_CLASSES = 35
# Max of two objects per cell, need to check every aspect ratio twice
NUM_ANCHOR_BOXES = 6
# Obtained from kmeans in kmeans_for_anchor_boxes.py
# Shape is (NBOXES, 2)
# Format is Width, Height in % of total image size
ANCHOR_BOXES = np.array([
    # Vertical discarded tiles and vertical revealed tiles have roughly this shape, aspect ratio 0.725
    [0.02632965, 0.06452057],
    [0.02632965, 0.06452057],
    # Horizontal discarded tiles and horizontal revealed tiles have about this shape, aspect ratio 1.3
    [0.03488724, 0.04753794],
    [0.03488724, 0.04753794],
    # Vertical tiles in your hand have this shape, aspect ratio 0.66
    [0.03629786, 0.0973151],
    [0.03629786, 0.0973151],
])
ANCHOR_BOXES_GRID_UNITS = np.array([[x*GRID_W, y*GRID_H] for x, y in ANCHOR_BOXES], dtype=np.float32)

# Each bounding box has [x,y,w,h,confidence]
YOLO_OUTPUT_SHAPE = (GRID_W, GRID_H, NUM_ANCHOR_BOXES, 5 + NUM_CLASSES)
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
    0+34: "f1"
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
    "f1": 0+34
}
BATCH_SIZE = 2
DS_BUFFER_SIZE = 0
TRAIN_VAL_TEST_SPLIT_RATIO_TUPLE = 0.8, 0.1, 0.1
GLOBAL_RNG_SEED = 4245
ROOT_DATASET_PATH = "C:\\Users\\NWerblun\\Desktop\\Projects and old school stuff\\" \
                    "mahjong_master\\ml_reinvented_wheel\\img\\"
IMG_FILETYPE = ".png"
LABEL_FILETYPE = ".txt"

LAMBDA_NO_OBJECT = 1.0
LAMBDA_OBJECT = 5.0
LAMBDA_COORD = 1.8
LAMBDA_CLASS = 1.5

NUM_EPOCHS = 40

# 0 = no prints
# 1 = print loss values
# 2 = print loss values + examples of guesses + all above
# 3 = print files being tested + all above
DEBUG_PRINT = 0


