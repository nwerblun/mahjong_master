IMG_W = 416
IMG_H = 416
GRID_W = 13
GRID_H = 13
NUM_CLASSES = 35
NUM_BOUNDING_BOXES = 4
# Each bounding box has [x,y,w,h,confidence]
# Total of bounding_boxes * 5 + num_classes = 55
YOLO_OUTPUT_SHAPE = (13, 13, ((NUM_BOUNDING_BOXES * 5) + NUM_CLASSES))
YOLO_TOTAL_DIMENSIONS = YOLO_OUTPUT_SHAPE[0] * YOLO_OUTPUT_SHAPE[1] * YOLO_OUTPUT_SHAPE[2]
LABEL_CLASS_INDEX_START = 0
LABEL_CLASS_INDEX_END = 35  # Assuming it will be used in slicing, so it will be exclusive on the end
LABEL_BBOX_INDEX_START = 35
LABEL_BBOX_INDEX_END = 39  # Assuming it will be used in slicing, so it will be exclusive on the end
LABEL_CONFIDENCE_INDEX = 40

PRED_CLASS_INDEX_START = 0
PRED_CLASS_INDEX_END = 35  # Assuming it will be used in slicing, so it will be exclusive on the end
PRED_CONFIDENCE_INDEX_START = PRED_CLASS_INDEX_END
PRED_CONFIDENCE_INDEX_END = PRED_CONFIDENCE_INDEX_START + NUM_BOUNDING_BOXES


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
BATCH_SIZE = 32
TRAIN_VAL_TEST_SPLIT_RATIO_TUPLE = 0.8, 0.1, 0.1
GLOBAL_RNG_SEED = 12039
ROOT_DATASET_PATH = "C:\\Users\\NWerblun\\Desktop\\Projects and old school stuff\\" \
                    "mahjong_master\\ml_reinvented_wheel\\img\\"
IMG_FILETYPE = ".png"
LABEL_FILETYPE = ".txt"

NO_OBJ_SCALE = 0.5
BBOX_XY_LOSS_SCALE = 5
BBOX_WH_LOSS_SCALE = 5

NUM_EPOCHS = 135


