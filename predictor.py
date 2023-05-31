import numpy as np
from yolo_model import predict_on_img_obj, load_model
from input_output_utils import get_pred_output_img
import yolo_globals as yg
from multiprocessing import Pipe, Process


class Predictor:
    def __init__(self, pipe):
        self.model = load_model(True)
        self.pipe = pipe

    def start(self):
        while True:
            if self.pipe.poll():
                from_parent = self.pipe.recv()
                request, args = from_parent
                if request == "predict":
                    self.img_to_prediction(args)
                elif request == "kill":
                    self.pipe.close()
                    return

    def img_to_prediction(self, img):
        res = predict_on_img_obj(img, True, self.model)
        xy, wh, conf, cls = res
        self.pred_to_processed_img_and_nms_output(img, xy, wh, conf, cls)
        return

    def pred_to_processed_img_and_nms_output(self, img, xy, wh, conf, cls,
                                             cls_thr=0.95, conf_thr=0.85, nms_iou_thr=0.35):
        res = get_pred_output_img(img, xy, wh, conf, cls,
                                  class_thresh=cls_thr,
                                  conf_thresh=conf_thr,
                                  nms_iou_thresh=nms_iou_thr)
        self.pipe.send(res)
        return


def start_predicting(pipe):
    p = Predictor(pipe)
    p.start()
    return

