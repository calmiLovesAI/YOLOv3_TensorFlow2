import tensorflow as tf
from yolo.bounding_box import bounding_box_predict
from configuration import IMAGE_WIDTH, IMAGE_HEIGHT, CATEGORY_NUM, SCALE_SIZE
from utils.nms import NMS


class Inference():
    def __init__(self, yolo_output, input_image_shape):
        super(Inference, self).__init__()
        self.yolo_output = yolo_output
        self.input_image_h = input_image_shape.shape[0]
        self.input_image_w = input_image_shape.shape[1]

    def __yolo_post_processing(self, feature, scale_type):
        box_xy, box_wh, confidence, class_prob = bounding_box_predict(feature_map=feature,
                                                                      scale_type=scale_type,
                                                                      is_training=False)
        boxes = self.__boxes_to_original_image(box_xy, box_wh)
        boxes = tf.reshape(boxes, shape=(-1, 4))
        box_scores = confidence * class_prob
        box_scores = tf.reshape(box_scores, shape=(-1, CATEGORY_NUM))
        return boxes, box_scores

    def __boxes_to_original_image(self, box_xy, box_wh):
        x_scale = IMAGE_WIDTH / self.input_image_w
        y_scale = IMAGE_HEIGHT / self.input_image_h
        x = box_xy[..., 0] / x_scale
        y = box_xy[..., 1] / y_scale
        w = box_wh[..., 0] / x_scale
        h = box_wh[..., 1] / y_scale
        xmin = x - w / 2
        ymin = y - h / 2
        xmax = x + w / 2
        ymax = y + h / 2
        boxes = tf.concat(values=[xmin, ymin, xmax, ymax], axis=-1)
        return boxes

    def get_final_boxes(self):
        boxes_list = []
        box_scores_list = []
        for i in range(len(SCALE_SIZE)):
            boxes, box_scores = self.__yolo_post_processing(feature=self.yolo_output[i],
                                                            scale_type=i)
            boxes_list.append(boxes)
            box_scores_list.append(box_scores)
        boxes_array = tf.concat(boxes_list, axis=0)
        box_scores_array = tf.concat(box_scores_list, axis=0)
        return NMS().nms(boxes=boxes_array, box_scores=box_scores_array)

