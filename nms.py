import tensorflow as tf
from configuration import IOU_THRESHOLD, CONFIDENCE_THRESHOLD, MAX_BOX_NUM


class NMS():
    def __init__(self, prediction, max_box_num):
        super(NMS, self).__init__()
        self.prediction = prediction
        self.max_box_num = max_box_num

    def get_box_coordinate(self):
        center_x, center_y, width, height, scores = tf.split(self.prediction,
                                                             num_or_size_splits=[1, 1, 1, 1, 1],
                                                             axis=-1)
        x1 = center_x - height / 2
        y1 = center_y - width / 2
        x2 = center_x + height / 2
        y2 = center_y + width / 2
        boxes = tf.concat(values=[x1, y1, x2, y2], axis=-1)
        return boxes, scores

    def nms(self):
        boxes, scores = self.get_box_coordinate()
        selected_indices = tf.image.non_max_suppression(boxes=boxes,
                                                        scores=scores,
                                                        max_output_size=MAX_BOX_NUM,
                                                        iou_threshold=IOU_THRESHOLD,
                                                        score_threshold=CONFIDENCE_THRESHOLD)
        selected_boxes = tf.gather(self.prediction, selected_indices)
        return selected_boxes
