import numpy as np


# Calculate the IOU between two boxes, both center points are (0, 0).
# The shape of anchors : [1, 9, 2]
# The shape of boxes : [N, 1, 2]
class IOU():
    def __init__(self, anchors, boxes):
        super(IOU, self).__init__()
        self.anchor_max = anchors / 2
        self.anchor_min = - self.anchor_max
        self.box_max = boxes / 2
        self.box_min = - self.box_max
        self.anchor_area = anchors[..., 0] * anchors[..., 1]
        self.box_area = boxes[..., 0] * boxes[..., 1]

    def calculate_iou(self):
        interset_min = np.maximum(self.box_min, self.anchor_min)
        interset_max = np.minimum(self.box_max, self.anchor_max)
        interset_wh = np.maximum(interset_max - interset_min, 0.0)
        interset_area = interset_wh[..., 0] * interset_wh[..., 1]    # w * h
        union_area = self.anchor_area + self.box_area - interset_area
        iou = interset_area / union_area  # shape : [N, 9]

        return iou


