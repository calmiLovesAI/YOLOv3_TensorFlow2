import tensorflow as tf
import numpy as np
from configuration import CATEGORY_NUM


class GenerateGroundTruth():
    def __init__(self, true_boxes, input_shape):
        super(GenerateGroundTruth, self).__init__()
        self.true_boxes = true_boxes
        self.input_shape = input_shape
        pass

    def generate_true_boxes(self):
        true_boxes = np.array(self.true_boxes, dtype=np.float32)
        input_shape = np.array(self.input_shape, dtype=np.int32)
        print(true_boxes)
        print(input_shape)
