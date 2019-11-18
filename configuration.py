# training
EPOCHS = 50
BATCH_SIZE = 8

# input image
IMAGE_HEIGHT = 416
IMAGE_WIDTH = 416
CHANNELS = 3

# Dataset
CATEGORY_NUM = 20
ANCHOR_NUM_EACH_SCALE = 3
COCO_ANCHORS = [[116, 90], [156, 198], [373, 326], [30, 61], [62, 45], [59, 119], [10, 13], [16, 30], [33, 23]]
COCO_ANCHOR_INDEX = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
SCALE_SIZE = [13, 26, 52]


PASCAL_VOC_DIR = "./dataset/VOCdevkit/VOC2012/"
# The 20 object classes of PASCAL VOC
PASCAL_VOC_CLASSES = {"person": 1, "bird": 2, "cat": 3, "cow": 4, "dog": 5,
                      "horse": 6, "sheep": 7, "aeroplane": 8, "bicycle": 9,
                      "boat": 10, "bus": 11, "car": 12, "motorbike": 13,
                      "train": 14, "bottle": 15, "chair": 16, "diningtable": 17,
                      "pottedplant": 18, "sofa": 19, "tvmonitor": 20}
TXT_DIR = "./data_process/data.txt"


train_ratio = 0.8

# loss
IGNORE_THRESHOLD = 0.5


# NMS
CONFIDENCE_THRESHOLD = 0.6
IOU_THRESHOLD = 0.5
MAX_BOX_NUM = 50

MAX_TRUE_BOX_NUM_PER_IMG = 20


# save model
save_model_dir = "saved_model/"
save_frequency = 5

test_picture_dir = "./test_data/1.jpg"
test_video_dir = ""
