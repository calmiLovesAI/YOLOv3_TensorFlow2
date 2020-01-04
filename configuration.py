# training
EPOCHS = 1000
BATCH_SIZE = 8
load_weights_before_training = False
load_weights_from_epoch = 10

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

use_dataset = "pascal_voc"      # "custom", "pascal_voc", "coco"

PASCAL_VOC_DIR = "./dataset/VOCdevkit/VOC2012/"
PASCAL_VOC_ANNOTATION = PASCAL_VOC_DIR + "Annotations"
PASCAL_VOC_IMAGE = PASCAL_VOC_DIR + "JPEGImages"
# The 20 object classes of PASCAL VOC
PASCAL_VOC_CLASSES = {"person": 1, "bird": 2, "cat": 3, "cow": 4, "dog": 5,
                      "horse": 6, "sheep": 7, "aeroplane": 8, "bicycle": 9,
                      "boat": 10, "bus": 11, "car": 12, "motorbike": 13,
                      "train": 14, "bottle": 15, "chair": 16, "diningtable": 17,
                      "pottedplant": 18, "sofa": 19, "tvmonitor": 20}

COCO_DIR = "./dataset/COCO/2017/"
COCO_CLASSES = {"person": 1, "bicycle": 2, "car": 3, "motorcycle": 4, "airplane": 5,
                "bus": 6, "train": 7, "truck": 8, "boat": 9, "traffic light": 10,
                "fire hydrant": 11, "stop sign": 13, "parking meter": 14, "bench": 15,
                "bird": 16, "cat": 17, "dog": 18, "horse": 19, "sheep": 20, "cow": 21,
                "elephant": 22, "bear": 23, "zebra": 24, "giraffe": 25, "backpack": 27,
                "umbrella": 28, "handbag": 31, "tie": 32, "suitcase": 33, "frisbee": 34,
                "skis": 35, "snowboard": 36, "sports ball": 37, "kite": 38, "baseball bat": 39,
                "baseball glove": 40, "skateboard": 41, "surfboard": 42, "tennis racket": 43,
                "bottle": 44, "wine glass": 46, "cup": 47, "fork": 48, "knife": 49, "spoon": 50,
                "bowl": 51, "banana": 52, "apple": 53, "sandwich": 54, "orange": 55, "broccoli": 56,
                "carrot": 57, "hot dog": 58, "pizza": 59, "donut": 60, "cake": 61, "chair": 62,
                "couch": 63, "potted plant": 64, "bed": 65, "dining table": 67, "toilet": 70,
                "tv": 72, "laptop": 73, "mouse": 74, "remote": 75, "keyboard": 76, "cell phone": 77,
                "microwave": 78, "oven": 79, "toaster": 80, "sink": 81, "refrigerator": 82,
                "book": 84, "clock": 85, "vase": 86, "scissors": 87, "teddy bear": 88,
                "hair drier": 89, "toothbrush": 90}



TXT_DIR = "./data_process/data.txt"

custom_dataset_dir = ""
custom_dataset_classes = {}



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
test_video_dir = "./test_data/test_video.mp4"
temp_frame_dir = "./test_data/temp.jpg"