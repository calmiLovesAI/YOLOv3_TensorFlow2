import tensorflow as tf
import cv2
from configuration import test_picture_dir, save_model_dir, CHANNELS, \
    IMAGE_HEIGHT, IMAGE_WIDTH, PASCAL_VOC_CLASSES, CATEGORY_NUM
from yolo.inference import Inference
from yolo.yolo_v3 import YOLOV3


def find_class_name(class_id):
    for k, v in PASCAL_VOC_CLASSES.items():
        if v == class_id:
            return k


# shape of boxes : (N, 4)  (xmin, ymin, xmax, ymax)
# shape of scores : (N,)
# shape of classes : (N,)
def draw_boxes_on_image(image, boxes, scores, classes):

    num_boxes = boxes.shape[0]
    for i in range(num_boxes):
        class_and_score = str(find_class_name(classes[i])) + ": " + str(scores[i, 0].numpy())
        cv2.rectangle(img=image, pt1=(boxes[i, 0], boxes[i, 1]), pt2=(boxes[i, 2], boxes[i, 3]), color=(255, 0, 0), thickness=10)
        cv2.putText(img=image, text=class_and_score, org=(boxes[i, 0], boxes[i, 1] - 10), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1.5, color=(0, 255, 255), thickness=5)
    return image


def single_image_inference(image_dir, model):
    image = tf.io.decode_jpeg(contents=tf.io.read_file(image_dir), channels=CHANNELS)
    h = image.shape[0]
    w = image.shape[1]
    input_image_shape = tf.constant([h, w], dtype=tf.dtypes.float32)
    img_tensor = tf.image.resize(image, [IMAGE_HEIGHT, IMAGE_WIDTH])
    img_tensor = tf.cast(img_tensor, tf.float32)
    img_tensor = img_tensor / 255.0
    img_tensor = tf.expand_dims(img_tensor, axis=0)
    yolo_output = model(img_tensor, training=False)
    boxes, scores, classes = Inference(yolo_output=yolo_output, input_image_shape=input_image_shape).get_final_boxes()
    image_with_boxes = draw_boxes_on_image(cv2.imread(image_dir), boxes, scores, classes)
    return image_with_boxes


if __name__ == '__main__':
    # GPU settings
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    # load model
    yolo_v3 = YOLOV3(out_channels=3 * (CATEGORY_NUM + 5))
    yolo_v3.load_weights(filepath=save_model_dir+"saved_model")
    # inference
    image = single_image_inference(image_dir=test_picture_dir, model=yolo_v3)

    cv2.namedWindow("detect result", flags=cv2.WINDOW_NORMAL)
    cv2.imshow("detect result", image)
    cv2.waitKey(0)