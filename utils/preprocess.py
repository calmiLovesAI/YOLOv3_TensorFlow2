import tensorflow as tf
from configuration import IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS, PASCAL_VOC_DIR
import os


def process_single_image(image_filename):
    img_raw = tf.io.read_file(image_filename)
    img_tensor = tf.image.decode_jpeg(img_raw, channels=CHANNELS)
    img_tensor = tf.image.resize_with_pad(image=img_tensor, target_height=IMAGE_HEIGHT, target_width=IMAGE_WIDTH)
    img_tensor = tf.cast(img_tensor, tf.float32)
    img = img_tensor / 255.0
    return img


def process_image_filenames(filenames):
    image_list = []
    for filename in filenames:
        image_path = os.path.join(PASCAL_VOC_DIR + "JPEGImages", filename)
        image_tensor = process_single_image(image_path)
        image_tensor = tf.expand_dims(image_tensor, axis=0)
        image_list.append(image_tensor)
    return tf.concat(values=image_list, axis=0)

