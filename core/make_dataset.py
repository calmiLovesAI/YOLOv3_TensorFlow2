import tensorflow as tf
from core.read_txt import read_txt
from configuration import PASCAL_VOC_DIR, IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS, BATCH_SIZE
import os


def preprocess_image(image_filename):
    img_raw = tf.io.read_file(image_filename)
    img_tensor = tf.image.decode_jpeg(img_raw, channels=CHANNELS)
    img_tensor = tf.image.resize(img_tensor, [IMAGE_HEIGHT, IMAGE_WIDTH])
    img_tensor = tf.cast(img_tensor, tf.float32)
    img = img_tensor / 255.0
    return img


def get_length_of_dataset(dataset):
    count = 0
    for _ in dataset:
        count += 1
    return count


def generate_dataset():
    image_names, boxes = read_txt()
    # image_names_tensor = tf.convert_to_tensor(image_names, dtype=tf.dtypes.string)
    boxes_tensor = tf.convert_to_tensor(boxes, dtype=tf.dtypes.float32)
    image_data = tf.data.Dataset.from_tensor_slices(image_names).map(preprocess_image)
    box_data = tf.data.Dataset.from_tensor_slices(boxes_tensor)
    dataset = tf.data.Dataset.zip((image_data, box_data))

    train_count = get_length_of_dataset(dataset)
    train_dataset = dataset.batch(batch_size=BATCH_SIZE)

    return train_dataset, train_count