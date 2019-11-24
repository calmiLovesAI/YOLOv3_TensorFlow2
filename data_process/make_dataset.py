import tensorflow as tf
from data_process.read_txt import read_txt
from configuration import BATCH_SIZE


def get_length_of_dataset(dataset):
    count = 0
    for _ in dataset:
        count += 1
    return count


def generate_dataset():
    image_names, boxes = read_txt()
    boxes_tensor = tf.convert_to_tensor(boxes, dtype=tf.dtypes.float32)
    image_data = tf.data.Dataset.from_tensor_slices(image_names)
    box_data = tf.data.Dataset.from_tensor_slices(boxes_tensor)
    dataset = tf.data.Dataset.zip((image_data, box_data))

    train_count = get_length_of_dataset(dataset)
    train_dataset = dataset.batch(batch_size=BATCH_SIZE)

    return train_dataset, train_count