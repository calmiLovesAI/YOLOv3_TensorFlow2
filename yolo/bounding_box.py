import tensorflow as tf
from configuration import ANCHOR_NUM_EACH_SCALE, CATEGORY_NUM, COCO_ANCHORS, COCO_ANCHOR_INDEX, IMAGE_HEIGHT


def get_coco_anchors(scale_type):
    index_list = COCO_ANCHOR_INDEX[scale_type - 1]
    return tf.convert_to_tensor(COCO_ANCHORS[index_list[0]: index_list[-1] + 1], dtype=tf.dtypes.float32)


def generate_grid_index(grid_dim):
    x = tf.range(grid_dim, dtype=tf.dtypes.float32)
    y = tf.range(grid_dim, dtype=tf.dtypes.float32)
    X, Y = tf.meshgrid(x, y)
    X = tf.reshape(X, shape=(-1, 1))
    Y = tf.reshape(Y, shape=(-1, 1))
    return tf.concat(values=[X, Y], axis=-1)


def bounding_box_predict(feature_map, scale_type):
    h = feature_map.shape[1]
    w = feature_map.shape[2]
    if h != w:
        raise ValueError("The shape[1] and shape[2] of feature map must be the same value.")
    downsampling_rate = IMAGE_HEIGHT // h
    area = h * w
    pred = tf.reshape(feature_map, shape=(-1, ANCHOR_NUM_EACH_SCALE * area, CATEGORY_NUM + 5))
    pred = tf.nn.sigmoid(pred)
    tx_ty, tw_th, confidence, classes = tf.split(pred, num_or_size_splits=[2, 2, 1, CATEGORY_NUM], axis=-1)
    center_index = generate_grid_index(grid_dim=h)
    center_index = tf.tile(center_index, [1, ANCHOR_NUM_EACH_SCALE])
    center_index = tf.reshape(center_index, shape=(1, -1, 2))

    center_coord = center_index + tx_ty
    anchors = tf.tile(get_coco_anchors(scale_type) / downsampling_rate, [area, 1])
    bw_bh = tf.math.exp(tw_th) * anchors

    predictions = tf.concat(values=[center_coord * downsampling_rate,
                                    bw_bh * downsampling_rate,
                                    confidence,
                                    classes],
                            axis=-1)

    return predictions