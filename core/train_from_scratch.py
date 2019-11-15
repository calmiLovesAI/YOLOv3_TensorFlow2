import tensorflow as tf
from yolo.yolo_v3 import YOLOV3
from configuration import CATEGORY_NUM, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS, EPOCHS, BATCH_SIZE, save_model_dir
from yolo.loss import YoloLoss
from core.make_dataset import generate_dataset
from yolo.make_label import GenerateLabel


def print_model_summary(network):
    network.build(input_shape=(None, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS))
    network.summary()


def generate_label_batch(true_boxes):
    true_label = GenerateLabel(true_boxes=true_boxes.numpy(), input_shape=[IMAGE_HEIGHT, IMAGE_WIDTH]).generate_label()
    return true_label


if __name__ == '__main__':
    # GPU settings
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    # dataset
    train_dataset, train_count = generate_dataset()

    net = YOLOV3(out_channels=3 * (CATEGORY_NUM + 5))
    print_model_summary(network=net)

    # loss and optimizer
    yolo_loss = YoloLoss()
    optimizer = tf.optimizers.RMSprop()

    # metrics
    loss_metric = tf.metrics.Mean()

    def train_step(image_batch, label_batch):
        with tf.GradientTape() as tape:
            yolo_output = net(image_batch, training=True)
            loss = yolo_loss(y_true=label_batch, y_pred=yolo_output)
        gradients = tape.gradient(loss, net.trainable_variables)
        optimizer.apply_gradients(grads_and_vars=zip(gradients, net.trainable_variables))
        loss_metric.update_state(values=loss)


    for epoch in range(EPOCHS):
        step = 0
        for images, boxes in train_dataset:
            step += 1
            labels = generate_label_batch(true_boxes=boxes)
            train_step(image_batch=images, label_batch=labels)
            print("Epoch: {}/{}, step: {}/{}, loss: {:.5f}".format(epoch + 1,
                                                                   EPOCHS,
                                                                   step,
                                                                   tf.math.ceil(train_count / BATCH_SIZE),
                                                                   loss_metric.result()))

        loss_metric.reset_states()

    tf.saved_model.save(obj=net, export_dir=save_model_dir)