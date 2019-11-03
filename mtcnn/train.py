import tensorflow as tf
import numpy as np

from mtcnn.model import ONet
from settings import *


def _recover_from_string(tensor):
    byte_string = tensor.numpy()
    img_b, label_b = tf.train.Feature.FromString(byte_string).bytes_list.value
    img, label = np.frombuffer(img_b, np.float32), np.frombuffer(label_b, np.float32)
    return img, label


def map_fun(shape):
    def fun(tensor):
        img, label = tf.py_function(_recover_from_string, [tensor], (tf.float32, tf.float32))
        return tf.reshape(img, shape), (tf.reshape(label[0], ()), tf.reshape(label, (5, )))
    return fun


def get_data(record_file, shape):
    if isinstance(shape, int):
        shape = (shape, shape, 3)
    dataset = tf.data.TFRecordDataset(record_file)
    return dataset.map(map_fun(shape))


def train(net, dataset, batch_size=128, steps=100, epochs=20, checkpoint_path=None):
    dataset = dataset.shuffle(batch_size*16).batch(batch_size).repeat()
    model = net()
    cp_callback = None
    if checkpoint_path is not None:
        checkpoint_path = os.path.join(CHECKPOINT_DIR, checkpoint_path)
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                         save_weights_only=True, period=10)
    if cp_callback is not None:
        cp_callback = [cp_callback]

    model.fit(dataset, steps_per_epoch=steps, epochs=epochs, callbacks=cp_callback)


if __name__ == '__main__':
    # data12 = get_data(TF_RECORD_12, 12)
    # data24 = get_data(TF_RECORD_24, 24)
    data48 = get_data(TF_RECORD_48, 48)

    # train(PNet, data12, checkpoint_path='pnet/cp-{epoch:04d}.ckpt', epochs=200)
    # train(RNet, data24, checkpoint_path='rnet/cp-{epoch:04d}.ckpt')
    train(ONet, data48, checkpoint_path='onet/cp-{epoch:04d}.ckpt')



