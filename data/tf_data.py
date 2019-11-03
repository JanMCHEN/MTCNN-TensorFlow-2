import tensorflow as tf
import cv2
import numpy as np

from data.pre_crop_img import gen_crop_im
from settings import *

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x


def gen_from_file(file_name, impath):
    with open(file_name, 'r') as f:
        for line in tqdm(f):
            img_name, *label = line.split()
            img = cv2.imread(os.path.join(impath, img_name+'.jpg'))

            if img is None:
                continue

            # 归一化
            img = (img - 127.5) / 128

            yield img, (label[0:1], label)


def tf_serialize_feature(img, label):
    """"""
    # Create a dictionary mapping the feature name to the tf.Example-compatible
    # data type.
    img = img.astype(np.float32)
    label = np.array(label, dtype=np.float32)

    feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[img.tostring(), label.tostring()]))

    return feature.SerializeToString()


def _tfrecord_save_from_gen(gen, tf12=None, tf24=None, tf48=None):
    for img, label in tqdm(gen):
        if tf12 is not None:
            img12 = cv2.resize(img, (12, 12))
            img12 = (img12 - 127.5) / 128
            tf12.write(tf_serialize_feature(img12, label))
        if tf24 is not None:
            img24 = cv2.resize(img, (24, 24))
            img24 = (img24 - 127.5) / 128
            tf24.write(tf_serialize_feature(img24, label))
        if tf48 is not None:
            img48 = cv2.resize(img, (48, 48))
            img48 = (img48 - 127.5) / 128
            tf48.write(tf_serialize_feature(img48, label))


def dataset_gen(file, impath, size=12):
    """
    从标签文件读取数据并以生成器形式返回给dataset, 可以直接用
    :param size: shape=(size, size, 3)
    :param file: 标注文件
    :param impath: 图片路径
    :return: 训练所需要的数据格式
    """
    gen_ = gen_from_file(file, impath)
    gen = lambda: gen_
    return tf.data.Dataset.from_generator(gen, (tf.float32, (tf.float32, tf.float32)),
                                          (tf.TensorShape([size, size, 3]), (tf.TensorShape(1), tf.TensorShape(5))))


def _tfrecord_save_from_dataset(dataset, save_path):
    tfrecord = tf.data.experimental.TFRecordWriter(save_path)
    tfrecord.write(dataset)


def tfrecord_save_all(nums=0):
    gen = gen_crop_im(nums)
    with tf.io.TFRecordWriter(TF_RECORD_12) as tf12, tf.io.TFRecordWriter(TF_RECORD_24) as tf24, \
            tf.io.TFRecordWriter(TF_RECORD_48) as tf48:
        _tfrecord_save_from_gen(gen, tf12, tf24, tf48)


def dataset_save(mode=(1, 1, 1)):
    if isinstance(mode, int):
        if mode == 0:
            mode = (1, 1, 1)
        elif mode == 1:
            mode = (1, 0, 0)
        elif mode == 2:
            mode = (0, 1, 0)
        elif mode == 3:
            mode = (0, 0, 1)
        else:
            mode = (0, 0, 0)
    if mode[0]:
        dataset = dataset_gen(LABEL_SAVE_FILE, CROP_SAVE_12)
        _tfrecord_save_from_dataset(dataset, TF_DATA_12)
    if mode[1]:
        dataset = dataset_gen(LABEL_SAVE_FILE, CROP_SAVE_24)
        _tfrecord_save_from_dataset(dataset, TF_DATA_24)
    if mode[2]:
        dataset = dataset_gen(LABEL_SAVE_FILE, CROP_SAVE_48)
        _tfrecord_save_from_dataset(dataset, TF_DATA_48)


if __name__ == '__main__':
    # tfrecord_save_all(1000)
    with tf.io.TFRecordWriter(TF_RECORD_48) as tf48:
        gen = gen_crop_im(2000)
        _tfrecord_save_from_gen(gen, tf48=tf48)
