"""
配置文件
"""
import os


# 项目目录
BASE_DIR = os.path.dirname(__file__)


# 数据集目录
DATA_DIR = 'D:/Documents/python/tensorFlowLearn/WIDER_train/'

# 数据集标注文件
ANN_FILE = os.path.join(DATA_DIR, 'wider_face_split', 'wider_face_train_bbx_gt.txt')

# 数据集图片目录
IMG_DIR = os.path.join(DATA_DIR, 'images')

# 裁剪图片存储目录
CROP_SAVE_12 = os.path.join(DATA_DIR, "crops12")
CROP_SAVE_24 = os.path.join(DATA_DIR, "crops24")
CROP_SAVE_48 = os.path.join(DATA_DIR, "crops48")
LABEL_SAVE_FILE = os.path.join(DATA_DIR, 'label', 'crops.txt')

# tf_record文件路径
TF_RECORD_12 = os.path.join(DATA_DIR, 'tfrecord', '12.tfrecord')
TF_RECORD_24 = os.path.join(DATA_DIR, 'tfrecord', '24.tfrecord')
TF_RECORD_48 = os.path.join(DATA_DIR, 'tfrecord', '48.tfrecord')
TF_DATA_12 = os.path.join(DATA_DIR, 'tfrecord', '12.dataset')
TF_DATA_24 = os.path.join(DATA_DIR, 'tfrecord', '24.dataset')
TF_DATA_48 = os.path.join(DATA_DIR, 'tfrecord', '48.dataset')

CHECKPOINT_DIR = os.path.join(BASE_DIR, 'data')

if not os.path.exists(CROP_SAVE_12):
    os.mkdir(CROP_SAVE_12)

if not os.path.exists(CROP_SAVE_24):
    os.mkdir(CROP_SAVE_24)

if not os.path.exists(CROP_SAVE_48):
    os.mkdir(CROP_SAVE_48)
