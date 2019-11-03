import cv2
import numpy as np
import numpy.random as npr

from settings import *
from utils.bbox import get_iou, gen_img_data, random_crop_box


def gen_neg_img(img, boxes, nums=100):
    """
    产生negative图片
    :param img: 原图
    :param boxes: 人脸框， shape=(-1, 4)
    :param nums: 总需产生个数
    :return: tuple， 裁剪后的图片和标签
    """
    while nums:
        # 随机裁剪
        crop_box = random_crop_box(*img.shape[:2])
        # calculate iou
        iou = get_iou(crop_box, boxes)

        crop_x, crop_y, crop_w, crop_h = crop_box

        # crop a part from inital image
        cropped_im = img[crop_x:crop_x+crop_w, crop_y:crop_y+crop_h]

        if np.max(iou) < 0.3:
            yield cropped_im, [0.] * 5
            nums -= 1


def gen_crop_im(nums=0, display=10):
    """
    生成器， 产生3种，pos=1、part=-1、neg=0的图片
    :param display: 多少张图片打印一次进度
    :return: generate， （img， label）
    """
    p_idx = 0  # positive
    n_idx = 0  # negative
    d_idx = 0  # part

    im_idx = 0

    for annotation in gen_img_data(ANN_FILE):

        # image path
        im_path = annotation[0]

        # bound
        boxes = np.array(annotation[1:], dtype=np.float32).reshape(-1, 4)

        # load image
        img = cv2.imread(os.path.join(IMG_DIR, im_path))

        height, width, channel = img.shape

        # neg-img
        neg_im = gen_neg_img(img, boxes)

        for im in neg_im:
            yield im
            n_idx += 1

        # for every bounding boxes
        for box in boxes:
            x1, y1, w, h = box

            # ignore small faces and those faces has left-top corner out of the image
            # in case the ground truth boxes of small faces are not accurate
            if min(w, h) < 20 or x1 < 0 or y1 < 0:
                continue

            # generate positive examples and part faces
            for i in range(100//len(boxes)+1):
                # pos and part face size [minsize*0.8,maxsize*1.25]
                size = npr.randint(int(min(w, h) * 0.8), np.ceil(1.25 * max(w, h)))

                # delta here is the offset of box center

                delta_x = npr.randint(-w * 0.2, w * 0.2)
                delta_y = npr.randint(-h * 0.2, h * 0.2)

                # show this way: nx1 = max(x1+w/2-size/2+delta_x)
                # x1+ w/2 is the central point, then add offset , then deduct size/2
                # deduct size/2 to make sure that the right bottom corner will be out of
                nx1 = int(max(x1 + w / 2 + delta_x - size / 2, 0))
                # show this way: ny1 = max(y1+h/2-size/2+delta_y)
                ny1 = int(max(y1 + h / 2 + delta_y - size / 2, 0))

                nx2 = nx1 + size
                ny2 = ny1 + size

                if nx2 > width or ny2 > height:
                    continue
                crop_box = np.array([nx1, ny1, size, size])
                # yu gt de offset
                offset_x1 = round((x1 - nx1) / size, 3)
                offset_y1 = round((y1 - ny1) / size, 3)

                offset_x2 = round((x1+w-nx1-size) / size, 3)
                offset_y2 = round((y1+h-ny1-size) / size, 3)
                # crop
                cropped_im = img[ny1: ny2, nx1: nx2]

                box_ = box.reshape(1, -1)
                iou = get_iou(crop_box, box_)

                if iou >= 0.65:
                    yield cropped_im, (1, offset_x1, offset_y1, offset_x2, offset_y2)
                    p_idx += 1

                elif iou >= 0.4:
                    yield cropped_im, (-1, offset_x1, offset_y1, offset_x2, offset_y2)
                    d_idx += 1

        im_idx += 1
        if im_idx == nums:
            break
        if im_idx % display == 0:
            print("%d images done, pos: %d part: %d neg: %d" % (im_idx, p_idx, d_idx, n_idx))


def crop_save(img, path, name, shape=None):
    """
    保存图片
    :param img: 图片数据
    :param path: 路径
    :param name: 名字
    :param shape: resize的shape
    :return:
    """
    if shape is not None:
        img = cv2.resize(img, shape, interpolation=cv2.INTER_LINEAR)

    cv2.imwrite(os.path.join(path, name), img)


def save_to_file(max_len=0):
    """
    以文件形式保存图片和标签
    :param max_len: 保存文件数， 0表示无限制
    :return:
    """
    gen = gen_crop_im()

    with open(LABEL_SAVE_FILE, 'w') as f:
        i = 0
        for im, label in gen:
            f.write(f'{i} ' + ' '.join(map(str, label)) + '\n')
            crop_save(im, CROP_SAVE_12, f'{i}.jpg', (12, 12))
            crop_save(im, CROP_SAVE_24, f'{i}.jpg', (24, 24))
            i += 1
            if max_len == i:
                break


if __name__ == '__main__':
    save_to_file(100)

