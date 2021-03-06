import numpy as np
import cv2
import numpy.random as npr


def img_show(imname, boxes, delay=0):
    img = cv2.imread(imname)

    for box in boxes:
        img = cv2.rectangle(img, tuple(box[:2]), (box[0]+box[2], box[1]+box[3]), (255, 0, 0))

    cv2.imshow('nan', img)
    cv2.waitKey(delay)


def get_iou(box, boxes):
    """Compute IoU between detect box and gt boxes
    Parameters:
    ----------
    box: numpy array , shape (4, ): x1, y1, x2, y2, score
        predicted boxes
    boxes: numpy array, shape (n, 4): x1, y1, x2, y2
        input ground truth boxes
    Returns:
    -------
    ovr: numpy.array, shape (n, )
        IoU
    """
    box_area = box[2] * box[3]

    areas = boxes[:, 2] * boxes[:, 3]

    xx1 = np.maximum(box[0], boxes[:, 0])

    yy1 = np.maximum(box[1], boxes[:, 1])

    xx2 = np.minimum(box[2] + box[0], boxes[:, 2] + boxes[:, 0])

    yy2 = np.minimum(box[3] + box[1], boxes[:, 3] + boxes[:, 1])

    # compute the width and height of the bounding box
    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)

    inter = w * h
    ovr = inter / (box_area + areas - inter)
    return ovr


def gen_img_data(label_path):
    """生成器，读取图片和标签，并以(name, box)形式返回"""
    with open(label_path, 'r') as annotations:
        while True:
            try:
                file_name = next(annotations).strip()
                nums = next(annotations).strip()

                if int(nums) == 0:
                    next(annotations)
                    continue

                boxes = []

                for i in range(int(nums)):
                    box = next(annotations).split()[: 4]
                    boxes.append(list(map(float, box)))

                yield file_name, boxes

            except StopIteration:
                break


def random_crop_box(w, h, minsize=12):
    """
    随机裁剪
    :param w: 原图宽
    :param h: 原图高
    :param minsize: 裁剪的最小尺寸
    :return: ndarray, 裁剪框
    """
    crop_x = npr.randint(0, w - 13)
    crop_y = npr.randint(0, h - 13)

    size = npr.randint(minsize, max(13, (w - crop_x) // 2, (h - crop_y) // 2))

    return np.array([crop_x, crop_y, size, size], dtype=np.int32)


def nms(dets, thresh, mode=0):
    """非极大值抑制剔除太相似的box"""
    x1, y1, x2, y2 = [dets[:, i] for i in range(4)]

    scores = dets[:, -1]

    areas = (x2-x1+1) * (y2-y1+1)
    # 将概率值从大到小排列
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        if mode == 0:
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
        elif mode == 1:
            ovr = inter / np.minimum(areas[i], areas[order[1:]])

        # 保留小于阈值的下标，因为order[0]拿出来做比较了，所以inds+1是原来对应的下标
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def pad(bboxes, h, w):
    '''将超出图像的box进行处理
    参数：
      bboxes:人脸框
      w,h:图像长宽
    返回值：
      dy, dx : 为调整后的box的左上角坐标相对于原box左上角的坐标
      edy, edx : n为调整后的box右下角相对原box左上角的相对坐标
      y, x : 调整后的box在原图上左上角的坐标
      ex, ex : 调整后的box在原图上右下角的坐标
      tmph, tmpw: 原始box的长宽
    '''
    # box的长宽
    tmpw, tmph = bboxes[:, 2] - bboxes[:, 0] + 1, bboxes[:, 3] - bboxes[:, 1] + 1
    num_box = bboxes.shape[0]

    dx, dy = np.zeros((num_box,)), np.zeros((num_box,))
    edx, edy = tmpw.copy() - 1, tmph.copy() - 1
    # box左上右下的坐标
    x, y, ex, ey = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
    # 找到超出右下边界的box并将ex,ey归为图像的w,h
    # edx,edy为调整后的box右下角相对原box左上角的相对坐标
    tmp_index = np.where(ex > w - 1)
    edx[tmp_index] = tmpw[tmp_index] + w - 2 - ex[tmp_index]
    ex[tmp_index] = w - 1

    tmp_index = np.where(ey > h - 1)
    edy[tmp_index] = tmph[tmp_index] + h - 2 - ey[tmp_index]
    ey[tmp_index] = h - 1
    # 找到超出左上角的box并将x,y归为0
    # dx,dy为调整后的box的左上角坐标相对于原box左上角的坐标
    tmp_index = np.where(x < 0)
    dx[tmp_index] = 0 - x[tmp_index]
    x[tmp_index] = 0

    tmp_index = np.where(y < 0)
    dy[tmp_index] = 0 - y[tmp_index]
    y[tmp_index] = 0

    return_list = [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph]
    return_list = [item.astype(np.int32) for item in return_list]

    return return_list


def convert_to_square(box):
    """
    将box转换成更大的正方形
    :param box: 预测的box,[n,5]
    :return: 调整后的正方形box，[n,5]
    """
    square_box = box.copy()
    h = box[:, 3] - box[:, 1] + 1
    w = box[:, 2] - box[:, 0] + 1
    # 找寻正方形最大边长
    max_side = np.maximum(w, h)

    square_box[:, 0] = box[:, 0] + w * 0.5 - max_side * 0.5
    square_box[:, 1] = box[:, 1] + h * 0.5 - max_side * 0.5
    square_box[:, 2] = square_box[:, 0] + max_side - 1
    square_box[:, 3] = square_box[:, 1] + max_side - 1
    return np.round(square_box)


if __name__ == '__main__':
    pass
