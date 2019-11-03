import os

import tensorflow as tf
import numpy as np
import cv2

from mtcnn.model import PNet, RNet, ONet
from settings import CHECKPOINT_DIR
from utils.bbox import nms, convert_to_square, pad


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


class Detector:
    def __init__(self, p_pro=0.6, r_pro=0.7, o_pro=0.7):
        assert 0 < p_pro < 1 and 0 < r_pro < 1 and 0 < o_pro < 1, '不懂就别动这俩'
        self._p_pro, self._r_pro, self._o_pro = p_pro, r_pro, o_pro
        self.pnet = PNet()
        self.rnet = RNet()
        self.onet = ONet()
        self._checkpoints()

    def _checkpoints(self):
        checkpoint1 = tf.train.latest_checkpoint(os.path.join(CHECKPOINT_DIR, 'pnet'))
        checkpoint2 = tf.train.latest_checkpoint(os.path.join(CHECKPOINT_DIR, 'rnet'))
        checkpoint3 = tf.train.latest_checkpoint(os.path.join(CHECKPOINT_DIR, 'onet'))

        self.pnet.load_weights(checkpoint1)
        self.rnet.load_weights(checkpoint2)
        self.onet.load_weights(checkpoint3)

    def _pnet1_detect(self, inputs, minsize=20, scale_factor=0.709):
        face_boxes = []
        scale = 12 / minsize

        img = self._img_resize(inputs, scale)

        # 图像金字得到所有预选框
        while min(img.shape[:2]) >= 12:
            cls = self.pnet.detect(tf.reshape(img, (1, *img.shape))).numpy()
            bbox = self._get_box(cls[0, :, :, 1:], scale)

            scale *= scale_factor
            img = self._img_resize(inputs, scale)
            if bbox is None:
                continue

            keep = nms(bbox, 0.3)

            face_boxes.append(bbox[keep])

        if not face_boxes:
            return
        face_boxes = np.vstack(face_boxes)

        # 将金字塔后的图片再进行一次抑制, 此时主要避免重合
        keep = nms(face_boxes, 0.5, 1)
        face_boxes = face_boxes[keep]

        return face_boxes

    def _pnet_detect(self, inputs, minsize=20, scale_factor=0.709):
        face_boxes = []
        scale = 12 / minsize

        img = self._img_resize(inputs, scale)

        # 图像金字得到所有预选框
        while min(img.shape[:2]) >= 12:
            cls, reg = self.pnet.predict(tf.reshape(img, (1, *img.shape)))
            face_reg = np.concatenate([reg, cls[:, :, :, 1:]], axis=-1).squeeze(axis=0)
            bbox = self._get_box(face_reg, scale)

            scale *= scale_factor
            img = self._img_resize(inputs, scale)
            if bbox is None:
                continue

            keep = nms(bbox, 0.3)

            face_boxes.append(bbox[keep])

        if not face_boxes:
            return
        face_boxes = np.vstack(face_boxes)

        # 将金字塔后的图片再进行一次抑制, 此时主要避免重合
        keep = nms(face_boxes, 0.5, 1)
        face_boxes = face_boxes[keep]

        # 对box进行校准
        # box的长宽
        bbw = face_boxes[:, 2] - face_boxes[:, 0] + 1
        bbh = face_boxes[:, 3] - face_boxes[:, 1] + 1
        # 对应原图的box坐标和分数
        boxes_c = np.vstack([face_boxes[:, 0] + face_boxes[:, 4] * bbw,
                             face_boxes[:, 1] + face_boxes[:, 5] * bbh,
                             face_boxes[:, 2] + face_boxes[:, 6] * bbw,
                             face_boxes[:, 3] + face_boxes[:, 7] * bbh,
                             face_boxes[:, -1]]).T
        return boxes_c

    def _rnet_detect(self, inputs, boxes):
        """
        通过PNet结果修正人脸框后送入RNet网络
        :param boxes:
        :return:
        """
        # 将pnet的box变成包含它的正方形，可以避免信息损失
        # boxes = convert_to_square(boxes)
        boxes = boxes.astype(np.int32)
        rnet_box = np.zeros([len(boxes), 24, 24, 3])
        for i, box in enumerate(boxes):
            img = inputs[box[1]: box[3], box[0]: box[2], :]
            rnet_box[i] = cv2.resize(img, (24, 24), interpolation=cv2.INTER_LINEAR)
        rnet_box = (rnet_box - 127.5) / 128
        cls, reg = self.rnet.predict(rnet_box)

        keep = np.where(cls[:, 1] > self._r_pro)[0]

        boxes = np.hstack([boxes[keep], cls[keep, 1:]])

        boxes = self._get_reg_box(boxes, reg[keep])
        keep = nms(boxes, 0.3)
        return boxes[keep]

    def _onet_detect(self, inputs, boxes):
        boxes = convert_to_square(boxes)
        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = pad(boxes, *inputs.shape[:2])
        onet_box = np.zeros([len(boxes), 48, 48, 3])
        for i in range(len(boxes)):
            tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.uint8)
            tmp[dy[i]:edy[i] + 1, dx[i]:edx[i] + 1, :] = inputs[y[i]:ey[i] + 1, x[i]:ex[i] + 1, :]
            onet_box[i] = cv2.resize(tmp, (48, 48))
        onet_box = (onet_box - 127.5) / 28
        cls, reg = self.onet.predict(onet_box)

        keep = np.where(cls[:, 1] > self._o_pro)[0]

        boxes[:, -1] = cls[:, 1]

        boxes = self._get_reg_box(boxes[keep], reg[keep])
        keep = nms(boxes, 0.3)
        return boxes[keep]

    def predict(self, inputs: np.ndarray, minsize=20, scale_factor=0.709, net=3):
        bbox = self._pnet1_detect(inputs, minsize, scale_factor)

        if net == 1:
            return bbox

        bbox = self._rnet_detect(inputs, bbox[:, :-1])

        if net == 2:
            return bbox

        bbox = self._onet_detect(inputs, bbox)

        return bbox

    def _get_box(self, det_box, scale, cellsize=12, stride=2):
        idx = np.where(det_box[:, :, -1] > self._p_pro)
        if idx[0].size == 0:
            return

        det_box = det_box[idx[0], idx[1], :]
        bbox = np.vstack([np.round((stride * idx[1]) / scale),
                          np.round((stride * idx[0]) / scale),
                          np.round((stride * idx[1] + cellsize) / scale),
                          np.round((stride * idx[0] + cellsize) / scale),
                          ])

        return np.hstack([bbox.T, det_box])

    @staticmethod
    def _get_reg_box(box, reg):
        # box的长宽
        bbw = box[:, 2] - box[:, 0] + 1
        bbh = box[:, 3] - box[:, 1] + 1
        # 对应原图的box坐标
        boxes_c = np.vstack([box[:, 0] + reg[:, 0] * bbw,
                             box[:, 1] + reg[:, 1] * bbh,
                             box[:, 2] + reg[:, 2] * bbw,
                             box[:, 3] + reg[:, 3] * bbh,
                             box[:, -1]]).T
        return boxes_c

    @staticmethod
    def _img_resize(img, scale, show=False):
        h, w, _ = img.shape

        # 垃圾opencv，什么都要反着来
        shape = (int(w*scale), int(h*scale))
        img = cv2.resize(img, shape, interpolation=cv2.INTER_LINEAR)
        if show:
            cv2.imshow('mi', img)
        return (img - 127.5) / 128


if __name__ == '__main__':
    det = Detector()
    img = cv2.imread('../test.jpg')
    a = det.predict(img)
    if a is None:
        a = np.zeros([1, 4])

    print(a.shape)
    x1, y1, x2, y2 = [a[:, i] for i in range(4)]
    for i, j, k, v in zip(x1, y1, x2, y2):
        img = cv2.rectangle(img, (int(i), int(j)), (int(k), int(v)), (255, 0, 0))

    cv2.imshow('aaa', img)
    cv2.waitKey(2000)




