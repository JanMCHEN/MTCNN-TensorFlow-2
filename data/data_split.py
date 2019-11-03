import cv2
import os

# from mtcnn.utils.bbox import random_crop_box, iou_calculate

BASE_DIR = r'D:\Documents\python\tensorFlowLearn\WIDER_train'


def gen_img_data(label, impath):
    """生成器，读取图片和标签，并以(name, box)形式返回"""
    with open(label, 'r') as annotations:
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
                    boxes.append(list(map(int, box)))
                
                yield file_name, boxes

            except StopIteration:
                break


def img_show(imname, boxes, delay=0):
    img = cv2.imread(imname)

    for box in boxes:
        img = cv2.rectangle(img, tuple(box[:2]), (box[0]+box[2], box[1]+box[3]), (255, 0, 0))

    cv2.imshow('nan', img)
    cv2.waitKey(delay)


if __name__ == '__main__':
    label = os.path.join(BASE_DIR, 'wider_face_split', 'wider_face_train_bbx_gt.txt')
    impath = os.path.join(BASE_DIR, 'images')

    data = gen_img_data(label, impath)
    for imname, pos in data:
        img_show(os.path.join(impath, imname), pos, 1000)
