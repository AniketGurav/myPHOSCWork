import colorsys
import random
import numpy as np
from yolov3.configs import *
import os
import cv2
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
from tensorflow.python.client import device_lib

import shutil
import numpy as np
from yolov3.configs import *

def read_class_names(class_file_name):
    # loads class name from a file
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names


def bboxes_iou(boxes1, boxes2):
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    ious = np.maximum(1.0 * inter_area / union_area, np.finfo(np.float32).eps)

    return ious


def nms(bboxes, iou_threshold, sigma=0.3, method='nms'):
    """
    :param bboxes: (xmin, ymin, xmax, ymax, score, class)

    Note: soft-nms, https://arxiv.org/pdf/1704.04503.pdf
          https://github.com/bharatsingh430/soft-nms
    """
    classes_in_img = list(set(bboxes[:, 5]))
    best_bboxes = []

    for cls in classes_in_img:
        cls_mask = (bboxes[:, 5] == cls)
        cls_bboxes = bboxes[cls_mask]
        # Process 1: Determine whether the number of bounding boxes is greater than 0
        while len(cls_bboxes) > 0:
            # Process 2: Select the bounding box with the highest score according to socre order A
            max_ind = np.argmax(cls_bboxes[:, 4])
            best_bbox = cls_bboxes[max_ind]
            best_bboxes.append(best_bbox)
            cls_bboxes = np.concatenate([cls_bboxes[: max_ind], cls_bboxes[max_ind + 1:]])
            # Process 3: Calculate this bounding box A and
            # Remain all iou of the bounding box and remove those bounding boxes whose iou value is higher than the threshold
            iou = bboxes_iou(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
            weight = np.ones((len(iou),), dtype=np.float32)

            assert method in ['nms', 'soft-nms']

            if method == 'nms':
                iou_mask = iou > iou_threshold
                weight[iou_mask] = 0.0

            if method == 'soft-nms':
                weight = np.exp(-(1.0 * iou ** 2 / sigma))

            cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
            score_mask = cls_bboxes[:, 4] > 0.
            cls_bboxes = cls_bboxes[score_mask]

    return best_bboxes


#def postprocess_boxes(pred_bbox, original_image,orgImage, input_size, score_threshold):
def postprocess_boxes(pred_bbox, orgImage,original_image, input_size, score_threshold):
    #print(" inside post process!!!")
    valid_scale = [0, np.inf]
    pred_bbox = np.array(pred_bbox)

    pred_xywh = pred_bbox[:, 0:4]
    pred_conf = pred_bbox[:, 4]
    pred_prob = pred_bbox[:, 5:]

    # 1. (x, y, w, h) --> (xmin, ymin, xmax, ymax)
    pred_coor = np.concatenate([pred_xywh[:, :2] - pred_xywh[:, 2:] * 0.5,
                                pred_xywh[:, :2] + pred_xywh[:, 2:] * 0.5], axis=-1)
    # 2. (xmin, ymin, xmax, ymax) -> (xmin_org, ymin_org, xmax_org, ymax_org)
    org_h, org_w = original_image.shape[:2]
    resize_ratio = min(input_size / org_w, input_size / org_h)

    dw = (input_size - resize_ratio * org_w) / 2
    dh = (input_size - resize_ratio * org_h) / 2

    pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
    pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio

    # 3. clip some boxes those are out of range
    pred_coor = np.concatenate([np.maximum(pred_coor[:, :2], [0, 0]),
                                np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1])], axis=-1)
    invalid_mask = np.logical_or((pred_coor[:, 0] > pred_coor[:, 2]), (pred_coor[:, 1] > pred_coor[:, 3]))
    pred_coor[invalid_mask] = 0

    # 4. discard some invalid boxes
    bboxes_scale = np.sqrt(np.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1))
    scale_mask = np.logical_and((valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1]))

    # 5. discard boxes with low scores
    classes = np.argmax(pred_prob, axis=-1)
    scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
    score_mask = scores > score_threshold
    mask = np.logical_and(scale_mask, score_mask)
    coors, scores, classes = pred_coor[mask], scores[mask], classes[mask]

    return np.concatenate([coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1)

from collections import namedtuple
Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')

def area(a, b):  # returns None if rectangles don't intersect
    dx = min(a.xmax, b.xmax) - max(a.xmin, b.xmin)
    dy = min(a.ymax, b.ymax) - max(a.ymin, b.ymin)
    if (dx>=0) and (dy>=0):
        return dx*dy


def draw_bbox(nm,counter,level, image,orgImgName, bboxes, df,CLASSES=YOLO_COCO_CLASSES, show_label=True, show_confidence=True,
              Text_colors=(255, 255, 0), rectangle_colors='', tracking=False):
    NUM_CLASS = read_class_names(CLASSES)
    num_classes = len(NUM_CLASS)
    #print("\n\t imag shape---:",image.shape)

    try:
        batch,image_h, image_w, _ = image.shape
    except Exception as e:
        batch,image_h, image_w= image.shape

    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    # print("hsv_tuples", hsv_tuples)
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    random.seed(0)
    random.shuffle(colors)
    random.seed(None)
    color1 = (0,0,255)

    thickness1 = 3

    try:
        os.mkdir("./crops/"+orgImgName)
    except Exception as e:
        pass

    """
        get the record corresponding to this file
    """
    print(" original image name:",orgImgName)
    filterinfDataframe = df[df['image_name'] ==orgImgName]

    filterinfDataframe1 =filterinfDataframe[["org_x1","org_y1","org_x2","org_y2","text"]]
    #g06-018o.png_25_[321, 1098, 404, 1174]

    filterinfDataframe1.to_csv("./crops/"+orgImgName+".csv")

    from copy import deepcopy
    bboxes2=deepcopy(bboxes)


    """
        THIS LOOP TAKES THE PREDICTION OF YOLO AND IF THERE IS OVERLAP THEN MERGES THEM
    """

    for i, bbox in enumerate(bboxes2):
        coor = np.array(bbox[:4], dtype=np.int32)
        (x1, y1), (x2, y2) = (coor[0], coor[1]), (coor[2], coor[3])

        ra = Rectangle(x1, y1, x2, y2)

        for j, bbox1 in enumerate(bboxes2):
            if i!=j:
                coor = np.array(bbox1[:4], dtype=np.int32)
                (x11, y11), (x22, y22) = (coor[0], coor[1]), (coor[2], coor[3])
                rb = Rectangle(x11, y11, x22, y22)

                if area(ra, rb):

                    if area(ra, rb)>1000:
                        x1,y1=min(x1,x11),min(y1,y11)
                        x2,y2=max(x2,x22),max(y2,y22)
                        bboxes[i]=np.array([x1,y1,x2,y2,0],dtype=np.int32)
                        #bboxes[i]=np.array([0,0,0,0,0],dtype=np.int32)
                        bboxes[j]=np.array([0,0,0,0,0],dtype=np.int32)


    for i, bbox in enumerate(bboxes):
        #print("\n\t i=",i)
        coor = np.array(bbox[:4], dtype=np.int32)
        score = bbox[4]
        class_ind = 0#int(bbox[5]) # phosc change
        #print("\n\t score:",score,"\t class_ind:",class_ind)
        bbox_color = rectangle_colors if rectangle_colors != '' else colors[class_ind]
        bbox_thick = int(0.6 * (image_h + image_w) / 1000)
        if bbox_thick < 1: bbox_thick = 1
        fontScale = 0.75 * bbox_thick
        (x1, y1), (x2, y2) = (coor[0], coor[1]), (coor[2], coor[3])

        if x1==0 and y1 ==0 and x2==0 and y2 ==0:
            continue

        #cropped_image = image[y1:y2, x1:x2]
        #cropped_image = image[y1-10:y2+10, x1+30:x2+30]
        cropped_image = image[y1-20:y2+20, x1-20:x2+20]

        #cropped_image = image[y1-10:y2+10, x1+10:x2+10]
        #cropped_image =cv2.resize(cropped_image,(250,50))

        s=[x1, y1, x2, y2]
        ra = Rectangle(x1, y1, x2, y2)

        for indx,row in filterinfDataframe1.iterrows():
            x11,y11,x22,y22,text=row
            #print("text:",x1,y1,x2,y2)

            rb = Rectangle(x11,y11,x22,y22)

            if area(ra, rb):
                #print("text:",text," ",x1,y1,x2,y2)
                #print(" overlap:",x11,y11,x22,y22)
                cv2.imwrite("./crops/"+orgImgName+"//"+orgImgName+"_"+str(i)+"_"+str(s)+"_"+text+'_.png', cropped_image)
                #input("check!!!")
                break


        image = cv2.rectangle(image, (x1-20, y1-20), (x2+20, y2+20), color1, 3)

        cv2.putText(image,str(i), (x1-10, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 2,255)
        #cv2.putText(image,str(i), (x1, y1))
        #cv2.imwrite(f'./pred/'+nm+'.png', image)

        #print("\n\t x1:",x1,"\t y1:",y1,"\t x2:",x2,"\t y2:",y2)
        # print("\n\t image shape:",image.shape)
        # put object rectangle
        #image=cv2.rectangle(image, (x1, y1), (x2, y2),(0,0,0),5)

        #input("check!!!")
    # Line thickness of 9 px
    # image = cv2.line(image,(0,0),(10,10), color1,5)
    # image = cv2.line(image,(x1,y1),(x2,y2), color1,1)

    if 0:#len(bboxes)>10 and counter<10:
        cv2.imwrite(f'./pred/'+orgImgName+'_5_.png', image)

    return image

