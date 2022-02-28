import sys
import os
from mmdet.apis import init_detector, inference_detector, show_result_pyplot, show_result_ins
import mmcv
import numpy as np
from scipy import ndimage
import cv2 as cv
from math import *
import pandas as pd

import matplotlib.pyplot as plt
import json
import csv
from shapely.geometry import Point, Polygon
from csv import reader

"""
# line exist test
def line_exist(xray_type, line_CLASSES, seg_CLASSES, sort_keypoints, target_line_sort_keypoints, csv_line_result):
    detect_line_num = len(target_line_sort_keypoints)
    minus = 0
    for pre_i in range(len(sort_keypoints)):
        for eval_i in range(len(seg_CLASSES)):
            if sort_keypoints[pre_i][0] == seg_CLASSES[eval_i]:
                if detect_line_num == 0:
                    print('')
                else:
                    if target_line_sort_keypoints[eval_i-minus][0] == line_CLASSES[eval_i]:
                        contours, hierarchy = cv.findContours(sort_keypoints[pre_i][1], cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
                        line_bool_val = []
                        for k in range(len(contours)):
                            poly = Polygon(contours[k].squeeze(1))
                            s_p = Point(target_line_sort_keypoints[eval_i-minus][1][0])
                            e_p = Point(target_line_sort_keypoints[eval_i-minus][1][1])
                            if s_p.within(poly) and e_p.within(poly):
                                line_bool_val.append('True')
                            else:
                                line_bool_val.append('False')
                        if 'True' in line_bool_val:
                            #print(line_CLASSES[eval_i], 'True')
                            csv_line_result.append(line_CLASSES[eval_i]+": True")
                        else:
                            #print(line_CLASSES[eval_i], 'False')
                            csv_line_result.append(line_CLASSES[eval_i]+": False")
                        detect_line_num = detect_line_num - 1

                    elif target_line_sort_keypoints[eval_i-minus][0] != line_CLASSES[eval_i]:
                        #print(line_CLASSES[eval_i], 'Nothing')
                        minus = minus + 1
"""

xray_type = sys.argv[1]
config_file = './SOLO/demo/cfg/' + xray_type + '/solov2_r101_dcn_fpn_8gpu_3x_custom.py'
checkpoint_file = './SOLO/demo/weight/'+ xray_type +'/epoch.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test a single image
dicom_list_str = []
with open(sys.argv[2], 'r') as read_obj:
    # pass the file object to reader() to get the reader object
    csv_reader = reader(read_obj)
    # Iterate over each row in the csv using reader object
    for row in csv_reader:
        # row variable is a list that represents a row in csv
        dicom_list_str = row

for kk in range(len(dicom_list_str)-1):
    img = "./eval/png/" + dicom_list_str[kk] + ".dcm.png"
    result = inference_detector(model, img)

    #############################################################################################################
    # segmentation base infomation
    cur_result = result[0]
    seg_label = cur_result[0]
    seg_label = seg_label.cpu().numpy().astype(np.uint8)
    cate_label = cur_result[1]
    cate_label = cate_label.cpu().numpy()
    score = cur_result[2].cpu().numpy()

    vis_inds = score > 0.3
    seg_label = seg_label[vis_inds]
    num_mask = seg_label.shape[0]
    cate_label = cate_label[vis_inds]
    cate_score = score[vis_inds]

    img = mmcv.imread(img)
    img_show = img.copy()
    h, w, _ = img.shape

    blank_image = np.zeros((h,w,3), np.uint8)
    blank_mask = np.zeros((h,w,3), np.uint8)

    mask_density = []
    direction_vector = []
    for idx in range(num_mask):
        cur_mask = seg_label[idx, :, :]
        cur_mask = mmcv.imresize(cur_mask, (w, h))
        cur_mask = (cur_mask > 0.5).astype(np.int32)
        mask_density.append(cur_mask.sum())

    orders = np.argsort(mask_density)
    seg_label = seg_label[orders]
    cate_label = cate_label[orders]
    cate_score = cate_score[orders]


    np.random.seed(42)
    color_masks = [
        np.random.randint(0, 256, (1, 3), dtype=np.uint8)
        for _ in range(num_mask)
    ]
    #############################################################################################################
    # segmentation class name
    CLASSES = []
    if xray_type == 'X1_FSAP':
        CLASSES = ('L_AP1', 'L_AP2', 'L_AP3', 'L_AP4', 'L_AP5', 'L_AP6', 'L_AP7', 'L_AP8', 'R_AP1', 'R_AP2', 'R_AP3', 'R_AP4', 'R_AP5', 'R_AP6', 'R_AP7', 'R_AP8')
    if xray_type == 'X2_FSLL':
        CLASSES = ('L_LAT2', 'L_LAT1', 'L_LAT3', 'L_LAT4', 'L_LAT5', 'L_LAT6')
    if xray_type == 'X2_FSLR':
        CLASSES = ('R_LAT2', 'R_LAT1', 'R_LAT3', 'R_LAT4', 'R_LAT5', 'R_LAT6')
    if xray_type == 'X3_FLOL':
        CLASSES = ('L_LOB1', 'L_LOB2')
    if xray_type == 'X3_FLOR':
        CLASSES = ('R_LOB1', 'R_LOB2')
    if xray_type == 'X4_HAV':
        CLASSES = ('R_HAV1', 'R_HAV2', 'L_HAV1', 'L_HAV2')
    if xray_type == 'X5_AWBAP': # error maybe
        CLASSES = ('R_AWB_AP1', 'R_AWB_AP2', 'R_AWB_AP3', 'L_AWB_AP1', 'L_AWB_AP2', 'L_AWB_AP3')
    if xray_type == 'X6_AWBLL':
        CLASSES = ('L_AWB_LAT1', 'L_AWB_LAT2')
    if xray_type == 'X6_AWBLR':
        CLASSES = ('R_AWB_LAT1', 'R_AWB_LAT2')
    if xray_type == 'X7_KWBAP':
        CLASSES = ('L_KWB_AP1', 'L_KWB_AP2', 'L_KWB_AP3', 'R_KWB_AP1', 'R_KWB_AP2', 'R_KWB_AP3')
    if xray_type == 'X8_KWBLL':
        CLASSES = ('L_KWB_LAT1', 'L_KWB_LAT2')
    if xray_type == 'X8_KWBLR':
        CLASSES = ('R_KWB_LAT1', 'R_KWB_LAT2')
    if xray_type == 'T1_TG':
        CLASSES = ('L_T1', 'L_T2', 'L_T3', 'R_T1', 'R_T2', 'R_T3', 'R_T4', 'L_T4')

    #############################################################################################################
    # segmentation vector calculation
    keypoints_dict = {}

    for idx in range(num_mask):
        class_name = idx
        idx = -(idx+1)
        cur_mask = seg_label[idx, :, :]
        cur_mask = mmcv.imresize(cur_mask, (w, h))
        cur_mask = (cur_mask > 0.5).astype(np.uint8)
        if cur_mask.sum() == 0:
            continue
        color_mask = color_masks[idx]
        blank_color_mask = color_masks[0]
        cur_mask_bool = cur_mask.astype(np.bool)
        img_show[cur_mask_bool] = img[cur_mask_bool] * 0.5 + color_mask * 0.5

        blank_image[cur_mask_bool] = img[cur_mask_bool] * 0.5 + blank_color_mask * 0.5 # draw mask image
        gray = cv.cvtColor(blank_image, cv.COLOR_BGR2GRAY)
        _, bw = cv.threshold(gray, 1, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
        blank_mask = bw
        blank_image = np.zeros((h,w,3), np.uint8) # initialize original image

        #cv.namedWindow("class_name", cv.WINDOW_NORMAL)
        #cv.resizeWindow("class_name", 1920, 1080)
        #cv.imshow("class_name", blank_mask)
        #cv.waitKey(0)
        #cv.imwrite(class_name+".png", blank_mask)
        cur_cate = cate_label[idx]
        if xray_type == 'X3_FLOL':
            if cur_cate == 2:
                cur_cate = 1
        if xray_type == 'X3_FLOR':
            if cur_cate == 1:
                cur_cate = 0
            if cur_cate == 3:
                cur_cate = 1
        if xray_type == 'X6_AWBLL' or xray_type == 'X6_AWBLR':
            if cur_cate == 2:
                cur_cate = 0
            if cur_cate == 3:
                cur_cate = 1
        if xray_type == 'X8_KWBLL' or xray_type == 'X8_KWBLR':
            if cur_cate == 2:
                cur_cate = 1
            if cur_cate == 3:
                cur_cate = 0
        keypoints_dict[CLASSES[cur_cate]] = blank_mask

    #cv.namedWindow("test", cv.WINDOW_NORMAL)
    #cv.resizeWindow("test", 1920, 1080)
    #cv.imshow("test", img_show)
    #cv.waitKey(1)

    sort_keypoints = sorted(keypoints_dict.items())
    #print(sort_keypoints)

    #############################################################################################################
    # json parsing
    json_file = "./eval/json/" + dicom_list_str[kk] + ".json"
    with open(json_file, "r", encoding="utf8") as f:
        contents = f.read()
        json_data = json.loads(contents)

    point_list = []
    point_dict = {}
    line_dict = {}
    for obj in json_data["ArrayOfannotation_info"]:
        label_n = obj['xyvalue']['label_val']['preset_detail_name']
        if obj['objectname'] == "LabelPolyLine":
            for points in obj['xyvalue']['pos_list']:
                point_list.append([int(points['X']), int(points['Y'])])

            mask1 = np.zeros((h,w),dtype = np.uint8)
            polygon1 = np.array(point_list)
            img = cv.fillPoly(mask1,[polygon1],(255,255,255))
            point_list.clear()
            point_dict[label_n] = img
        if obj['objectname'] == "LabelCenterLine":
            line_list = []
            val_x = obj['xyvalue']['start_c_pos']['X']
            val_y = obj['xyvalue']['start_c_pos']['Y']
            line_list.append([float(val_x),float(val_y)])
            val_x = obj['xyvalue']['end_c_pos']['X']
            val_y = obj['xyvalue']['end_c_pos']['Y']
            line_list.append([float(val_x),float(val_y)])
            line_dict[label_n] = line_list

    target_sort_keypoints = sorted(point_dict.items())
    #print(target_sort_keypoints)
    target_line_sort_keypoints = sorted(line_dict.items())
    #print(target_line_sort_keypoints)

    #############################################################################################################
    # segmentation IoU calculation
    iou_list = []
    cp_iou_list = []
    Overlapping_list = []
    Combined_list = []
    obj_check = []
    for base_i in range(len(CLASSES)):
        json_chk = 0
        for target_i in range(len(target_sort_keypoints)):
            detect_chk = 0
            if CLASSES[base_i] == target_sort_keypoints[target_i][0]:
                json_chk = 1
                for pre_i in range(len(sort_keypoints)):
                    if sort_keypoints[pre_i][0] == target_sort_keypoints[target_i][0]:
                        detect_chk = 1
                        intersection = np.logical_and(target_sort_keypoints[target_i][1], sort_keypoints[pre_i][1])
                        union = np.logical_or(target_sort_keypoints[target_i][1], sort_keypoints[pre_i][1])
                        Overlapping_val = np.sum(intersection)
                        Combined_val = np.sum(union)
                        iou_score = Overlapping_val / Combined_val
                        significant_figures = "{:.2f}".format(iou_score)
                        Overlapping_figures = "{:.2f}".format(Overlapping_val)
                        Combined_figures = "{:.2f}".format(Combined_val)
                        iou_list.append(significant_figures)
                        cp_iou_list.append(sort_keypoints[pre_i][0]+": "+significant_figures)
                        obj_check.append(sort_keypoints[pre_i][0]+": True")
                        Overlapping_list.append(sort_keypoints[pre_i][0]+": "+Overlapping_figures)
                        Combined_list.append(sort_keypoints[pre_i][0]+": "+Combined_figures)
                        print(sort_keypoints[pre_i][0],iou_score)
                if detect_chk == 0:
                    iou_list.append("0")
                    cp_iou_list.append(target_sort_keypoints[target_i][0]+": detect missing")
                    obj_check.append(target_sort_keypoints[target_i][0]+": detect missing")
                    Overlapping_list.append(target_sort_keypoints[target_i][0]+": detect missing")
                    Combined_list.append(target_sort_keypoints[target_i][0]+": detect missing")
                    print(target_sort_keypoints[target_i][0], 'detect missing')
        if json_chk == 0:
            cp_iou_list.append(CLASSES[base_i]+": json nothing")
            obj_check.append(CLASSES[base_i]+": False")
            Overlapping_list.append(CLASSES[base_i]+": json nothing")
            Combined_list.append(CLASSES[base_i]+": json nothing")
            print(CLASSES[base_i], 'json nothing')
    if len(iou_list) == 0:
        mean_IoU = 0
    else:
        float_iou_list = [float(x) for x in iou_list]
        mean_IoU = sum(float_iou_list)/len(iou_list)

    #############################################################################################################
    """
    csv_line_result = []
    line_CLASSES = ()
    seg_CLASSES = ()
    if xray_type == 'X1_FSAP':
        line_CLASSES = ('L_V1', 'L_V2', 'L_V3', 'L_V4', 'L_V5', 'L_V6', 'L_V7', 'R_V1', 'R_V2', 'R_V3', 'R_V4', 'R_V5', 'R_V6', 'R_V7')
        seg_CLASSES = ('L_AP1', 'L_AP2', 'L_AP3', 'L_AP4', 'L_AP5', 'L_AP7', 'L_AP8', 'R_AP1', 'R_AP2', 'R_AP3', 'R_AP4', 'R_AP5', 'R_AP7', 'R_AP8')

    if xray_type == 'X2_FSLR' or xray_type == 'X2_FSLL':
        if xray_type == 'X2_FSLL':
            line_CLASSES = ('L_V1', 'L_V2', 'L_V3', 'L_V4')
            seg_CLASSES = ('L_LAT1', 'L_LAT3', 'L_LAT4', 'L_LAT6')
        else:
            line_CLASSES = ('R_V1', 'R_V2', 'R_V3', 'R_V4')
            seg_CLASSES = ('R_LAT1', 'R_LAT3', 'R_LAT4', 'R_LAT6')

    if xray_type == 'X4_HAV':
        line_CLASSES = ('L_V1', 'L_V2', 'R_V1', 'R_V2')
        seg_CLASSES = ('L_HAV1', 'L_HAV2', 'R_HAV1', 'R_HAV2')

    if xray_type == 'X5_AWBAP':
        line_CLASSES = ('L_V1', 'R_V1')
        seg_CLASSES = ('L_AWB_AP1', 'R_AWB_AP1')

    if xray_type == 'X6_AWBLL' or xray_type == 'X6_AWBLR':
        if xray_type == 'X6_AWBLL':
            line_CLASSES = ('L_V1',)
            seg_CLASSES = ('L_AWB_LAT1',)
        else:
            line_CLASSES = ('R_V1',)
            seg_CLASSES = ('R_AWB_LAT1',)

    if xray_type == 'X7_KWBAP':
        line_CLASSES = ('L_V1', 'L_V2', 'R_V1', 'R_V2')
        seg_CLASSES = ('L_KWB_AP1', 'L_KWB_AP2', 'R_KWB_AP1', 'R_KWB_AP2')

    if xray_type == 'X8_KWBLL' or xray_type == 'X8_KWBLR':
        if xray_type == 'X8_KWBLL':
            line_CLASSES = ('L_V1', 'L_V2')
            seg_CLASSES = ('L_KWB_LAT1', 'L_KWB_LAT2')
        else:
            line_CLASSES = ('R_V1', 'R_V2')
            seg_CLASSES = ('R_KWB_LAT1', 'R_KWB_LAT2')

    if xray_type == 'T1_TG':
        line_CLASSES = ('L_V1', 'L_V2', 'L_V3', 'R_V1', 'R_V2', 'R_V3')
        seg_CLASSES = ('L_T1', 'L_T1', 'L_T2', 'R_T1', 'R_T1', 'R_T2')

    if len(line_CLASSES) != 0 or len(seg_CLASSES) != 0:
        line_exist(xray_type, line_CLASSES, seg_CLASSES, sort_keypoints, target_line_sort_keypoints, csv_line_result)
    """
    #############################################################################################################

    base_n = os.path.basename(json_file)
    file_n = base_n.split('.')
    with open('seg.csv', 'a', newline='') as fd:
        wr = csv.writer(fd)
        wr.writerow([file_n[0],Overlapping_list,Combined_list,cp_iou_list,"{:.2f}".format(mean_IoU)])

    with open('inspection.csv', 'a', newline='') as fd:
        wr = csv.writer(fd)
        wr.writerow([file_n[0],obj_check])
