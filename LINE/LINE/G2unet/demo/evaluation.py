import os
from collections.abc import Iterable
import argparse
from functools import partial
from PIL import Image
from PIL import ImageDraw, ImageFont

from tqdm import tqdm
import scipy.io as sio
from scipy.optimize import linear_sum_assignment as assign
import numpy as np

from model.utils import mkdir, toYaml, dis2, colorRGB, getPointsFromHeatmap, get_config
import csv
from csv import reader
import json
import cv2
import shutil

PATH_DIC = {
    'cephalometric': '../data/ISBI2015_ceph/raw',
    'hand': '../data/hand/jpg',
    'chest': '../data/chest/pngs',
}

FONT_PATH = './times.ttf'
THRESHOLD = [2, 2.5, 3, 4, 6, 9, 10]
CEPH_PHYSICAL_FACTOR = 0.46875
WRIST_WIDTH = 50  # mm
DRAW_TEXT_SIZE_FACTOR = { 'cephalometric': 1.13, 'hand': 1, 'chest': 1.39}



def np2py(obj):
    if isinstance(obj, Iterable):
        return [np2py(i) for i in obj]
    elif isinstance(obj, np.generic):
        return np.asscalar(obj)
    else:
        return obj


def radial(pt1, pt2, factor=1):
    if  not isinstance(factor,Iterable):
        factor = [factor]*len(pt1)
    return sum(((i-j)*s)**2 for i, j,s  in zip(pt1, pt2, factor))**0.5


def draw_text(image, text, factor=1):
    width = round(40*factor)
    padding = round(10*factor)
    margin = round(5*factor)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(FONT_PATH, width)
    text_size = draw.textsize(text, font)
    text_w = padding
    text_h = image.height-width-padding
    text_w = text_h = padding
    pos = [text_w, text_h, text_w + text_size[0], text_h+text_size[1]]
    draw.rectangle(pos, fill='#000000')  # 用于填充
    draw.text((text_w, text_h), text, fill='#00ffff', font=font)  # blue
    return image


def cal_all_distance(points, gt_points, factor=1):
    '''
    points: [(x,y,z...)]
    gt_points: [(x,y,z...)]
    return : [d1,d2, ...]
    '''
    n1 = len(points)
    n2 = len(gt_points)
    if n1 == 0:
        print("[Warning]: Empty input for calculating mean and std")
        return 0, 0
    if n1 != n2:
        raise Exception("Error: lengthes dismatch, {}<>{}".format(n1, n2))
    return [radial(p, q, factor) for p, q in zip(points, gt_points)]


def assigned_distance(points, gt_points, factor=1):
    n = len(points)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            mat[i, j] = radial(points[i], gt_points[j])*factor
    return [mat[i, j] for i, j in zip(*assign(mat))]


def get_sdr(distance_list, threshold=THRESHOLD):
    ''' successfully detection rate (pixel)
    '''
    ret = {}
    n = len(distance_list)
    for th in threshold:
        ret[th] = sum(d <= th for d in distance_list)/n
    return ret


def saveLabels(path, points, size):
    with open(path, 'w') as f:
        f.write('{}\n'.format(len(points)))
        for pt in points:
            ratios = ['{:.4f}'.format(x/X) for x,X in zip(pt,size)]
            f.write(' '.join(ratios)+'\n')

def evaluate(input_path, output_path, save_img=False, assigned=False, IS_DRAW_TEXT=False):
    mkdir(output_path)
    dataset = os.path.basename(input_path).lower()
    image_path_pre = PATH_DIC[dataset]
    #print('\n'+'-'*20+dataset+'-'*20)
    print('input : ', input_path)
    print('output: ', output_path)
    print('image : ', image_path_pre)
    gen = [gt_p for gt_p in os.listdir(input_path) if gt_p.endswith('_gt.npy')]
    pbar = tqdm(gen, ncols=80)
    data_num = len(gen)
    out_label_path = os.path.join(output_path, 'labels')
    mkdir(out_label_path)
    out_gt_path = os.path.join(output_path, 'gt_laels')
    mkdir(out_gt_path)
    if save_img:
        out_img_path = os.path.join(output_path, 'images')
        mkdir(out_img_path)
    physical_factor = 1
    if dataset == 'cephalometric':
        physical_factor = CEPH_PHYSICAL_FACTOR
    distance_list = []
    pixel_dis_list = []
    assigned_list = []
    for i, gt_p in enumerate(pbar):
        pbar.set_description('{:03d}/{:03d}: {}'.format(i+1, data_num, gt_p))
        name = gt_p[:-7]
        heatmaps = np.load(os.path.join(input_path, name+'_output.npy'))
        img_size = heatmaps.shape[1:]
        cur_points = getPointsFromHeatmap(heatmaps)
        gt_map = np.load(os.path.join(input_path, gt_p))
        cur_gt = getPointsFromHeatmap(gt_map)


        if dataset == 'hand':
            physical_factor = WRIST_WIDTH/radial(cur_gt[0], cur_gt[4])
        cur_distance_list = cal_all_distance(cur_points, cur_gt, physical_factor)
        cur_pixel_dis = cal_all_distance(cur_points, cur_gt, 1)
        distance_list += cur_distance_list
        pixel_dis_list += cur_pixel_dis
        if assigned:
            assigned_list += assigned_distance(cur_points,
                                               cur_gt, physical_factor)
        saveLabels(out_label_path+'/'+name+'.txt', cur_points, img_size)
        saveLabels(out_gt_path+'/'+name+'.txt', cur_gt, img_size)
        if save_img:
            if dataset == 'cephalometric':
                img_path = image_path_pre+'/'+name+'.bmp'
            elif dataset == 'hand':
                img_path = image_path_pre+'/'+name+'.jpg'
            else:
                img_path = image_path_pre+'/'+name+'.png'
            img = Image.open(img_path)
            img = img.resize(img_size)
            img = np.array(img)
            if len(img.shape) == 2:
                img = np.stack([img, img, img], axis=-1)
            img = np.transpose(img, (1, 0, 2))
            for p, q in zip(cur_points, cur_gt):
                colorRGB(img, [q], partial(dis2, q), 20, [0, 255, 0])
                colorRGB(img, [p], partial(dis2, p), 20, [255, 0, 0])
            img = np.transpose(img, (1, 0, 2))
            img = Image.fromarray(img)
            mre = np.mean(cur_distance_list)
            mre_str = '{:.3f}'.format(mre)
            if IS_DRAW_TEXT:
                img = draw_text(img, mre_str, DRAW_TEXT_SIZE_FACTOR[dataset])
            img.save(out_img_path+'/'+name + '_' + mre_str+'.png')
    if assigned:
        print('assigned...')
    return assigned_list if assigned else distance_list, pixel_dis_list


def analysis(li1, dataset):
    #print('\n'+'-'*20+dataset+'-'*20)
    summary = {}
    mean1, std1, = np.mean(li1), np.std(li1)
    sdr1 = get_sdr(li1)
    n = len(li1)
    summary['LANDMARK_NUM'] = n
    summary['MRE'] = np2py(mean1)
    summary['STD'] = np2py(std1)
    summary['SDR'] = {k: np2py(v) for k, v in sdr1.items()}
    #print('MRE:', mean1)
    #print('STD:', std1)
    #print('SDR:')
    #for k in sorted(sdr1.keys()):
        #print('     {}: {}'.format(k, sdr1[k]))
    return summary


def get_args():
    parser = argparse.ArgumentParser()
    # optinal
    parser.add_argument("-s", "--save_img", action='store_true')
    parser.add_argument("-d", "--draw_text", action='store_true')
    parser.add_argument("-o", "--output", type=str)
    parser.add_argument("-C", "--config", type=str)
    parser.add_argument("-a", "--assigned", action='store_true')
    parser.add_argument("-T", "--type", help='x-ray type')
    parser.add_argument("-F", "--file_list", help='x-ray file list')
    # required
    parser.add_argument("-i", "--input", type=str, required=True)
    return parser.parse_args()

def removeAllFiles(dirpath):
    if os.path.exists(dirpath):
        for file in os.scandir(dirpath):
            os.remove(file.path)

def removedirs(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)


if __name__ == "__main__":
    args = get_args()

    print("labeling checking")
    dicom_list_str = []
    with open(args.file_list, 'r') as read_obj:
        # pass the file object to reader() to get the reader object
        csv_reader = reader(read_obj)
        # Iterate over each row in the csv using reader object
        for row in csv_reader:
            # row variable is a list that represents a row in csv
            dicom_list_str = row
    print(dicom_list_str)

    if len(dicom_list_str) != 0:
        dic = {}
        pixel_dic = {}
        if not args.output:
            output = os.path.join('.eval', args.input.replace('/', '_'))
        for d in os.listdir(args.input):
            inp = os.path.join(args.input, d)
            if os.path.isdir(inp):
                phy_dis, pixel_dis = evaluate(inp, os.path.join(output, d), args.save_img,
                                              args.assigned, args.draw_text)
                phy_dis = np2py(phy_dis)
                pixel_dis = np2py(pixel_dis)
                dic[d] = phy_dis
                pixel_dic[d+'_pixel'] = pixel_dis
        toYaml(output+'/distance.yaml', dic)
        summary = {}
        li_total = []
        for d, phy_dis in dic.items():
            pixel_dis = pixel_dic[d+'_pixel']
            summary[d] = analysis(phy_dis, d)
            li_total += pixel_dis
        summary['total'] = analysis(li_total, 'total')
        toYaml(output+'/summary.yaml', summary)

        # line classes
        eval_loss = 0
        line_CLASSES = ()
        if args.type == 'X1_FSAP':
            line_CLASSES = ('L_E7', 'L_E8', 'L_V1', 'L_V2', 'L_V3', 'L_V4', 'L_V5', 'L_V6', 'L_V7', 'R_E7', 'R_E8', 'R_V1', 'R_V2', 'R_V3', 'R_V4', 'R_V5', 'R_V6', 'R_V7')
            eval_loss = 2
            false_loss = 5
        if args.type == 'X2_FSLL':
            line_CLASSES = ('L_E1', 'L_E2', 'L_V1', 'L_V2', 'L_V4')
            eval_loss = 2
            false_loss = 5
        if args.type == 'X2_FSLR':
            line_CLASSES = ('R_E1', 'R_E2', 'R_V1', 'R_V2', 'R_V4')
            eval_loss = 2
            false_loss = 5
        if args.type == 'X4_HAV':
            line_CLASSES = ('L_V1', 'L_V2', 'R_V1', 'R_V2')
            eval_loss = 2
            false_loss = 5
        if args.type == 'X5_AWBAP':
            line_CLASSES = ('L_E1', 'L_E2', 'L_V1', 'R_E1', 'R_E2', 'R_V1')
            eval_loss = 2
            false_loss = 5
        if args.type == 'X6_AWBLL':
            line_CLASSES = ('L_E1', 'L_V1')
            eval_loss = 2
            false_loss = 5
        if args.type == 'X6_AWBLR':
            line_CLASSES = ('R_E1', 'R_V1')
            eval_loss = 2
            false_loss = 5
        if args.type == 'X7_KWBAP':
            line_CLASSES = ('L_E1', 'L_E2', 'L_V1', 'L_V2', 'R_E1', 'R_E2', 'R_V1', 'R_V2')
            eval_loss = 2
            false_loss = 5
        if args.type == 'X8_KWBLL':
            line_CLASSES = ('L_V1', 'L_V2')
            eval_loss = 2
            false_loss = 5
        if args.type == 'X8_KWBLR':
            line_CLASSES = ('R_V1', 'R_V2')
            eval_loss = 2
            false_loss = 5
        if args.type == 'T1_TG':
            line_CLASSES = ('L_E1', 'L_E2', 'L_E3', 'L_V1', 'L_V2', 'L_V3', 'R_E1', 'R_E2', 'R_E3', 'R_V1', 'R_V2', 'R_V3')
            eval_loss = 2
            false_loss = 5

        # labeling checking
        for f_list in range(len(dicom_list_str)-1):
            obj_check = []
            pre_img = cv2.imread("./G2unet/data/ISBI2015_ceph/raw/"+dicom_list_str[f_list]+".bmp", cv2.IMREAD_COLOR)
            pre_h, pre_w, pre_c = pre_img.shape

            with open("./eval/json/"+dicom_list_str[f_list]+".json", "r", encoding="utf8") as f:
                contents = f.read()
                json_data = json.loads(contents)

            keypoints_dict = {}
            # gt_labeling find
            for obj in json_data["ArrayOfannotation_info"]:
                if obj['objectname'] == "LabelCenterLine":
                    val_list = []
                    label_info = obj['xyvalue']['label_val']['preset_detail_name'] # label name
                    val_x = obj['xyvalue']['start_c_pos']['X']#/w
                    val_y = obj['xyvalue']['start_c_pos']['Y']#/h
                    val_list.append(int(val_x))
                    val_list.append(int(val_y))
                    val_x = obj['xyvalue']['end_c_pos']['X']#/w
                    val_y = obj['xyvalue']['end_c_pos']['Y']#/h
                    val_list.append(int(val_x))
                    val_list.append(int(val_y))
                    keypoints_dict[label_info] = val_list

                if obj['objectname'] == "LabelLine":
                    val_list = []
                    label_info = obj['xyvalue']['label_val']['preset_detail_name'] # label name
                    val_x = obj['xyvalue']['start_pos']['X']#/w
                    val_y = obj['xyvalue']['start_pos']['Y']#/h
                    val_list.append(int(val_x))
                    val_list.append(int(val_y))
                    val_x = obj['xyvalue']['end_pos']['X']#/w
                    val_y = obj['xyvalue']['end_pos']['Y']#/h
                    val_list.append(int(val_x))
                    val_list.append(int(val_y))
                    keypoints_dict[label_info] = val_list

            gt_sort_keypoints = sorted(keypoints_dict.items())
            #print(gt_sort_keypoints)


            # pre_labeling find
            with open("./.eval/.runs_GU2Net_runs_results_test_epoch/cephalometric/labels/"+dicom_list_str[f_list]+".txt", 'r') as file:
                pre_output = [ line.strip().split(' ') for line in file.readlines()]
            pre_output = [[float(j) for j in i] for i in pre_output]
            del pre_output[0]
            pre_keypoints_dict = {}
            i = 0
            for compare_n in range(int(len(pre_output)/2)):
                pre_keypoints_dict[line_CLASSES[compare_n]] = [int(pre_output[i][0]*pre_w), int(pre_output[i][1]*pre_h), int(pre_output[i+1][0]*pre_w), int(pre_output[i+1][1]*pre_h)]
                #pre_img = cv2.line(pre_img,(int(pre_output[i][0]*pre_w),int(pre_output[i][1]*pre_h)),(int(pre_output[i+1][0]*pre_w),int(pre_output[i+1][1]*pre_h)),(0,0,255),5)
                i=i+2
            pre_sort_keypoints_dict = sorted(pre_keypoints_dict.items())
            #print(pre_sort_keypoints_dict)
            # raw images
            #os.makedirs(".\\eval\\image_results", exist_ok=True)
            #cv2.imwrite(".\\eval\\image_results\\" + dicom_list_str[f_list] + ".png", pre_img)



            TPv = 0
            FPv = 0
            FNv = 0
            for base_i in range(len(line_CLASSES)):
                json_chk = 0
                for target_i in range(len(gt_sort_keypoints)):
                    detect_chk = 0
                    if line_CLASSES[base_i] == gt_sort_keypoints[target_i][0]:
                        json_chk = 1
                        for pre_i in range(len(pre_sort_keypoints_dict)):
                            if pre_sort_keypoints_dict[pre_i][0] == gt_sort_keypoints[target_i][0]:
                                detect_chk = 1
                                obj_check.append(pre_sort_keypoints_dict[pre_i][0]+": True")
                                #f1-score calculation
                                loss_val = 0
                                pre_numerator = abs(pre_sort_keypoints_dict[pre_i][1][1] - pre_sort_keypoints_dict[pre_i][1][3])
                                pre_denominator = abs(pre_sort_keypoints_dict[pre_i][1][0] - pre_sort_keypoints_dict[pre_i][1][2])
                                if pre_numerator == 0 or pre_denominator == 0:
                                    pre_lean = 0
                                else:
                                    pre_lean = pre_numerator / pre_denominator
                                gt_numerator = abs(gt_sort_keypoints[target_i][1][1] - gt_sort_keypoints[target_i][1][3])
                                gt_denominator = abs(gt_sort_keypoints[target_i][1][0] - gt_sort_keypoints[target_i][1][2])
                                if gt_numerator == 0 or gt_denominator == 0:
                                    gt_lean = 0
                                else:
                                    gt_lean = gt_numerator / gt_denominator
                                loss_val = abs(gt_lean - pre_lean)
                                #print(loss_val)
                                if loss_val > eval_loss:
                                    FPv = FPv + 1
                                if loss_val > false_loss:
                                    FNv = FNv + 1
                                if loss_val <= eval_loss or loss_val <= false_loss:
                                    TPv = TPv + 1

                        if detect_chk == 0:
                            obj_check.append(gt_sort_keypoints[target_i][0]+": False")

                if json_chk == 0:
                    obj_check.append(line_CLASSES[base_i]+": False")

            #print(dicom_list_str[f_list], obj_check)
            #print(TPv,FPv,FNv)
            precision = 0
            recall = 0
            f1_score = 0
            if TPv != 0:
                precision = TPv/(TPv+FPv)
                recall = TPv/(TPv+FNv)
            #print(precision,recall)
            if precision > 0 and recall > 0:
                f1_score = 2*((precision*recall)/(precision+recall))*100
                print("F1-score: ", f1_score)

            if len(gt_sort_keypoints) == len(pre_sort_keypoints_dict):
                if f1_score != 0:
                    with open('line.csv', 'a', newline='') as fd:
                        wr = csv.writer(fd)
                        wr.writerow([dicom_list_str[f_list],TPv,FPv,FNv,precision,recall,f1_score])
                with open('inspection.csv', 'a', newline='') as fd:
                    wr = csv.writer(fd)
                    wr.writerow([dicom_list_str[f_list],obj_check])

            else:
                with open('inspection.csv', 'a', newline='') as fd:
                    wr = csv.writer(fd)
                    wr.writerow([dicom_list_str[f_list],obj_check])


        # analysis folder and file remove
        dir_path = "./.eval"
        removedirs(dir_path)
        dir_path = "./.runs"
        removedirs(dir_path)
        dir_1 = "./G2unet/data/ISBI2015_ceph/raw"
        dir_2 = "./G2unet/data/ISBI2015_ceph/400_senior"
        dir_3 = "./G2unet/data/ISBI2015_ceph/400_junior"
        removeAllFiles(dir_1)
        removeAllFiles(dir_2)
        removeAllFiles(dir_3)
