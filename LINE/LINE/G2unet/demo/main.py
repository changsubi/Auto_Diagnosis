import os
import argparse
import torch
from model.runner import Runner
import csv
from csv import reader
from PIL import Image
import shutil
import cv2

def get_args():
    parser = argparse.ArgumentParser()
    # optional
    parser.add_argument("-C", "--config", help='configuration path')
    parser.add_argument("-c", "--checkpoint", help='model path')
    parser.add_argument("-g", "--cuda_devices", default='0')
    parser.add_argument("-m", "--model", type=str)
    parser.add_argument("-l", "--localNet", type=str)
    parser.add_argument("-n", "--name_list", nargs='+', type=str)
    parser.add_argument("-e", "--epochs", type=int)
    parser.add_argument("-L", "--lr", type=float)
    parser.add_argument("-w", "--weight_decay", type=float)
    parser.add_argument("-s", "--sigma", type=int)
    parser.add_argument("-x", "--mix_step", type=int)
    parser.add_argument("-u", "--use_background_channel", action='store_true')
    parser.add_argument("-T", "--type", help='x-ray type')
    parser.add_argument("-F", "--file_list", help='x-ray file list')
    # required
    parser.add_argument("-r", "--run_name", type=str, required=True)
    parser.add_argument("-d", "--run_dir", type=str, required=True, default='.runs')
    parser.add_argument(
        "-p", "--phase", choices=['train', 'validate', 'test'], required=True)
    return parser.parse_args()


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    args = get_args()
    args.config = "./G2unet/demo/cfg/" + args.type + "/config.yaml"
    args.checkpoint = "./G2unet/demo/weight/" + args.type + "/best_GU2Net_runs_epoch.pt"
    label_count = 0
    if args.type == 'X1_FSAP':
        label_count = 36
    if args.type == 'X2_FSLL':
        label_count = 10
    if args.type == 'X2_FSLR':
        label_count = 10
    if args.type == 'X4_HAV':
        label_count = 8
    if args.type == 'X5_AWBAP':
        label_count = 12
    if args.type == 'X6_AWBLL':
        label_count = 4
    if args.type == 'X6_AWBLR':
        label_count = 4
    if args.type == 'X7_KWBAP':
        label_count = 16
    if args.type == 'X8_KWBLL':
        label_count = 4
    if args.type == 'X8_KWBLR':
        label_count = 4
    if args.type == 'T1_TG':
        label_count = 24
    if label_count != 0:
        # test a single image
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
            for f_list in range(len(dicom_list_str)-1):
                #img = Image.open(".\\eval\\png\\"+dicom_list_str[f_list]+".dcm.png")
                #img.save(".\\G2unet\\data\\ISBI2015_ceph\\raw\\"+dicom_list_str[f_list]+".bmp")
                img = cv2.imread("./eval/png/"+dicom_list_str[f_list]+".dcm.png")
                cv2.imwrite("./G2unet/data/ISBI2015_ceph/raw/"+dicom_list_str[f_list]+".bmp", img)
                f = open("./G2unet/data/ISBI2015_ceph/400_senior/"+dicom_list_str[f_list]+".txt","w")
                for f_label in range(label_count):
                    f.write(str(1) + "," + str(1)+"\n")
                f.close()
                shutil.copy("./G2unet/data/ISBI2015_ceph/400_senior/"+dicom_list_str[f_list]+".txt", "./G2unet/data/ISBI2015_ceph/400_junior/"+dicom_list_str[f_list]+".txt")

            Runner(args).run()
    else:
        print("no line data")
