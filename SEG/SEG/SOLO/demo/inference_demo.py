from mmdet.apis import init_detector, inference_detector, show_result_pyplot, show_result_ins
import mmcv
import numpy as np


config_file = '../configs/solov2/solov2_r101_dcn_fpn_8gpu_3x_custom.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
checkpoint_file = '../work_dirs/solov2_release_r101_dcn_fpn_8gpu_3x/epoch_36.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test a single image
img = '111.png'
result = inference_detector(model, img)
show_result_ins(img, result, model.CLASSES, score_thr=0.2, out_file="demo_out.jpg")
