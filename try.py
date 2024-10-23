'''
Author: error: git config user.name && git config user.email & please set dead value or install git
Date: 2022-09-28 14:38:18
LastEditors: error: git config user.name && git config user.email & please set dead value or install git
LastEditTime: 2022-09-28 14:43:01
FilePath: /yolov5/try.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch
import numpy as np

# list the path of the two kind of weight file below
src_file = '/home/hzl/STAT/dataset_process/prepare_data/convert_weight/pretrained_model/resnet18_updown.pt'
dst_file = 'pretrained_model/try.weight'
####################################################### load the .pth file #######################################################

net = torch.load(src_file,map_location=torch.device('cpu'))
print(1)