'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2023-03-29 15:21:26
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2023-07-10 10:12:38
FilePath: /mmdetection/tools/feature_visualization.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import cv2
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from utils.global_var import set_value, get_value

def featuremap_2_heatmap(feature_map):
    assert isinstance(feature_map, torch.Tensor)
    #feature_map=feature_map[:,:,2:254,2:254]
    feature_map = feature_map.detach()
    #feature_map=F.sigmoid(feature_map)
    heatmap = feature_map[:,0,:,:]*0
    heatmaps = []
    for c in range(feature_map.shape[1]):
        heatmap+=feature_map[:,c,:,:]
    heatmap = heatmap.cpu().numpy()
    heatmap = np.mean(heatmap, axis=0)

    heatmap = np.maximum(heatmap, 0)
    # heatmap /= np.max(heatmap)
    if heatmap.max()!=heatmap.min():
        heatmap = (heatmap-heatmap.min())/(heatmap.max()-heatmap.min())
    else :
        heatmap = heatmap
    
    heatmaps.append(heatmap)

    return heatmaps

def draw_feature_map(features,save_dir = '/STAT/wzs/Project/yolov5/LL/hrnet_feat_img',name = 'reg'):
    i=0
    if isinstance(features,torch.Tensor):
        for heat_maps in features:
            heat_maps=heat_maps.unsqueeze(0)
            heatmaps = featuremap_2_heatmap(heat_maps)
            # 这里的h,w指的是你想要把特征图resize成多大的尺寸
            heatmaps[0]=heatmaps[0].astype(np.float32)
            heatmaps[0] = cv2.resize(heatmaps[0], (1024, 1024))  
            for heatmap in heatmaps:
                #heatmap = (255-np.uint8(255 * heatmap))
                heatmap = np.uint8(255 * heatmap)
                # 下面这行将热力图转换为RGB格式 ，如果注释掉就是灰度图
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                # heatmap = cv2.bitwise_not(heatmap)
                heat_maps = torch.mean(heat_maps,dim=0,keepdim=True)
                img = get_value('input_img')
                img = img[i]
                img=img.permute(1,2,0).cpu().numpy()
                img=img*255.0
                superimposed_img = heatmap*0.4+img
                cv2.imwrite(os.path.join(save_dir,name +str(i)+'.png'), superimposed_img)
                i=i+1
                #plt.imshow(superimposed_img,cmap='gray')
                #plt.show()
    else:
        for featuremap in features:
            heatmaps = featuremap_2_heatmap(featuremap)
            # heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # 将热力图的大小调整为与原始图像相同
            for heatmap in heatmaps:
                heatmap = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                # superimposed_img = heatmap * 0.5 + img*0.3
                superimposed_img = heatmap
                #plt.imshow(superimposed_img,cmap='gray')
                #plt.show()
                # 下面这些是对特征图进行保存，使用时取消注释
                #cv2.imshow("1",superimposed_img)
                #cv2.waitKey(0)
                #cv2.destroyAllWindows()
                cv2.imwrite(os.path.join(save_dir,name +str(i)+'.png'), superimposed_img)
                i=i+1

def draw_feature_map1(features,save_dir = '/STAT/ll/code_10.7/SFR_LOC/ll/SFR/SAR_AIR1/img/',name = 'new_reg_feat'):
    i=0
    if isinstance(features,torch.Tensor):
        for heat_maps in features:
            heat_maps=heat_maps.unsqueeze(0)
            heatmaps = featuremap_2_heatmap(heat_maps)
            # 这里的h,w指的是你想要把特征图resize成多大的尺寸
            heatmaps[0] = cv2.resize(heatmaps[0], (1024, 1024))  
            for heatmap in heatmaps:
                # heatmap = (255-np.uint8(255 * heatmap))
                heatmap = np.uint8(255 * heatmap)
                # 下面这行将热力图转换为RGB格式 ，如果注释掉就是灰度图
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                # heatmap = cv2.bitwise_not(heatmap)
                heat_maps = torch.mean(heat_maps,dim=0,keepdim=True)
                img = get_value('input_img')
                img = img[i]
                img=img.permute(1,2,0).cpu().numpy()
                superimposed_img = heatmap*0.4+img
                cv2.imwrite(os.path.join(save_dir,name +str(i)+'.png'), superimposed_img)
                i=i+1
                #plt.imshow(superimposed_img,cmap='gray')
                #plt.show()
    else:
        for featuremap in features:
            heatmaps = featuremap_2_heatmap(featuremap)
            # heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # 将热力图的大小调整为与原始图像相同
            for heatmap in heatmaps:
                heatmap = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                # superimposed_img = heatmap * 0.5 + img*0.3
                superimposed_img = heatmap
                #plt.imshow(superimposed_img,cmap='gray')
                #plt.show()
                # 下面这些是对特征图进行保存，使用时取消注释
                #cv2.imshow("1",superimposed_img)
                #cv2.waitKey(0)
                #cv2.destroyAllWindows()
                cv2.imwrite(os.path.join(save_dir,name +str(i)+'.png'), superimposed_img)
                i=i+1

                
def draw_feature_map2(features,save_dir = '/STAT/ll/code_10.7/SFR_LOC/ll/SFR/SAR_AIR1/img/',name = 'loc_feat'):
    i=0
    if isinstance(features,torch.Tensor):
        for heat_maps in features:
            heat_maps=heat_maps.unsqueeze(0)
            heatmaps = featuremap_2_heatmap(heat_maps)
            # 这里的h,w指的是你想要把特征图resize成多大的尺寸
            heatmaps[0] = cv2.resize(heatmaps[0], (1024, 1024))  
            for heatmap in heatmaps:
                # heatmap = (255-np.uint8(255 * heatmap))
                heatmap = np.uint8(255 * heatmap)
                # 下面这行将热力图转换为RGB格式 ，如果注释掉就是灰度图
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                # heatmap = cv2.bitwise_not(heatmap)
                heat_maps = torch.mean(heat_maps,dim=0,keepdim=True)
                img = get_value('input_img')
                img = img[i]
                img=img.permute(1,2,0).cpu().numpy()
                superimposed_img = heatmap*0.4+img
                cv2.imwrite(os.path.join(save_dir,name +str(i)+'.png'), superimposed_img)
                i=i+1
                #plt.imshow(superimposed_img,cmap='gray')
                #plt.show()
    else:
        for featuremap in features:
            heatmaps = featuremap_2_heatmap(featuremap)
            # heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # 将热力图的大小调整为与原始图像相同
            for heatmap in heatmaps:
                heatmap = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                # superimposed_img = heatmap * 0.5 + img*0.3
                superimposed_img = heatmap
                #plt.imshow(superimposed_img,cmap='gray')
                #plt.show()
                # 下面这些是对特征图进行保存，使用时取消注释
                #cv2.imshow("1",superimposed_img)
                #cv2.waitKey(0)
                #cv2.destroyAllWindows()
                cv2.imwrite(os.path.join(save_dir,name +str(i)+'.png'), superimposed_img)
                i=i+1