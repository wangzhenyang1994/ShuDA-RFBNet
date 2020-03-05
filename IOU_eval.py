#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 10:01:29 2019

@author: jjldr
"""

import torch
import torch.utils.data as data
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from dataset import bdd100k

import numpy as np
import cv2
import os
import os.path as osp
from PIL import Image
import argparse

from models.RFB_Net_shuffle import build_net

parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Evaluation')
parser.add_argument('--trained_model',
                    default='weights/VOC.pth', type=str,
                    help='Trained state_dict file path to open')
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = build_net("test",11).to(device)
net.load_state_dict({k.replace('module.',''):v for k,v in torch.load(args.trained_model).items()})
net.eval()
val_dataset = bdd100k("image_list/val_images.txt",'/cluster/home/it_stu27/bdd100k',512,False)
print(len(val_dataset))



def find_all_png(folder):
    paths = []
    for root, _, files in os.walk(folder, topdown=True):
        paths.extend([osp.join(root, f)
                      for f in files if osp.splitext(f)[1] == '.png'])
    return paths
def fast_hist(gt, prediction, n):
    k = (gt >= 0) & (gt < n)
    return np.bincount(
        n * gt[k].astype(int) + prediction[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
    ious = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    ious[np.isnan(ious)] = 0
    return ious

def evaluate_segmentation(gt_dir, result_dir, num_classes, key_length):
    gt_dict = dict([(osp.split(p)[1][:key_length], p)
                    for p in find_all_png(gt_dir)])
    result_dict = dict([(osp.split(p)[1][:key_length], p)
                        for p in find_all_png(result_dir)])
    result_gt_keys = set(gt_dict.keys()) & set(result_dict.keys())
    if len(result_gt_keys) != len(gt_dict):
        raise ValueError('Result folder only has {} of {} ground truth files.'
                         .format(len(result_gt_keys), len(gt_dict)))
    print('Found', len(result_dict), 'results')
    print('Evaluating', len(gt_dict), 'results')
    hist = np.zeros((num_classes, num_classes))
    i = 0
    gt_id_set = set()
    for key in sorted(gt_dict.keys()):
        gt_path = gt_dict[key]
        result_path = result_dict[key]
        gt = np.asarray(Image.open(gt_path, 'r'))
        gt_id_set.update(np.unique(gt).tolist())
        prediction = np.asanyarray(Image.open(result_path, 'r'))
        hist += fast_hist(gt.flatten(), prediction.flatten(), num_classes)
        i += 1
        if i % 100 == 0:
            print('Finished', i, per_class_iu(hist) * 100)
            
            
    #print(gt_id_set)
    #gt_id_set.remove(255)
    print('GT id set', gt_id_set)
    ious = per_class_iu(hist) * 100
    miou = np.mean(ious[list(gt_id_set)])
    return miou, list(ious)

if __name__=="__main__":
    for i in range(len(val_dataset)):
        image,label,seg_label = val_dataset[i]
        img_id = val_dataset.ids[i]
        image = Variable(image.unsqueeze(0)).cuda()
        seg_label = Variable(seg_label.type(torch.LongTensor)).cuda()
        output,_,_ = net(image)
        output = F.upsample(output,size=(720,1280),mode="bilinear")
        output = output.data.cpu().numpy()
        pred_label_img = np.argmax(output,axis=1)
        pred_label_img = pred_label_img.astype(np.uint8)
        pred_label_img = pred_label_img[0]
        path = "/cluster/home/it_stu27/bdd100k/results/img_seg/"+img_id+"_seg_img.png"
        cv2.imwrite(path,pred_label_img)
        print("***** process {} of {} *****".format(i,len(val_dataset)))
    gt_dir = "/cluster/home/it_stu27/bdd100k/drivable_maps/labels/val"
    result_dir = "/cluster/home/it_stu27/bdd100k/results/img_seg"
    miou,list_iou = evaluate_segmentation(gt_dir,result_dir,3,17)
    print("MIOU of val dataset is: {},list_iou is: {}".format(miou,list_iou))
        
        
        
    
