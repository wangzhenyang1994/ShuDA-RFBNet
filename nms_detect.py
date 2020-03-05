#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 11:04:47 2019

@author: jjldr
"""
import torch
import numpy as np
from utils.nms_wrapper import nms

def nms_detect(boxes,scores,num_classes,max_per_image,thresh):
    scale=torch.Tensor([1280,720,1280,720]).cuda()
    #scale=torch.Tensor([1280,720,1280,720])
    boxes*=scale
    boxes=boxes.cpu().numpy()
    scores=scores.cpu().numpy()
    #print(boxes.shape,scores.shape)
    all_boxes=[[[] for _ in range(1)]
                 for _ in range(num_classes)]
    for j in range(1,num_classes):
        inds = np.where(scores[:, j] > thresh)[0]
        if len(inds) == 0:
            all_boxes[j][0] = np.empty([0, 5], dtype=np.float32)
            continue
        c_bboxes = boxes[inds]
        c_scores = scores[inds, j]
        #print(c_bboxes.shape,c_scores.shape)
        c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(
                np.float32, copy=False)

        keep = nms(c_dets, 0.45, force_cpu=False)
        c_dets = c_dets[keep, :]
        all_boxes[j][0] = c_dets
    if max_per_image > 0:
        image_scores = np.hstack([all_boxes[j][0][:, -1] for j in range(1,num_classes)])
        if len(image_scores) > max_per_image:
            image_thresh = np.sort(image_scores)[-max_per_image]
            for j in range(1, num_classes):
                keep = np.where(all_boxes[j][0][:, -1] >= image_thresh)[0]
                all_boxes[j][0] = all_boxes[j][0][keep, :]
    
    return all_boxes
    