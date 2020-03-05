#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 16:46:51 2018

@author: jjldr
"""
import os 
import numpy as np
import cv2

import torch
from torch.utils.data import Dataset

from augmentations import SSDAugmentation
import transform_data
import transform_data_alter
#from data import BaseTransform


class bdd100k(Dataset):
    def __init__(self,list_path,root_path,img_size,is_training=True,is_debug=False):
        self.img_files=[]
        self.label_files=[]
        self.seg_label_files=[]
        self.name = 'bdd100k'
        
        self.is_debug=is_debug
        for path in open(list_path,'r'):
            path = os.path.join(root_path,path)
            label_path=path.replace("images","labels").replace(".png",".txt").replace(".jpg",".txt").strip()
            seg_label_path=path.replace("images/100k","drivable_maps/labels").replace(".png","_drivable_id.png").replace(".jpg","_drivable_id.png").strip()
            
            if os.path.isfile(label_path) & os.path.isfile(seg_label_path):
                self.img_files.append(path)
                self.label_files.append(label_path)
                self.seg_label_files.append(seg_label_path)
            else:
                print("no label found. skip it:{}".format(label_path))
        print("Total images: {}".format(len(self.img_files)))
        self.img_size=img_size
        self.max_object=50
        self.ids=list()
        for i in range(len(self.img_files)):
            self.ids.append(self.img_files[i][-22:-5])
        
        self.transforms=transform_data.Compose()
        if is_training:
            self.transforms.add(transform_data.ImageBaseAug())
            
        self.transforms.add(transform_data.ResizeImage((512,512)))
        self.transforms.add(transform_data.ToTensor(self.max_object, self.is_debug))
        
                
    def __getitem__(self,index):
        img_path=self.img_files[index].rstrip()
        img=cv2.imread(img_path)
        if img is None:
            raise Exception("Read image error: {}".format(img_path))
        ori_h,ori_w=img.shape[:2]
        #print(img,img.shape)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        
        label_path=self.label_files[index].rstrip()
        if os.path.exists(label_path):
            size=os.path.getsize(label_path)
            if size==0:
                print("{} is null.".format(label_path))
            labels=np.loadtxt(label_path).reshape(-1,5)
        else:
            print("label does  not exists:{}".format(label_path))
            labels=np.zeros((1,5),np.float32)
        
        
        labels[:,1]=labels[:,1]/ori_w
        labels[:,3]=labels[:,3]/ori_w
        labels[:,2]=labels[:,2]/ori_h
        labels[:,4]=labels[:,4]/ori_h
        
        new_labels=np.zeros_like(labels)
        new_labels[:,0:4]=labels[:,1:]
        new_labels[:,4]=labels[:,0]
        
        seg_path=self.seg_label_files[index].rstrip()
        label_img=cv2.imread(seg_path,0)
        #print(label_img,label_img.shape)
#        #label_img=cv2.resize(label_img,(448,448),interpolation=cv2.INTER_NEAREST)
#        
#        if self.transform is not None:
#            img,boxes,gt_labels=self.transform(img,labels[:,1:],labels[:,0])
#            
#            img = img[:, :, (2, 1, 0)]
#            target = np.hstack((boxes, np.expand_dims(gt_labels, axis=1)))
#        return torch.from_numpy(img).permute(2, 0, 1), target, ori_h, ori_w
        sample={'image':img,"label":new_labels,"seg_label":label_img}
        if self.transforms is not None:
            sample=self.transforms(sample)
        sample["image_path"]=img_path
        sample["origin_size"]=str([ori_w,ori_h])
        return sample['image'],sample['label'],sample['seg_label']
    
    def __len__(self):
        return len(self.img_files)

class bdd100kalter(Dataset):
    def __init__(self,list_path,root_path,img_size,is_training=True,is_debug=False):
        self.img_files=[]
        self.label_files=[]
        self.name = 'bdd100k'
        
        self.is_debug=is_debug
        for path in open(list_path,'r'):
            path = os.path.join(root_path,path)
            label_path=path.replace("images","labels").replace(".png",".txt").replace(".jpg",".txt").strip()
            
            if os.path.isfile(label_path):
                self.img_files.append(path)
                self.label_files.append(label_path)
            else:
                print("no label found. skip it:{}".format(label_path))
        print("Total images: {}".format(len(self.img_files)))
        self.img_size=img_size
        self.max_object=50
        self.ids=list()
        for i in range(len(self.img_files)):
            self.ids.append(self.img_files[i][-22:-5])
        
        self.transforms=transform_data_alter.Compose()
        if is_training:
            self.transforms.add(transform_data_alter.ImageBaseAug())
            
        self.transforms.add(transform_data_alter.ResizeImage((512,512)))
        self.transforms.add(transform_data_alter.ToTensor(self.max_object, self.is_debug))
        
                
    def __getitem__(self,index):
        img_path=self.img_files[index].rstrip()
        img=cv2.imread(img_path)
        if img is None:
            raise Exception("Read image error: {}".format(img_path))
        ori_h,ori_w=img.shape[:2]
        #print(img,img.shape)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        
        label_path=self.label_files[index].rstrip()
        if os.path.exists(label_path):
            size=os.path.getsize(label_path)
            if size==0:
                print("{} is null.".format(label_path))
            labels=np.loadtxt(label_path).reshape(-1,5)
        else:
            print("label does  not exists:{}".format(label_path))
            labels=np.zeros((1,5),np.float32)
        
        
        labels[:,1]=labels[:,1]/ori_w
        labels[:,3]=labels[:,3]/ori_w
        labels[:,2]=labels[:,2]/ori_h
        labels[:,4]=labels[:,4]/ori_h
        
        new_labels=np.zeros_like(labels)
        new_labels[:,0:4]=labels[:,1:]
        new_labels[:,4]=labels[:,0]
        
         #print(label_img,label_img.shape)
#        #label_img=cv2.resize(label_img,(448,448),interpolation=cv2.INTER_NEAREST)
#        
#        if self.transform is not None:
#            img,boxes,gt_labels=self.transform(img,labels[:,1:],labels[:,0])
#            
#            img = img[:, :, (2, 1, 0)]
#            target = np.hstack((boxes, np.expand_dims(gt_labels, axis=1)))
#        return torch.from_numpy(img).permute(2, 0, 1), target, ori_h, ori_w
        sample={'image':img,"label":new_labels}
        if self.transforms is not None:
            sample=self.transforms(sample)
        sample["image_path"]=img_path
        sample["origin_size"]=str([ori_w,ori_h])
        return sample['image'],sample['label']
    
    def __len__(self):
        return len(self.img_files)

#transform=BaseTransform(448, (104/256.0, 117/256.0, 123/256.0))
#MEANS = (74, 75, 71)
##MEANS=(104, 117, 123)
##transform=SSDAugmentation(448,MEANS)
#dataset=bdd100k("image_list/train_images.txt",448)
#len(dataset)
##print(dataset.ids)
#rgb_value=[(42,42,128),(0,0,255),(0,156,255),(0,255,255),(0,255,0),(255,255,0),(255,0,0),(255,0,255),(18,153,255),(0,0,0)]
#labelmap = (  # always index 0
#    'bus', 'traffic light', 'traffic sign',
#    'person', 'bike', 'truck', 'motor', 
#    'car', 'train', 'rider')
#FONT = cv2.FONT_HERSHEY_SIMPLEX
#for i in range(len(dataset)):
#    if i%100==0:
#        print("{}/{}".format(i,len(dataset)))
#    img,labels,seg_img=dataset[i]
##    img=sample["image"]
##    labels=sample["label"]
##    seg_img=sample["seg_label"]
##    img,target,h,w=dataset[i]
#    img=img.permute(1,2,0).numpy()*255
#    seg_img=seg_img.numpy()
#    
#    #print(seg_img,seg_img.shape)
#    img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
#    h, w = img.shape[:2]
#    
#    
#    
#    
#    for label in labels:
#        color=rgb_value[int(label[4])-1]
#        label_img=cv2.rectangle(img,(int(label[0]*w),int(label[1]*h)),(int(label[2]*w),int(label[3]*h)),color)
#        cv2.putText(label_img, labelmap[int(label[4] - 1)], (int(label[0]*w),int(label[1]*h)),
#                            FONT, 1, (255, 255, 255), 1, cv2.LINE_AA)
#    cv2.imshow("label_img",img)
#    #cv2.imshow("seg_img",seg_img)
#    cv2.imwrite("s_data/BGR/"+dataset.ids[i]+".jpg",img)
#    if i>100:
#        break
##    cv2.imwrite("labels/"+dataset.ids[i]+".jpg",label_img)
##    if cv2.waitKey(100)&0xFF is ord('q'):
##        break
#    key=cv2.waitKey(1)&0xFF 
#    if key==ord(" "):
#        cv2.waitKey(0)
#    elif key==ord("q"):
#        break
#cv2.destroyWindow("label_img")
#
