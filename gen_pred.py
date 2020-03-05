#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 19:19:29 2019

@author: jjldr
"""

from __future__ import print_function
import torch
from torch.autograd import Variable
import cv2
import time
# from imutils.video import FPS, WebcamVideoStream
import argparse
import numpy as np
from box_utils import label_img_to_color
from data import BaseTransform, COCO_512
from dataset import bdd100k,bdd100kalter
from layers.functions import PriorBox, Detect
from nms_detect import nms_detect
from models.RFB_Net_shuffle import build_net


parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--weights', default='/home/jjldr/BDD100K/weights/VOC.pth',
                    type=str, help='Trained state_dict file path')
parser.add_argument('--cuda', default=True, type=bool,
                    help='Use cuda in live demo')
args = parser.parse_args()

COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
FONT = cv2.FONT_HERSHEY_SIMPLEX
cfg = COCO_512
priorbox = PriorBox(cfg)
with torch.no_grad():
    priors = priorbox.forward()
    if args.cuda:
        priors = priors.cuda()

def cv2_demo(net, transform):
    def predict(frame):
        #height, width = frame.shape[:2]
        # height, width = 720, 1280
#        x=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
#        x=cv2.resize(x, (448,448), interpolation=cv2.INTER_LINEAR)
#        x=x.astype(np.float32)
#        x=x/255.0
#        x = torch.from_numpy(x).permute(2, 0, 1).to(device)
        
        x=frame
        x=x.to(device)
        x = Variable(x.unsqueeze(0))
        
        frame=frame.numpy().transpose(1,2,0)
        frame=frame*255.0
        frame=cv2.resize(frame,(1280,720))
        frame=cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)

        
        seg, loc, conf = net(x)  # forward pass
        # st=time.time()
        boxes,scores=detector.forward(loc,conf,priors)
        # end=time.time()
        # print("detector time",end-st)
        boxes=boxes[0]
        scores=scores[0]
        out=nms_detect(boxes,scores,11,300,0.05)
        for i in range(len(out)):
            if len(out[i][0])==0:
                continue
            else:
                j=0
                pred_box=out[i][0]
                #print(pred_box)
                while pred_box[j,4]>=0.6:
                    pt=pred_box[j,0:4]
                    cv2.rectangle(frame,
                          (int(pt[0]),int(pt[1])),
                          (int(pt[2]),int(pt[3])),
                          rgb_value[i-1],2)
                    cv2.putText(frame, labelmap[i - 1], (int(pt[0]), int(pt[1])),
                            FONT, 1, (255, 255, 255), 1, cv2.LINE_AA)
                    if len(pred_box)-1>j:
                        j+=1        
                    else:
                        break
#        seg_out,y = net(x)  # forward pass
#        detections = y.data
#        # scale each detection back up to the image
#        scale = torch.Tensor([width, height, width, height])
#        #scale = np.array([width, height, width, height])
#        #print(detections.shape,detections[0,:,:,0].shape)
#        st=time.time()
#        for i in range(detections.size(1)):
#            
#            conf_detect=detections[0, i, :, 0] >= 0.6
#            conf_detect=conf_detect.unsqueeze(1).expand(200,5)
#            detected=detections[0, i, :, :][conf_detect].view(-1,5)
#
#            for j in range(len(detected)):
#                
#                pt = (detected[ j, 1:] * scale).cpu().numpy()
#                #print(detections[0, i, j, 0],int(pt[0]), int(pt[1]),int(pt[2]), int(pt[3]))
#                cv2.rectangle(frame,
#                              (int(pt[0]), int(pt[1])),
#                              (int(pt[2]), int(pt[3])),
#                              rgb_value[i-1], 2)

#        cv2.imshow('frame',frame)
#        cv2.waitKey(0)
#        draw_box=time.time()
#        print("draw box time:{}".format(draw_box-st))
        seg = seg.data.cpu().numpy()
        pred_label_img=np.argmax(seg,axis=1)
        pred_label_img=pred_label_img.astype(np.uint8)
        pred_label_img=pred_label_img[0]
        pred_label_img_color=label_img_to_color(pred_label_img)
        #color_time=time.time()
#        print("color time:{}".format(color_time-draw_box))
        pred_label_img_color=cv2.resize(pred_label_img_color, (1280,720), interpolation=cv2.INTER_LINEAR)
        overlayed_img=0.8*frame+0.2*pred_label_img_color
        overlayed_img=overlayed_img.astype(np.uint8)
        
#        draw_seg=time.time()
#        print(" draw seg time:{}".format(draw_seg-draw_box))
        return overlayed_img

    # start video stream thread, allow buffer to fill
    val_dataset=bdd100k('./image_list/val_images.txt','/cluster/home/it_stu27/bdd100k',512,False)
    detector=Detect(11,0,cfg)
    
    
    for i in range(len(val_dataset)):
        st_time=time.time()
        image,_,_=val_dataset[i]
        #frame = stream.read()
        # key = cv2.waitKey(1) & 0xFF

        # update FPS counter
        #fps.update()
        
        
        frame = predict(image)

        # keybindings for display
        # if key == ord('p'):  # pause
        #     while True:
        #         key2 = cv2.waitKey(1) or 0xff
        #         cv2.imshow('frame', frame)
        #         if key2 == ord('p'):  # resume
        #             break
        # cv2.imshow('frame', frame)
        # if key == 27:  # exit
        #     break
        en_time=time.time()
        print("inference time is:{}".format(en_time-st_time))

        img_id=val_dataset.ids[i]
#        image=Variable(image.unsqueeze(0)).cuda()
#        seg_label=Variable(seg_label.type(torch.LongTensor)).cuda()
#        output,_=net(image)
#        output=F.upsample(output,size=(720,1280),mode="bilinear")
#        output=output.data.cpu().numpy()
#        pred_label_img=np.argmax(output,axis=1)
#        pred_label_img=pred_label_img.astype(np.uint8)
#        pred_label_img=pred_label_img[0]
        path="/cluster/home/it_stu27/bdd100k/results/img_detect_seg2/"+img_id+"_pred_seg_img.png"
        cv2.imwrite(path,frame)
        print("***** process {} of {} *****".format(i,len(val_dataset)))


if __name__ == '__main__':
    labelmap = (  # always index 0
    'bus', 'traffic light', 'traffic sign',
    'person', 'bike', 'truck', 'motor', 
    'car', 'train', 'rider')
    rgb_value=[(42,42,128),(0,0,255),(0,156,255),(0,255,255),(0,255,0),(255,255,0),(255,0,0),(255,0,255),(18,153,255),(0,0,0)]
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = build_net('test', 11).to(device)    # initialize SSD
    #net.load_state_dict(torch.load("4.pth"))
    net.load_state_dict({k.replace('module.',''):v for k,v in torch.load(args.weights).items()})
    transform = BaseTransform(512, (104/256.0, 117/256.0, 123/256.0))
    
    #fps = FPS().start()
    cv2_demo(net.eval(), transform)
    
    cv2.destroyAllWindows()
