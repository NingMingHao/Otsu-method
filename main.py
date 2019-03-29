#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 16:24:56 2019

@author: minghao
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2

def calculate_between_group_variance(threshold):
    eps = 1e-10 #avoid divide by zero
    
    omiga0 = omiga_arr[threshold-1]
    Ex0 = Ex_arr[threshold-1]
    between_group_variance = omiga0*(Ex0-EX)**2/(1-omiga0+eps)
    return between_group_variance

img = cv2.imread('origin.jpg',0)


retval,dst = cv2.threshold(img,0,255,cv2.THRESH_OTSU)


plt.imshow(img,cmap=plt.cm.gray)

#plt.imshow(img==255,cmap=plt.cm.gray) #to show how many pixels are 255
plt.imsave('255.jpg',img==255,cmap=plt.cm.gray)
plt.imsave('0.jpg',img==0,cmap=plt.cm.gray)

plt.figure()
img_flattened = img.flatten()
#img_hist = plt.hist(img_flattened,bins=255)
img_hist = np.histogram(img_flattened,bins=255)
brightness_list = img_hist[1][1:].astype(int) #ignore the first element
hist_counts = img_hist[0]
hist_counts[-1] = hist_counts[-2]
hist_counts[0] = hist_counts[1]

plt.bar(brightness_list,hist_counts)
plt.savefig('hist.jpg')
normed_hist_counts = hist_counts/hist_counts.sum()
weighted_prob = brightness_list*normed_hist_counts

omiga_arr = normed_hist_counts.cumsum()
Ex_arr = weighted_prob.cumsum()
EX = Ex_arr[-1]

inter_class_variance_arr = np.array(list(map(calculate_between_group_variance,brightness_list)))

plt.figure()
plt.plot(brightness_list,inter_class_variance_arr)
plt.savefig('inter_class_variance.jpg')

final_threshold = inter_class_variance_arr.argmax()+1

plt.figure()
binary_img = img>final_threshold
plt.imshow(binary_img,cmap=plt.cm.gray)
plt.imsave('binary_reault.jpg',binary_img,cmap=plt.cm.gray)
    