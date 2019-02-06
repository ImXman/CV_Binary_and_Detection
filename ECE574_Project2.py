# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 15:18:52 2019

@author: Yang Xu
"""
##-----------------------------------------------------------------------------
##
import cv2
import numpy as np
import seaborn as sns
import binary_processing as bp

##-----------------------------------------------------------------------------
##read image as grayscale
bimg = cv2.imread('airplane.jpg',0)
##plot pixel value density
pixel=bimg.flatten()
sns.kdeplot(pixel, label='pixel', shade=True)

##-----------------------------------------------------------------------------
##create masks
##in this task, I simplify the object by assuming airplane '___' shape
##However, use masks with different size for left and right sub-images
erode_mask = np.zeros((11,11))
erode_mask[4:7,:]=1
dilat_mask = np.zeros((11,17))
dilat_mask[4:7,:]=1

mask_right = np.zeros((7,11))
mask_right[2:5,:]=1

##-----------------------------------------------------------------------------
##implementation of detection
##The left sub-image has dense and small airplanes, therefore, erosion
##followed by dilation. In practice, foreground pixels that have less 60% match
##with mask will be erode. 
bimg_left = bimg[:,:360]

bi_left = bp.thresholding(bimg_left,[180,245])
airplane_l = bp.morpho_operator(bi_left,erode_mask,0.6)
airplane_l = bp.dilation(airplane_l,dilat_mask)
#airplane_l[airplane_l==1]=255
#airplane_l=airplane_l.astype(np.uint8)

#cv2.namedWindow('image')
#cv2.imshow('image',airplane_l)
#cv2.waitKey()
#scipy.misc.imsave('airplane_left.jpg',airplane_l)

##connect components and count number and size
con_left = bp.connect8(airplane_l)
obj_left = bp.object_detection(con_left)

##But for the right sub-image, first dilate image and then erode it. In this
##case, size of connected component will be considered if it can be regarded
##as an airplane.
bimg_right = bimg[:,360:]

bi_right = bp.thresholding(bimg_right,[180,245])
airplane_r = bp.dilation(bi_right,mask_right)
airplane_r = bp.erosion(airplane_r,mask_right)
#airplane_r[airplane_r==1]=255
#airplane_r=airplane_r.astype(np.uint8)

#cv2.namedWindow('image')
#cv2.imshow('image',airplane_r)
#cv2.waitKey()
#scipy.misc.imsave('airplane_right.jpg',airplane_r)

##connect components and count number and size. For right sub-image, object
##size smaller than 150 will be removed
con_right = bp.connect8(airplane_r)
obj_right = bp.object_detection(con_right)
obj_right_new = obj_right[obj_right[:,1]>=150]

objs = np.vstack((obj_left[1:,:],obj_right_new[1:,:]))
sns.distplot(objs[1:,1],bins=30)
