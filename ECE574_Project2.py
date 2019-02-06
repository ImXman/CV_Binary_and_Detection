# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 15:18:52 2019

@author: Yang Xu
"""
##-----------------------------------------------------------------------------
##load modules
import seaborn as sns
import numpy as np
#import scipy.misc
import cv2

##-----------------------------------------------------------------------------
##thresholding
##it thresholds grayscale image to binary image
##To have better performance, good estimation on pixel range of goal objects
##is necessary
def thresholding(img,threshold=[180,245]):
    
    ##a is the low value threshold and b is the high value threshold
    a,b=threshold[0],threshold[1]
            
    bi_img=img.copy()
    bi_img[bi_img<a]=0
    bi_img[bi_img>b]=0
    bi_img[bi_img!=0]=1
    
    return bi_img

##erosion
##it returns erosed binary image
def erosion(bi_img,mask):
    
    n,m=bi_img.shape
    a,b=mask.shape
    
    if a%2==0:
        mask=np.vstack((np.zeros((1,b)),mask))
    if b%2==0:
        mask=np.column_stack((np.zeros((a+1,1)),mask))
        
    a,b=mask.shape
    ##add padding spaces to keep image the same size after convolution
    pad1 = int((a-1)/2)
    pad2 = int((b-1)/2)
    pixel = np.zeros((n+(a-1),m+(b-1)))
    pixel[pad1:pad1+n,pad2:pad2+m] = bi_img[:,:]
    bi_erode=np.ones((n,m))
    
    for i in range(n):
        for j in range(m):
            if  (1 in (mask-pixel[i:i+pad1*2+1,j:j+pad2*2+1])): 
                bi_erode[i,j]=0
    
    return bi_erode

##dilation
##it returns dilated binary image
def dilation(bi_img,mask):
    
    n,m=bi_img.shape
    a,b=mask.shape
    
    if a%2==0:
        mask=np.vstack((np.zeros((1,b)),mask))
    if b%2==0:
        mask=np.column_stack((np.zeros((a+1,1)),mask))
        
    a,b=mask.shape
    ##add padding spaces to keep image the same size after convolution
    pad1 = int((a-1)/2)
    pad2 = int((b-1)/2)
    pixel = np.zeros((n+(a-1),m+(b-1)))    
    pixel[pad1:pad1+n,pad2:pad2+m] = bi_img[:,:]
    bi_dilate=np.zeros((n,m))
    
    for i in range(n):
        for j in range(m):        
            g=np.count_nonzero((mask-pixel[i:i+pad1*2+1,j:j+pad2*2+1])==1)
            k=np.count_nonzero(mask == 1)
            if g < k:
                bi_dilate[i,j]=1
    
    return bi_dilate

##In real life proble, it is very hard to 100% match foreground pixels to mask
##, and it is unpractical to dilate image if only 1 pixel match the mask.
##Therefore, in this project, I introduced this function to perform erosion and
##dilation by simplying changing the parameter similarity. When similarity is
##high or close to 1, it performs a similar erosion function; However, when
##similarity is low and close to 0, it works as an alternative for dilation.
def morpho_operator(bi_img,mask,similarity=0.9):
    
    n,m=bi_img.shape
    a,b=mask.shape
    
    if a%2==0:
        mask=np.vstack((np.zeros((1,b)),mask))
    if b%2==0:
        mask=np.column_stack((np.zeros((a+1,1)),mask))
        
    a,b=mask.shape
    ##add padding spaces to keep image the same size after convolution
    pad1 = int((a-1)/2)
    pad2 = int((b-1)/2)
    pixel = np.zeros((n+(a-1),m+(b-1)))    
    pixel[pad1:pad1+n,pad2:pad2+m] = bi_img[:,:]
    morpho=np.zeros((n,m))
    
    for i in range(n):
        for j in range(m):
            g=np.count_nonzero((mask-pixel[i:i+pad1*2+1,j:j+pad2*2+1])==1)
            k=np.count_nonzero(mask == 1)
            if ((1-g/k) >= similarity):
                morpho[i,j]=1
    
    return morpho

##this connects pixels in 8 directions and label them the same
##In this project, I used sequential method to connect components
def connect8(bi_img):
    
    n,m=bi_img.shape
    con_img=np.zeros((n+2,m+2))
    pixel = np.zeros((n+2,m+2))    
    pixel[1:n+1,1:m+1] = bi_img[:,:]
    l=0
    for i in range(n):
        for j in range(m):
            if bi_img[i,j]==1:
                if l == 0:
                    l=l+1
                    con_img[i+1,j+1]=l
                else:
                    if pixel[i:i+2,j:j+2].sum()>=2:
                        a=pixel[i:i+2,j:j+2].reshape(4).tolist()
                        b=con_img[i:i+2,j:j+2].reshape(4).tolist()
                        con_img[i+1,j+1]=b[a.index(1)]
                    else:
                        l=l+1
                        con_img[i+1,j+1]=l
                        
    ##
    for i in range(n):
        for j in range(m):
            pixel = con_img.copy()
            g=np.count_nonzero(pixel[i:i+2,j:j+2]==0)
            if g<=2:
                a=pixel[i:i+2,j:j+2]
                ls = np.min(a[np.nonzero(a)])
                a = a.reshape(4).tolist()
                for label in a:
                    if label!=0:
                        con_img[con_img==label]=ls                                        
                    
    return con_img

##it takes in component connected images and counts the number of object
##and reports their sizes
def object_detection(con_img):
    
    label, counts = np.unique(con_img, return_counts=True)
    return np.array((label, counts)).T

##-----------------------------------------------------------------------------
##read image as grayscale
bimg = cv2.imread('airplane.jpg',0)
bimg_left = bimg[:,:360]
bimg_right = bimg[:,360:]
##plot pixel value density
pixel=bimg.flatten()
sns.kdeplot(pixel, label='pixel', shade=True)

##-----------------------------------------------------------------------------
##create masks
##in this task, I simplify the object by assuming airplane '-' shape
##However, use masks with different size for left and right sub-images
erode_mask = np.zeros((11,11))
erode_mask[4:7,:]=1
dilat_mask = np.zeros((11,17))
dilat_mask[4:7,:]=1

dilat_right = np.zeros((7,11))
dilat_right[2:5,:]=1

##-----------------------------------------------------------------------------
##implementation of detection
##The left sub-image has dense and small airplanes, therefore, erosion
##followed by dilation. In practice, foreground pixels that have less 60% match
##with mask will be erode. 
bi_left = thresholding(bimg_left,[180,245])
airplane_l = morpho_operator(bi_left,erode_mask,0.6)
airplane_l = dilation(airplane_l,dilat_mask)
#airplane_l[airplane_l==1]=255
#airplane_l=airplane_l.astype(np.uint8)

#cv2.namedWindow('image')
#cv2.imshow('image',airplane_l)
#cv2.waitKey()
#scipy.misc.imsave('airplane_left.jpg',airplane_l)

##connect components and count number and size
con_left = connect8(airplane_l)
obj_left = object_detection(con_left)

##But for the right sub-image, first dilate image and then erode it. In this
##case, size of connected component will be considered if it can be regarded
##as an airplane.
bi_right = thresholding(bimg_right,[180,245])
airplane_r = dilation(bi_right,dilat_right)
airplane_r = erosion(airplane_r,dilat_right)
#airplane_r[airplane_r==1]=255
#airplane_r=airplane_r.astype(np.uint8)

#cv2.namedWindow('image')
#cv2.imshow('image',airplane_r)
#cv2.waitKey()
#scipy.misc.imsave('airplane_right.jpg',airplane_r)

##connect components and count number and size
con_right = connect8(airplane_r)
obj_right = object_detection(con_right)
obj_right_new = obj_right[obj_right[:,1]>=150]

