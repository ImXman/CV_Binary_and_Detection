# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 17:08:04 2019

@author: Yang Xu
"""

##-----------------------------------------------------------------------------
##load modules
import numpy as np

##-----------------------------------------------------------------------------
##thresholding
##it thresholds grayscale image to binary image
##To have better performance, good estimation on pixel range of goal objects
##is necessary.
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
    ##in case mask is not in odd shape
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
    ##in case mask is not in odd shape
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
    ##in case mask is not in odd shape
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
    ##label foreground pixel
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
                        
    ##connect components
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