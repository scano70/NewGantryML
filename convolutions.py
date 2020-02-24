import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.filters import convolve

def rotate_image(im,theta):
    rows,cols=im.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2),theta,1)
    return cv2.warpAffine(im,M,(cols,rows))

def norm_im(im):
    return (im - np.mean(im))/np.std(im)

def convolv(im,filter):
    filter_shape = np.array(filter).shape
    filter_pad=[np.ceil(filter_shape[0]/2),np.ceil(filter_shape[1]/2)]
    filter_pad = map(int,filter_pad)
    print 'filter_shape',filter_shape
    print 'filter_pad',filter_pad
    im_shape=im.shape
    conv = np.zeros(im.shape)
    for i in range(filter_pad[0],im_shape[0]-filter_pad[0]):
        for j in range(filter_pad[1],im_shape[1]-filter_pad[1]):
            conv[i,j]=np.sum(np.multiply(filter,im[i-filter_pad[0]:i+filter_pad[0]+1,j-filter_pad[1]:j+filter_pad[1]+1]))
    return conv

def line_pixels(s,m,y0):
    points=[]
    m_sign=m/abs(m)
    for i in range(s):
        #if y0+m*i >= s or y0+m*i <0 : continue
        #points.append([int(y0+m*i),i])
        for j in range(abs(int(m))+1) :
            if y0+m*i+m_sign*j >= s or y0+m*i+m_sign*j < 0: continue
            #print y0+m_sign*m*i+m_sign*j,i
            points.append([int(y0+m*i+m_sign*j),i])
    return points

def line_filter(s,m,y0):
    points = line_pixels(s,m,y0)
    #print points
    filter = np.zeros((s,s))
    #print filter.shape
    for p in points:
        filter[p[0],p[1]]=1
#        if p[0]+1 < 250 :
#            filter[p[0]+1,p[1]]=1
#        if p[0]-1 >= 0 :
#            filter[p[0]-1,p[1]]=1
    return filter

#im = cv2.imread('line_imgs/lines_439.png')
im_train = cv2.imread('Fiducial/fifth/Pic_63.png')
#im_test = cv2.imread('Fiducial/third/Pic_12.png')
im_test = cv2.imread('Fiducial/fifth/Pic_64.png')
im_train=im_train[:,:,0]
im_test=im_test[:,:,0]
print im_train.shape
plt.ion()

im_train = norm_im(im_train)
im_test = norm_im(im_test)

f=im_train[1665:1766,1755:1856]

#conv=convolve(im_test,f)
#plt.imshow(conv)
#plt.show()

f=im_train[1515:1916,1605:2006]
f30=rotate_image(f,30)[100:300,100:300]
f60=rotate_image(f,60)[100:300,100:300]
f90=rotate_image(f,90)[100:300,100:300]
f120=rotate_image(f,120)[100:300,100:300]
f150=rotate_image(f,150)[100:300,100:300]
f180=rotate_image(f,180)[100:300,100:300]
f210=rotate_image(f,210)[100:300,100:300]
f240=rotate_image(f,240)[100:300,100:300]
f270=rotate_image(f,270)[100:300,100:300]
f300=rotate_image(f,300)[100:300,100:300]
f330=rotate_image(f,330)[100:300,100:300]

plt.subplot(3,4,1),plt.imshow(f[100:300,100:300])
plt.subplot(3,4,2),plt.imshow(rotate_image(f,30)[100:300,100:300])
plt.subplot(3,4,3),plt.imshow(rotate_image(f,60)[100:300,100:300])
plt.subplot(3,4,4),plt.imshow(rotate_image(f,90)[100:300,100:300])
plt.subplot(3,4,5),plt.imshow(rotate_image(f,120)[100:300,100:300])
plt.subplot(3,4,6),plt.imshow(rotate_image(f,150)[100:300,100:300])
plt.subplot(3,4,7),plt.imshow(rotate_image(f,180)[100:300,100:300])
plt.subplot(3,4,8),plt.imshow(rotate_image(f,210)[100:300,100:300])
plt.subplot(3,4,9),plt.imshow(rotate_image(f,240)[100:300,100:300])
plt.subplot(3,4,10),plt.imshow(rotate_image(f,270)[100:300,100:300])
plt.subplot(3,4,11),plt.imshow(rotate_image(f,300)[100:300,100:300])
plt.subplot(3,4,12),plt.imshow(rotate_image(f,330)[100:300,100:300])
