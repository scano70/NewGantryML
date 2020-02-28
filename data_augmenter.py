import matplotlib.pyplot as plt
import cv2
import numpy as np
import random as r

plt.ion()

im = plt.imread('Fiducial/fifth/Pic_63.png')
ax = plt.gca()
fig = plt.gcf()

width=200

def rotate_image(im,theta):
    M = cv2.getRotationMatrix2D((width,width),theta,1)
    return cv2.warpAffine(im,M,(width*2,width*2))

def translate_image(im,x,y):
    T = np.float32([[1, 0, y], [0, 1, x]])
    return cv2.warpAffine(im, T, (2*width,2*width))
    
def onclick(event):
    if event.xdata != None and event.ydata != None:
        plt.imshow(im[int(event.ydata)-width:int(event.ydata)+width,int(event.xdata)-width:int(event.xdata)+width])
        for theta in range(0,360,20):
            print 'theta: ',theta
            for x in range(-25,26,25):
                for y in range(-25,26,25):
                    T = np.float32([[1, 0, y], [0, 1, x]])
                    temp=im[int(event.ydata)-width:int(event.ydata)+width,int(event.xdata)-width:int(event.xdata)+width]
                    temp=rotate_image(temp,theta)
                    temp=translate_image(temp,y,x)
                    plt.imsave('test_x{0}_y{1}_th{2}.png'.format(event.ydata+y,event.xdata+x,theta),temp[width/2:3*width/2,width/2:3*width/2])

        #for i in range(900) :
        #    x=r.uniform(0,im.shape[0])
        #    y=r.uniform(0,im.shape[1])
                    
while raw_input('continue? ') != 'n':
    implot = ax.imshow(im)
    fig.canvas.mpl_connect('button_press_event', onclick)
