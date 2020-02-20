import cv2
import matplotlib.pyplot as plt
import numpy as np

def get_key(my_dict,val):
    for key, value in my_dict.items():
         if val == value:
             return key
    return None

def first_nonzero(my_list):
    ind=0
    for ai,a in enumerate(my_list):
        if a > 0 :
            ind=ai
            break
    for ai,a in enumerate(my_list[::-1]):
        if a > 0 :
            if ai > ind :
                return len(my_list)-ai
            else :
                return ind
    return 0

for i in np.arange(0,500):
    img = cv2.imread('line_imgs/lines_{0}.png'.format(i),0)
    edges = cv2.Canny(img,100,100)

    if edges is None :
        print 'canny failed'
        continue

    top = edges[0,:]/255.
    bot = edges[-1,:]/255.
    left = edges[:,0]/255.
    right = edges[:,-1]/255.

    ind = np.arange(0,250,1)

    top_sum = sum(top) ; #print top_sum
    bot_sum = sum(bot) ; #print bot_sum
    left_sum = sum(left) ; #print left_sum
    right_sum = sum(right) ; #print right_sum

    sums = [top_sum,bot_sum,left_sum,right_sum]
    sums_dict = {'t':top_sum,
                 'b':bot_sum,
                 'l':left_sum,
                 'r':right_sum}
    ind_dict = {'t':top,
                'b':bot,
                'r':right,
                'l':left}

    #print sums_dict

    if 0 in sums :
        sums.pop(sums.index(0))
        if 0 in sums :
            sums.pop(sums.index(0))
        else :
            sums.pop(sums.index(max(sums)))
    else :
        sums.pop(sums.index(max(sums)))
        sums.pop(sums.index(max(sums)))

    #print sums
    x1_side = get_key(sums_dict,sums[0])
    if x1_side == None :
        print 'failed to find x1_side'
        continue
    x1_ind = first_nonzero(ind_dict[x1_side])
    x2_side = get_key(sums_dict,sums[1])
    if x1_side == None :
        print 'failed to find x1_side'
        continue
    x2_ind = first_nonzero(ind_dict[x2_side])

    x1=[0]*2
    x2=[0]*2

    if x1_side == 'r' :
        x1=[250,x1_ind]
        #print 'x1 (250,{0})'.format(x1_ind)
    if x1_side == 'l' :
        x1=[0,x1_ind]
        #print 'x1 (0,{0})'.format(x1_ind)
    if x1_side == 't' :
        x1=[x1_ind,0]
        #print 'x1 ({0},0)'.format(x1_ind)
    if x1_side == 'b' :
        x1=[x1_ind,250]
        #print 'x1 ({0},250)'.format(x1_ind)

    if x2_side == 'r' :
        x2=[250,x2_ind]
        #print 'x2 (250,{0})'.format(x2_ind)
    if x2_side == 'l' :
        x2=[0,x2_ind]
        #print 'x2 (0,{0})'.format(x2_ind)
    if x2_side == 't' :
        x2=[x2_ind,0]
        #print 'x2 ({0},0)'.format(x2_ind)
    if x2_side == 'b' :
        x2=[x2_ind,250]
        #print 'x2 ({0},250)'.format(x2_ind)


    xs=zip(*[x1,x2])[0]
    ys=zip(*[x1,x2])[1]

    plt.subplot(231)
    plt.imshow(img,cmap = 'gray')
    plt.xlim(0,250)
    plt.ylim(250,0)
    plt.plot(xs,ys,color='m')
    # plt.subplot(232)
    # plt.imshow(edges,cmap = 'gray')
    # plt.xlim(0,250)
    # plt.ylim(250,0)
    # plt.plot(xs,ys,color='m')
    # plt.subplot(233),plt.plot(ind,top)
    # plt.subplot(234),plt.plot(ind,bot)
    # plt.subplot(235),plt.plot(ind,left)
    # plt.subplot(236),plt.plot(ind,right)
    plt.savefig('predictions/lines_{0}.png'.format(i))
    plt.clf()
    #plt.show()

    #if raw_input('continue')=='n' : break
