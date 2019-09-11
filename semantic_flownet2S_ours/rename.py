from __future__ import division
import os
import numpy as np
from scipy import misc as ms
import sys
import re
import cv2

###################################################

def write_kitti_png(path, flow, valid=None):
    temp = np.ones((flow.shape[0], flow.shape[1], 3), dtype=np.float64)
    temp[:, :, :2] = flow.astype(np.float64) * 64.0 + 2**15
    if valid is not None:
        temp[:, :, 2] = valid
    temp = temp.astype('uint16')
    write_PNG_u16(path, temp)

def write_PNG_u16(path, flow):
    """ Does not check if input flow is multichannel. """
    print(flow.shape)
    ret = cv2.imwrite(path, flow[..., ::-1])
    if not ret:
        print('Flow not written')

def read_flow_flo(filename):
    """ Read flo file and return flow array. """
    f = open(filename, 'rb')
    magic = np.fromfile(f, np.float32, count=1)
    data2d = None

    if magic != 202021.25:
        print('Magic number incorrect. Invalid .flo file.')
    else:
        w = np.fromfile(f, np.int32, count=1)
        h = np.fromfile(f, np.int32, count=1)
        # print("Reading %d x %d flo file" % (h, w))
        # data2d = np.fromfile(f, np.float32, count=2 * w * h)
        # data2d = np.resize(data2d, (h, w, 2))
        # Numpy bullshit adendum
        data2d = np.fromfile(f, np.float32, count=int(2 * w * h))
        # reshape data into 3D array (columns, rows, channels)
        data2d = np.resize(data2d, (int(h), int(w), 2))
    f.close()
    return data2d

def crop_center(img,cropx,cropy):
    y,x,s = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx]

###################################################

filenames = os.listdir('flow_noc_test')

print('############ ground truth renaming.... ############')
filenames = np.sort(filenames)
print(filenames)

for i,name in enumerate (filenames):
    print(i, name)
    print ('%06d'%i)
    tempImage = ms.imread('flow_noc_test/'+name)
    tempImage = crop_center(tempImage, 1216, 320)
    print(tempImage.shape)
    ms.imsave('GT_test/'+('%06d'%i)+'.png', tempImage)

filenames = os.listdir('predicted')

print('############ predicted conversion.... ############')

filenames = np.sort(filenames)
print(filenames)

for i,name in enumerate (filenames):
    print(i, name)
    data2d = read_flow_flo('predicted/'+name)
    print(data2d.shape)
    name = name.split('.')[0]
    name = name + '.png'
    write_kitti_png('converted/'+name, data2d)



