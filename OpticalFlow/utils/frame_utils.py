import numpy as np
from os.path import *
from scipy.misc import imread
#from imageio import imread
from . import flow_utils 

def read_gen(file_name):
    ext = splitext(file_name)[-1]
    if ext == '.png' or ext == '.jpeg' or ext == '.ppm' or ext == '.jpg':
        im = imread(file_name)

        if im.shape[2] > 3:###img.shape[0]：图像的垂直尺寸（高度）img.shape[1]：图像的水平尺寸（宽度）img.shape[2]：图像的通道数
            return im[:,:,:3]
        else:
            return im
    elif ext == '.bin' or ext == '.raw':
        return np.load(file_name)
    elif ext == '.flo':
        return flow_utils.readFlow(file_name).astype(np.float32)
    return []
