import os
import warnings
import cv2
import torch
import numpy as np
import argparse
import time
from models import FlowNet2  # the path is depended on where you create this module
from utils.flow_utils import flow2img
from utils.frame_utils import read_gen
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
save_path= '../FlowNet/flownet/shiliangtu'

def writeFlow(name, flow):
    f = open(name, 'wb')  # 打开name文件以二进制格式、只写模式打开文件，一般用于非文本文件
    f.write('PIEH'.encode('utf-8'))  # 将utf-8编码的字符串str以二进制流写入文件
    np.array([flow.shape[1], flow.shape[0]], dtype=np.int32).tofile(f)
    flow = flow.astype(np.float32)
    flow.tofile(f)
    f.flush()
    f.close()


def flow_calculate(flow):
    # flow (ndarray): Array of optical flow.
    # 光流表示的是图像之间的对应像素的位移
    # 一般是两个通道，分别表示X轴， Y轴
    # 典型的光流size： H，W，2
    assert flow.ndim == 3 and flow.shape[-1] == 2
    dx = flow[:, :, 0].copy()
    dy = flow[:, :, 1].copy()
    rad = np.sqrt(dx ** 2 + dy ** 2)
    return rad


def load_flow_to_numpy(path):
    with open(path, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        assert (202021.25 == magic), 'Magic number incorrect. Invalid .flo file'
        w = np.fromfile(f, np.int32, count=1)[0]
        h = np.fromfile(f, np.int32, count=1)[0]
        data = np.fromfile(f, np.float32, count=2 * w * h)
    data2D = np.reshape(data, (h, w, 2))
    # print(data2D[:10,:10,0])
    return data2D


def sparse_flow(img_bg, flow, X=None, Y=None, stride=384):
    # flow = flow.data.cpu().numpy().transpose(2, 1, 0)
    flow = flow.copy()
    flow[:, :, 0] = -flow[:, :, 0]
    if X is None:
        height, width, _ = flow.shape
        xx = np.arange(0, width, stride)  # 三个参数 起点为0，终点为height，步长为stride
        yy = np.arange(0, height, stride)
        # xx = np.arange(0, height, stride)  # 三个参数 起点为0，终点为height，步长为stride
        # yy = np.arange(0, width, stride)
        X, Y = np.meshgrid(xx, yy)  # meshgrid() 用于生成网格采样点矩阵
        X = X.flatten()  # 对多维数据的降维函数
        Y = Y.flatten()

        # sample
        sample_0 = flow[:, :, 0][yy]
        sample_0 = sample_0.T
        sample_x = sample_0[xx]
        sample_x = sample_x.T
        sample_1 = flow[:, :, 1][yy]
        sample_1 = sample_1.T
        sample_y = sample_1[xx]
        sample_y = sample_y.T


        # # sample
        # sample_0 = flow[:, :, 0][xx]
        # sample_0 = sample_0.T
        # sample_x = sample_0[yy]
        # sample_x = sample_x.T
        # sample_1 = flow[:, :, 1][xx]
        # sample_1 = sample_1.T
        # sample_y = sample_1[yy]
        # sample_y = sample_y.T

        sample_x = sample_x[:, :, np.newaxis]
        sample_y = sample_y[:, :, np.newaxis]
        new_flow = np.concatenate([sample_x, sample_y], axis=2)  # 能够一次完成多个数组的拼接
    flow_x = new_flow[:, :, 0].flatten()
    flow_y = new_flow[:, :, 1].flatten()

    # display
    ax = plt.gca()
    ax.xaxis.set_ticks_position('top')

    ax.invert_yaxis()
    # plt.quiver(X,Y, flow_x, flow_y, angles="xy", color="#666666")
    # ax.quiver(X, Y, -flow_x, flow_y, color="#ffff00")
    ax.quiver(X, Y, -flow_x, -flow_y, scale=60, color="#00FF00")
    ax.imshow(img_bg, extent=[0, 384, 384, 0], origin='upper')
    ax.grid()
    # ax.legend()
    plt.axis('off')
    plt.draw()
    plt.show()

if __name__ == '__main__':
    # obtain the necessary args for construct the flownet framework
    parser = argparse.ArgumentParser()
    parser.add_argument('--fp16', action='store_true', help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')
    parser.add_argument("--rgb_max", type=float, default=255.)

    args = parser.parse_args()

    # initial a Net
    net = FlowNet2(args).cuda()
    # load the state_dict
    dict = torch.load("FlowNet2_checkpoint.pth.tar")
    net.load_state_dict(dict["state_dict"], strict=False)

    # load the image pair, you can find this operation in dataset.py
    pim1 = read_gen("images/test42.jpg")
    pim2 = read_gen("images/test43.jpg")

    images = [pim1, pim2]
    images = np.array(images).transpose(3, 0, 1, 2)
    im = torch.from_numpy(images.astype(np.float32)).unsqueeze(0).cuda()

    # process the image pair to obtian the flow
    result = net(im).squeeze()

    data = result.data.cpu().numpy().transpose(1, 2, 0)
    result1 = np.array(data).reshape(-1, 2)
    img = flow2img(data)  # 将光流数据转为图像
    # cv2.imwrite(save_path  + '.png', img)
    rad = flow_calculate(data)
    # np.savetxt(save_path + str(i) + ".txt", rad, fmt='%s', delimiter=',')
    img_RGB = cv2.cvtColor(pim1, cv2.COLOR_BGR2RGB)
    sparse_flow(img_RGB, data)