import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.cm as cm
import cv2
import torch
import os
import math
import matplotlib.pyplot as plt

from models.matching import Matching
from models.utils import AverageTimer, read_image, make_matching_plot
from make_img_list import make_img_list
import xlsxwriter as xw


def xw_toExcel(data, fileName):  # xlsxwriter库储存数据到excel
    workbook = xw.Workbook(fileName)  # 创建工作簿
    worksheet1 = workbook.add_worksheet("sheet1")  # 创建子表
    worksheet1.activate()  # 激活表
    title = ['图像1', '图像2', 'x位移', 'y位移', "arctan值", "角度"]  # 设置表头
    worksheet1.write_row('A1', title)  # 从A1单元格开始写入表头
    i = 2  # 从第二行开始写入数据
    for j in range(len(data)):
        insertData = [data[j]["img0"], data[j]["img1"], data[j]["x_move"], data[j]["y_move"], data[j]["tan"],
                      data[j]["angle"]]
        row = 'A' + str(i)
        worksheet1.write_row(row, insertData)
        i += 1
    workbook.close()  # 关闭表
    print("表格写入完成！")


# "-------------数据用例-------------"
# testData = [{"img0": "test1.jpg", "img1": "test2.jpg", "x_move": 100, "y_move": 200, "angle":20}]
# fileName = '测试.xlsx'
# xw_toExcel(testData, fileName)

torch.set_grad_enabled(False)


class stitcher:
    def __init__(self):
        # self.input_dir = r"data/1/ly/"  # 待拼接图像数据地址
        # self.input_dir = r"data/6.10/cube/"  # 待拼接图像数据地址
        self.input_dir = r"images12/"  # 待拼接图像数据地址
        self.input_pairs_path = self.input_dir + "stitching_list.txt"  # 待拼接图像列表
        self.output_viz_dir = self.input_dir + "match"  # 保存matches图片地址
        self.nms_radius = 4  # 4
        self.keypoint_threshold = 0.005  # SuperPoint检测特征点阈值 0.005
        self.max_keypoints = 1024  # 1024
        self.superglue = 'outdoor'  # model工作模式
        self.sinkhorn_iterations = 100  # Sinkhorn算法迭代次数（实验部分！）
        self.match_threshold = 0.80  # SuperGlue匹配特征点阈值（实验部分！）
        self.resize = [-1, -1]  # -1为保留原图像尺寸
        self.viz_extension = 'png'
        self.resize_float = False
        self.force_cpu = False
        self.do_viz = False
        self.show_keypoints = True
        self.fast_viz = False
        self.opencv_display = True


stitcher = stitcher()
if os.path.exists(stitcher.input_pairs_path):
    os.remove(stitcher.input_pairs_path)

make_img_list(stitcher.input_dir, stitcher.input_pairs_path)

with open(stitcher.input_pairs_path, 'r') as f:
    pairs = [l.split() for l in f.readlines()]

img_first_stitching = cv2.imread(stitcher.input_dir + pairs[0][0])

if len(stitcher.resize) == 2 and stitcher.resize[1] == -1:
    resize = stitcher.resize[0:1]

device = 'cuda' if torch.cuda.is_available() and not stitcher.force_cpu else 'cpu'
print('使用加速： \"{}\"'.format(device))
config = {
    'superpoint': {
        'nms_radius': stitcher.nms_radius,
        'keypoint_threshold': stitcher.keypoint_threshold,
        'max_keypoints': stitcher.max_keypoints
    },
    'superglue': {
        'weights': stitcher.superglue,
        'sinkhorn_iterations': stitcher.sinkhorn_iterations,
        'match_threshold': stitcher.match_threshold,
    }
}
matching = Matching(config).eval().to(device)

Input_dir = Path(stitcher.input_dir)
print('加载图像数据地址： \"{}\"'.format(Input_dir))
output_viz_dir = Path(stitcher.output_viz_dir)
# output_dir.mkdir(exist_ok=True, parents=True)
print('将匹配图像保存到： \"{}\"'.format(output_viz_dir))
timer = AverageTimer(newline=True)

x_move, y_move = [], []
print("正在运行中......")
count = 0  # 拼接图像计数器(用于拼接断续重连)
dict_list = []
for i, pair in enumerate(pairs):
    dict = {}
    name0, name1 = pair[:2]  # 注意：.txt文件中不能有空行，否则此处会报错
    dict["img0"] = name0
    dict["img1"] = name1
    print("正在匹配", name0, name1)
    stem0, stem1 = Path(name0).stem, Path(name1).stem
    # matches_path = output_dir / '{}_{}_matches.npz'.format(stem0, stem1)
    # if not os.path.exists(output_viz_dir):
    #     os.makedirs(output_viz_dir)
    viz_path = output_viz_dir / '{}_{}_matches.{}'.format(stem0, stem1, stitcher.viz_extension)

    if len(pair) >= 5:
        rot0, rot1 = int(pair[2]), int(pair[3])
    else:
        rot0, rot1 = 0, 0

    # 加载图像(for SuperGlue).
    image0, inp0, scales0 = read_image(
        Input_dir / name0, device, resize, rot0, stitcher.resize_float)
    image1, inp1, scales1 = read_image(
        Input_dir / name1, device, resize, rot1, stitcher.resize_float)
    if image0 is None or image1 is None:
        print('问题图像组: {} {}'.format(
            stitcher.input_dir / name0, stitcher.input_dir / name1))
        exit(1)
    timer.update('load_image')

    # 图像匹配部分
    pred = matching({'image0': inp0, 'image1': inp1})
    pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
    kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
    matches, conf = pred['matches0'], pred['matching_scores0']
    timer.update('matcher')

    # 保存匹配点
    valid = matches > -1
    mkpts0 = kpts0[valid]
    mkpts1 = kpts1[matches[valid]]
    mconf = conf[valid]
    diff = mkpts0 - mkpts1
    # print(diff)
    euclidean_distance = np.sqrt(np.sum(np.square(diff), axis=1))
    mean_euclidean_distance = np.mean(euclidean_distance)
    # print(mean_euclidean_distance)

    # np.savetxt('mean_euclidean_distance_shOD.txt', [mean_euclidean_distance], fmt='%.5f', newline='\n')
    with open('12.txt', 'a') as f:
        f.write(str(mean_euclidean_distance) + '\n')
        # f.write('No'+str(count)+'  '+str(mean_euclidean_distance) + '\n')
        count += 1

    if stitcher.do_viz:
        # Visualize the matches.
        color = cm.jet(mconf)
        text = [
            # 'SuperGlue',
            # 'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
            # 'Matches: {}'.format(len(mkpts0)),
        ]
        if rot0 != 0 or rot1 != 0:
            text.append('Rotation: {}:{}'.format(rot0, rot1))

        # Display extra parameter info.
        k_thresh = matching.superpoint.config['keypoint_threshold']
        m_thresh = matching.superglue.config['match_threshold']
        small_text = [
            # 'Keypoint Threshold: {:.4f}'.format(k_thresh),
            # 'Match Threshold: {:.2f}'.format(m_thresh),
            # 'Image Pair: {}:{}'.format(stem0, stem1),
        ]

        make_matching_plot(
            image0, image1, kpts0, kpts1, mkpts0, mkpts1, color,
            text, viz_path, stitcher.show_keypoints,
            stitcher.fast_viz, stitcher.opencv_display, 'Matches', small_text)

        timer.update('viz_match')
    # h, status = cv2.findHomography(mkpts0, mkpts1)

    try:
        H, SS = cv2.estimateAffinePartial2D(mkpts0, mkpts1)
        assert H is not None
    except:
        dict["x_move"] = "None"
        dict["y_move"] = "None"
        dict["tan"] = "None"
        dict["angle"] = "None"
    else:
        dict["x_move"] = H[0, 2]
        dict["y_move"] = H[1, 2]
        tan = math.atan(H[1, 2] / H[0, 2])
        angle = tan / math.pi * 180
        dict["tan"] = tan
        dict["angle"] = angle
    dict_list.append(dict)

testData = dict_list
fileName = stitcher.input_dir + '测试数据6.10.xlsx'
xw_toExcel(testData, fileName)

# 假设mkpts0、mkpts1和diff分别表示匹配点在第一张图像中的坐标、匹配点在第二张图像中的坐标以及两张图像之间的差值矢量
img = cv2.imread('images12/test1.jpg')
'''
在img上绘制每个特征点的矢量箭头
for i in range(len(mkpts0)):
    pt1 = (int(mkpts0[i, 0]), int(mkpts0[i, 1]))   # 箭头起点坐标
    pt2 = (int(mkpts0[i, 0]-diff[i, 0]), int(mkpts0[i, 1]-diff[i, 1]))   # 箭头终点坐标
    cv2.arrowedLine(img, pt1, pt2, color=(0, 0, 255), thickness=2)
img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.axis('off')
plt.savefig("arrow.png")

cv2.imshow('image with arrows', img)
cv2.imwrite('arrow.jpg', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
def arrow(img):
    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    sum_mkpts0 = np.sum(mkpts0, axis=0)
    avg_mkpts0= sum_mkpts0 / len(mkpts0)

    sum_diff = np.sum(diff, axis=0)
    avg_diff = sum_diff / len(mkpts0)
    # print(avg_diff)
    pt1 = (avg_mkpts0[0], avg_mkpts0[1])  # 箭头起点坐标
    pt2 = (int(pt1[0] - avg_diff[0]), int(pt1[1] - avg_diff[1]))  # 箭头终点坐标
    cv2.arrowedLine(img, pt1, pt2, color=(255, 0, 0), thickness=2)
    plt.imshow(img)
    plt.axis('off')
    plt.savefig("arrow_1.png")

arrow(cv2.imread('images12/test1.jpg'))
