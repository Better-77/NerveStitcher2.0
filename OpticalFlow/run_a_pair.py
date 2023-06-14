import torch
import numpy as np
import argparse
import os
import cv2
import matplotlib.pyplot as plt

from utils.flow_utils import flow2img

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from utils.frame_utils import read_gen
from models import FlowNet2  # the path is depended on where you create this module


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
    pim1 = read_gen("images/test2.jpg")
    pim2 = read_gen("images/test1.jpg")
    '''
    crop_size = (145, 85)
    pim_1 = cv2.resize(pim1, crop_size, interpolation=cv2.INTER_CUBIC)
    pim_2 = cv2.resize(pim2, crop_size, interpolation=cv2.INTER_CUBIC)
    '''
    images = [pim1, pim2]
    images = np.array(images).transpose(3, 0, 1, 2)
    im = torch.from_numpy(images.astype(np.float32)).unsqueeze(0).cuda()

    # process the image pair to obtian the flow
    result = net(im).squeeze()


    # save flow, I reference the code in scripts/run-flownet.py in flownet2-caffe project
    def writeFlow(name, flow):
        f = open(name, 'wb')
        f.write('PIEH'.encode('utf-8'))
        np.array([flow.shape[1], flow.shape[0]], dtype=np.int32).tofile(f)
        flow = flow.astype(np.float32)
        flow.tofile(f)
        f.flush()
        f.close()


    def readFlow(name):
        f = open(name, 'rb')
        header = f.read(4).decode('utf-8')
        if header != 'PIEH':
            raise Exception('Flow file header does not contain PIEH')
        width = np.fromfile(f, np.int32, 1).squeeze()
        height = np.fromfile(f, np.int32, 1).squeeze()
        flow = np.fromfile(f, np.float32, width * height * 2).reshape((height, width, 2))
        return flow


    data = result.data.cpu().numpy().transpose(1, 2, 0)
    result1 = np.array(data).reshape(-1, 2)

    np.savetxt(r"flo_txt_car.txt", result1, fmt='%s', delimiter=',')

    img = flow2img(data)
    #cv2.imwrite('flow.png', img)
    plt.axis('off')
    plt.imshow(img)
    plt.savefig('flow.jpg')
    writeFlow("flow.flo", data)
    flow = readFlow("flow.flo")

    # Calculate the magnitude of the optical flow vectors
    magnitude = np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2)

    # Calculate the average speed
    final_speed = np.mean(magnitude)
    print(final_speed)

    flow_x = flow[:, :, 0]
    flow_y = flow[:, :, 1]

    # Calculate the x and y components of the motion speed separately if needed
    avg_flow_x = np.mean(flow_x)
    avg_flow_y = np.mean(flow_y)

    # Determine the direction of motion based on the sign of the average x and y components
    if avg_flow_x > 0 and avg_flow_y < 0:
        print("The motion is towards the top right corner of the image.")
    elif avg_flow_x > 0 and avg_flow_y > 0:
        print("The motion is towards the bottom right corner of the image.")
    elif avg_flow_x < 0 and avg_flow_y < 0:
        print("The motion is towards the top left corner of the image.")
    elif avg_flow_x < 0 and avg_flow_y > 0:
        print("The motion is towards the bottom left corner of the image.")
    else:
        print("There is no motion in the image.")

