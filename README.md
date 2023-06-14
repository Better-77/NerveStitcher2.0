NerveStitcher2.0：Automatic Stitching and Detection Algorithm for Corneal Nerve Images Based on Optical Flow Information
=======
Introduction
-------
NerveStitcher2.0 is an improvement on the NerveStitcher automatic stitching algorithm for correcting mis-stitched images, effectively improving the stitching accuracy of the algorithm. NerveStitcher2.0 differs from NerveStitcher in terms of feature extraction and matching algorithms by extracting the optical flow information of each pixel in the image pairs and avoiding stitching errors caused by matching based on local background features. NerveStitcher2.0 calculates the difference between the displacement of feature points that successfully match between the two image sets and the optical flow field displacement. Stitching results beyond our confidence interval are classified as mis-stitched images. In experiments on 3868 IVCM corneal nerve images from 5 patients, 979 stitching results were found to exceed our confidence interval, resulting in an approximately 25.31% increase in accuracy compared to NerveStitcher.

NerveStitcher2.0 was developed by Yu Qiao of TGU-UOW Lab on the basis of [FlowNet2.0](https://github.com/NVIDIA/flownet2-pytorch) and [SuperGlue](https://github.com/magicleap/SuperGluePretrainedNetwork).

Official Website: [TGU-UOW](https://tgu-uow.gitee.io/)

Usage
------

