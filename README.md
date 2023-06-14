NerveStitcher2.0：Automatic Stitching and Detection Algorithm for Corneal Nerve Images Based on Optical Flow Information
=======
Introduction
-------
NerveStitcher2.0 is an improvement on the NerveStitcher automatic stitching algorithm for correcting mis-stitched images, effectively improving the stitching accuracy of the algorithm. NerveStitcher2.0 differs from NerveStitcher in terms of feature extraction and matching algorithms by extracting the optical flow information of each pixel in the image pairs and avoiding stitching errors caused by matching based on local background features. NerveStitcher2.0 calculates the difference between the displacement of feature points that successfully match between the two image sets and the optical flow field displacement. Stitching results beyond our confidence interval are classified as mis-stitched images. In experiments on 3868 IVCM corneal nerve images from 5 patients, 979 stitching results were found to exceed our confidence interval, resulting in an approximately 25.31% increase in accuracy compared to NerveStitcher.

NerveStitcher2.0 was developed by Yu Qiao of TGU-UOW Lab on the basis of [FlowNet2.0](https://github.com/NVIDIA/flownet2-pytorch) and [SuperGlue](https://github.com/magicleap/SuperGluePretrainedNetwork).

Official Website: [TGU-UOW](https://tgu-uow.gitee.io/)

Usage
------

It is recommended that you download the FlowNet 2.0 pre-training model before using it.[FlowNet2.0.CheckPoint.path.tar](https://drive.google.com/file/d/1hF8vS6YeHkx3j2pfCeQqqZGwA_PJq_Da/view)

Image Stitching
-------
Image size for the stitching section is 384*384 and the files are named in English with a sorted naming convention. NerveStitcher is compatible with **.jpg .png .tiff** format images.
Please refer to stitching.py , modify **input_dir**，**input_pairs_path** (the address of the image data to be stitched),**output_viz_dir** (the address where the final result is saved).**arrow** function visualises the vector information arrows of matching feature points on the input **img**.


