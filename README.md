NerveStitcher2.0：Automatic Stitching and Detection Algorithm for Corneal Nerve Images Based on Optical Flow Information
=======
Introduction
-------
NerveStitcher2.0 is an improvement on the NerveStitcher automatic stitching algorithm for correcting mis-stitched images, effectively improving the stitching accuracy of the algorithm. NerveStitcher2.0 differs from NerveStitcher in terms of feature extraction and matching algorithms by extracting the optical flow information of each pixel in the image pairs and avoiding stitching errors caused by matching based on local background features. NerveStitcher2.0 calculates the difference between the displacement of feature points that successfully match between the two image sets and the optical flow field displacement. Stitching results beyond our confidence interval are classified as mis-stitched images. In experiments on 3868 IVCM corneal nerve images from 5 patients, 979 stitching results were found to exceed our confidence interval, resulting in an approximately 25.31% increase in accuracy compared to NerveStitcher.

NerveStitcher2.0 was developed by Yu Qiao of TGU-UOW Lab on the basis of [FlowNet2.0](https://github.com/NVIDIA/flownet2-pytorch) and [SuperGlue](https://github.com/magicleap/SuperGluePretrainedNetwork).

Official Website: [TGU-UOW](https://tgu-uow.gitee.io/)

Usage
------

It is recommended that you download the FlowNet 2.0 pre-training model before using it. [FlowNet2.0.CheckPoint.path.tar](https://drive.google.com/file/d/1hF8vS6YeHkx3j2pfCeQqqZGwA_PJq_Da/view)

Image Stitching
-------
Image size for the stitching section is 384*384 and the files are named in English with a sorted naming convention. NerveStitcher is compatible with **.jpg .png .tiff** format images.


Please refer to **stitch_eval.py** Complete feature point extraction and matching of the input image, modify **input_dir**, **input_pairs_path** (the address of the image data to be stitched), **output_viz_dir** (the address where the final result is saved). The displacement of the images **(mean_euclidean_distance)** calculated based on the motion vectors of the matched feature points is saved in a text file, which can be named by modifying in **line 153**. **arrow** function visualises the vector information arrows of matching feature points on the input **img**. 

**stitch_img.py** responsible for completing the stitching of image pairs, modify **input_dir** and **xlsx_dir**(the address of the image data to be stitched). The folder structure is based on the example file **images/stitch_img**, where the **"images/result"** folder holds the results of each stitching.

Optical Flow Information Estimation And Visualisation
-------
The relevant program for extracting optical flow information is in folder **OpticalFlow**.

"run_a_pair.py" can extract optical flow information from image pairs obtained from consecutive frames, and calculate the displacement speed of the images based on the optical flow field, modify **dict** is [FlowNet2.0.CheckPoint.path.tar](https://drive.google.com/file/d/1hF8vS6YeHkx3j2pfCeQqqZGwA_PJq_Da/view), modify **pim1 pim2** refers to the input image pairs that you want to extract optical flow information from, can roughly predict the motion direction of the images based on the extracted optical flow vectors. "**flow.jpg**" represents the result of dense visualization of the extracted optical flow information.

If you want to perform the above operations on multiple sets of image pairs, **"run_much.py"** can help you process all the images in a folder. You just need to modify the **"image_folder"** parameter.The displacement of the image pairs **(magnitude)** calculated based on the extracted optical flow field is saved in a text file, which can be named by modifying in **line 72**.

**"Arrow_pair.py"** can convert the extracted optical flow information into sparse optical flow and represent the motion direction of the images more directly through arrow visualization. You need to modify **pim1** and **pim2** in lines 122 and 123 to the images you want to input. The **stride** parameter in **sparse_flow** represents the step size of the visualization sampling points.











