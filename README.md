NerveStitcher2.0：Automatic Stitching and Detection Algorithm for Corneal Nerve Images Based on Optical Flow Information
=======
Introduction
-------
NerveStitcher2.0 is an improvement on the NerveStitcher automatic stitching algorithm for correcting mis-stitched images, effectively improving the stitching accuracy of the algorithm. By extracting the optical flow information of each pixel in the IVCM corneal nerve image pair and calculating the image displacement based on the optical flow field, the difference between the displacement of the matched feature points using SuperGlue in NerveStitcher is calculated by interpolation. Any stitching result beyond the confidence interval is considered to be a mis-stitched image.
