# VISTA

## Image segmentation
In the first step of the process, frames to be transmitted are first segmented into static objects and moving objects. [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) is used here for this purpose. Run "..\VISTA\mmsegmentation-master\demo\img_seg.py" to do segmentation in VISTA.

## Dynamic JSCC
 Joint source channel coding based on paper _Deep Joint Source-Channel Coding for Wireless Image Transmission with Adaptive Rate Control_. Source github page https://github.com/mingyuyng/Dynamic_JSCC. 

## Video Frame Interpolation
Interpolate lost frames based on received ones to recover the full video based on paper
[Video Frame Interpolation with Transformer](https://arxiv.org/abs/2205.07230). Source github page https://github.com/dvlab-research/VFIformer. Run py file "..\VISTA\VFIformer-main\video_interpolation_paper.py" to perform task in VISTA. 

## Results
Sample video and simulation results can be found in the _video_ file.

