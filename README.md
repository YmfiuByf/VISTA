# VISTA

**VISTA** (**Vi**deo Transmission over A **S**emantic Communication Approach) is a semantic video transmission framework for wireless environments.  
Instead of transmitting all pixels of every frame, VISTA focuses on transmitting **semantics** of the video, aiming to reduce bandwidth consumption while preserving visual quality.

The framework combines:

- **Semantic Segmentation** to separate static background and dynamic objects
- **Dynamic JSCC** for channel-adaptive semantic transmission
- **Video Frame Interpolation** to recover missing behavior frames at the receiver

This repository contains the implementation of the VISTA pipeline and the components used in the paper.

---

## Overview

Traditional wireless video transmission sends pixel-level information and can be costly in bandwidth and processing time.  
VISTA addresses this by splitting video frames into:

- **Environment segment**: static background
- **Behavior segment**: moving objects
- **SLG (Semantic Location Graph)**: semantic labels and object locations across frames

Only the necessary semantic information is transmitted, and the receiver reconstructs the video with interpolation and semantic guidance.

<p align="center">
  <img src="results/fig1_transceiver.png" alt="VISTA transceiver diagram" width="95%">
</p>

<p align="center"><em>Figure 1. Transceiver design of VISTA.</em></p>

---

## Pipeline

### 1. Semantic Segmentation

In the first step, video frames are segmented into static background and moving objects.

VISTA uses **MMSegmentation** for semantic segmentation.

Run:

```bash
..\VISTA\mmsegmentation-master\demo\img_seg.py

This stage prepares the semantic components needed for transmission.

2. Dynamic JSCC

The segmented semantic features are transmitted through a dynamic joint source-channel coding (JSCC) module, which adapts to channel conditions.

This part is based on the paper:

Deep Joint Source-Channel Coding for Wireless Image Transmission with Adaptive Rate Control

Reference implementation:
https://github.com/mingyuyng/Dynamic_JSCC

3. Video Frame Interpolation

At the receiver side, missing behavior frames are recovered by frame interpolation, guided by transmitted semantic information.

This part is based on the paper:

Video Frame Interpolation with Transformer

Reference implementation:
https://github.com/dvlab-research/VFIformer

Run:

..\VISTA\VFIformer-main\video_interpolation_paper.py
Main Idea

The key idea of VISTA is to avoid transmitting redundant visual information.

The static environment only needs limited transmission

The dynamic behavior segments are sampled and transmitted more efficiently

The SLG helps preserve object semantics and spatial relations

The frame interpolation module reconstructs missing intermediate behavior frames

This design improves transmission efficiency while maintaining good reconstruction quality, especially under noisy wireless conditions.

Results
PSNR under Different SNRs

The figure below shows the PSNR performance of recovered frames under different channel SNRs and interpolation ratios.

<p align="center"> <img src="results/fig2_psnr.png" alt="PSNR performance under different SNRs" width="70%"> </p> <p align="center"><em>Figure 2. PSNR of recovered video frames versus SNR.</em></p>

From the paper, VISTA achieves stronger robustness in low-SNR settings and shows better visual quality than the conventional scheme in challenging channel conditions.

Visual Comparison

The following comparison shows reconstructed video frames under different methods and interpolation settings.

<p align="center"> <img src="results/fig3_visual_comparison.png" alt="Visual comparison on VIRAT frame" width="95%"> </p> <p align="center"><em>Figure 3. Visual comparison of reconstructed frames.</em></p>

VISTA preserves object structures more clearly and produces better perceptual quality than the compared baseline methods.

Processing Time

The figure below compares total processing time for 20 consecutive video frames.

<p align="center"> <img src="results/fig4_processing_time.png" alt="Processing time comparison" width="70%"> </p> <p align="center"><em>Figure 4. Total processing time under different interpolation ratios.</em></p>

VISTA significantly reduces processing time compared with the conventional scheme, while also remaining more efficient than the JSCC-VFI baseline.

Transmission Bits

The figure below compares total transmission bits for 20 consecutive video frames.

<p align="center"> <img src="results/fig5_transmission_bits.png" alt="Transmission bits comparison" width="70%"> </p> <p align="center"><em>Figure 5. Total transmission bits under different interpolation ratios.</em></p>

One of the main advantages of VISTA is its large reduction in required transmission bits, showing the benefit of transmitting semantics instead of raw pixel-level information.

Project Structure

A possible structure of this project is:

VISTA/
├── mmsegmentation-master/
│   └── demo/
│       └── img_seg.py
├── VFIformer-main/
│   └── video_interpolation_paper.py
├── results/
│   ├── fig1_transceiver.png
│   ├── fig2_psnr.png
│   ├── fig3_visual_comparison.png
│   ├── fig4_processing_time.png
│   └── fig5_transmission_bits.png
└── README.md
Requirements

This project depends on the following major components:

Python 3.9

MMSegmentation

Dynamic JSCC implementation

VFIformer

Please install the dependencies required by each module before running the full pipeline.

Reference

If you use this project, please cite the paper:

@article{liang2023vista,
  title={VISTA: Video Transmission over A Semantic Communication Approach},
  author={Liang, Chengsi and Deng, Xiangyi and Sun, Yao and Cheng, Runze and Xia, Le and Niyato, Dusit and Imran, Muhammad Ali},
  journal={arXiv preprint arXiv},
  year={2023}
}
Acknowledgements

This project builds upon the following open-source works:

MMSegmentation

Dynamic JSCC

VFIformer

Their excellent implementations made this project possible.
