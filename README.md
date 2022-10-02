## TransResNet: Integrating the Strengths of ViTs and CNNs for High Resolution Medical Image Segmentation via Feature Grafting (BMVC 2022)
[Muhammad Hamza Sharif](https://github.com/Sharifmhamza/), [Dmitry Demidov](https://github.com/Talal-Algumaei/), [Asif Hanif](https://github.com/asif-hanif/), [Mohammad Yaqub](https://scholar.google.co.uk/citations?hl=en&user=9dfn5GkAAAAJ&view_op=list_works&sortby=pubdate/), and [Min Xu](https://xulabs.github.io/)

[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)]()
[![supplement](https://img.shields.io/badge/Supplementary-Material-red)]()
[![video](https://img.shields.io/badge/Video-Presentation-F9D371)]()
[![slides](https://img.shields.io/badge/Presentation-Slides-B762C1)]()
[![Summary](https://img.shields.io/badge/Summary-Slide-87CEEB)]()

<hr />

> **Abstract:** *High-resolution images are preferable in medical imaging domain as they significantly improve the diagnostic capability of the underlying method. In particular, high resolution helps substantially in improving automatic image segmentation. However, most of the existing deep learning-based techniques for medical image segmentation are optimized for input images having small spatial dimensions and perform poorly on
high-resolution images. To address this shortcoming, we propose a parallel-in-branch architecture called TransResNet, which incorporates Transformer and CNN in a parallel manner to extract features from multi-resolution images independently. In TransResNet, we introduce Cross Grafting Module (CGM), which generates the grafted features, enriched in both global semantic and low-level spatial details, by combining the feature maps from Transformer and CNN branches through fusion and self-attention mechanism. Moreover, we use these grafted features in the decoding process, increasing the information flow for better prediction of the segmentation mask. Extensive experiments on ten datasets demonstrate that TransResNet achieves either state-of-the-art
or competitive results on several segmentation tasks, including skin lesion, retinal vessel, and polyp segmentation.* 
<hr />

## Network Architecture

  <img src="https://github.com/Sharifmhamza/TransResNet/blob/main/Architecture.png" align="center">
  <img src="https://github.com/Sharifmhamza/TransResNet/blob/main/CGM.png" align="center"/> 




