## TransResNet: Integrating the Strengths of ViTs and CNNs for High Resolution Medical Image Segmentation via Feature Grafting (BMVC 2022) (Will updated soon)
[Muhammad Hamza Sharif](https://github.com/Sharifmhamza/), [Dmitry Demidov](https://github.com/demidovd98/), [Asif Hanif](https://github.com/asif-hanif/), [Mohammad Yaqub](https://scholar.google.co.uk/citations?hl=en&user=9dfn5GkAAAAJ&view_op=list_works&sortby=pubdate/), and [Min Xu](https://xulabs.github.io/)

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

<img src="https://github.com/Sharifmhamza/TransResNet/blob/main/Final-Architecture.png" align="center">
<p align="center"><img src = "https://github.com/Sharifmhamza/TransResNet/blob/main/Final-Cross%20Grafting%20Module.png" width="500"></p>
<p align="center"><strong>Cross Grafting Module</strong></p>


## Installation

See [INSTALL.md](install.md) for the installation of dependencies required to run TransResNet.

## Demo

To test the pre-trained TransResNet models of [Skin-Segmentation](), [Polyp-Segmentation](), and [Retinal-Vessel-Segmentation]() on your images you can either use the following command as:
```
python 
```
Example usage to perform Skin-Segmentation on a directory of images:
```
python 
```
Example usage to perform Retinal-Vessel-Segmentation on a directory of images:
```
python 
```
Example usage to perform Polyp-Segmentation on a directory of images:
```
python 
```
## Training and Evaluation

Training and Testing instructions for Skin, Polyp, and Retinal-Vessel segmentation are provided in their respective directories. Here is a summary table containing hyperlinks for easy navigation:

<table>
  <tr>
    <th align="left">Task</th>
    <th align="center">Training Instructions</th>
    <th align="center">Testing Instructions</th>
  </tr>
  <tr>
    <td align="left">Skin Segmentation</td>
    <td align="center"><a href="https://github.com/Sharifmhamza/TransResNet/blob/main/Skin-Segmentation/skin.md">Link</a></td>
    <td align="center"><a href="https://github.com/Sharifmhamza/TransResNet/blob/main/Skin-Segmentation/skin.md">Link</a></td>
  </tr>
  <tr>
    <td>Polyp Segmentation</td>
    <td align="center"><a href="https://github.com/Sharifmhamza/TransResNet/blob/main/Polyp-Segmentation/polyp.md">Link</a></td>
    <td align="center"><a href="https://github.com/Sharifmhamza/TransResNet/blob/main/Polyp-Segmentation/polyp.md">Link</a></td>
  </tr>
  <tr>
    <td>Retinal Vessel Segmentation</td>
    <td align="center"><a href="https://github.com/Sharifmhamza/TransResNet/blob/main/Retinal-Vessel-Segmentation/retinal_vessel.md">Link</a></td>
    <td align="center"><a href="https://github.com/Sharifmhamza/TransResNet/blob/main/Retinal-Vessel-Segmentation/retinal_vessel.md">Link</a></td>
  </tr>

</table>

## Quantitative Results
Experiments are performed for different medical segmentation tasks including, polyp, skin, and retinal-vessel.

<details>
<summary><strong>Skin Segmentation</strong> (click to expand) </summary>
<p align="left"><img src = "https://github.com/Sharifmhamza/TransResNet/blob/main/Results/Skin-results.png" width="400"></p>
</details>

<details>
<summary><strong>Retinal Vessel Segmentation</strong> (click to expand) </summary>
<p align="left"><img src = "https://github.com/Sharifmhamza/TransResNet/blob/main/Results/Retinal-vessel-results.png" width="400"></p></details>
</details>

<details>
<summary><strong>Polyp Segmentation</strong> (click to expand) </summary>
<p align="center"><img src = "https://github.com/Sharifmhamza/TransResNet/blob/main/Results/Polyp-results.png"></p></details>
</details>

## Qualitative Results
<details>
<summary><strong>Qualitative Results</strong> (click to expand)</summary>
<p align="center"><img src = "https://github.com/Sharifmhamza/TransResNet/blob/main/Results/Qualitative%20Results.png"></p></details>
</details>


## Citation
If you use TransResNet, please consider citing:

    @inproceedings{Sharif2022TransResNet,
        title={TransResNet: Integrating the Strengths of ViTs and CNNs for High Resolution Medical Image Segmentation via Feature Grafting}, 
        author={Muhammad Hamza Sharif and Dmitry Demidov and Asif Hanif and Mohammad Yaqub and Min Xu},
        booktitle={BMVC},
        year={2022}
    }


## Contact
Should you have any question, please contact sharifmhamza@gmail.com


