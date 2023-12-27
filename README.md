# TCNet
Learn More and Learn Usefully: Truncation Compensation Network for Semantic Segmentation of High-Resolution Remote Sensing Images

# Introduction
This is the official code of [Learn More and Learn Usefully: Truncation Compensation Network for Semantic Segmentation of High-Resolution Remote Sensing Images](https://github.com/LiZhangwhu/TCNet/tree/main/TCNet). 

Recent methods for semantic segmentation of high-resolution remote sensing images incorporate a downsized global image as an auxiliary input to alleviate the loss of global context caused by cropping operations. Nonetheless, these methods still face two key challenges: one is the detailed features are weakened when resizing the global auxiliary image through down-sampling; and the other one is the noise brought by the global auxiliary image (GAI), which reduces the network’s discriminability of useful and useless information. As shown in the comparison.mp4 below. Using the Potsdam dataset, we trained three models: FCN, previous method with GAI, and our model with GUAI. From the red and blue boxes in comparison.mp4, we observe that there are large-scale misclassification results in previous method due to the direct introduction of the entire GAI into the network.

https://private-user-images.githubusercontent.com/109151568/292878979-8df3ca8f-4960-4cd3-9141-6693953bd6ba.mp4?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTEiLCJleHAiOjE3MDM1OTk3NjgsIm5iZiI6MTcwMzU5OTQ2OCwicGF0aCI6Ii8xMDkxNTE1NjgvMjkyODc4OTc5LThkZjNjYThmLTQ5NjAtNGNkMy05MTQxLTY2OTM5NTNiZDZiYS5tcDQ_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBSVdOSllBWDRDU1ZFSDUzQSUyRjIwMjMxMjI2JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDIzMTIyNlQxNDA0MjhaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT1lYzNlMWQyN2E0ODgyMTdkNWMzMzk5ODM4NzcwOTZjMDE1ZTNiYmQ1NTk2Mzc5YjFjNzVlZDJjNWMxNTQ5MGU2JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZhY3Rvcl9pZD0wJmtleV9pZD0wJnJlcG9faWQ9MCJ9.tzSuAVn2xjaMrb5J1JYrAWrxiHfBBeHW4Nzmnjb7Vds

To tackle these challenges, we propose a Truncation Compensation network (TCNet) for semantic segmentation of high-resolution remote sensing images. Compared with previous methods, we compensate the truncated features in the local image while minimize noise interference to accentuate the learning of useful information (GUAI, as shown in the figure below). Upon comparison of the second and third rows, our approch noticeably diminishes the adverse effects stemming from irrelevant information. Intuitively, the removal of interference information substantially enhances the network's ability to discriminate between useful and useless information within GAI.

![fig1](https://github.com/LiZhangwhu/TCNet/blob/main/TCNet/pic/fig1.png)

Besides, we also design a related category semantic enhancement module (RSEM) to alleviate the information loss caused by downsampling and a global-local contextual cross-fusion module (CFM) to enrich local image semantic segmentation with long-distance contextual information. We illustrate the overall framework of TCNet in the Figure. 
 ![fig2](https://github.com/LiZhangwhu/TCNet/blob/main/TCNet/pic/fig2.png)

Comprehensive experiments conducted on the ISPRS, BLU, and GID datasets demonstrate the superior performance of our proposed method compared to alternative approaches. We present some segmentation results of our method in TCNet.mp4.

https://private-user-images.githubusercontent.com/109151568/292879027-e4f45834-bffc-41af-92cd-f5527ab7c85a.mp4?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTEiLCJleHAiOjE3MDM2NDM5MDQsIm5iZiI6MTcwMzY0MzYwNCwicGF0aCI6Ii8xMDkxNTE1NjgvMjkyODc5MDI3LWU0ZjQ1ODM0LWJmZmMtNDFhZi05MmNkLWY1NTI3YWI3Yzg1YS5tcDQ_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBSVdOSllBWDRDU1ZFSDUzQSUyRjIwMjMxMjI3JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDIzMTIyN1QwMjIwMDRaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT1iYjg3MDhhYjdlMzk4ODliOGQxODZiNGZhNjFmMGU4YjZlOGM3MWE4MTY5YjczNmUyNGZmNmY3OTkzOTAxNWQ5JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZhY3Rvcl9pZD0wJmtleV9pZD0wJnJlcG9faWQ9MCJ9.mMgMWetpUCJ-FdyEOaV8lxk5edZbyTjDBS03H3_rig4

# Quick start
## Download
Clone this repo to local
`git clone https://github.com/LiZhangwhu/TCNet.git`
## Installation
Run the following command to quickly configure the required environment.
`conda env create -f tcn.ymal`
## Data
Download the data via the link below.  
BLU: `https://rslab.disi.unitn.it/dataset/BLU/`  
GID: `https://x-ytong.github.io/project/GID.html`  
Potsdam: `https://www.isprs.org/education/benchmarks/UrbanSemLab/2d-sem-label-potsdam.aspx`

Your directory tree should be look like this:
```
$SEG_ROOT/data
├── BLU
│   ├── train
│   │   ├── image
│   │   └── label
│   ├── val
│   │   ├── image
│   │   └── label
│   └── test
│   │   ├── image
│   │   └── label
├── GID
│   ├── train
│   │   ├── image
│   │   └── label
│   ├── val
│   │   ├── image
│   │   └── label
│   └── test
│   │   ├── image
│   │   └── label
├── Potsdam
│   ├── train
│   ├── test
│   ├── val
│   ├── dsm
│   └── Labels_noBoundary
```
## How to train
You just need to change your data path in `Train.py` to start training the model.  
```
conda activate tcn
python Train.py
```
At present, you can run the BLU data set directly, if you want to run the GID dataset or Potsdam dataset, please uncomment the code that load the GID or Potsdam data in `Train.py`.
## How to evaluate
You just need to change your data path and your checkpoints path in `Eval.py` to start training the model.
```
conda activate tcn
python Eval.py
```
# Citation
Please cite the following paper, if you use this code.  
```
 @article{Zhang2023learn,  
  title={Learn More and Learn Usefully: Truncation Compensation Network for Semantic Segmentation of High-Resolution Remote Sensing Images},  
  author={Li Zhang, Zhenshan Tan, Guo Zhang, Wen Zhang, Zhijiang Li},  
  journal={IEEE TRANSACTIONS ON GEOSCIENCE AND REMOTE SENSING}  
  year={2023}  
}
```
# Reference
Our code is following the work "Looking Outside the Window: Wide-Context Transformer for the Semantic Segmentation of High-Resolution Remote Sensing Images".
```
@ARTICLE{ding2106looking,
  author={Ding, Lei and Lin, Dong and Lin, Shaofu and Zhang, Jing and Cui, Xiaojie and Wang, Yuebin and Tang, Hao and Bruzzone, Lorenzo},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Looking Outside the Window: Wide-Context Transformer for the Semantic Segmentation of High-Resolution Remote Sensing Images}, 
  year={2022},
  volume={60},
  number={},
  pages={1-13},
  doi={10.1109/TGRS.2022.3168697}}
@article{chen2019collaborative,
  author={Chen, Wuyang and Jiang, Ziyu and Wang, Zhangyang and Cui, Kexin and Qian, Xiaoning},
  booktitle={2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)}, 
  title={Collaborative Global-Local Networks for Memory-Efficient Segmentation of Ultra-High Resolution Images}, 
  year={2019},
  volume={},
  number={},
  pages={8916-8925},
  doi={10.1109/CVPR.2019.00913}}
@article{ding2020semantic,
  title={Semantic segmentation of large-size VHR remote sensing images using a two-stage multiscale training architecture},
  author={Ding, Lei and Zhang, Jing and Bruzzone, Lorenzo},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  volume={58},
  number={8},
  pages={5367--5376},
  year={2020},
  publisher={IEEE}
}
```
