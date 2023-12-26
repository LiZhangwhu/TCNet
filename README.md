# TCNet
Learn More and Learn Usefully: Truncation Compensation Network for Semantic Segmentation of High-Resolution Remote Sensing Images

# Introduction
This is the official code of [Learn More and Learn Usefully: Truncation Compensation Network for Semantic Segmentation of High-Resolution Remote Sensing Images](https://github.com/LiZhangwhu/TCNet/tree/main/TCNet). 
Recent methods for semantic segmentation of high-resolution remote sensing images incorporate a downsized global image as an auxiliary input to alleviate the loss of global context caused by cropping operations. Nonetheless, these methods still face two key challenges: one is the detailed features are weakened when resizing the global auxiliary image through down-sampling; and the other one is the noise brought by the global auxiliary image, which reduces the network’s discriminability of useful and useless information. To tackle these challenges, we propose a Truncation Compensation network (TCNet) for semantic segmentation of high-resolution remote sensing images. Compared with previous methods, we compensate the truncated features in the local image while minimize noise interference to accentuate the learning of useful information. we also design a related category semantic enhancement module to alleviate the information loss caused by downsampling and a global-local contextual cross-fusion module to enrich local image semantic segmentation with long-distance contextual information.

<video width="320" height="240" controls>
  <source src="[video.mp4](https://github.com/LiZhangwhu/TCNet/blob/main/TCNet/video/FCN.mp4)" type="video/mp4">
  <img src="video.jpg">
</video>

 
# Installation
`conda env create -f tcn.ymal`
# Data
All the data formats are unchanged, and the file structure of the data is shown in folder `Data`.  
BLU: `https://rslab.disi.unitn.it/dataset/BLU/`  
GID: `https://x-ytong.github.io/project/GID.html`  
Potsdam: `https://www.isprs.org/education/benchmarks/UrbanSemLab/2d-sem-label-potsdam.aspx`
# How to train
You just need to change your data path in `Train.py` to start training the model.  
At present, you can run the BLU data set directly, if you want to run the GID dataset or Potsdam dataset, please uncomment the code that load the GID or Potsdam data in `Train.py`.
# How to evaluate
You just need to change your data path and your checkpoints path in `Eval.py` to start training the model.
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
Our paper is following the work "Looking Outside the Window: Wide-Context Transformer for the Semantic Segmentation of High-Resolution Remote Sensing Images".
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
```
