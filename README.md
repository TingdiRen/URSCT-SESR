# URSCT-SESR: Reinforced Swin-Convs Transformer for Simultaneous Underwater Sensing Scene Image Enhancement and Super-resolution
[Tingdi Ren](http://www.adilifer.com/), Haiyongxu, Gangyi Jiang, Mei Yu, Xuan Zhang, Biao Wang, and Ting Luo.


---


[![arXiv](https://img.shields.io/badge/IEEE-Paper-%3CCOLOR%3E.svg)](https://ieeexplore.ieee.org/document/9881581)
[![GitHub Stars](https://img.shields.io/github/stars/TingdiRen/URSCT-SESR?style=social)](https://github.com/TingdiRen/URSCT-SESR)
[![download](https://img.shields.io/github/downloads/TingdiRen/URSCT-SESR/total.svg)](https://github.com/TingdiRen/URSCT-SESR/releases)
![visitors](https://visitor-badge.glitch.me/badge?page_id=TingdiRen/URSCT-SESR)
[ <a href="https://colab.research.google.com/gist/JingyunLiang/a5e3e54bc9ef8d7bf594f6fee8208533/swinir-demo-on-real-world-image-sr.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="google colab logo"></a>](https://colab.research.google.com/gist/JingyunLiang/a5e3e54bc9ef8d7bf594f6fee8208533/swinir-demo-on-real-world-image-sr.ipynb)


This repository is the official PyTorch implementation of URSCT-SESR: Reinforced Swin-Convs Transformer for Simultaneous Underwater Sensing Scene Image Enhancement and Super-resolution
<img width="1000" src="README_figs/network.png">

<img width="180" src="README_figs/RAW.png"><img width="180" src="README_figs/CF.png"><img width="180" src="README_figs/IBLA.png"><img width="180" src="README_figs/HL.png"><img width="180" src="README_figs/WATERNET.png"> <img width="180" src="README_figs/UWCNN.png"><img width="180" src="README_figs/UCOLOR.png"><img width="180" src="README_figs/USHAPED.png"><img width="180" src="README_figs/OUR.png">


## Contents

1. [Training](#Training)
2. [Testing](#Testing)
3. [Download](#Download)
4. [Citation](#Citation)



## Training

### 1. Put your dataset into the folder "_data_" as follows:
_URSCT-SESR_<br />
├─ other files and folders<br />
├─ _data_<br />
│&ensp;&ensp;├─ _train\_data_<br />
│&ensp;&ensp;│&ensp;&ensp;├─ _input_<br />
│&ensp;&ensp;│&ensp;&ensp;&ensp;&ensp;&ensp;├─ _fig1.png_<br />
│&ensp;&ensp;│&ensp;&ensp;&ensp;&ensp;&ensp;├─ ...<br />
│&ensp;&ensp;│&ensp;&ensp;├─ _target_<br />
│&ensp;&ensp;│&ensp;&ensp;&ensp;&ensp;&ensp;├─ _fig1.png_<br />
│&ensp;&ensp;│&ensp;&ensp;&ensp;&ensp;&ensp;├─ ...<br />
│&ensp;&ensp;├─ _val\_data_<br />
│&ensp;&ensp;│&ensp;&ensp;&ensp;&ensp;&ensp;├─ ...<br />
│&ensp;&ensp;├─ _test\_data_<br />
│&ensp;&ensp;│&ensp;&ensp;&ensp;&ensp;&ensp;├─ ...

### 2. Configure the _\*.yaml_:
If you want to train with the default setting, _\*\_DIR_ of _TRAINING_ and _TEST_ is the main option you need to edit.

(1) _Enh&SR\_opt.yaml_ for Simultaneous Underwater Sensing Scene Image Enhancement and Super-resolution

(2) _Enh\_opt.yaml_ for Underwater Sensing Scene Image Enhancement only

### 3. Run _\*\_train.py_

## Testing

### 1. As reported above, put your dataset for testing and model we provided into the folders as follows:
_URSCT-SESR_<br />
├─ other files and folders<br />
├─ _checkpoints_<br />
│&ensp;&ensp;├─ _model\_test_ (same as configurated above)<br />
│&ensp;&ensp;│&ensp;&ensp;├─ _models_<br />
│&ensp;&ensp;│&ensp;&ensp;&ensp;&ensp;&ensp;├─ _model\_bestSSIM.pth_ (downloaded model)<br />
├─ _data_<br />
│&ensp;&ensp;├─ _train\_data_<br />
│&ensp;&ensp;├─ _val\_data_<br />
│&ensp;&ensp;├─ _test\_data_<br />
│&ensp;&ensp;│&ensp;&ensp;├─ _input_<br />
│&ensp;&ensp;│&ensp;&ensp;&ensp;&ensp;&ensp;├─ _fig1.png_<br />
│&ensp;&ensp;│&ensp;&ensp;&ensp;&ensp;&ensp;├─ ...<br />
│&ensp;&ensp;│&ensp;&ensp;├─ _target_<br />
│&ensp;&ensp;│&ensp;&ensp;&ensp;&ensp;&ensp;├─ _fig1.png_<br />
│&ensp;&ensp;│&ensp;&ensp;&ensp;&ensp;&ensp;├─ ...<br />

### 2. Run _\*\_eval.py_

## Download
### Model
(1) [GoogleDrive](https://drive.google.com/drive/folders/1ljhYcXwbdJ0fYzlF9UbEbZwr5zKk1ERb?usp=sharing)

(2) [BaiduDisk](https://pan.baidu.com/s/1SSwjv37uvwR7Zilq0s3YHw) (Password: SESR)

### Dataset
(1) LSUI (UIE): [Data](https://lintaopeng.github.io/_pages/UIE%20Project%20Page.html) [Paper](https://arxiv.org/abs/2111.11843) [Homepage](https://lintaopeng.github.io/_pages/UIE%20Project%20Page.html)

(2) UIEB (UIE):  [Data](https://li-chongyi.github.io/proj_benchmark.html) [Paper](https://ieeexplore.ieee.org/document/8917818) [Homepage](https://li-chongyi.github.io/proj_benchmark.html)

(3) SQUID (UIE): [Data](http://csms.haifa.ac.il/profiles/tTreibitz/datasets/ambient_forwardlooking/index.html) [Paper](https://ieeexplore.ieee.org/abstract/document/9020130) [Homepage](http://csms.haifa.ac.il/profiles/tTreibitz/datasets/ambient_forwardlooking/index.html)

(4) UFO (SESR): [Data](https://irvlab.cs.umn.edu/resources/ufo-120-dataset) [Paper](https://arxiv.org/abs/2002.01155) [Homepage](https://irvlab.cs.umn.edu/projects/deep-sesr)

(5) USR (SR): [Data](https://irvlab.cs.umn.edu/resources/usr-248-dataset) [Paper](https://arxiv.org/abs/1909.09437) [Homepage](https://irvlab.cs.umn.edu/projects/srdrm)

## Citation
    @article{ren2022reinforced,
	  title={Reinforced Swin-convs Transformer for Simultaneous Underwater Sensing Scene Image Enhancement and Super-resolution},
	  author={Ren, Tingdi and Xu, Haiyong and Jiang, Gangyi and Yu, Mei and Zhang, Xuan and Wang, Biao and Luo, Ting},
	  journal={IEEE Transactions on Geoscience and Remote Sensing},
	  year={2022},
	  publisher={IEEE}
	}
