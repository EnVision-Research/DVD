<h2 align="center"> DVD: Deterministic Video Depth Estimation with Generative Priors</h2>
<div align="center">

_**[Hongfei Zhang](https://github.com/soyouthinkyoucantell)<sup>1*</sup>, [Harold H. Chen](https://haroldchen19.github.io/)<sup>1,2*</sup>, [Chenfei Liao](https://chenfei-liao.github.io/)<sup>1*</sup>, [Jing He](https://jingheya.github.io/)<sup>1*</sup>, [Zixi Zhang](https://scholar.google.com/citations?hl=en&user=BbZ0mwoAAAAJ)<sup>1</sup>, [Haodong Li](https://haodong2000.github.io/)<sup>3</sup>, [Yihao Liang](https://scholar.google.com/citations?user=rlKejNUAAAAJ&hl=en)<sup>4</sup>,
<br>
[Kanghao Chen](https://khao123.github.io/)<sup>1</sup>, [Bin Ren](https://amazingren.github.io/)<sup>5</sup>, [Xu Zheng](https://zhengxujosh.github.io/)<sup>1</sup>, [Shuai Yang](https://andysonys.github.io/)<sup>1</sup>, [Kun Zhou](https://redrock303.github.io/)<sup>6</sup>, [Yinchuan Li](https://scholar.google.com/citations?user=M6YfuCTSaKsC&hl=en)<sup>7</sup>, [Nicu Sebe](https://disi.unitn.it/~sebe/)<sup>8</sup>,
<br>
[Ying-Cong Chen](https://www.yingcong.me/)<sup>1,2†</sup>,**_
<br><br>
<sup>*</sup>Equal Contribution; <sup>†</sup>Corresponding Author
<br>
<sup>1</sup>HKUST(GZ), <sup>2</sup>HKUST, <sup>3</sup>UCSD, <sup>4</sup>Princeton University, <sup>5</sup>MBZUAI, <sup>6</sup>SZU, <sup>7</sup>Knowin, <sup>8</sup>UniTrento,

<h5 align="center"> If you like our project, please give us a star ⭐ on GitHub for latest update.  </h2>

 <a href='https://arxiv.org/abs/2511.13704'><img src='https://img.shields.io/badge/arXiv-xxxx.xxxx-b31b1b.svg'></a>
 [![Project Page](https://img.shields.io/badge/DVD-Website-green?logo=googlechrome&logoColor=green)](https://haroldchen19.github.io/TiViBench-Page/)
<br>

</div>



## Introduction


## News

## Installation

### Install from source code:


```
git clone https://github.com/EnVision-Research/DVD.git
cd DVD
conda create -n DVD python=3.10 -y 
conda activate dvd 
pip install -e .
```

### Install SageAttention (For Speedup):
```
pip install sageattention
```
### Download the checkpoint from Huggingface

```
mkdir ckpt
cd ckpt 
huggingface
```

If you encounter issues during installation, it may be caused by the packages we depend on. Please refer to the documentation of the package that caused the problem.

* [torch](https://pytorch.org/get-started/locally/)
* [sentencepiece](https://github.com/google/sentencepiece)
* [cmake](https://cmake.org)
* [cupy](https://docs.cupy.dev/en/stable/install.html)

## 3 Inference

### 3.1. For AIGC or Open World Evaluation (Stable Setting)
```
bash infer_bash/openworld.sh
```

### 3.2. For Academic Purpose (Paper Setting)

#### 3.2.1 Image Inference

For depth estimation, you can download the [evaluation datasets](https://share.phys.ethz.ch/~pf/bingkedata/marigold/evaluation_dataset/) (depth) by the following commands (referred to Marigold).

Run the image inference script

```
bash infer_bash/image.sh
```

#### 3.2.2 Video Inference

Download the [KITTI Dataset](https://www.cvlibs.net/datasets/kitti/), [Bonn Dataset](https://www.ipb.uni-bonn.de/data/rgbd-dynamic-dataset/index.html), [ScanNet Dataset](http://www.scan-net.org/).

Run the video inference script
```
bash infer_bash/video.sh
```



## Training 
