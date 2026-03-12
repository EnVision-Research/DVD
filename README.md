<h2 align="center"> DVD: Deterministic Video Depth Estimation with Generative Priors</h2>
<div align="center">

_**[Hongfei Zhang](https://x.com/hongfeizhang0xF)<sup>1*</sup>, [Harold H. Chen](https://haroldchen19.github.io/)<sup>1,2*</sup>, [Chenfei Liao](https://chenfei-liao.github.io/)<sup>1*</sup>, [Jing He](https://jingheya.github.io/)<sup>1*</sup>, [Zixin Zhang](https://scholar.google.com/citations?hl=en&user=BbZ0mwoAAAAJ)<sup>1</sup>, [Haodong Li](https://haodong2000.github.io/)<sup>3</sup>, [Yihao Liang](https://scholar.google.com/citations?user=rlKejNUAAAAJ&hl=en)<sup>4</sup>,
<br>
[Kanghao Chen](https://khao123.github.io/)<sup>1</sup>, [Bin Ren](https://amazingren.github.io/)<sup>5</sup>, [Xu Zheng](https://zhengxujosh.github.io/)<sup>1</sup>, [Shuai Yang](https://andysonys.github.io/)<sup>1</sup>, [Kun Zhou](https://redrock303.github.io/)<sup>6</sup>, [Yinchuan Li](https://scholar.google.com/citations?user=M6YfuCTSaKsC&hl=en)<sup>7</sup>, [Nicu Sebe](https://disi.unitn.it/~sebe/)<sup>8</sup>,
<br>
[Ying-Cong Chen](https://www.yingcong.me/)<sup>1,2†</sup>,**_
<br><br>
<sup>*</sup>Equal Contribution; <sup>†</sup>Corresponding Author
<br>
<sup>1</sup>HKUST(GZ), <sup>2</sup>HKUST, <sup>3</sup>UCSD, <sup>4</sup>Princeton University, <sup>5</sup>MBZUAI, <sup>6</sup>SZU, <sup>7</sup>Knowin, <sup>8</sup>UniTrento,

<h5 align="center"> If you like our project, please give us a star ⭐ on GitHub for latest update.  </h2>

 <a href='#'><img src='https://img.shields.io/badge/arXiv-TBD-b31b1b.svg'></a>
 [![Project Page](https://img.shields.io/badge/DVD-Website-green?logo=googlechrome&logoColor=green)](https://dvd-project.github.io/)
<br>

</div>

![framework](assets/teaser.png)



## 👋 Introduction

Welcome to the official repository for **DVD: Deterministic Video Depth**! 

While recent generative foundation models have shown remarkable potential in zero-shot depth estimation, they inherently suffer from the *ambiguity-hallucination dilemma* due to their stochastic sampling process. **DVD** fundamentally shifts this paradigm. We present the first deterministic framework that elegantly adapts pre-trained Video Diffusion Models (like WanV2.1) into single-pass depth regressors. 

By cleanly stripping away generative stochasticity, DVD unites the semantic richness of generative foundation models with the structural stability of discriminative regressors.

### ✨ Key Highlights

* 🚀 **Extreme Data Efficiency:** DVD effectively unlocks profound generative priors using only **367K frames**—which is **163× less** task-specific training data than leading discriminative baselines like VDA (60M frames).
* ⏱️ **Deterministic & Fast:** Bypasses iterative ODE integration. Inference is performed in a single forward pass, ensuring absolute temporal stability without generative hallucinations.
* 📐 **Unparalleled Structural Fidelity:** Powered by our Latent Manifold Rectification (LMR), DVD achieves state-of-the-art high-frequency boundary precision (Boundary Recall & F1) compared to overly smoothed baselines.
* 🎥 **Long-Video Inference:** Equipped with our training-free *Global Affine Coherence* module, DVD seamlessly stitches sliding windows to support unconstrained long-video rollouts with zero scale drift.

> **TL;DR:** If you want state-of-the-art video depth estimation that is highly detailed, temporally stable across long videos, and exceptionally data-efficient, **DVD** is what you need.

---

## 📢 News

## 🛠️ Installation

### 📦 Install from source code (Basic Dependency):


```
git clone https://github.com/EnVision-Research/DVD.git
cd DVD
conda create -n dvd python=3.10 -y 
conda activate dvd 
pip install -e .
```

### 🏃 Install SageAttention (For Speedup):
```
pip install sageattention # DO NOT USE THIS FOR TRAINING!!!
```
### 🤗 Download the checkpoint from Huggingface

#### 1. Make dir and login to huggingface
```
mkdir ckpt
huggingface-cli login # Or hf auth login 
```
####  2. Download the checkpoint

```
huggingface-cli download FayeHongfeiZhang/DVD --revision main --local-dir ckpt
```

#### 3. The final structure shoule be like
```
DVD
├── ckpt/
├──── model_config.yaml
├──── model.safetensors
├── configs/
├── examples/
├── ...
```

### 💅🏻 Potential Issue (from [DiffSynth Studio](https://github.com/modelscope/DiffSynth-Studio))

If you encounter issues during installation, it may be caused by the packages we depend on. Please refer to the documentation of the package that caused the problem.

* [torch](https://pytorch.org/get-started/locally/)
* [sentencepiece](https://github.com/google/sentencepiece)
* [cmake](https://cmake.org)
* [cupy](https://docs.cupy.dev/en/stable/install.html)

## 🕹️ Inference

### 🤹🏼‍♂️ For AIGC or Open World Evaluation (Stable Setting)
```
bash infer_bash/openworld.sh
```

### 👩🏼‍🏫 For Academic Purpose (Paper Setting)


####  1. Video Inference

1.1. Download the [KITTI Dataset](https://www.cvlibs.net/datasets/kitti/), [Bonn Dataset](https://www.ipb.uni-bonn.de/data/rgbd-dynamic-dataset/index.html), [ScanNet Dataset](http://www.scan-net.org/).

1.2. Format the dataset as the structure below
```
kitti_depth
├── rgb
├──── 2011_09_26
├──── ...
├── depth
├──── train
├──── val

rgbd_bonn_dataset
├── rgbd_bonn_balloon
├── rgbd_bonn_balloon_tracking
├── ...

scannet
├── scene0000_00
├── scene0000_01
├── ...
```

1.3. Reconfig the bash (**$VIDEO_BASE_DATA_DIR**) and run the video inference script
```
bash infer_bash/video.sh
```

####  2. Image Inference

2.1. Download the [evaluation datasets](https://share.phys.ethz.ch/~pf/bingkedata/marigold/evaluation_dataset/) (depth) by the following commands (referred to [Marigold](https://github.com/prs-eth/Marigold)).

2.2.
Reconfig the bash (**$IMAGE_BASE_DATA_DIR**) and run the image inference script

```
bash infer_bash/image.sh
```

## 🔥Training 

#### Please refer to this [document](train_script/train.MD) for details on training.

## Reference

If you find our work useful in your research, please consider citing:
```bib
@article{zhang2026dvd,
  title={DVD: Deterministic Video Depth Estimation with Generative Priors},
  author={Zhang, Hongfei and Chen, Harold Haodong and Liao, Chenfei and He, Jing and Zhang, Zixin and Li, Haodong and Liang, Yihao and Chen, Kanghao and Ren, Bin and Zheng, Xu and Yang, Shuai and Zhou, Kun and Li, Yinchuan and Sebe, Nicu and Chen, Ying-Cong},
  journal={arXiv preprint},
  year={2026}
}
```


