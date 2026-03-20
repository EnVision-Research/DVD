# 🔥 Training Guidelines for DVD

This document provides a comprehensive guide to training **DVD (Deterministic Video Depth)**. 

## 1. 📂 Key Files Overview

Before starting, it is helpful to understand the core scripts involved in the training process:

* `train_script/train_video_new.sh`,`examples/wanvideo/model_training/train_with_accelerate_video.py`: The main entry point for the training loop.
* `examples/wanvideo/model_training/WanTrainingModule.py`: Handles training and validation logic. Please note that we only validate on a single window during training. Please consider using the inference script to perform more validation to save time if needed.
* `examples/dataset`: Handles dataset (both train and val).
* `train_config/normal_config/video_config_new.yaml`: Contains all hyperparameters, including learning rate, batch size, dataset config, and so on.
* `diffsynth/pipelines/wan_video_new_determine.py`: The core model architecture.

---

## 2. 🗄️ Dataset Preparation



As mentioned in our paper, DVD requires only **367K frames** to unlock generative priors. We mainly [Hypersim](https://github.com/apple/ml-hypersim) (image), [TartanAir](https://theairlab.org/tartanair-dataset/) (video) and [Virtual KITTI](https://europe.naverlabs.com/proxy-virtual-worlds-vkitti-2/) (video,image) for training. For detailed preprocessing instructions, please refer to [Lotus](https://github.com/EnVision-Research/Lotus).

Please download the raw datasets from their official websites and organize them as follows:

```
vkitti
├── Scene01
├── Scene02
├── ...

hypersim
├── test
├── train
└── val

ttr
├── abandonedfactory
├── abandonedfactory_night
├── amusement
├── ...
```


---

## 3. ⚙️ Configuration

All training hyperparameters are centralized in `configs/train_config.yaml`. 
Key parameters you might want to adjust based on your hardware:

* `batch_size`: Reduce this if you encounter Out-Of-Memory (OOM) errors.
* `gradient_accumulation_steps`: Increase this to maintain the effective batch size if you reduce `batch_size`.
* `use_gradient_checkpointing` : Set this to `True` if you are facing OOM errors.
* `learning_rate`: Default is set to `1e-4`.
* `{test/train}_{min/max}_num_frame`: The number of frames processed in one clip (default is e.g., 45-45).
* `denoise_step`: The $\tau$ condition in our paper.
* `grad_loss` , `grad_co`: The LMR and the $\lambda_{LMR}$ in our paper.
* `lora_rank`: Set to 512 following [Lotus-2](https://lotus-2.github.io/).
* `init_validate`: Whether to perform initial validation before training.
* `log_step`: Interval for logging training state.
* `prob`: The ratio for Hypersim(img), Virtual KITTI(img), TartanAir(vid), Virtual KITTI(vid).
* `batch_size`: The batch for **image**. The default batch_size for video is 1.
* `dataset settings`: Please refer to `examples/dataset` for more details.

---

## 4. 🚀 Launching the Training

Make sure you have downloaded the base weights (e.g., Wan2.1) before starting (You will automatically download the weight if you are using the training script we provide).

### Multi-GPU / Distributed Training (Recommended)
We use `accelerate` for multi-GPU training. To train on 4 GPUs:
```bash
bash train_script/train_video_new.sh
```
You might also alter the files under `train_config/accelerate_config` to change the GPU configuration(e.g., for single GPU or DEEPSPEED).


---
## 5. 📊  Checkpoints


### Resuming from a Checkpoint
Checkpoints are saved automatically in the `output_path` directory every `validate_step`. If your training is interrupted, you can resume it by specifying the `training_state_dir` and setting `resume` and `load_optimizer` to `True`. Please also set `global_step` if possible. Then just simply rerun the training script:
```bash
bash train_script/train_video_new.sh
```

Please refer to [this line in the training script](../examples/wanvideo/model_training/train_with_accelerate_video.py#L474) for more details.
