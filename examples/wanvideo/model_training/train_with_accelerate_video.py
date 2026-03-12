import argparse
import gc
import os
import random
from datetime import timedelta
from itertools import cycle

import torch
from accelerate import Accelerator, accelerator
from accelerate.utils import InitProcessGroupKwargs, set_seed
from omegaconf import OmegaConf
from safetensors.torch import load_file
from tqdm import tqdm

from examples.dataset import (HypersimDataset, KITTI_VID_Dataset, NYUv2Dataset,
                              Scannet_VID_Dataset, TartanAir_VID_Dataset,
                              VKITTI_VID_Dataset, VKITTIDataset)
# Import modules
from examples.wanvideo.model_training.DiffusionTrainingModule import \
    DiffusionTrainingModule
from examples.wanvideo.model_training.WanTrainingModule import (
    Validation, WanTrainingModule)

from .training_loss import GradientLoss3DSeparate

process_group_kwargs = InitProcessGroupKwargs(timeout=timedelta(minutes=30))


os.environ["TOKENIZERS_PARALLELISM"] = "false"


def custom_collate_fn(batch):
    collated = {}
    for key in batch[0].keys():
        values = [d[key] for d in batch]
        if isinstance(values[0], torch.Tensor):
            collated[key] = torch.stack(values)
        elif isinstance(values[0], str):
            collated[key] = values
        elif values[0] is None:
            collated[key] = None
        else:
            raise TypeError(
                f"Unsupported type for key '{key}': {type(values[0])}")
    return collated


def get_data(data, args):
    # print(f"data {data if isinstance(data,str) else type(data)}")
    input_data = {
        "images": data["images"],
        "disparity": data["disparity"],
        # Extra images
        "extra_images": data.get("extra_images", None),
        "extra_image_frame_index": data.get("extra_image_frame_index", None),
        # Shape
        "batch_size": data["images"].shape[0],
        "num_frames": data["images"].shape[1],
        "height": data["images"].shape[-2],
        "width": data["images"].shape[-1],
    }
    return input_data


class ModelLogger:
    def __init__(
        self, output_path, remove_prefix_in_ckpt=None, state_dict_converter=lambda x: x
    ):
        self.output_path = output_path
        import time
        os.makedirs(self.output_path, exist_ok=True)


def launch_training_task(
    accelerator,
    start_epoch,
    global_step,
    args,
    dataset_range,
    train_dataloader_list,
    test_loader_dict,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    num_epochs: int = 1,
    validate_step: int = 500,
    log_step: int = 10,
):
    validator = Validation()
    accelerator.print(
        f"Initial accelerator with gradient accumulation steps: {accelerator.gradient_accumulation_steps}"
    )
    accelerator.print(
        f"Using {accelerator.num_processes} processes for training.")

    accelerator.print(
        f"accelerator.state.deepspeed_plugin: {accelerator.state.deepspeed_plugin}")

    accelerator.print(f"Validate every {validate_step} steps.")

    if args.init_validate:
        accelerator.print(
            f"Starting validation with model at epoch {start_epoch}, global step {global_step}"
        )
        model.pipe.dit.eval()
        # if accelerator.is_main_process:
        validator.validate(
            accelerator=accelerator,
            dataset_range=dataset_range,
            pipe=model.pipe,
            global_step=global_step,
            args=args,
            test_loader_dict=test_loader_dict,
            output_path=model_logger.output_path,
        )

        model.pipe.scheduler.set_timesteps(
            training=True,
            denoise_step=args.denoise_step,
        )
        model.pipe.dit.train()
    accelerator.wait_for_everyone()

    optimizer.zero_grad()
    accumulate_depth_loss = 0.0
    accumulate_grad_loss = 0.0

    acm_cnt = 0
    rank = accelerator.process_index

    loader_iter_list = [iter(_train_dataloader)
                        for _train_dataloader in train_dataloader_list]
    prob = args.get('prob', [1 for _ in range(len(train_dataloader_list))])
    grad_loss = GradientLoss3DSeparate()

    pick_ranges = []
    start = 0
    for p in prob:
        end = start + p
        pick_ranges.append((start, end))
        start = end
    set_seed(42)

    print(f"{rank} Entering training loop...")

    for epoch_id in range(num_epochs):
        for small_batch_step in tqdm(range(100_000), desc=f"Epoch {epoch_id + 1}/{num_epochs}", disable=not accelerator.is_main_process):

            select_pos = random.choices(
                population=range(len(train_dataloader_list)),
                weights=prob,
                k=1
            )[0]

            data = None
            try:
                data = next(loader_iter_list[select_pos])
            except StopIteration:
                print(
                    f"GPU used up dataset {select_pos}, setting up new one...")
                loader_iter_list[select_pos] = iter(
                    train_dataloader_list[select_pos])
                data = next(loader_iter_list[select_pos])

            # Forward and backward pass
            with accelerator.accumulate(model):
                input_data = get_data(data, args=args)
                res_dict = model(input_data, args=args)
                depth_gt = res_dict['depth_gt']
                pred = res_dict['pred']

                # from torchvision.utils import save_image
                pred_rgb, pred_depth = None, None

                if isinstance(pred, tuple):
                    pred_depth, pred_rgb = pred
                else:
                    pred_depth = pred
                
                loss = torch.nn.functional.mse_loss(
                    depth_gt, pred_depth)
                
                
                accumulate_depth_loss += loss.item()

                if args.get('grad_loss', False):
                    _grad_loss = grad_loss(pred_depth, depth_gt)
                    _grad_t, _grad_h, _grad_w = _grad_loss
                    grad_co = args.get('grad_co', 1)
                    use_latent_flow = args.get('use_latent_flow', True)
                    if not use_latent_flow:
                        _grad_t = 0
                    loss += grad_co * (_grad_t+_grad_h+_grad_w)
                accumulate_grad_loss += loss.item()
                
                # accelerator.print(f"Microstep {small_batch_step} total loss: {loss.item()} pred_depth shape: {pred_depth.shape}, gt_depth shape: {depth_gt.shape}")
                accelerator.backward(loss)
                acm_cnt += 1

                # Update optimizer and scheduler
                if accelerator.sync_gradients:
                    
                    if args.get('clip_grad_norm', True):
                        accelerator.clip_grad_norm_(
                            model.trainable_modules(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                    global_step += 1
                    # print(f"Step at {global_step} microstep {small_batch_step}")

                    # Calculate the average loss across all processes
                    if global_step % log_step == 0:
                        accumulate_depth_loss /= acm_cnt
                        accumulate_grad_loss /= acm_cnt

                        accumulate_grad_loss = accumulate_grad_loss - accumulate_depth_loss

                        print(
                            f"GPU {rank} step {global_step}: depth loss = {accumulate_depth_loss}, grad_loss = {accumulate_grad_loss}, learning rate : {scheduler.get_last_lr()[0]:.8f}"
                        )
                        accumulate_depth_loss = 0.0
                        accumulate_grad_loss = 0.0
                        acm_cnt = 0

                    if (global_step) % validate_step == 0:
                        model.pipe.dit.eval()
                        print(f"GPU {rank} saving training state...")
                        accelerator.save_state(
                            os.path.join(model_logger.output_path,
                                         f"checkpoint-step-{global_step}")
                        )
                        if accelerator.is_main_process:

                            torch.save(
                                {"global_step": global_step},
                                os.path.join(
                                    model_logger.output_path, "trainer_state.pt")
                            )
                            accelerator.print(
                                f"Checkpoint saved at step {global_step}")
                        validator.validate(
                            accelerator=accelerator,
                            pipe=model.pipe,
                            dataset_range=dataset_range,
                            global_step=global_step,
                            args=args,
                            test_loader_dict=test_loader_dict,
                            output_path=model_logger.output_path,
                        )

                        model.pipe.scheduler.set_timesteps(
                            training=True,
                            denoise_step=args.denoise_step,
                        )
                        model.pipe.dit.train()
                    accelerator.wait_for_everyone()

        accelerator.end_training()


if __name__ == "__main__":
    def get_config():
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--config", type=str, default=None, help="Path to config yaml"
        )
        args = parser.parse_args()
        cfg = OmegaConf.load(args.config)
        return cfg

    cfg = get_config()
    args = cfg

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        kwargs_handlers=[process_group_kwargs],
    )
    accelerator.print(OmegaConf.to_yaml(cfg))

    # set_seed(42)

    # Save args
    os.makedirs(args.output_path, exist_ok=True)
    args_save_path = os.path.join(args.output_path, "args.yaml")

    if accelerator.is_main_process:
        accelerator.print(f"Saving args to {args_save_path}")
        with open(args_save_path, "w") as f:
            f.write(OmegaConf.to_yaml(args))
    accelerator.wait_for_everyone()

    # Load model
    model = WanTrainingModule(
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        trainable_models=args.trainable_models,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        lora_rank=args.lora_rank,
        lora_base_model=args.lora_base_model,
        args=args,
        accelerator=accelerator,
    )

    model.set_training_param()

    model_logger = ModelLogger(
        args.output_path,
    )

    optimizer = torch.optim.AdamW(
        model.trainable_modules(), lr=args.learning_rate)
    world_size = accelerator.num_processes

    if args.warmup_steps > 0:
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.5, total_iters=args.warmup_steps*world_size
        )
    else:
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)

    # Image training dataset
    vikitt_train_dataset = VKITTIDataset(
        args.train_data_dir_vkitti,
        norm_type=args.norm_type,
        train_ratio=args.train_ratio,
    )
    hypersim_train_dataset = HypersimDataset(
        data_dir=args.train_data_dir_hypersim,
        resolution=args.resolution_hypersim,
        random_flip=args.random_flip,
        norm_type=args.norm_type,
        truncnorm_min=args.truncnorm_min,
        align_cam_normal=args.align_cam_normal,
        split="train",
        train_ratio=args.train_ratio,

    )
    hypersim_train_dataloader = torch.utils.data.DataLoader(
        hypersim_train_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=2,
        collate_fn=custom_collate_fn,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True,
        drop_last=True,
    )
    vkitti_train_dataloader = torch.utils.data.DataLoader(
        vikitt_train_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=2,
        collate_fn=custom_collate_fn,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True,
        drop_last=True,
    )

    # Video training dataset
    ttr_vid_train_dataset = TartanAir_VID_Dataset(
        data_dir=args.train_data_dir_ttr_vid,
        random_flip=args.random_flip,
        norm_type=args.norm_type,
        max_num_frame=args.max_num_frame,
        min_num_frame=args.min_num_frame,
        max_sample_stride=args.max_sample_stride,
        min_sample_stride=args.min_sample_stride,
        train_ratio=args.train_ratio,

    )
    vikitt_vid_train_dataset = VKITTI_VID_Dataset(
        root_dir=args.train_data_dir_vkitti_vid,
        norm_type=args.norm_type,
        max_num_frame=args.max_num_frame,
        min_num_frame=args.min_num_frame,
        max_sample_stride=args.max_sample_stride,
        min_sample_stride=args.min_sample_stride,
        train_ratio=args.train_ratio,

    )
    # Enlarge the video dataset
    ttr_vid_train_dataset.data_list = ttr_vid_train_dataset.data_list * 100
    vikitt_vid_train_dataset.data_list = vikitt_vid_train_dataset.data_list * 100
    
    accelerator.print(
        f"Enlarged length of ttr and vkitti: {len(ttr_vid_train_dataset)}, {len(vikitt_vid_train_dataset)}")

    ttr_vid_train_dataloader = torch.utils.data.DataLoader(
        ttr_vid_train_dataset,
        shuffle=True,
        batch_size=1,
        num_workers=2,
        collate_fn=custom_collate_fn,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
    )

    vkitti_vid_train_dataloader = torch.utils.data.DataLoader(
        vikitt_vid_train_dataset,
        shuffle=True,
        batch_size=1,
        num_workers=2,
        collate_fn=custom_collate_fn,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
    )

    # Test set
    kitti_vid_test_dataset = KITTI_VID_Dataset(
        data_root=args.kitti_vid_test_data_root,
        max_num_frame=args.test_max_num_frame,
        min_num_frame=args.test_min_num_frame,
        max_sample_stride=args.test_max_sample_stride,
        min_sample_stride=args.test_min_sample_stride,
    )
    scannet_vid_test_dataset = Scannet_VID_Dataset(
        data_root=args.scannet_vid_test_data_root,
        split_ls=args.scannet_split_ls,
        test=False,
        max_num_frame=args.test_max_num_frame,
        min_num_frame=args.test_min_num_frame,
        max_sample_stride=args.test_max_sample_stride,
        min_sample_stride=args.test_min_sample_stride,
    )
    nyuv2_test_dataset = NYUv2Dataset(
        data_root=args.nyuv2_test_data_root,
        test=False,
    )
    kitti_vid_test_dataloader = torch.utils.data.DataLoader(
        kitti_vid_test_dataset,
        shuffle=False,
        batch_size=args.test_batch_size,
        num_workers=2,
        collate_fn=custom_collate_fn,
        pin_memory=True,
        persistent_workers=True,

    )
    scannet_vid_test_dataloader = torch.utils.data.DataLoader(
        scannet_vid_test_dataset,
        shuffle=False,
        batch_size=args.test_batch_size,
        num_workers=2,
        collate_fn=custom_collate_fn,
        pin_memory=True,
        persistent_workers=True,

    )
    nyuv2_test_dataloader = torch.utils.data.DataLoader(
        # Assuming dataset returns a dict
        nyuv2_test_dataset,
        shuffle=False,
        batch_size=args.test_batch_size,
        num_workers=2,
        collate_fn=custom_collate_fn,
        pin_memory=True,
        persistent_workers=True,

    )

    start_epoch, global_step = 0, 0
    model, optimizer, scheduler, hypersim_train_dataloader, vkitti_train_dataloader, ttr_vid_train_dataloader, vkitti_vid_train_dataloader, kitti_vid_test_dataloader, scannet_vid_test_dataloader, nyuv2_test_dataloader = (
        accelerator.prepare(
            model, optimizer, scheduler, hypersim_train_dataloader, vkitti_train_dataloader, ttr_vid_train_dataloader, vkitti_vid_train_dataloader, kitti_vid_test_dataloader, scannet_vid_test_dataloader, nyuv2_test_dataloader
        )
    )

    if args.resume and args.training_state_dir is not None:

        _training_state_dir = args.training_state_dir
        if os.path.exists(_training_state_dir) and 'checkpoint' in _training_state_dir:

            accelerator.print(
                f"Resuming training state from {args.training_state_dir}...")
            # assign training state dir must come with a given global step
            global_step = args.global_step

        else:  # use the global step to find
            g_s_path = os.path.join(
                args.training_state_dir,  'trainer_state.pt')
            if os.path.exists(g_s_path):
                g_s_pt = torch.load(g_s_path)
                global_step = g_s_pt['global_step']

            args.training_state_dir = os.path.join(
                args.training_state_dir, 'checkpoint-step-{}'.format(global_step))

        if args.load_optimizer:
            accelerator.load_state(args.training_state_dir)
            for pg in optimizer.param_groups:
                pg['lr'] = args.learning_rate

        else:
            unwrapped_model = accelerator.unwrap_model(model)
            ckpt_path = os.path.join(
                args.training_state_dir, "model.safetensors")
            state_dict = load_file(ckpt_path, device="cpu")
            missing, unexpected = unwrapped_model.load_state_dict(
                state_dict, strict=False)
            assert len(unexpected) == 0

        if global_step > 0:
            accelerator.print(f"Resuming from global step {global_step}")
        else:
            accelerator.print(f"Training from scratch...")
        accelerator.print("Training state loaded.")
        accelerator.wait_for_everyone()

    # Hard code to the order of 'hypersim', 'vikitti', ttr', 'vkitti'
    dataloader_list = [hypersim_train_dataloader, vkitti_train_dataloader,
                       ttr_vid_train_dataloader, vkitti_vid_train_dataloader]

    if hasattr(model, 'module'):
        model = model.module

    test_loader_dict = {
        'kitti': kitti_vid_test_dataloader,
        'scannet': scannet_vid_test_dataloader,
        'nyuv2': nyuv2_test_dataloader
    }

    dataset_range = {
        'kitti': [1e-5, 80],
        'scannet': [1e-3, 10],
        'nyuv2': [1e-3, 10],
    }

    launch_training_task(
        accelerator=accelerator,
        train_dataloader_list=dataloader_list,
        test_loader_dict=test_loader_dict,
        dataset_range=dataset_range,
        start_epoch=start_epoch,
        global_step=global_step,
        model=model,
        model_logger=model_logger,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=args.num_epochs,
        validate_step=args.validate_step,
        log_step=args.log_step,
        args=args,
    )
