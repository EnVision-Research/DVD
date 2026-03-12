import json
import logging
import math
import os

import numpy as np
import torch
from tabulate import tabulate
from tqdm import tqdm

from diffsynth import save_video
from diffsynth.data.video import save_video_ffmpeg
from diffsynth.pipelines.wan_video_new_determine import (ModelConfig,
                                                         WanVideoPipeline)
from diffsynth.util import metric
from diffsynth.util.alignment import (align_depth_least_square_video,
                                      depth2disparity, disparity2depth)
from diffsynth.util.metric import MetricTracker
from examples.wanvideo.model_training.DiffusionTrainingModule import \
    DiffusionTrainingModule


class WanTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        args,
        accelerator,
        model_paths=None,
        model_id_with_origin_paths=None,
        trainable_models=None,
        lora_rank=32,
        lora_base_model='dit',
        lora_target_modules='q,k,v,o,ffn.0,ffn.2',
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,

    ):
        super().__init__()

        # Load models
        model_configs = []
        if model_paths is not None:
            model_paths = json.loads(model_paths)
            model_configs += [ModelConfig(path=path) for path in model_paths]
        if model_id_with_origin_paths is not None:
            model_id_with_origin_paths = model_id_with_origin_paths.split(",")
            model_configs += [
                ModelConfig(
                    model_id=i.split(
                        ":")[0], origin_file_pattern=i.split(":")[1]
                )
                for i in model_id_with_origin_paths
            ]
            
        self.pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cpu",
            model_configs=model_configs,
        )

        # DiT related
        self.pipe.scheduler.set_timesteps(
            training=True,
            denoise_step=args.denoise_step,
        )
        accelerator.print(f"Denoise step: {args.denoise_step}")
        accelerator.print("Timesteps:", self.pipe.scheduler.timesteps)
        accelerator.print(
            f"Training weights: {self.pipe.scheduler.linear_timesteps_weights}"
        )

        # LoRA related
        if lora_base_model is not None:
            model = self.add_lora_to_model(
                getattr(self.pipe, lora_base_model),
                target_modules=lora_target_modules.split(","),
                lora_rank=lora_rank,
                adapter_name="default",
            )
            setattr(self.pipe, lora_base_model, model)

        self.validate_time = 0

        # Freeze untrainable models
        self.pipe.freeze_except(
            [] if trainable_models is None else trainable_models.split(",")
        )

        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.accelerator = accelerator
        self.args = args

    def merge_lora_layer(self):
        from peft.tuners.lora.layer import Linear as LoraLinear
        merged_count = 0
        for name, module in self.pipe.dit.named_modules():
            if isinstance(module, LoraLinear):
                module.merge()
                merged_count += 1
        print(f"Merged {merged_count} LoRA layers into base weights.")

    def set_training_param(self):
        args = self.args
        accelerator = self.accelerator
        for _, param in self.named_parameters():
            param.requires_grad = False
        if args.lora_base_model is not None:  # train only lora
            for k, v in self.named_parameters():
                # if "dit" in k and 'lora' in k and 'video' in k:
                if "dit" in k and 'lora' in k:
                    v.requires_grad = True
                else:
                    v.requires_grad = False
        if args.train_all:
            for k, v in self.named_parameters():
                if 'dit' in k:
                    v.requires_grad = True

        accelerator.print(
            f"Trainable model parameters: {sorted(list(self.trainable_param_names()))}")
        accelerator.print(
            f"Total trainable parameters: {sum(p.numel() for p in self.trainable_modules())}")

    def forward_preprocess(self, data):
        inputs_shared = {
            # Assume you are using this pipeline for inference,
            # please fill in the input parameters.
            # Shape
            "height": data["height"],
            "width": data["width"],
            "num_frames": data["num_frames"],
            "batch_size": data["batch_size"],
            # Image and disp
            "input_video": data["images"],
            "input_disp": data["disparity"],
            # Extra param
            'mode': 'regression',
            # "extra_images": data.get("extra_images", None),
            # "extra_image_frame_index": data.get("extra_image_frame_index", None),
            "input_image": data["images"][:, 0],
            "extra_images": data["images"][:, 1:],
            "extra_image_frame_index": torch.ones([data["batch_size"], data["num_frames"]]),
            # Please do not modify the following parameters
            # unless you clearly know what this will cause.
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
        }
        batch_size = inputs_shared["batch_size"]
        inputs_posi = {
            "prompt": data.get("prompt", [""] * data.get("batch_size")),
            "prompt_num": batch_size,
        }
        inputs_nega = {
            "negative_prompt": data.get(
                "negative_prompt", [""] * data.get("batch_size")
            ),
            "prompt_num": batch_size,
        }
        for unit in self.pipe.units:
            # print(f"Processing unit: {unit.__class__.__name__}")
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(
                unit, self.pipe, inputs_shared, inputs_posi, inputs_nega
            )

        return {**inputs_shared, **inputs_posi}

    def forward(self, data, inputs=None, args=None):
        # if inputs is None:
        inputs = self.forward_preprocess(data)
        models = {
            name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models
        }
        res_dict = self.pipe.training_predict( **models, **inputs )
        return res_dict


class Validation():
    def __init__(self):
        self.validate_time = 0

    def validate(
        self,
        accelerator,
        global_step,
        dataset_range,
        args,
        pipe,
        test_loader_dict=None,
        output_path=None,
    ):
        self.validate_time += 1
        rank = accelerator.process_index

        print(
            f"GPU {rank} starting validation # {self.validate_time} at step {global_step}...")

        eval_metrics = [
            "abs_relative_difference",
            "squared_relative_difference",
            "rmse_linear",
            "rmse_log",
            "log10",
            "delta1_acc",
            "delta2_acc",
            "delta3_acc",
            "i_rmse",
            "silog_rmse",
            # "pixel_mean",
            # "pixel_var"
        ]

        def generate_depth(input_rgb):
            # Prompt
            input_data = {}
            input_data["video"] = input_rgb
            input_data["frame"] = input_rgb.shape[1]
            input_data["batch_size"] = input_rgb.shape[0]
            input_data["height"], input_data["width"] = input_rgb.shape[-2:]
            # print(f"input rgb shape {input_rgb.shape}")
            videos = pipe(
                prompt=[""] * input_data["batch_size"],
                negative_prompt=[""] * input_data["batch_size"],
                mode=args.mode,
                height=input_data["height"],
                width=input_data["width"],
                num_frames=input_data["frame"],
                batch_size=input_data["batch_size"],
                input_image=input_rgb[:, 0],
                extra_images=input_rgb,
                extra_image_frame_index=torch.ones(
                    [input_data["batch_size"], input_data["frame"]]).to(pipe.device),
                input_video=input_rgb,
                denoise_step=args.denoise_step,
                cfg_scale=1,
                seed=0,
                tiled=False,
                num_inference_steps=args.validation_scheduler_timesteps,
            )

            return {k: np.array(v) for k, v in videos.items() if v is not None}

        def get_data(batch, dataset_min, dataset_max):
            '''
            This function is to adapt for the older version of dataset format proposed by Marigold (Ke et al).
            '''
            rgb, depth, valid_mask, sample_id = None, None, None, None
            if 'rgb_int' in batch.keys():
                rgb = batch['rgb_int'].to(torch.float32)/255.0
            else:
                rgb = batch['images']

            if 'depth_raw_linear' in batch.keys():
                depth = batch['depth_raw_linear']
            else:
                depth = batch['disparity']

            if 'valid_mask_raw' in batch.keys():
                valid_mask = batch['valid_mask_raw']
            else:
                valid_mask = batch.get('eval_mask', torch.ones(rgb.shape))
                _range_mask = torch.logical_and((depth > dataset_min), (
                    depth < dataset_max)).bool()
                valid_mask = torch.logical_and(valid_mask, _range_mask)

            if 'index' in batch.keys():
                sample_id = batch['index']
            else:
                sample_id = batch['sample_idx']

            if rgb.ndim == 4:
                rgb = rgb.unsqueeze(1)
                depth = depth.unsqueeze(1)
                valid_mask = valid_mask.unsqueeze(1)

            return rgb, depth, valid_mask, sample_id

        rank = accelerator.process_index
        world_size = accelerator.num_processes
        save_root = os.path.join(output_path, f"val_step_{global_step}")

        for test_set_name, test_dataloader in test_loader_dict.items():
            _dataset_min = dataset_range[test_set_name][0]
            _dataset_max = dataset_range[test_set_name][1]
            metric_funcs = [getattr(metric, _met) for _met in eval_metrics]
            metric_tracker = MetricTracker(*[m.__name__ for m in metric_funcs])
            metric_tracker.reset()
            save_dir = os.path.join(save_root, test_set_name)
            os.makedirs(save_dir, exist_ok=True)
            per_sample_filename = os.path.join(
                save_dir, f"per_sample_metrics_{rank}.json"
            )

            metrics_filename = f"eval_metrics_{rank}.txt"
            _save_to = os.path.join(save_dir, metrics_filename)

            dirs_to_make = ['rgb', 'pred_depth', 'gt_depth', 'pred_depth_rgb',
                            'gt_disp',  'valid_mask', 'pred_rgb']
            for _dir in dirs_to_make:
                os.makedirs(os.path.join(save_dir, _dir), exist_ok=True)

            for idx, batch in enumerate(tqdm(test_dataloader)):
                if idx > 1:
                    break
                with torch.no_grad():
                    input_rgb, input_depth, valid_mask, sample_idx = get_data(
                        batch, _dataset_min, _dataset_max)

                    # valid_mask = torch.logical_and(valid_mask, _range_mask)
                    res_dict = generate_depth(input_rgb)
                    pred_depth = res_dict.get("depth")
                    # print(
                    #     f"Pred_depth shape {pred_depth.shape}, range{pred_depth.min()}, {pred_depth.max()}")
                    pred_rgb = res_dict.get('rgb', None)

                    _input_depth = (
                        input_depth.cpu().numpy().transpose(0, 1, 3, 4, 2)
                    )  # B T H W 3
                    _input_rgb = (
                        input_rgb.cpu().numpy().transpose(0, 1, 3, 4, 2)
                    )
                    valid_mask = (
                        valid_mask.cpu().numpy().transpose(0, 1, 3, 4, 2)
                    )

                    bs = pred_depth.shape[0]
                    for b in range(bs):
                        _sample_idx = sample_idx[b].item()
                        _pred_depth = pred_depth[b]  # T H W C
                        _valid_mask = valid_mask[b]  # T H W C

                        _pred_rgb = pred_rgb[b] if pred_rgb is not None else None
                        _gt_depth = _input_depth[b]  # T H W C

                        _rgb_depth = _pred_depth
                        _pred_depth = np.mean(
                            _pred_depth, axis=-1, keepdims=True)  # T H W 1
                        _gt_depth = np.mean(_gt_depth, axis=-1,
                                            keepdims=True)  # T H W 1
                        _valid_mask = np.mean(_valid_mask, axis=-1,
                                              keepdims=True).astype(bool)
                        save_idx = _sample_idx
                        # print(f"{save_idx} mask mean: {_valid_mask.mean()}")

                        _gt_disparity, _gt_non_neg_mask = depth2disparity(
                            depth=_gt_depth, return_mask=True
                        )

                        pred_non_neg_mask = _pred_depth > 0
                        valid_nonnegative_mask = _gt_non_neg_mask & pred_non_neg_mask & _valid_mask

                        _vis_disp = (_gt_disparity - _gt_disparity[valid_nonnegative_mask].min()) / (
                            _gt_disparity[valid_nonnegative_mask].max(
                            ) - _gt_disparity[valid_nonnegative_mask].min() + 1e-8
                        )

                        _vis_depth = (_gt_depth - _gt_depth[valid_nonnegative_mask].min()) / (
                            _gt_depth[valid_nonnegative_mask].max(
                            ) - _gt_depth[valid_nonnegative_mask].min() + 1e-8
                        )
                        SAVE_DICT = {
                            'rgb': _input_rgb[b],
                            'pred_depth': _pred_depth,
                            'gt_depth': _vis_depth,
                            'pred_depth_rgb': _rgb_depth,
                            'gt_disp': _vis_disp,
                            'valid_mask': valid_nonnegative_mask.copy().astype(np.float32),
                            'pred_rgb': _pred_rgb
                        }

                        for k, v in SAVE_DICT.items():
                            if v is not None:
                                save_path = os.path.join(
                                    save_dir, k, f"{save_idx:05d}.mp4")
                                save_video(
                                    v,
                                    save_path,
                                    fps=5,
                                    quality=6,
                                )

                        disparity_pred, scale, shift = align_depth_least_square_video(
                            gt_arr=_gt_disparity,
                            pred_arr=_pred_depth,
                            valid_mask_arr=valid_nonnegative_mask,
                            return_scale_shift=True,
                            max_resolution=None,
                        )
                        # convert to depth
                        disparity_pred = np.clip(
                            disparity_pred, a_min=1e-3, a_max=None
                        )  # avoid 0 disparity

                        depth_pred = disparity2depth(disparity_pred)
                        # depth_pred = np.clip(depth_pred,a_min=1e-6,a_max=None)
                        # TODO
                        depth_pred = np.clip(
                            depth_pred, _dataset_min, _dataset_max)
                        depth_pred = np.clip(
                            depth_pred, a_min=1e-6, a_max=None)

                        sample_metric = {
                            'idx': f"{save_idx:05d}",
                        }

                        _valid_mask_ts = torch.from_numpy(
                            _valid_mask
                        ).to(accelerator.device).squeeze()
                        depth_pred_ts = torch.from_numpy(
                            depth_pred).to(accelerator.device).squeeze()
                        gt_depth_ts = torch.from_numpy(
                            _gt_depth).to(accelerator.device).squeeze()

                        for met_func in metric_funcs:
                            _metric_name = met_func.__name__
                            _metric = met_func(
                                depth_pred_ts, gt_depth_ts, _valid_mask_ts
                            ).item()
                            sample_metric[_metric_name] = f"{_metric:.6f}"
                            metric_tracker.update(_metric_name, _metric)

                        # Save per-sample metric
                        with open(per_sample_filename, "a+") as f:
                            f.write(json.dumps(sample_metric) + "\n")

            eval_text = f"Evaluation metrics:\n"
            eval_text += tabulate(
                [
                    metric_tracker.result().keys(),
                    metric_tracker.result().values(),
                ]
            )
            eval_text += "\n"
            with open(_save_to, "a+") as f:
                f.write(eval_text)

            local_metrics = metric_tracker.result()  # dict
            metric_names = list(local_metrics.keys())
            metric_values = torch.tensor(
                list(local_metrics.values()), device=accelerator.device)
            # print(f"GPU get values {metric_values}")
            # Gather metrics from all GPUs
            num_gpus = accelerator.num_processes
            all_metrics = accelerator.gather(metric_values).view(num_gpus, -1)
            # print(f"After reduce {all_metrics}")

            # Only main process saves aggregated metrics
            if accelerator.is_main_process:
                avg_metrics = all_metrics.float().mean(dim=0).tolist()
                summary = {name: value for name,
                           value in zip(metric_names, avg_metrics)}
                summary_path = os.path.join(
                    save_root, f"{test_set_name}_summary_metrics_step_{global_step}.json")
                with open(summary_path, "w") as f:
                    json.dump(summary, f, indent=4)

            accelerator.print(
                f"Metrics per sample saving to {per_sample_filename}")
            accelerator.print(f"Validation videos saved to {save_dir}")
            # save_executor.shutdown(wait=True)
            # accelerator.wait_for_everyone()
