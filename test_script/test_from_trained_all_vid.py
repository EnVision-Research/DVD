# Core imports
import argparse
import json
import os
from datetime import datetime

import numpy as np
import torch
# Accelerate & Config imports
from accelerate import Accelerator
from omegaconf import OmegaConf
from safetensors.torch import load_file
from tabulate import tabulate
from tqdm import tqdm

# DiffSynth imports
from diffsynth import save_video
from diffsynth.util import metric
from diffsynth.util.alignment import (align_depth_least_square_video,
                                      depth2disparity, disparity2depth)
from diffsynth.util.metric import MetricTracker
# Dataset & Model imports
from examples.dataset.video_dataset import get_vid_eval_dataset
from examples.wanvideo.model_training.WanTrainingModule import \
    WanTrainingModule


# ---------------------------------------------------------------------------
# Math & Alignment Helpers
# ---------------------------------------------------------------------------
def compute_scale_and_shift_full(prediction, target, mask):
    """Computes optimal scale and shift for alignment using least squares."""
    prediction, target, mask = prediction.astype(
        np.float32), target.astype(np.float32), mask.astype(np.float32)
    a_00 = np.sum(mask * prediction * prediction)
    a_01 = np.sum(mask * prediction)
    a_11 = np.sum(mask)
    b_0 = np.sum(mask * prediction * target)
    b_1 = np.sum(mask * target)

    det = a_00 * a_11 - a_01 * a_01
    if det != 0:
        x_0 = (a_11 * b_0 - a_01 * b_1) / det
        x_1 = (-a_01 * b_0 + a_00 * b_1) / det
    else:
        x_0, x_1 = 1.0, 0.0
    return x_0, x_1


def compute_scale(prediction, target, mask):
    """Computes optimal scale (no shift) for alignment."""
    prediction, target, mask = prediction.astype(
        np.float32), target.astype(np.float32), mask.astype(np.float32)
    a_00 = np.sum(mask * prediction * prediction)
    b_0 = np.sum(mask * prediction * target)
    return b_0 / (a_00 + 1e-6)

# ---------------------------------------------------------------------------
# Data Processing Helpers
# ---------------------------------------------------------------------------


def custom_collate_fn(batch):
    if not batch or batch[0] is None:
        return None
    collated = {}
    for key in batch[0].keys():
        values = [sample[key] for sample in batch]
        if all(v is None for v in values):
            collated[key] = None
        elif isinstance(values[0], torch.Tensor):
            collated[key] = torch.stack(values, dim=0)
        elif isinstance(values[0], (int, float)):
            collated[key] = torch.tensor(values)
        else:
            collated[key] = values
    return collated


def get_data(batch, dataset_min, dataset_max, max_frame):
    """Adapts dataset format for the older version proposed by Ke et al."""
    rgb = batch['rgb_int'].to(torch.float32) / \
        255.0 if 'rgb_int' in batch else batch['images']
    depth = batch['depth_raw_linear'] if 'depth_raw_linear' in batch else batch['disparity']

    if 'valid_mask_raw' in batch:
        valid_mask = batch['valid_mask_raw']
    else:
        valid_mask = batch.get('eval_mask', torch.ones_like(rgb))
        _range_mask = torch.logical_and(
            (depth > dataset_min), (depth < dataset_max)).bool()
        valid_mask = torch.logical_and(valid_mask, _range_mask)

    sample_id = batch['index'] if 'index' in batch else batch['sample_idx']

    if rgb.ndim == 4:
        rgb, depth, valid_mask = rgb.unsqueeze(
            1), depth.unsqueeze(1), valid_mask.unsqueeze(1)

    print(f"_max_frame: {max_frame}")
    return rgb[:, :max_frame], depth[:, :max_frame], valid_mask[:, :max_frame], sample_id

# ---------------------------------------------------------------------------
# Generation Helpers
# ---------------------------------------------------------------------------


def get_window_index(T, window_size, overlap):
    if T <= window_size:
        return [(0, T)]
    res = [(0, window_size)]
    start = window_size - overlap
    while start < T:
        end = start + window_size
        if end < T:
            res.append((start, end))
            start += window_size - overlap
        else:
            res.append((start, T))
            break
    return res


def pad_time_mod4(_input_rgb):
    B, T, H, W, C = _input_rgb.shape
    remainder = T % 4
    if remainder != 1:
        pad_len = (4 - remainder + 1) % 4
        pad_frames = _input_rgb[:, -1:, :, :, :].repeat(1, pad_len, 1, 1, 1)
        _input_rgb = torch.cat([_input_rgb, pad_frames], dim=1)
    return _input_rgb, T


def generate_depth_sliced(model, args, input_rgb, window_size, overlap, return_overlap_err=True, scale_only=False):
    B, T, C, H, W = input_rgb.shape
    depth_windows = get_window_index(T, window_size, overlap)
    print(f"depth_windows {depth_windows}")

    depth_res_list = []
    for start, end in depth_windows:
        print(f"Handling window {start} - {end}", end='\t')
        _input_rgb_slice = input_rgb[:, start:end]
        _input_rgb_slice, origin_T = pad_time_mod4(_input_rgb_slice)
        _input_frame = _input_rgb_slice.shape[1]
        _input_height, _input_width = _input_rgb_slice.shape[-2:]

        videos = model.pipe(
            prompt=[""] * B, negative_prompt=[""] * B,
            mode=args.mode,
            height=_input_height,
            width=_input_width,
            num_frames=_input_frame,
            batch_size=B,
            # input_image=_input_rgb_slice[:,
            #                              0],
            # extra_images=_input_rgb_slice,
            # extra_image_frame_index=torch.ones(
            #     [B, _input_frame]).to(model.pipe.device),
            input_video=_input_rgb_slice,
            denoise_step=args.denoise_step,
            cfg_scale=1,
            seed=0,
            tiled=False,
        )
        depth_res_list.append(videos['depth'][:, :origin_T])

    depth_list_aligned = None
    overlap_mae_list = []

    for i, t in enumerate(depth_res_list):
        if i == 0:
            depth_list_aligned = t
            continue

        ref_frames = depth_list_aligned[:, -overlap:]
        curr_frames = t[:, :overlap]
        mask = (np.ones_like(ref_frames) == 1)

        if scale_only:
            scale, shift = compute_scale(curr_frames, ref_frames, mask), 0.0
        else:
            scale, shift = compute_scale_and_shift_full(
                curr_frames, ref_frames, mask)

        aligned_t = t * scale + shift
        aligned_t[aligned_t < 0] = 0
        curr_overlap_aligned = aligned_t[:, :overlap]

        # MAE computation
        diff = np.abs(curr_overlap_aligned - ref_frames)
        mae_per_b = diff.mean(axis=tuple(range(1, diff.ndim)))
        mae_scalar = float(mae_per_b.mean())
        overlap_mae_list.append(mae_scalar)

        # Smooth blending
        alpha = np.linspace(0, 1, overlap, dtype=np.float32).reshape(
            [1, overlap] + [1] * (ref_frames.ndim - 2))
        smooth_overlap = (1 - alpha) * ref_frames + \
            alpha * curr_overlap_aligned

        depth_list_aligned = np.concatenate(
            [depth_list_aligned[:, :-overlap], smooth_overlap,
                aligned_t[:, overlap:]], axis=1
        )
        print(f"Apply scale/shift {scale} {shift} | overlap MAE(after align)={mae_scalar:.6f} "
              f"| Total depth range {depth_list_aligned.min()} - {depth_list_aligned.max()}")

    overlap_mae_mean = float(np.mean(overlap_mae_list)
                             ) if overlap_mae_list else 0.0
    out = {'depth': depth_list_aligned}
    if return_overlap_err:
        out.update({'overlap_mae_list': overlap_mae_list,
                   'overlap_mae_mean': overlap_mae_mean})
    return out


def create_directories(base_dir, subdirs):
    for subdir in subdirs:
        os.makedirs(os.path.join(base_dir, subdir), exist_ok=True)


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to checkpoint directory")
    parser.add_argument('--window_size', type=int, default=45)
    parser.add_argument('--overlap', type=int, default=9)
    parser.add_argument("--frame", type=int, required=True, default=1)
    parser.add_argument("--num", type=int, default=10)
    parser.add_argument('--base_data_dir', type=str, required=True)
    parser.add_argument('--model_config', default='ckpt/model_config.yaml')

    cli_args = parser.parse_args()

    state_dir = cli_args.ckpt
    args_path = cli_args.model_config

    if not os.path.exists(state_dir) or not os.path.exists(args_path):
        raise FileNotFoundError(f"State directory or args.yaml missing.")

    try:
        args = OmegaConf.load(args_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load configuration: {e}")

    print(f"Parsed args: {args}")
    accelerator = Accelerator()

    # Init Model
    model = WanTrainingModule(
        accelerator=accelerator, model_id_with_origin_paths=args.model_id_with_origin_paths,
        trainable_models=None, use_gradient_checkpointing=False,
        lora_rank=args.lora_rank, lora_base_model=args.lora_base_model, args=args,
    )

    ckpt_path = os.path.join(state_dir, "model.safetensors")
    dit_state_dict = {k.replace("pipe.dit.", ""): v for k, v in load_file(
        ckpt_path, device="cpu").items() if "pipe.dit." in k}
    model.pipe.dit.load_state_dict(dit_state_dict, strict=True)
    model.merge_lora_layer()
    model = model.to('cuda')

    VIDEO_DATASET_CONFIGS = {
        # "vid_bonn": "configs/vid_config/vid_bonn.yaml",
        "vid_kitti": "configs/vid_config/vid_kitti.yaml",
        # "vid_sintel": "configs/vid_config/vid_sintel.yaml",
        # 'vid_scannet': 'configs/vid_config/vid_scannet.yaml'
    }

    eval_metrics = [
        "abs_relative_difference", "squared_relative_difference", "rmse_linear",
        "rmse_log", "log10", "delta1_acc", "delta2_acc", "delta3_acc",
        "i_rmse", "silog_rmse", 'relative_temporal_diff', 'boundary_metrics',
    ]
    metric_funcs = [getattr(metric, _met) for _met in eval_metrics]

    frame_setting = {
        "min_num_frame": cli_args.frame, "max_num_frame": cli_args.frame,
        "max_sample_stride": 1, "min_sample_stride": 1,
    }

    exp_name = datetime.now().strftime("%m-%d-%H_") + \
        f"frame_{cli_args.frame}_sample_{cli_args.num}_window_size_{cli_args.window_size}_overlap_{cli_args.overlap}"
    print(exp_name)

    # -----------------------------------------------------------------------
    # Dataset Loop
    # -----------------------------------------------------------------------
    for dst_name, config_path in VIDEO_DATASET_CONFIGS.items():
        cfg_data = OmegaConf.load(config_path)
        overall_overlap_mae_sum, overall_overlap_mae_cnt = 0.0, 0
        print(f"Cfg :{cfg_data}")

        dataset = get_vid_eval_dataset(
            cfg_data, cli_args.base_data_dir, **frame_setting)
        test_dataloader = torch.utils.data.DataLoader(
            dataset, shuffle=False, batch_size=1, num_workers=2,
            pin_memory=True, prefetch_factor=2, collate_fn=custom_collate_fn,
        )

        # Setup paths to avoid overwriting existing eval_metrics.txt
        output_path = os.path.join(state_dir, 'test', exp_name, dst_name)
        i = 1
        while os.path.exists(os.path.join(output_path, "eval_metrics.txt")):
            output_path = os.path.join(
                state_dir, 'test', exp_name, f"{dst_name}_{i}")
            i += 1

        save_dir = output_path
        print(f"Dataset: {dst_name}, output_path: {save_dir} ")

        create_directories(save_dir, [
                           'rgb', 'pred', 'depth', 'depth_rgb', 'gt_disp', 'valid_mask', 'pred_disp_npz', 'error'])
        metrics_filename = os.path.join(save_dir, "eval_metrics.txt")
        per_sample_filename = os.path.join(
            save_dir, "_per_sample_metrics.json")

        # Setup Metric Tracker
        _record_funcs = []
        for m in metric_funcs:
            if m.__name__ == "boundary_metrics":
                _record_funcs.extend(["f1", "precision", "recall"])
            else:
                _record_funcs.append(m.__name__)
        metric_tracker = MetricTracker(*_record_funcs)
        metric_tracker.reset()

        # Batch Inference
        for idx, batch in enumerate(tqdm(test_dataloader)):
            if idx >= cli_args.num:
                break
            with torch.no_grad():
                if batch is None:
                    continue

                input_rgb, input_depth, valid_mask, sample_idx = get_data(
                    batch, dataset.min_depth, dataset.max_depth, frame_setting['max_num_frame'])
                print(f"input_rgb shape: {input_rgb.shape}")
                import time 
                torch.cuda.synchronize()
                start_time = time.time()
                
                res_dict = generate_depth_sliced(
                    model, args, input_rgb, cli_args.window_size, cli_args.overlap)
                torch.cuda.synchronize()
                end_time = time.time()
                print(f"Inference time (s): {(end_time - start_time):.6f}")
                
                overlap_mae_mean = res_dict.get("overlap_mae_mean", None)
                if overlap_mae_mean is not None:
                    overall_overlap_mae_sum += float(overlap_mae_mean)
                    overall_overlap_mae_cnt += 1
                print(f"[overlap] mean MAE(after align) = {overlap_mae_mean}")

                pred_depth_batch = res_dict.get("depth")
                pred_rgb_batch = res_dict.get('rgb')
                print(f"pred depth shape: {pred_depth_batch.shape} ")

                # Format conversions
                _input_depth_batch = input_depth.cpu().numpy().transpose(0, 1, 3, 4, 2)
                _input_rgb_batch = input_rgb.cpu().numpy().transpose(0, 1, 3, 4, 2)
                valid_mask_batch = valid_mask.cpu().numpy().transpose(0, 1, 3, 4, 2)

                for b in range(pred_depth_batch.shape[0]):
                    _sample_idx = sample_idx[b].item()
                    _pred_depth = pred_depth_batch[b]
                    _pred_rgb = pred_rgb_batch[b] if pred_rgb_batch is not None else None
                    _gt_depth, _gt_rgb, _valid_mask = _input_depth_batch[
                        b], _input_rgb_batch[b], valid_mask_batch[b]

                    _rgb_depth = _pred_depth
                    _pred_depth = np.mean(_pred_depth, axis=-1, keepdims=True)
                    _gt_depth = np.mean(_gt_depth, axis=-1, keepdims=True)
                    _valid_mask = np.mean(
                        _valid_mask, axis=-1, keepdims=True).astype(bool)

                    # Outlier rejection
                    _temporal_mask = _valid_mask.mean(axis=(-1, -2, -3))
                    if np.any(_temporal_mask == 0):
                        print(f"Outlier found in {_sample_idx}")
                        continue

                    save_idx = _sample_idx
                    print(f"Saving npz shape {_pred_depth.shape}")

                    # Disparity and masks
                    _gt_disparity, _gt_non_neg_mask = depth2disparity(
                        depth=_gt_depth, return_mask=True)
                    pred_non_neg_mask = _pred_depth > 0
                    valid_nonnegative_mask = _valid_mask & _gt_non_neg_mask & pred_non_neg_mask

                    # Scaling for visualization
                    _vis_disp = (_gt_disparity - _gt_disparity[valid_nonnegative_mask].min()) / \
                                (_gt_disparity[valid_nonnegative_mask].max(
                                ) - _gt_disparity[valid_nonnegative_mask].min() + 1e-8)
                    _vis_depth = (_gt_depth - _gt_depth[valid_nonnegative_mask].min()) / \
                                 (_gt_depth[valid_nonnegative_mask].max(
                                 ) - _gt_depth[valid_nonnegative_mask].min() + 1e-8)

                    # Alignment
                    disparity_pred, scale, shift = align_depth_least_square_video(
                        gt_arr=_gt_disparity, pred_arr=_pred_depth,
                        valid_mask_arr=valid_nonnegative_mask, return_scale_shift=True
                    )
                    print(f"Apply scale and shift {scale} {shift}")

                    disparity_pred = np.clip(
                        disparity_pred, a_min=1e-3, a_max=None)
                    depth_pred = np.clip(disparity2depth(
                        disparity_pred), a_min=dataset.min_depth, a_max=dataset.max_depth)
                    depth_pred = np.clip(depth_pred, a_min=1e-6, a_max=None)

                    # Relative error map
                    abs_rel_map = np.zeros_like(_gt_depth, dtype=np.float32)
                    abs_rel_map[valid_nonnegative_mask] = (np.abs(
                        depth_pred - _gt_depth)[valid_nonnegative_mask] / _gt_depth[valid_nonnegative_mask])

                    # ----------------- Save Videos -----------------
                    videos_to_save = [
                        (valid_nonnegative_mask.copy().astype(
                            np.float32), 'valid_mask'),
                        (_vis_disp, 'gt_disp'),
                        (_rgb_depth, 'depth_rgb'),
                        (_gt_rgb, 'rgb'),
                        (_pred_depth, 'pred'),
                        (_vis_depth, 'depth'),
                        (abs_rel_map, 'error')
                    ]
                    for v_data, folder_name in videos_to_save:
                        save_video(v_data, os.path.join(
                            save_dir, folder_name, f"{save_idx:05d}.mp4"), fps=15, quality=6)

                    # ----------------- Prints -----------------
                    print(f"Sample {save_idx}")
                    disp_pred_masked = disparity_pred[valid_nonnegative_mask]
                    gt_disp_masked = _gt_disparity[valid_nonnegative_mask]
                    depth_pred_masked = depth_pred[valid_nonnegative_mask]
                    gt_depth_masked = _gt_depth[valid_nonnegative_mask]
                    vis_disp_masked = _vis_disp[valid_nonnegative_mask]

                    print(
                        f"Predicted disparity: {disp_pred_masked.min()} - {disp_pred_masked.max()}")
                    print(
                        f"Ground truth disparity: {gt_disp_masked.min()} - {gt_disp_masked.max()}")
                    print(
                        f"Predicted depth: {depth_pred_masked.min()} - {depth_pred_masked.max()}")
                    print(
                        f"Ground truth depth: {gt_depth_masked.min()} - {gt_depth_masked.max()}")
                    print(
                        f"Visible disparity range: {vis_disp_masked.min()} - {vis_disp_masked.max()}")
                    print(
                        f"Visible depth range: {_vis_depth.min()} - {_vis_depth.max()}")

                    # ----------------- Metrics -----------------
                    sample_metric = {'idx': f"{save_idx:05d}"}

                    _valid_mask_ts = torch.from_numpy(
                        _valid_mask).to(accelerator.device).squeeze()
                    depth_pred_ts = torch.from_numpy(
                        depth_pred).to(accelerator.device).squeeze()
                    gt_depth_ts = torch.from_numpy(_gt_depth).to(
                        accelerator.device).squeeze()
                    gt_rgb_ts = torch.from_numpy(_gt_rgb).to(
                        accelerator.device).squeeze().permute(0, 3, 1, 2)

                    print(
                        f"Shape of mask,pred,gt,gt_rgb {_valid_mask_ts.shape}, {depth_pred_ts.shape}, {gt_depth_ts.shape}, {gt_rgb_ts.shape}")

                    for met_func in metric_funcs:
                        _metric_name = met_func.__name__
                        if _metric_name != 'boundary_metrics':
                            _metric = met_func(
                                depth_pred_ts, gt_depth_ts, _valid_mask_ts).item()
                            sample_metric[_metric_name] = f"{_metric:.6f}"
                            metric_tracker.update(_metric_name, _metric)
                        else:
                            _metric = met_func(
                                depth_pred_ts, gt_rgb_ts, _valid_mask_ts)
                            for k, v in _metric.items():
                                sample_metric[k] = v
                                metric_tracker.update(k, v)

                    try:
                        with open(per_sample_filename, "a+") as f:
                            f.write(json.dumps(sample_metric) + "\n")
                        print(f"Metrics this sample {sample_metric}")
                    except IOError as e:
                        print(
                            f"Warning: Failed to write per-sample metrics: {e}")
                    print(
                        f"Metrics per sample saving to {per_sample_filename}")

        # -------------------- Final Metric Saving --------------------
        eval_text = "Evaluation metrics:\n" + \
            tabulate([metric_tracker.result().keys(),
                     metric_tracker.result().values()]) + "\n"
        if overall_overlap_mae_cnt > 0:
            eval_text += f"\nOverlap MAE(after align) mean over samples: {overall_overlap_mae_sum/overall_overlap_mae_cnt:.6f}\n"
        else:
            eval_text += "\nOverlap MAE(after align) mean over samples: N/A\n"

        try:
            with open(metrics_filename, "a+") as f:
                f.write(eval_text)
        except IOError as e:
            print(f"Warning: Failed to write evaluation metrics: {e}")

        print(f"Evaluation metrics saved to {metrics_filename}")
        print(f"Validation videos saved to {save_dir}\n" + "="*50)
