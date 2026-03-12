import argparse
import os
from datetime import datetime

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from omegaconf import OmegaConf
from safetensors.torch import load_file
from tqdm import tqdm

from diffsynth import save_video
from examples.wanvideo.model_training.WanTrainingModule import \
    WanTrainingModule


# =============================
# Helper: Math & Alignment
# =============================
def compute_scale_and_shift(curr_frames, ref_frames, mask=None):
    """Computes scale and shift for overlap alignment."""
    if mask is None:
        mask = np.ones_like(ref_frames)

    a_00 = np.sum(mask * curr_frames * curr_frames)
    a_01 = np.sum(mask * curr_frames)
    a_11 = np.sum(mask)
    b_0 = np.sum(mask * curr_frames * ref_frames)
    b_1 = np.sum(mask * ref_frames)

    det = a_00 * a_11 - a_01 * a_01
    if det != 0:
        scale = (a_11 * b_0 - a_01 * b_1) / det
        shift = (-a_01 * b_0 + a_00 * b_1) / det
    else:
        scale, shift = 1.0, 0.0

    return scale, shift


# =============================
# Helper: Video Processing
# =============================
def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()

    video_np = np.stack(frames)
    video_tensor = torch.from_numpy(
        video_np).permute(0, 3, 1, 2).float() / 255.0

    return video_tensor.unsqueeze(0), fps   # [1, T, C, H, W], fps


def resize_for_training_scale(video_tensor, target_h=480, target_w=640):
    B, T, C, H, W = video_tensor.shape
    ratio = max(target_h / H, target_w / W)
    new_H = int(np.ceil(H * ratio))
    new_W = int(np.ceil(W * ratio))

    # Align to 16
    new_H = (new_H + 15) // 16 * 16
    new_W = (new_W + 15) // 16 * 16

    if new_H == H and new_W == W:
        return video_tensor, (H, W)

    video_reshape = video_tensor.view(B * T, C, H, W)
    resized = F.interpolate(video_reshape, size=(
        new_H, new_W), mode="bilinear", align_corners=False)
    resized = resized.view(B, T, C, new_H, new_W)
    return resized, (H, W)


def resize_depth_back(depth_np, orig_size):
    orig_H, orig_W = orig_size
    depth_tensor = torch.from_numpy(depth_np).permute(0, 3, 1, 2).float()
    depth_tensor = F.interpolate(depth_tensor, size=(
        orig_H, orig_W), mode='bilinear', align_corners=False)
    return depth_tensor.permute(0, 2, 3, 1).cpu().numpy()


def pad_time_mod4(video_tensor):
    """Pads the temporal dimension to satisfy 4n+1 requirement."""
    B, T, C, H, W = video_tensor.shape
    remainder = T % 4
    if remainder != 1:
        pad_len = (4 - remainder + 1) % 4
        pad_frames = video_tensor[:, -1:, :, :, :].repeat(1, pad_len, 1, 1, 1)
        video_tensor = torch.cat([video_tensor, pad_frames], dim=1)
    return video_tensor, T


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
            # Last window ensures full window_size length if possible
            start = max(0, T - window_size)
            res.append((start, T))
            break
    return res


# =============================
# Core Inference
# =============================
def generate_depth_sliced(model, input_rgb, window_size=45, overlap=9, scale_only=False):
    B, T, C, H, W = input_rgb.shape
    depth_windows = get_window_index(T, window_size, overlap)
    print(f"depth_windows {depth_windows}")

    depth_res_list = []

    # 1. Inference per window
    for start, end in tqdm(depth_windows, desc="Inferencing Slices"):
        _input_rgb_slice = input_rgb[:, start:end]

        # Ensure 4n+1 padding
        _input_rgb_slice, origin_T = pad_time_mod4(_input_rgb_slice)
        _input_frame = _input_rgb_slice.shape[1]
        _input_height, _input_width = _input_rgb_slice.shape[-2:]

        outputs = model.pipe(
            prompt=[""] * B,
            negative_prompt=[""] * B,
            mode=model.args.mode,
            height=_input_height,
            width=_input_width,
            num_frames=_input_frame,
            batch_size=B,
            input_image=_input_rgb_slice[:, 0],
            extra_images=_input_rgb_slice,
            extra_image_frame_index=torch.ones(
                [B, _input_frame]).to(model.pipe.device),
            input_video=_input_rgb_slice,
            cfg_scale=1,
            seed=0,
            tiled=False,
            denoise_step=model.args.denoise_step,
        )
        # Drop the padded frames
        depth_res_list.append(outputs['depth'][:, :origin_T])

    # 2. Overlap Alignment
    depth_list_aligned = None
    prev_end = None

    for i, (t, (start, end)) in enumerate(zip(depth_res_list, depth_windows)):
        print(f"Handling window {i} start: {start}, end: {end}")

        if i == 0:
            depth_list_aligned = t
            prev_end = end
            continue

        curr_start = start
        real_overlap = prev_end - curr_start

        if real_overlap > 0:
            ref_frames = depth_list_aligned[:, -real_overlap:]
            curr_frames = t[:, :real_overlap]

            if scale_only:
                scale = np.sum(curr_frames * ref_frames) / \
                    (np.sum(curr_frames * curr_frames) + 1e-6)
                shift = 0.0
            else:
                scale, shift = compute_scale_and_shift(curr_frames, ref_frames)

            scale = np.clip(scale, 0.7, 1.5)

            aligned_t = t * scale + shift
            aligned_t[aligned_t < 0] = 0

            # Debugging Output
            curr_overlap_aligned = aligned_t[:, :real_overlap]
            diff = np.abs(curr_overlap_aligned - ref_frames)
            mae_scalar = float(
                diff.mean(axis=tuple(range(1, diff.ndim))).mean())

            print(f"\n[Overlap {i}]")
            print(f"real_overlap = {real_overlap}")
            print(f"scale = {scale:.8f}, shift = {shift:.8f}")
            print(
                f"aligned curr range = {aligned_t.min():.6f} ~ {aligned_t.max():.6f}")
            print(f"overlap MAE(after align) = {mae_scalar:.6f}")

            # Smooth blending
            alpha = np.linspace(0, 1, real_overlap, dtype=np.float32).reshape(
                1, real_overlap, 1, 1, 1)
            smooth_overlap = (1 - alpha) * ref_frames + \
                alpha * aligned_t[:, :real_overlap]

            depth_list_aligned = np.concatenate(
                [depth_list_aligned[:, :-real_overlap], smooth_overlap,
                    aligned_t[:, real_overlap:]], axis=1
            )
        else:
            # Fallback if no overlap exists
            depth_list_aligned = np.concatenate(
                [depth_list_aligned, t], axis=1)

        print(
            f"Total depth range after concat = {depth_list_aligned.min():.6f} ~ {depth_list_aligned.max():.6f}")
        prev_end = end

    # Crop to original length
    return depth_list_aligned[:, :T]


# =============================
# Main Script
# =============================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--input_video", type=str, required=True)
    parser.add_argument("--output_dir", type=str,
                        default="./inference_results")
    parser.add_argument('--model_config', default='ckpt/model_config.yaml')
    parser.add_argument("--window_size", type=int, default=81)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument("--overlap", type=int, default=9)
    parser.add_argument('--grayscale', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    yaml_args = OmegaConf.load(args.model_config)

    # Prepare Data
    input_tensor, origin_fps = read_video(args.input_video)
    print("Original shape:", input_tensor.shape)

    input_tensor, orig_size = resize_for_training_scale(
        input_tensor, args.height, args.width)
    print("Resized shape:", input_tensor.shape)
    print(f"input range {input_tensor.min()} - {input_tensor.max()}")

    # Model Init
    accelerator = Accelerator()
    model = WanTrainingModule(
        accelerator=accelerator,
        model_id_with_origin_paths=yaml_args.model_id_with_origin_paths,
        trainable_models=None,
        use_gradient_checkpointing=False,
        lora_rank=yaml_args.lora_rank,
        lora_base_model=yaml_args.lora_base_model,
        args=yaml_args,
    )

    ckpt_path = os.path.join(args.ckpt, "model.safetensors")
    state_dict = load_file(ckpt_path, device="cpu")
    dit_state_dict = {k.replace("pipe.dit.", ""): v for k,
                      v in state_dict.items() if "pipe.dit." in k}
    model.pipe.dit.load_state_dict(dit_state_dict, strict=True)
    model.merge_lora_layer()
    model = model.to("cuda")

    # Inference
    depth = generate_depth_sliced(
        model, input_tensor, args.window_size, args.overlap)[0]
    print(
        f"depth range shape {depth.min()} - {depth.max()}, shape {depth.shape}")

    # Post Process
    depth = resize_depth_back(depth, orig_size)
    print(f"after resizing {depth.min()} - {depth.max()}, {depth.shape}")

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    exp_name = datetime.now().strftime("%m-%d-%H%M")
    base_name = os.path.basename(args.input_video).split('.')[0]
    gray_scale = 'gray' if args.grayscale else 'color'
    out_prefix = os.path.join(
        args.output_dir, f"{base_name}_{exp_name}_{gray_scale}")

    print(f"Saving to {out_prefix}_depth_vis.mp4")
    d_min, d_max = depth.min(), depth.max()
    vis_depth = (depth - d_min) / (d_max - d_min + 1e-8)
    save_video(vis_depth, f"{out_prefix}_depth_vis.mp4",
               fps=origin_fps, quality=6, grayscale=args.grayscale)

    print("Inference completed successfully!")


if __name__ == "__main__":
    main()
