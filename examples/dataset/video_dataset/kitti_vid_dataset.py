#!/usr/bin/env python3

"""
KITTI Dataset class for loading depth, images, and valid masks.
Based on the structure of existing Sintel dataset class.
"""
import json
import os
import random
import re

import imageio
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset


def depth_read(filename):
    # loads depth map D from png file
    # and returns it as a numpy array,
    # for details see readme.txt

    depth_png = np.array(Image.open(filename))
    # make sure we have a proper 16bit depth map here.. not 8bit!
    assert (np.max(depth_png) > 255)

    depth = depth_png.astype(np.float32) / 256.
    depth[depth_png == 0] = -1.
    return depth


def kitti_benchmark_crop(input_img):
    """
    Crop images to KITTI benchmark size
    Args:
        `input_img` (torch.Tensor): Input image to be cropped.

    Returns:
        torch.Tensor:Cropped image.
    """
    KB_CROP_HEIGHT = 352
    KB_CROP_WIDTH = 1216

    height, width = input_img.shape[-2:]
    top_margin = int(height - KB_CROP_HEIGHT)
    left_margin = int((width - KB_CROP_WIDTH) / 2)
    if 2 == len(input_img.shape):
        out = input_img[
            top_margin: top_margin + KB_CROP_HEIGHT,
            left_margin: left_margin + KB_CROP_WIDTH,
        ]
    elif 3 == len(input_img.shape):
        out = input_img[
            :,
            top_margin: top_margin + KB_CROP_HEIGHT,
            left_margin: left_margin + KB_CROP_WIDTH,
        ]
    elif 4 == len(input_img.shape):
        out = input_img[
            :,
            :,
            top_margin: top_margin + KB_CROP_HEIGHT,
            left_margin: left_margin + KB_CROP_WIDTH,
        ]

    return out


def create_valid_mask(depth):
    """
    Create valid mask from depth map.
    Valid pixels are those with depth > 0 (not -1).
    """
    return (depth > 0).astype(np.float32)


class KITTI_VID_Dataset(Dataset):
    """
    KITTI Dataset for loading images, depth maps, and valid masks.

    Returns items with:
    - img: RGB image
    - depth: Depth map
    - scene_name: Scene/drive name
    - valid_mask: Binary mask indicating valid depth pixels
    """

    def __init__(
        self,
        data_root,
        split="train",
        min_depth=0.1,
        max_depth=80.0,
        valid_mask_crop='eigen',
        transform=None,
        camera_id="image_02",  # Use left color camera by default
        test=True,
        max_num_frame=None,
        min_num_frame=None,
        max_sample_stride=None,
        min_sample_stride=None,
        **kwargs
    ):
        self.test = test
        self.data_root = data_root
        self.transform = transform
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.camera_id = camera_id
        self.valid_mask_crop = valid_mask_crop
        self.data_list = []

        self.max_num_frame = max_num_frame
        self.min_num_frame = min_num_frame
        self.max_sample_stride = max_sample_stride
        self.min_sample_stride = min_sample_stride
        self.num_frames = list(range(min_num_frame, max_num_frame + 1))
        self.strides = list(range(min_sample_stride, max_sample_stride + 1))

        # Create metadata if it doesn't exist
        self.metadata = self._create_metadata(split)

    def _create_metadata(self, split="train"):
        """
        Create metadata dictionary by scanning the data_root directory.
        Prioritize depth folders, then check corresponding RGB images.
        """

        def extract_date_from_path(path):
            """
            从 KITTI 路径中提取日期，例如：
            'kitti_depth/2011_09_26/2011_09_26_drive_0001_sync/...' -> '2011_09_26'
            """
            match = re.search(r"(\d{4}_\d{2}_\d{2})", path)
            if match:
                return match.group(1)
            return None

        metadata = {}

        split_depth_root = os.path.join(self.data_root, "depth", rf"{split}")

        if not os.path.exists(split_depth_root):
            print(f"No {split} directory found at {split_depth_root}")
            return metadata

        rgb_d_dir = []
        for scene_name in sorted(os.listdir(split_depth_root)):
            # print(f"Processing scene: {scene_name}")
            scene_depth_path = os.path.join(
                split_depth_root,
                scene_name,
                "proj_depth",
                "groundtruth",
                self.camera_id,
            )

            scene_name_date = extract_date_from_path(scene_name)
            scene_final_path = os.path.join(
                self.data_root,
                "rgb",
                scene_name_date,
                scene_name,
                self.camera_id,
                "data",
            )
            # print(
            #     f"Scene final path: {scene_final_path}, depth path: {scene_depth_path}"
            # )
            rgb_d_dir.append((scene_name, scene_final_path, scene_depth_path))

        rgb_d_dir = sorted(rgb_d_dir, key=lambda x: x[0])

        for scene_name, scene_final_path, scene_depth_path in rgb_d_dir:
            res_dict = {
                'scene': scene_name,
                'cam': self.camera_id,
                'rgb_dir': scene_final_path,
                'depth_dir': scene_depth_path
            }
            self.data_list.append(res_dict)

        self.data_list = sorted(self.data_list, key=lambda x: x['scene'])

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        """
        Return a scene sample with all frames as tensors:
        images (T, C, H, W), depths (T, 1, H, W), valid_mask (T, 1, H, W)
        """
        try:
            idx = idx % len(self.data_list)
            scene_name = self.data_list[idx]["scene"]
            cam = self.data_list[idx]["cam"]
            rgb_dir = self.data_list[idx]["rgb_dir"]
            depth_dir = self.data_list[idx]["depth_dir"]

            img_path_list = []
            depth_path_list = []
            for rgb_file in os.listdir(rgb_dir):
                if not rgb_file.endswith(".png"):
                    continue
                # _rgb_path = os.path.join(rgb_dir, rgb_file)
                img_path_list.append(rgb_file)

            for depth_file in os.listdir(depth_dir):
                if not depth_file.endswith(".png"):
                    continue
                # _depth_path = os.path.join(depth_dir, depth_file)
                depth_path_list.append(depth_file)

            common = sorted(list(set(img_path_list) & set(depth_path_list)))
            img_path_list = common
            depth_path_list = img_path_list

            # print(f"first ten element in img list {img_path_list[:10]}")
            # print(f"first ten element in depth list {depth_path_list[:10]}")
            assert len(img_path_list) == len(depth_path_list)
            img_path_list = [os.path.join(rgb_dir, _img_path)
                             for _img_path in img_path_list]
            depth_path_list = [os.path.join(depth_dir, _depth_path)
                               for _depth_path in depth_path_list]

            # Handling index
            total_frames = len(img_path_list)
            if total_frames < self.min_num_frame:
                raise ValueError(
                    f"Not enough frames for scene {scene_name}. Need at least {self.min_num_frame} frames, but got {total_frames}."
                )

            _sample_idx = None
            for _ in range(1000):
                _stride = random.choice(self.strides)
                _num_frames = random.choice(self.num_frames)
                # if _num_frames % 4 != 1:
                #     continue
                _total_frames_req = _stride * (_num_frames - 1) + 1
                if _total_frames_req > total_frames:
                    continue
                if not self.test:
                    start_idx = random.randint(
                        0, total_frames - _total_frames_req)
                else:
                    start_idx = 0

                end_idx = start_idx + _total_frames_req
                _sample_idx = list(range(start_idx, end_idx, _stride))
                break
            if _sample_idx is None:
                raise ValueError(
                    f"Fail to sample frames for scene {scene_name}.")

            _sample_img_path = [img_path_list[i] for i in _sample_idx]
            _sample_depth_path = [depth_path_list[i] for i in _sample_idx]

            img_list = []
            depth_list = []
            # 遍历 scene 内的所有帧
            for _idx, (_img_path, _depth_path) in enumerate(
                zip(_sample_img_path, _sample_depth_path)
            ):
                # Load image
                img = Image.open(_img_path).convert("RGB")
                img = np.array(img)
                depth_np = depth_read(_depth_path)
                img_list.append(img)
                depth_list.append(depth_np)

            img_np = np.array(img_list).astype(np.float32) / 255.0
            depth_np = np.array(depth_list)

            image = torch.from_numpy(img_np).permute(0, 3, 1, 2)
            depth = torch.from_numpy(depth_np).unsqueeze(
                3).permute(0, 3, 1, 2).repeat(1, 3, 1, 1)

            image = kitti_benchmark_crop(image)
            depth = kitti_benchmark_crop(depth)

            valid_mask_tensor = self._get_valid_mask(depth)

            sample = {
                'sample_idx': torch.tensor(idx),
                "images": image,
                "disparity": depth,
                "eval_mask": valid_mask_tensor,
                "scene_name": scene_name,
            }
            return sample
        except Exception as e:
            print(f"Error in __getitem__: {e}")
            return self.__getitem__(idx+1)

    def _get_valid_mask(self, depth: torch.Tensor):
        # reference: https://github.com/cleinc/bts/blob/master/pytorch/bts_eval.py
        valid_mask = torch.logical_and(
            (depth > self.min_depth), (depth < self.max_depth)
        ).bool()
        # print(f"valid_mask shape {valid_mask.shape}")
        if self.valid_mask_crop is not None:
            eval_mask = torch.zeros_like(valid_mask).bool()
            # print(f"Eval mask shape {eval_mask.shape}")
            gt_height, gt_width = eval_mask.shape[-2:]

            if "garg" == self.valid_mask_crop:
                eval_mask[
                    :,
                    :,
                    int(0.40810811 * gt_height): int(0.99189189 * gt_height),
                    int(0.03594771 * gt_width): int(0.96405229 * gt_width),
                ] = 1
            elif "eigen" == self.valid_mask_crop:
                eval_mask[
                    :,
                    :,
                    int(0.3324324 * gt_height): int(0.91351351 * gt_height),
                    int(0.0359477 * gt_width): int(0.96405229 * gt_width),
                ] = 1

            # eval_mask.reshape(valid_mask.shape)
            valid_mask = torch.logical_and(valid_mask, eval_mask)
        return valid_mask
