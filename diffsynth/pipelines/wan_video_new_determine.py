import glob
import os
import time
import types
import warnings
from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import torch
from einops import rearrange, reduce, repeat
# from modelscope import snapshot_download
from huggingface_hub import snapshot_download
from PIL import Image
from tqdm import tqdm
from typing_extensions import Literal

from ..models import ModelManager, load_state_dict
from ..models.wan_video_dit import RMSNorm, WanModel, sinusoidal_embedding_1d
from ..models.wan_video_image_encoder import WanImageEncoder
from ..models.wan_video_motion_controller import WanMotionControllerModel
# from ..model.
from ..models.wan_video_text_encoder import (T5LayerNorm, T5RelativeEmbedding,
                                             WanTextEncoder)
from ..models.wan_video_vace import VaceWanModel
from ..models.wan_video_vae import (CausalConv3d, RMS_norm, Upsample,
                                    WanVideoVAE)
from ..schedulers.flow_match import FlowMatchScheduler
# from ..prompters import WanPrompter
from ..vram_management import (AutoWrappedLinear, AutoWrappedModule,
                               WanAutoCastLayerNorm, enable_vram_management)


class BasePipeline(torch.nn.Module):

    def __init__(
        self,
        device="cuda",
        torch_dtype=torch.float16,
        height_division_factor=64,
        width_division_factor=64,
        time_division_factor=None,
        time_division_remainder=None,
    ):
        super().__init__()
        # The device and torch_dtype is used for the storage of intermediate variables, not models.
        self.device = device
        self.torch_dtype = torch_dtype
        # The following parameters are used for shape check.
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        self.vram_management_enabled = False

    def to(self, *args, **kwargs):
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(
            *args, **kwargs
        )
        if device is not None:
            self.device = device
        if dtype is not None:
            self.torch_dtype = dtype
        super().to(*args, **kwargs)
        return self

    def check_resize_height_width(self, height, width, num_frames=None):
        # Shape check
        # print(
        #     f"height, width, time division factor: {self.height_division_factor}, {self.width_division_factor}, {self.time_division_factor}, time division remainder: {self.time_division_remainder}"
        # )
        assert (
            height % self.height_division_factor == 0
        ), f"height {height} is not divisible by {self.height_division_factor}."

        assert (
            width % self.width_division_factor == 0
        ), f"width {width} is not divisible by {self.width_division_factor}."
        assert (num_frames is not None) and (
            (num_frames + self.time_division_factor) % self.time_division_factor
            == self.time_division_remainder
        ), f"num_frames {num_frames} is not divisible by {self.time_division_factor} with remainder {self.time_division_remainder}."
        return height, width, num_frames

    def preprocess_image(
        self,
        image,
        torch_dtype=None,
        device=None,
        pattern="B C H W",
        min_value=-1,
        max_value=1,
    ):
        # Transform a PIL.Image to torch.Tensor
        # print(f"Image size: {image.size}, dtype: {image.mode}")
        # assert isinstance(image, torch.Tensor), "Image must be a torch.Tensor."
        # C H W
        if isinstance(image, torch.Tensor):
            # C H W
            # print(f"Image shape {image.shape}")
            assert (len(image.shape) == 3 and image.shape[0] == 3) or (
                len(image.shape) == 4 and image.shape[1] == 3
            ), "Image tensor must be in 3 H W or B 3 H W format."
            image = image.to(
                dtype=torch_dtype or self.torch_dtype, device=device or self.device
            )
            image = image * ((max_value - min_value)) + min_value
            if len(image.shape) == 3:
                image = image.unsqueeze(0)  # Add batch dimension
        else:
            image = torch.Tensor(np.array(image, dtype=np.float32))
            image = image.to(
                dtype=torch_dtype or self.torch_dtype, device=device or self.device
            )
            image = image * ((max_value - min_value) / 255) + min_value
            image = repeat(
                image, f"H W C -> {pattern}", **({"B": 1} if "B" in pattern else {})
            )
        return image

    def preprocess_video(
        self,
        video,
        torch_dtype=None,
        device=None,
        pattern="B C T H W",
        min_value=-1,
        max_value=1,
    ):
        video = [
            self.preprocess_image(
                image,
                torch_dtype=torch_dtype,
                device=device,
                min_value=min_value,
                max_value=max_value,
            )
            for image in video
        ]
        video = torch.stack(video, dim=pattern.index("T") // 2)
        return video

    def vae_output_to_image(
        self, vae_output, pattern="B C H W", min_value=-1, max_value=1
    ):
        # Transform a torch.Tensor to PIL.Image
        if pattern != "H W C":
            vae_output = reduce(
                vae_output, f"{pattern} -> H W C", reduction="mean")

        # image = ((vae_output - min_value) * (255 / (max_value - min_value))).clip(
        #     0, 255
        # )
        image = (vae_output - min_value) * (255.0 / (max_value - min_value))
        image = image.clamp(0.0, 255.0)

        image = image.to(device="cpu", dtype=torch.float32)
        image = image.numpy()
        # image = Image.fromarray(image.numpy())
        return image

    def vae_output_to_video(
        self, vae_output, pattern="B C T H W", min_value=-1, max_value=1
    ):
        # Transform a torch.Tensor to list of PIL.Image
        # if pattern != "T H W C":
        #     vae_output = reduce(
        #         vae_output, f"{pattern} -> T H W C", reduction="mean")
        if vae_output.ndim == 5:  # B C T H W
            assert (
                vae_output.shape[1] == 3
            ), f"vae_output shape {vae_output.shape} is not valid. Expected 5D tensor with 3 channels on the second dimension."
            vae_output = vae_output.permute(0, 2, 3, 4, 1)
            # print(f"vae_output shape after permute: {vae_output.shape}")
            video = vae_output.to(device="cpu", dtype=torch.float32).numpy()
            video = (video + 1.0) / 2.0
            # print(f"Video range before clip: {video.min()} to {video.max()}")
            video = video.clip(0.0, 1.0)

        #     for _video in vae_output:
        #         video.append(
        #             [
        #                 self.vae_output_to_image(
        #                     image,
        #                     pattern="H W C",
        #                     min_value=min_value,
        #                     max_value=max_value,
        #                 )
        #                 for image in _video
        #             ]
        #         )
        # else:
        #     raise ValueError(
        #         f"Invalid vae_output shape {vae_output.shape}. Expected 5D tensor."
        #     )
        return video

    def load_models_to_device(self, model_names=None):
        if model_names is None:
            model_names = []
        if self.vram_management_enabled:
            # offload models
            for name, model in self.named_children():
                if name not in model_names:
                    if (
                        hasattr(model, "vram_management_enabled")
                        and model.vram_management_enabled
                    ):
                        for module in model.modules():
                            if hasattr(module, "offload"):
                                module.offload()
                    else:
                        model.cpu()
            torch.cuda.empty_cache()
            # onload models
            for name, model in self.named_children():
                if name in model_names:
                    if (
                        hasattr(model, "vram_management_enabled")
                        and model.vram_management_enabled
                    ):
                        for module in model.modules():
                            if hasattr(module, "onload"):
                                module.onload()
                    else:
                        model.to(self.device)

    def generate_noise(
        self,
        shape,
        seed=None,
        rand_device="cpu",
        rand_torch_dtype=torch.float32,
        device=None,
        torch_dtype=None,
    ):
        # Initialize Gaussian noise
        generator = (
            None if seed is None else torch.Generator(
                rand_device).manual_seed(seed)
        )
        # TODO multi-res noise
        noise = torch.randn(
            shape, generator=generator, device=rand_device, dtype=rand_torch_dtype
        )
        noise = noise.to(
            dtype=torch_dtype or self.torch_dtype, device=device or self.device
        )
        return noise

    def enable_cpu_offload(self):
        warnings.warn(
            "`enable_cpu_offload` will be deprecated. Please use `enable_vram_management`."
        )
        self.vram_management_enabled = True

    def get_vram(self):
        return torch.cuda.mem_get_info(self.device)[1] / (1024**3)

    def freeze_except(self, model_names):
        for name, model in self.named_children():
            if name in model_names:
                print(f"Unfreezing model {name}.")
                print(
                    f"Model parameters: {sum(p.numel() for p in model.parameters())}")
                model.train()
                model.requires_grad_(True)
            else:
                print(f"Freezing model {name}.")
                print(
                    f"Model parameters: {sum(p.numel() for p in model.parameters())}")
                model.eval()
                model.requires_grad_(False)


@dataclass
class ModelConfig:
    path: Union[str, list[str]] = None
    model_id: str = None
    origin_file_pattern: Union[str, list[str]] = None
    download_resource: str = "ModelScope"
    offload_device: Optional[Union[str, torch.device]] = None
    offload_dtype: Optional[torch.dtype] = None

    def download_if_necessary(
        self, local_model_path="./models", skip_download=False, use_usp=False
    ):
        if self.path is None:
            # Check model_id and origin_file_pattern
            if self.model_id is None:
                raise ValueError(
                    f"""No valid model files. Please use `ModelConfig(path="xxx")` or `ModelConfig(model_id="xxx/yyy", origin_file_pattern="zzz")`."""
                )

            # Skip if not in rank 0
            if use_usp:
                import torch.distributed as dist

                skip_download = dist.get_rank() != 0

            # Check whether the origin path is a folder
            if self.origin_file_pattern is None or self.origin_file_pattern == "":
                self.origin_file_pattern = ""
                allow_file_pattern = None
                is_folder = True
            elif isinstance(
                self.origin_file_pattern, str
            ) and self.origin_file_pattern.endswith("/"):
                allow_file_pattern = self.origin_file_pattern + "*"
                is_folder = True
            else:
                allow_file_pattern = self.origin_file_pattern
                is_folder = False

            # Download
            if not skip_download:
                downloaded_files = glob.glob(
                    self.origin_file_pattern,
                    root_dir=os.path.join(local_model_path, self.model_id),
                )
                # snapshot_download(
                #     self.model_id,
                #     local_dir=os.path.join(local_model_path, self.model_id),
                #     allow_file_pattern=allow_file_pattern,
                #     ignore_file_pattern=downloaded_files,
                #     local_files_only=False,
                # )
                snapshot_download(
                    self.model_id,
                    repo_type="model",  # 如果是dataset要改成"dataset"
                    local_dir=os.path.join(local_model_path, self.model_id),
                    allow_patterns=allow_file_pattern,
                    ignore_patterns=downloaded_files,    # 注意这里是 patterns
                    # ignore_file_pattern=downloaded_files,
                    # local_files_only=False,
                    local_files_only=False,
                    resume_download=True,   # 支持断点续传

                )

            # Let rank 1, 2, ... wait for rank 0
            if use_usp:
                import torch.distributed as dist

                dist.barrier(device_ids=[dist.get_rank()])

            # Return downloaded files
            if is_folder:
                self.path = os.path.join(
                    local_model_path, self.model_id, self.origin_file_pattern
                )
            else:
                self.path = glob.glob(
                    os.path.join(
                        local_model_path, self.model_id, self.origin_file_pattern
                    )
                )
            if isinstance(self.path, list) and len(self.path) == 1:
                self.path = self.path[0]


class WanVideoPipeline(BasePipeline):

    def __init__(self, device="cuda", torch_dtype=torch.bfloat16, tokenizer_path=None):
        super().__init__(
            device=device,
            torch_dtype=torch_dtype,
            height_division_factor=16,
            width_division_factor=16,
            time_division_factor=4,
            time_division_remainder=1,
        )
        self.scheduler = FlowMatchScheduler(
            shift=5, sigma_min=0.0, extra_one_step=True)
        # self.prompter = WanPrompter(tokenizer_path=tokenizer_path)
        self.text_encoder: WanTextEncoder = None
        self.image_encoder: WanImageEncoder = None
        # self.pose_encoder: CameraPoseEncoder = None
        self.dit: WanModel = None
        self.vae: WanVideoVAE = None
        self.motion_controller: WanMotionControllerModel = None
        self.vace: VaceWanModel = None
        self.in_iteration_models = ("dit", "motion_controller", "vace")
        self.unit_runner = PipelineUnitRunner()

        self.units = [
            WanVideoUnit_ShapeChecker(),  # check if the shape if ok
            # WanVideoUnit_NoiseInitializer(),
            WanVideoUnit_InputVideoEmbedder(),
            WanVideoUnit_PromptEmbedder(),
            WanVideoUnit_ImageEmbedder(),
            # WanVideoUnit_FunReference(),
            # WanVideoUnit_CameraPoseEmbedder(),
            # WanVideoUnit_SpeedControl(),
            # WanVideoUnit_VACE(),
            WanVideoUnit_UnifiedSequenceParallel(),
            # WanVideoUnit_TeaCache(),
            # WanVideoUnit_CfgMerger(),
        ]

        self.model_fn = model_fn_wan_video

    def training_predict(self, **inputs):
        timestep_id = torch.tensor([0])
        # print(f"timestep_id: {timestep_id}")
        timestep = self.scheduler.timesteps[timestep_id].to(
            dtype=self.torch_dtype, device=self.device
        )
        # print(f"Selected timestep {timestep}")
        inputs["latents"] = inputs['rgb_latents']
        training_target = self.scheduler.training_target(
            inputs["depth_latents"], inputs["rgb_latents"], timestep
        )
        noise_pred = self.model_fn(**inputs, timestep=timestep)

        return {
            'rgb_gt': inputs['rgb_latents'],
            "depth_gt": training_target,
            "pred": noise_pred,
            "weight": self.scheduler.training_weight(timestep),
        }

    def enable_vram_management(
        self, num_persistent_param_in_dit=None, vram_limit=None, vram_buffer=0.5
    ):
        self.vram_management_enabled = True
        if num_persistent_param_in_dit is not None:
            vram_limit = None
        else:
            if vram_limit is None:
                vram_limit = self.get_vram()
            vram_limit = vram_limit - vram_buffer
        if self.text_encoder is not None:
            dtype = next(iter(self.text_encoder.parameters())).dtype
            enable_vram_management(
                self.text_encoder,
                module_map={
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Embedding: AutoWrappedModule,
                    T5RelativeEmbedding: AutoWrappedModule,
                    T5LayerNorm: AutoWrappedModule,
                },
                module_config=dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                vram_limit=vram_limit,
            )
        if self.dit is not None:
            dtype = next(iter(self.dit.parameters())).dtype
            device = "cpu" if vram_limit is not None else self.device
            enable_vram_management(
                self.dit,
                module_map={
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv3d: AutoWrappedModule,
                    torch.nn.LayerNorm: WanAutoCastLayerNorm,
                    RMSNorm: AutoWrappedModule,
                    torch.nn.Conv2d: AutoWrappedModule,
                },
                module_config=dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device=device,
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                max_num_param=num_persistent_param_in_dit,
                overflow_module_config=dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                vram_limit=vram_limit,
            )
        if self.vae is not None:
            dtype = next(iter(self.vae.parameters())).dtype
            enable_vram_management(
                self.vae,
                module_map={
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv2d: AutoWrappedModule,
                    RMS_norm: AutoWrappedModule,
                    CausalConv3d: AutoWrappedModule,
                    Upsample: AutoWrappedModule,
                    torch.nn.SiLU: AutoWrappedModule,
                    torch.nn.Dropout: AutoWrappedModule,
                },
                module_config=dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device=self.device,
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
            )
        if self.image_encoder is not None:
            dtype = next(iter(self.image_encoder.parameters())).dtype
            enable_vram_management(
                self.image_encoder,
                module_map={
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv2d: AutoWrappedModule,
                    torch.nn.LayerNorm: AutoWrappedModule,
                },
                module_config=dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=dtype,
                    computation_device=self.device,
                ),
            )
        if self.motion_controller is not None:
            dtype = next(iter(self.motion_controller.parameters())).dtype
            enable_vram_management(
                self.motion_controller,
                module_map={
                    torch.nn.Linear: AutoWrappedLinear,
                },
                module_config=dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=dtype,
                    computation_device=self.device,
                ),
            )
        if self.vace is not None:
            device = "cpu" if vram_limit is not None else self.device
            enable_vram_management(
                self.vace,
                module_map={
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv3d: AutoWrappedModule,
                    torch.nn.LayerNorm: AutoWrappedModule,
                    RMSNorm: AutoWrappedModule,
                },
                module_config=dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device=device,
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                vram_limit=vram_limit,
            )

    def initialize_usp(self):
        import torch.distributed as dist
        from xfuser.core.distributed import (init_distributed_environment,
                                             initialize_model_parallel)

        dist.init_process_group(backend="nccl", init_method="env://")
        init_distributed_environment(
            rank=dist.get_rank(), world_size=dist.get_world_size()
        )
        initialize_model_parallel(
            sequence_parallel_degree=dist.get_world_size(),
            ring_degree=1,
            ulysses_degree=dist.get_world_size(),
        )
        torch.cuda.set_device(dist.get_rank())

    def enable_usp(self):
        from xfuser.core.distributed import get_sequence_parallel_world_size

        from ..distributed.xdit_context_parallel import (usp_attn_forward,
                                                         usp_dit_forward)

        for block in self.dit.blocks:
            block.self_attn.forward = types.MethodType(
                usp_attn_forward, block.self_attn
            )
        self.dit.forward = types.MethodType(usp_dit_forward, self.dit)
        self.sp_size = get_sequence_parallel_world_size()
        self.use_unified_sequence_parallel = True

    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = "cuda",
        model_configs: list[ModelConfig] = None,
        tokenizer_config: ModelConfig = ModelConfig(
            model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/*"
        ),
        local_model_path: str = "./models",
        skip_download: bool = False,
        redirect_common_files: bool = True,
        use_usp=False,
    ):
        # Redirect model path
        if model_configs is None:
            model_configs = []
        if redirect_common_files:
            redirect_dict = {
                "models_t5_umt5-xxl-enc-bf16.pth": "Wan-AI/Wan2.1-T2V-1.3B",
                "Wan2.1_VAE.pth": "Wan-AI/Wan2.1-T2V-1.3B",
                "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth": "Wan-AI/Wan2.1-I2V-14B-480P",
            }
            for model_config in model_configs:
                if (
                    model_config.origin_file_pattern is None
                    or model_config.model_id is None
                ):
                    continue
                if (
                    model_config.origin_file_pattern in redirect_dict
                    and model_config.model_id
                    != redirect_dict[model_config.origin_file_pattern]
                ):
                    print(
                        f"To avoid repeatedly downloading model files, ({model_config.model_id}, {model_config.origin_file_pattern}) is redirected to ({redirect_dict[model_config.origin_file_pattern]}, {model_config.origin_file_pattern}). You can use `redirect_common_files=False` to disable file redirection."
                    )
                    model_config.model_id = redirect_dict[
                        model_config.origin_file_pattern
                    ]

        # Initialize pipeline
        pipe = WanVideoPipeline(device=device, torch_dtype=torch_dtype)
        if use_usp:
            pipe.initialize_usp()

        # Download and load models
        model_manager = ModelManager()
        for model_config in model_configs:
            model_config.download_if_necessary(
                local_model_path, skip_download=skip_download, use_usp=use_usp
            )
            model_manager.load_model(
                model_config.path,
                device=model_config.offload_device or device,
                torch_dtype=model_config.offload_dtype or torch_dtype,
            )

        # Load models
        # pipe.text_encoder = model_manager.fetch_model("wan_video_text_encoder")
        pipe.dit = model_manager.fetch_model("wan_video_dit")

        pipe.vae = model_manager.fetch_model("wan_video_vae")
        pipe.image_encoder = model_manager.fetch_model(
            "wan_video_image_encoder")
        pipe.motion_controller = model_manager.fetch_model(
            "wan_video_motion_controller"
        )
        pipe.vace = model_manager.fetch_model("wan_video_vace")

        # Initialize tokenizer
        tokenizer_config.download_if_necessary(
            local_model_path, skip_download=skip_download
        )
        # pipe.prompter.fetch_models(pipe.text_encoder)
        # pipe.prompter.fetch_tokenizer(tokenizer_config.path)

        # Unified Sequence Parallel
        if use_usp:
            pipe.enable_usp()
        return pipe

    # @torch.no_grad()
    @torch.inference_mode()
    def __call__(
        self,
        # Prompt
        prompt: str,
        negative_prompt: Optional[str] = "",
        # Image-to-video
        input_image: Optional[Image.Image] = None,
        # First-last-frame-to-video
        end_image: Optional[Image.Image] = None,
        # Video-to-video
        input_video: Optional[list[Image.Image]] = None,
        denoising_strength: Optional[float] = 1.0,
        # ControlNet
        reference_image: Optional[Image.Image] = None,
        extra_images: Optional[List[Image.Image]] = None,
        extra_image_frame_index: Optional[List[int]] = None,
        # VACE
        vace_video: Optional[list[Image.Image]] = None,
        vace_video_mask: Optional[Image.Image] = None,
        vace_reference_image: Optional[Image.Image] = None,
        vace_scale: Optional[float] = 1.0,
        # Randomness
        seed: Optional[int] = None,
        rand_device: Optional[str] = "cpu",
        # Shape
        mode: Optional[str] = "regression",
        batch_size: Optional[int] = 1,
        height: Optional[int] = 480,
        width: Optional[int] = 720,
        frame_mask: Optional[torch.Tensor] = None,
        num_frames=41,
        # Classifier-free guidance
        cfg_scale: Optional[float] = 1,
        cfg_merge: Optional[bool] = False,
        # Scheduler
        num_inference_steps: Optional[int] = 1,
        sigma_shift: Optional[float] = 5.0,
        denoise_step=1,
        # Speed control
        motion_bucket_id: Optional[int] = None,
        # VAE tiling
        tiled: Optional[bool] = False,
        tile_size: Optional[tuple[int, int]] = (30, 52),
        tile_stride: Optional[tuple[int, int]] = (15, 26),
        # Sliding window
        sliding_window_size: Optional[int] = None,
        sliding_window_stride: Optional[int] = None,
        # Teacache
        tea_cache_l1_thresh: Optional[float] = None,
        tea_cache_model_id: Optional[str] = "",
        # progress_bar
        progress_bar_cmd=tqdm,
    ):
        self.scheduler.set_timesteps(
            num_inference_steps=num_inference_steps,
            denoising_strength=denoising_strength,
            shift=sigma_shift,
            denoise_step=denoise_step,
        )

        # Inputs
        inputs_posi = {
            "prompt": prompt,
            "prompt_num": batch_size,
            "tea_cache_l1_thresh": tea_cache_l1_thresh,
            "tea_cache_model_id": tea_cache_model_id,
            "num_inference_steps": num_inference_steps,
        }
        inputs_nega = {
            "negative_prompt": negative_prompt,
            "prompt_num": batch_size,
            "tea_cache_l1_thresh": tea_cache_l1_thresh,
            "tea_cache_model_id": tea_cache_model_id,
            "num_inference_steps": num_inference_steps,
        }

        inputs_shared = {
            "batch_size": batch_size,
            "input_image": input_image,
            "end_image": end_image,
            "input_video": input_video,
            "denoising_strength": denoising_strength,
            "reference_image": reference_image,
            "vace_video": vace_video,
            "vace_video_mask": vace_video_mask,
            "vace_reference_image": vace_reference_image,
            "vace_scale": vace_scale,
            "seed": seed,
            "rand_device": rand_device,
            'mode': mode,
            "height": height,
            "width": width,
            "frame_mask": frame_mask,
            "num_frames": num_frames,
            "cfg_scale": cfg_scale,
            "cfg_merge": cfg_merge,
            "sigma_shift": sigma_shift,
            "motion_bucket_id": motion_bucket_id,
            "tiled": tiled,
            "tile_size": tile_size,
            "tile_stride": tile_stride,
            "sliding_window_size": sliding_window_size,
            "sliding_window_stride": sliding_window_stride,
            "extra_images": extra_images,
            "extra_image_frame_index": extra_image_frame_index,
        }
        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(
                unit, self, inputs_shared, inputs_posi, inputs_nega
            )

        models = {name: getattr(self, name)
                  for name in self.in_iteration_models}

        for timestep in self.scheduler.timesteps:
            timestep = timestep.unsqueeze(0).to(
                dtype=self.torch_dtype, device=self.device
            )
            # torch.cuda.synchronize()
            # start_time = time.time()
            noise_pred_posi = self.model_fn(
                **models, **inputs_shared, **inputs_posi, timestep=timestep
            )
            # torch.cuda.synchronize()
            # end_time = time.time()
            # print(f"Model forward time: {end_time - start_time}")
            noise_pred = noise_pred_posi

            inputs_shared["latents"] = self.scheduler.step(
                model_output=noise_pred,
                sample=inputs_shared["latents"],
            )
            
        rgb, depth = None, None
        if isinstance(inputs_shared['latents'], tuple):
            rgb, depth = inputs_shared['latents']
        else:
            depth = inputs_shared['latents']

        # VACE (TODO: remove it)
        if vace_reference_image is not None:
            inputs_shared["latents"] = inputs_shared["latents"][:, :, 1:]

        # torch.cuda.synchronize()
        # start_time = time.time()
        depth_video = self.vae.decode(
            depth,
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        # torch.cuda.synchronize()
        # end_time = time.time()
        # print(f"VAE decoding time: {end_time - start_time}")
        depth_video = self.vae_output_to_video(depth_video)
        rgb_video = None
        if rgb is not None:
            rgb_video = self.vae.decode(
                depth,
                device=self.device,
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride,
            )
            rgb_video = self.vae_output_to_video(rgb_video)

        return {
            'depth': depth_video,
            'rgb': rgb_video
        }


class PipelineUnit:
    def __init__(
        self,
        seperate_cfg: bool = False,
        take_over: bool = False,
        input_params: tuple[str] = None,
        input_params_posi: dict[str, str] = None,
        input_params_nega: dict[str, str] = None,
        onload_model_names: tuple[str] = None,
    ):
        self.seperate_cfg = seperate_cfg
        self.take_over = take_over
        self.input_params = input_params
        self.input_params_posi = input_params_posi
        self.input_params_nega = input_params_nega
        self.onload_model_names = onload_model_names

    def process(
        self, pipe: WanVideoPipeline, inputs: dict, positive=True, **kwargs
    ) -> dict:
        raise NotImplementedError("`process` is not implemented.")


class PipelineUnitRunner:
    def __init__(self):
        pass

    def __call__(
        self,
        unit: PipelineUnit,
        pipe: WanVideoPipeline,
        inputs_shared: dict,
        inputs_posi: dict,
        inputs_nega: dict,
    ) -> tuple[dict, dict]:
        if unit.take_over:
            # Let the pipeline unit take over this function.
            inputs_shared, inputs_posi, inputs_nega = unit.process(
                pipe,
                inputs_shared=inputs_shared,
                inputs_posi=inputs_posi,
                inputs_nega=inputs_nega,
            )
        elif unit.seperate_cfg:
            # Positive side
            processor_inputs = {
                name: inputs_posi.get(name_)
                for name, name_ in unit.input_params_posi.items()
            }
            if unit.input_params is not None:
                for name in unit.input_params:
                    processor_inputs[name] = inputs_shared.get(name)
            processor_outputs = unit.process(pipe, **processor_inputs)
            inputs_posi.update(processor_outputs)
            # Negative side
            if inputs_shared["cfg_scale"] != 1:
                processor_inputs = {
                    name: inputs_nega.get(name_)
                    for name, name_ in unit.input_params_nega.items()
                }
                if unit.input_params is not None:
                    for name in unit.input_params:
                        processor_inputs[name] = inputs_shared.get(name)
                processor_outputs = unit.process(pipe, **processor_inputs)
                inputs_nega.update(processor_outputs)
            else:
                inputs_nega.update(processor_outputs)
        else:
            processor_inputs = {
                name: inputs_shared.get(name) for name in unit.input_params
            }
            processor_outputs = unit.process(pipe, **processor_inputs)
            inputs_shared.update(processor_outputs)
        return inputs_shared, inputs_posi, inputs_nega


class WanVideoUnit_ShapeChecker(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("height", "width", "num_frames"))

    def process(self, pipe: WanVideoPipeline, height, width, num_frames):
        #     print(
        #         f"Init WanVideoPipeline with height={height}, width={width}, num_frames={num_frames}."
        #     )
        height, width, num_frames = pipe.check_resize_height_width(
            height, width, num_frames
        )
        # print(
        #     f"Resized WanVideoPipeline to height={height}, width={width}, num_frames={num_frames}."
        # )
        return {"height": height, "width": width, "num_frames": num_frames}


class WanVideoUnit_NoiseInitializer(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=(
                "batch_size",
                "height",
                "width",
                "num_frames",
                "seed",
                "rand_device",
                "vace_reference_image",
            )
        )

    def process(
        self,
        pipe: WanVideoPipeline,
        batch_size,
        height,
        width,
        num_frames,
        seed,
        rand_device,
        vace_reference_image,
    ):
        # print(f"num frames {num_frames}")
        length = (num_frames - 1) // 4 + 1
        if vace_reference_image is not None:
            length += 1
        # TODO
        noise = pipe.generate_noise(
            (batch_size, 16, length, height // 8, width // 8),
            seed=seed,
            rand_device=rand_device,
        )
        # print(f"Noise shape {noise.shape} ")

        return {"noise": noise, "latents": noise}


class WanVideoUnit_InputVideoEmbedder(PipelineUnit):  # For training only
    def __init__(self):
        super().__init__(
            input_params=(
                'mode',
                'seed',
                'rand_device',
                "batch_size",
                "height",
                "width",
                "num_frames",
                "input_video",
                "input_disp",
                "noise",
                "tiled",
                "tile_size",
                "tile_stride",
                "vace_reference_image",
            ),
            onload_model_names=("vae",),
        )

    def process(
        self,
        pipe,
        mode,
        seed,
        rand_device,
        batch_size,
        height,
        width,
        num_frames,
        input_video,
        input_disp,
        noise,
        tiled,
        tile_size,
        tile_stride,
        vace_reference_image,
    ):
        assert mode in ['generation',
                        'regression'], f"mode {mode} is not supported"
        length = (num_frames - 1) // 4 + 1
        # inference part        
        if not pipe.scheduler.training:
            if mode == 'generation':
                # only need noise
                noise = pipe.generate_noise(
                    (batch_size, 16, length, height // 8, width // 8),
                    seed=seed,
                    rand_device=rand_device,
                )
                return {'latents': noise}
            else:
                # only need rgb latent
                video_list = []
                for _input_video in input_video:
                    _preprocessed_video = pipe.preprocess_video(_input_video)
                    video_list.append(_preprocessed_video)
                videos_tensor = torch.cat(video_list, dim=0)
                # print(f"videos_tensor shape: {videos_tensor.shape}")
                input_rgb_latents = pipe.vae.encode(
                    videos_tensor,
                    device=pipe.device,
                    tiled=tiled,
                    tile_size=tile_size,
                    tile_stride=tile_stride,
                ).to(dtype=pipe.torch_dtype, device=pipe.device)
                return {"latents": input_rgb_latents}
        
        disp_list = []
        for _input_disp in input_disp:
            _preprocessed_disp = pipe.preprocess_video(_input_disp)
            disp_list.append(_preprocessed_disp)
        disp_tensor = torch.cat(disp_list, dim=0)
        input_disp_latents = pipe.vae.encode(
            disp_tensor,
            device=pipe.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        ).to(dtype=pipe.torch_dtype, device=pipe.device)
        
        # Training
        if mode == 'generation':
            # need noise + depth
            noise = pipe.generate_noise(
                (batch_size, 16, length, height // 8, width // 8),
                seed=seed,
                rand_device=rand_device,
            )
            return {'rgb_latents': noise, 'depth_latents': input_disp_latents}
        else:
            # need rgb + depth
            video_list = []
            for _input_video in input_video:
                _preprocessed_video = pipe.preprocess_video(_input_video)
                video_list.append(_preprocessed_video)
            videos_tensor = torch.cat(video_list, dim=0)
            input_rgb_latents = pipe.vae.encode(
                videos_tensor,
                device=pipe.device,
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride,
            ).to(dtype=pipe.torch_dtype, device=pipe.device)
            # del videos_tensor
            return {
                "rgb_latents": input_rgb_latents,
                "depth_latents": input_disp_latents,
            }


class WanVideoUnit_PromptEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={
                "prompt": "prompt",
                "positive": "positive",
                "prompt_num": "prompt_num",
            },
            input_params_nega={
                "prompt": "negative_prompt",
                "positive": "positive",
                "prompt_num": "prompt_num",
            },
            onload_model_names=("text_encoder",),
        )

    def process(self, pipe: WanVideoPipeline, prompt, positive, prompt_num) -> dict:
        # pipe.load_models_to_device(self.onload_model_names)
        prompt_emb = []
        # print(f"Encoding prompt: {prompt}")
        # if isinstance(prompt, str):
        #     prompt = [prompt] * prompt_num
        # prompt_emb = None
        # for _prompt in prompt:
        #     _prompt_emb = pipe.prompter.encode_prompt(
        #         _prompt, positive=positive, device=pipe.device
        #     )
        #     prompt_emb = _prompt_emb
        #     break
        # prompt_emb = prompt_emb.repeat(prompt_num,1,1)
        # # prompt_emb = torch.cat(prompt_emb, dim=0)
        # prompt_emb = prompt_emb.to(dtype=pipe.torch_dtype, device=pipe.device)
        # print(f"Prompt embedding shape: {prompt_emb.shape}")
        zero_pad = torch.zeros([prompt_num, 512, 4096])
        zero_pad = zero_pad.to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"context": zero_pad}
        # return {"context": prompt_emb}


class WanVideoUnit_ImageEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=(
                "input_image",
                "end_image",
                "num_frames",
                "height",
                "width",
                "tiled",
                "tile_size",
                "tile_stride",
                "extra_images",
                "extra_image_frame_index",
            ),
            onload_model_names=("image_encoder", "vae"),
        )

    def process(
        self,
        pipe: WanVideoPipeline,
        input_image,
        end_image,
        num_frames,
        height,
        width,
        tiled,
        tile_size,
        tile_stride,
        extra_images,
        extra_image_frame_index,
    ):
        # print(f"input image shape{input_image.shape} ")
        if not pipe.dit.has_image_input:
            return {}
        if input_image is None:
            return {}
        # pipe.load_models_to_device(self.onload_model_names)
        image = pipe.preprocess_image(input_image).to(pipe.device)  # B C H W
        batch_size = image.shape[0]
        clip_context = pipe.image_encoder.encode_image([image])
        msk = torch.ones(
            batch_size, num_frames, height // 8, width // 8, device=pipe.device
        )

        # print(
        #     f"tiled, tile size, tile stride: {tiled}, {tile_size}, {tile_stride}")
        # Assmue that one must have a input image
        vae_input = torch.concat(
            [
                image.unsqueeze(2),  # B C 1 H W
                torch.zeros(batch_size, 3, num_frames - 1, height, width).to(
                    image.device
                ),
            ],
            dim=2,
        )  # B C F H W

        vae_input = vae_input.permute(0, 2, 1, 3, 4).contiguous()  # B F C H W

        if (
            extra_images is not None
            and extra_image_frame_index is not None
        ):
            # print(f"extra images shape {extra_images.shape}")
            for _videoid, _video in enumerate(extra_images):
                # _video F C H W
                for idx, image in enumerate(_video):
                    if idx == 0:
                        continue
                    image = pipe.preprocess_image(
                        image).to(pipe.device)  # 1 C H W
                    vae_input[_videoid, idx] = image.squeeze(0)

            mask = extra_image_frame_index[:, :, None, None].to(
                pipe.device)  # B F 1 1
            mask = mask.expand(
                batch_size, mask.shape[1], height // 8, width // 8
            )  # B F H W

            msk = msk * mask
        else:
            msk[:, 1:] = 0

        msk = torch.concat(
            [torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1
        )
        msk = msk.view(
            batch_size, msk.shape[1] // 4, 4, height // 8, width // 8
        )  # B F C(4) H W
        msk = msk.transpose(1, 2)
        vae_input = vae_input.permute(0, 2, 1, 3, 4).contiguous()  # B C F H W
        y = pipe.vae.encode(
            vae_input.to(dtype=pipe.torch_dtype, device=pipe.device),
            device=pipe.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        # print(f"y shape after VAE encode: {y.shape}")
        # print(f"after VAE encode, y shape: {y.shape}")
        y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
        # print()
        y = torch.concat([msk, y], dim=1)  # B 16+4 F H W
        # print(f"after concat, y shape: {y.shape}")
        # y = y.unsqueeze(0)
        clip_context = clip_context.to(
            dtype=pipe.torch_dtype, device=pipe.device)
        # print(f"clip context shape: {clip_context.shape}")
        y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"clip_feature": clip_context, "y": y}


class WanVideoUnit_VACE(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=(
                "vace_video",
                "vace_video_mask",
                "vace_reference_image",
                "vace_scale",
                "height",
                "width",
                "num_frames",
                "tiled",
                "tile_size",
                "tile_stride",
            ),
            onload_model_names=("vae",),
        )

    def process(
        self,
        pipe: WanVideoPipeline,
        vace_video,
        vace_video_mask,
        vace_reference_image,
        vace_scale,
        height,
        width,
        num_frames,
        tiled,
        tile_size,
        tile_stride,
    ):
        if (
            vace_video is not None
            or vace_video_mask is not None
            or vace_reference_image is not None
        ):
            # pipe.load_models_to_device(["vae"])
            if vace_video is None:
                vace_video = torch.zeros(
                    (1, 3, num_frames, height, width),
                    dtype=pipe.torch_dtype,
                    device=pipe.device,
                )
            else:
                vace_video = pipe.preprocess_video(vace_video)

            if vace_video_mask is None:
                vace_video_mask = torch.ones_like(vace_video)
            else:
                vace_video_mask = pipe.preprocess_video(
                    vace_video_mask, min_value=0, max_value=1
                )

            inactive = vace_video * (1 - vace_video_mask) + 0 * vace_video_mask
            reactive = vace_video * vace_video_mask + 0 * (1 - vace_video_mask)
            inactive = pipe.vae.encode(
                inactive,
                device=pipe.device,
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride,
            ).to(dtype=pipe.torch_dtype, device=pipe.device)
            reactive = pipe.vae.encode(
                reactive,
                device=pipe.device,
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride,
            ).to(dtype=pipe.torch_dtype, device=pipe.device)
            vace_video_latents = torch.concat((inactive, reactive), dim=1)

            vace_mask_latents = rearrange(
                vace_video_mask[0, 0], "T (H P) (W Q) -> 1 (P Q) T H W", P=8, Q=8
            )
            vace_mask_latents = torch.nn.functional.interpolate(
                vace_mask_latents,
                size=(
                    (vace_mask_latents.shape[2] + 3) // 4,
                    vace_mask_latents.shape[3],
                    vace_mask_latents.shape[4],
                ),
                mode="nearest-exact",
            )

            if vace_reference_image is None:
                pass
            else:
                vace_reference_image = pipe.preprocess_video(
                    [vace_reference_image])
                vace_reference_latents = pipe.vae.encode(
                    vace_reference_image,
                    device=pipe.device,
                    tiled=tiled,
                    tile_size=tile_size,
                    tile_stride=tile_stride,
                ).to(dtype=pipe.torch_dtype, device=pipe.device)
                vace_reference_latents = torch.concat(
                    (vace_reference_latents, torch.zeros_like(
                        vace_reference_latents)),
                    dim=1,
                )
                vace_video_latents = torch.concat(
                    (vace_reference_latents, vace_video_latents), dim=2
                )
                vace_mask_latents = torch.concat(
                    (torch.zeros_like(
                        vace_mask_latents[:, :, :1]), vace_mask_latents),
                    dim=2,
                )

            vace_context = torch.concat(
                (vace_video_latents, vace_mask_latents), dim=1)
            return {"vace_context": vace_context, "vace_scale": vace_scale}
        else:
            # print(f"No VACE video, mask or reference image provided, skipping VACE.")
            return {"vace_context": None, "vace_scale": vace_scale}


class WanVideoUnit_UnifiedSequenceParallel(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=())

    def process(self, pipe: WanVideoPipeline):
        if hasattr(pipe, "use_unified_sequence_parallel"):
            if pipe.use_unified_sequence_parallel:
                return {"use_unified_sequence_parallel": True}
        return {}


class WanVideoUnit_CfgMerger(PipelineUnit):
    def __init__(self):
        super().__init__(take_over=True)
        self.concat_tensor_names = ["context",
                                    "clip_feature", "y", "reference_latents"]

    def process(self, pipe: WanVideoPipeline, inputs_shared, inputs_posi, inputs_nega):
        if not inputs_shared["cfg_merge"]:
            # print(f"Skipping CFG merge, cfg_merge is set to False.")
            return inputs_shared, inputs_posi, inputs_nega
        for name in self.concat_tensor_names:
            tensor_posi = inputs_posi.get(name)
            tensor_nega = inputs_nega.get(name)
            tensor_shared = inputs_shared.get(name)
            if tensor_posi is not None and tensor_nega is not None:
                inputs_shared[name] = torch.concat(
                    (tensor_posi, tensor_nega), dim=0)
            elif tensor_shared is not None:
                inputs_shared[name] = torch.concat(
                    (tensor_shared, tensor_shared), dim=0
                )
        inputs_posi.clear()
        inputs_nega.clear()
        return inputs_shared, inputs_posi, inputs_nega


class TeaCache:
    def __init__(self, num_inference_steps, rel_l1_thresh, model_id):
        self.num_inference_steps = num_inference_steps
        self.step = 0
        self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = None
        self.rel_l1_thresh = rel_l1_thresh
        self.previous_residual = None
        self.previous_hidden_states = None

        self.coefficients_dict = {
            "Wan2.1-T2V-1.3B": [
                -5.21862437e04,
                9.23041404e03,
                -5.28275948e02,
                1.36987616e01,
                -4.99875664e-02,
            ],
            "Wan2.1-T2V-14B": [
                -3.03318725e05,
                4.90537029e04,
                -2.65530556e03,
                5.87365115e01,
                -3.15583525e-01,
            ],
            "Wan2.1-I2V-14B-480P": [
                2.57151496e05,
                -3.54229917e04,
                1.40286849e03,
                -1.35890334e01,
                1.32517977e-01,
            ],
            "Wan2.1-I2V-14B-720P": [
                8.10705460e03,
                2.13393892e03,
                -3.72934672e02,
                1.66203073e01,
                -4.17769401e-02,
            ],
        }
        if model_id not in self.coefficients_dict:
            supported_model_ids = ", ".join(
                [i for i in self.coefficients_dict])
            raise ValueError(
                f"{model_id} is not a supported TeaCache model id. Please choose a valid model id in ({supported_model_ids})."
            )
        self.coefficients = self.coefficients_dict[model_id]

    def check(self, dit: WanModel, x, t_mod):
        modulated_inp = t_mod.clone()
        if self.step == 0 or self.step == self.num_inference_steps - 1:
            should_calc = True
            self.accumulated_rel_l1_distance = 0
        else:
            coefficients = self.coefficients
            rescale_func = np.poly1d(coefficients)
            self.accumulated_rel_l1_distance += rescale_func(
                (
                    (modulated_inp - self.previous_modulated_input).abs().mean()
                    / self.previous_modulated_input.abs().mean()
                )
                .cpu()
                .item()
            )
            if self.accumulated_rel_l1_distance < self.rel_l1_thresh:
                should_calc = False
            else:
                should_calc = True
                self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = modulated_inp
        self.step += 1
        if self.step == self.num_inference_steps:
            self.step = 0
        if should_calc:
            self.previous_hidden_states = x.clone()
        return not should_calc

    def store(self, hidden_states):
        self.previous_residual = hidden_states - self.previous_hidden_states
        self.previous_hidden_states = None

    def update(self, hidden_states):
        hidden_states = hidden_states + self.previous_residual
        return hidden_states


class TemporalTiler_BCTHW:
    def __init__(self):
        pass

    def build_1d_mask(self, length, left_bound, right_bound, border_width):
        x = torch.ones((length,))
        if not left_bound:
            x[:border_width] = (torch.arange(border_width) + 1) / border_width
        if not right_bound:
            x[-border_width:] = torch.flip(
                (torch.arange(border_width) + 1) / border_width, dims=(0,)
            )
        return x

    def build_mask(self, data, is_bound, border_width):
        _, _, T, _, _ = data.shape
        t = self.build_1d_mask(T, is_bound[0], is_bound[1], border_width[0])
        mask = repeat(t, "T -> 1 1 T 1 1")
        return mask

    def run(
        self,
        model_fn,
        sliding_window_size,
        sliding_window_stride,
        computation_device,
        computation_dtype,
        model_kwargs,
        tensor_names,
        batch_size=None,
    ):
        tensor_names = [
            tensor_name
            for tensor_name in tensor_names
            if model_kwargs.get(tensor_name) is not None
        ]
        tensor_dict = {
            tensor_name: model_kwargs[tensor_name] for tensor_name in tensor_names
        }
        B, C, T, H, W = tensor_dict[tensor_names[0]].shape
        if batch_size is not None:
            B *= batch_size
        data_device, data_dtype = (
            tensor_dict[tensor_names[0]].device,
            tensor_dict[tensor_names[0]].dtype,
        )
        value = torch.zeros(
            (B, C, T, H, W), device=data_device, dtype=data_dtype)
        weight = torch.zeros(
            (1, 1, T, 1, 1), device=data_device, dtype=data_dtype)
        for t in range(0, T, sliding_window_stride):
            if (
                t - sliding_window_stride >= 0
                and t - sliding_window_stride + sliding_window_size >= T
            ):
                continue
            t_ = min(t + sliding_window_size, T)
            model_kwargs.update(
                {
                    tensor_name: tensor_dict[tensor_name][:, :, t:t_:, :].to(
                        device=computation_device, dtype=computation_dtype
                    )
                    for tensor_name in tensor_names
                }
            )
            model_output = model_fn(**model_kwargs).to(
                device=data_device, dtype=data_dtype
            )
            mask = self.build_mask(
                model_output,
                is_bound=(t == 0, t_ == T),
                border_width=(sliding_window_size - sliding_window_stride,),
            ).to(device=data_device, dtype=data_dtype)
            value[:, :, t:t_, :, :] += model_output * mask
            weight[:, :, t:t_, :, :] += mask
        value /= weight
        model_kwargs.update(tensor_dict)
        return value


def model_fn_wan_video(
    dit: WanModel,
    motion_controller: WanMotionControllerModel = None,
    vace: VaceWanModel = None,
    latents: torch.Tensor = None,
    timestep: torch.Tensor = None,
    context: torch.Tensor = None,
    clip_feature: Optional[torch.Tensor] = None,
    y: Optional[torch.Tensor] = None,
    reference_latents=None,
    vace_context=None,
    vace_scale=1.0,
    tea_cache: TeaCache = None,
    use_unified_sequence_parallel: bool = False,
    motion_bucket_id: Optional[torch.Tensor] = None,
    sliding_window_size: Optional[int] = None,
    sliding_window_stride: Optional[int] = None,
    cfg_merge: bool = False,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
    **kwargs,
):
    if sliding_window_size is not None and sliding_window_stride is not None:
        model_kwargs = dict(
            dit=dit,
            motion_controller=motion_controller,
            vace=vace,
            latents=latents,
            timestep=timestep,
            context=context,
            clip_feature=clip_feature,
            y=y,
            reference_latents=reference_latents,
            vace_context=vace_context,
            vace_scale=vace_scale,
            tea_cache=tea_cache,
            use_unified_sequence_parallel=use_unified_sequence_parallel,
            motion_bucket_id=motion_bucket_id,
        )
        return TemporalTiler_BCTHW().run(
            model_fn_wan_video,
            sliding_window_size,
            sliding_window_stride,
            latents.device,
            latents.dtype,
            model_kwargs=model_kwargs,
            tensor_names=["latents", "y"],
            batch_size=2 if cfg_merge else 1,
        )

    if use_unified_sequence_parallel:
        import torch.distributed as dist
        from xfuser.core.distributed import (get_sequence_parallel_rank,
                                             get_sequence_parallel_world_size,
                                             get_sp_group)

    # x = latents
    # print(f"Receving x with shape{x.shape}")
    # print(f"timesteps {timestep}", end=" ")
    t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
    t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
    # print(f"t_mod shape: {t_mod.shape}")
    # print(f"first ten element{t_mod[0][:10]}")
    if motion_bucket_id is not None and motion_controller is not None:
        t_mod = t_mod + \
            motion_controller(motion_bucket_id).unflatten(1, (6, dit.dim))
    context = dit.text_embedding(context)

    # c_b, c_c, c_f, c_h, c_w = x.shape

    # Merged cfg
    if latents.shape[0] != context.shape[0]:
        latents = torch.concat([latents] * context.shape[0], dim=0)
        # print(f"Merging x to shape {x.shape}")

    if timestep.shape[0] != context.shape[0]:
        timestep = torch.concat([timestep] * context.shape[0], dim=0)
    # import pdb
    # pdb.set_trace()
    if dit.has_image_input:
        latents = torch.cat([latents, y], dim=1)  # (b, c_x + c_y, f, h, w)
        clip_embdding = dit.img_emb(clip_feature)
        context = torch.cat([clip_embdding, context], dim=1)

    latents, (f, h, w) = dit.patchify(latents, None)
    _shortcut = latents
    freqs = (
        torch.cat(
            [
                dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        )
        .reshape(f * h * w, 1, -1)
        .to(latents.device)
    )

    if tea_cache is not None:
        tea_cache_update = tea_cache.check(dit, latents, t_mod)
    else:
        tea_cache_update = False

    if use_unified_sequence_parallel:
        if dist.is_initialized() and dist.get_world_size() > 1:
            latents = torch.chunk(latents, get_sequence_parallel_world_size(), dim=1)[
                get_sequence_parallel_rank()
            ]

    if tea_cache_update:
        latents = tea_cache.update(latents)
    else:
        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward

        for idx, block in enumerate(dit.blocks):
            if use_gradient_checkpointing_offload:
                with torch.autograd.graph.save_on_cpu():
                    latents = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        latents,
                        context,
                        t_mod,
                        freqs,
                        use_reentrant=False,
                    )
            elif use_gradient_checkpointing:
                latents = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    latents,
                    context,
                    t_mod,
                    freqs,
                    use_reentrant=False,
                )
            else:
                latents = block(latents, context, t_mod, freqs)

            if vace_context is not None and idx in vace.vace_layers_mapping:
                current_vace_hint = vace_hints[vace.vace_layers_mapping[idx]]
                if (
                    use_unified_sequence_parallel
                    and dist.is_initialized()
                    and dist.get_world_size() > 1
                ):
                    current_vace_hint = torch.chunk(
                        current_vace_hint, get_sequence_parallel_world_size(), dim=1
                    )[get_sequence_parallel_rank()]
                latents = latents + current_vace_hint * vace_scale
        if tea_cache is not None:
            tea_cache.store(latents)

    latents = dit.head(latents, t)

    if use_unified_sequence_parallel:
        if dist.is_initialized() and dist.get_world_size() > 1:
            latents = get_sp_group().all_gather(latents, dim=1)
    # Remove reference latents
    if reference_latents is not None:
        latents = latents[:, reference_latents.shape[1]:]
        f -= 1

    latents = dit.unpatchify(latents, (f, h, w))
    return latents
