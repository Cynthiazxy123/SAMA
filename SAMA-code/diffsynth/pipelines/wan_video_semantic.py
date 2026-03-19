# wan_video_pipeline.py (cleaned: removed SigLIP model path / online SigLIP semantic branch)
# NOTE:
# - 训练阶段默认只走在线 siglip_model_path（离线 siglip_feats / siglip_feat_path 已关闭）
# - 推理阶段不暴露 use_siglip_semantic / siglip_model_path 参数
# - 修复：from_pretrained 未定义变量、InputVideoEmbedder debug 未定义变量、audio_encoder 未初始化 等

import os
import time
import warnings
import math
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from einops import rearrange
from tqdm import tqdm
from typing_extensions import Literal

from ..utils import BasePipeline, ModelConfig, PipelineUnit, PipelineUnitRunner
from ..models import ModelManager, load_state_dict
from ..models.wan_video_dit import WanModel, RMSNorm, sinusoidal_embedding_1d
from ..models.wan_video_dit_s2v import rope_precompute  # keep existing import
# If your repo defines model_fn_wans2v elsewhere, import it here.
# Try uncommenting if needed:
# from ..models.wan_video_dit_s2v import model_fn_wans2v

from ..models.wan_video_text_encoder import WanTextEncoder, T5RelativeEmbedding, T5LayerNorm
from ..models.wan_video_vae import WanVideoVAE, RMS_norm, CausalConv3d, Upsample
from ..models.wan_video_image_encoder import WanImageEncoder
from ..models.wan_video_vace import VaceWanModel

from ..models.wan_video_motion_controller import WanMotionControllerModel
from ..models.wan_video_animate_adapter import WanAnimateAdapter
from ..schedulers.flow_match import FlowMatchScheduler
from ..prompters import WanPrompter
from ..vram_management import (
    enable_vram_management,
    AutoWrappedModule,
    AutoWrappedLinear,
    WanAutoCastLayerNorm,
)
from ..lora import GeneralLoRALoader


class WanVideoPipeline(BasePipeline):
    def __init__(
        self,
        device="cuda",
        torch_dtype=torch.bfloat16,
        tokenizer_path=None,
        source_concat_mode="token",
        channel_concat_patch_init: Literal["copy", "zeros", "random"] = "copy",
        use_source_attention_mask: bool = False,
    ):
        super().__init__(
            device=device,
            torch_dtype=torch_dtype,
            height_division_factor=16,
            width_division_factor=16,
            time_division_factor=4,
            time_division_remainder=1,
        )
        self.scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
        self.prompter = WanPrompter(tokenizer_path=tokenizer_path)
        self.source_concat_mode = source_concat_mode
        self.use_source_attention_mask = use_source_attention_mask
        if channel_concat_patch_init not in {"copy", "zeros", "random"}:
            raise ValueError(f"Unsupported channel_concat_patch_init='{channel_concat_patch_init}'")
        self.channel_concat_patch_init = channel_concat_patch_init

        self.text_encoder: WanTextEncoder = None
        self.image_encoder: WanImageEncoder = None
        self.dit: WanModel = None
        self.dit2: WanModel = None
        self.vae: WanVideoVAE = None
        self.motion_controller: WanMotionControllerModel = None
        self.vace: VaceWanModel = None
        self.vace2: VaceWanModel = None
        self.animate_adapter: WanAnimateAdapter = None
        self.siglip_encoder = None
        self.siglip_model_path: Optional[str] = None

        self.in_iteration_models = ("dit", "motion_controller", "vace", "animate_adapter")
        self.in_iteration_models_2 = ("dit2", "motion_controller", "vace2", "animate_adapter")
        self._wan_debug_timing: dict[str, float] = {}

        self.unit_runner = PipelineUnitRunner()
        self.units = [
            WanVideoUnit_ShapeChecker(),
            WanVideoUnit_NoiseInitializer(),
            WanVideoUnit_PromptEmbedder(),
            WanVideoUnit_InputVideoEmbedder(),
            WanVideoUnit_CfgMerger(),
        ]
        self.post_units = []
        self.model_fn = model_fn_wan_video

    def preprocess_video(
        self,
        video,
        torch_dtype=None,
        device=None,
        pattern="B C T H W",
        min_value=-1,
        max_value=1,
    ):
        """
        支持三种输入形式：
        1) 单条视频: List[PIL.Image] -> (1, C, T, H, W)
        2) 批量视频: List[List[PIL.Image]] -> (B, C, T, H, W)
        3) Tensor:   尽量统一到 (B, C, T, H, W)

        兼容 preprocess_image 返回 (C,H,W) 或 (1,C,H,W) 的情况。
        """
        if video is None:
            raise ValueError("video is None.")
        if isinstance(video, (list, tuple)) and len(video) == 0:
            raise ValueError("video is empty.")

        if torch_dtype is None:
            torch_dtype = self.torch_dtype
        if device is None:
            device = self.device

        # 1) 单条样本：List[PIL.Image]
        if isinstance(video, list) and len(video) > 0 and isinstance(video[0], Image.Image):
            frames = []
            for frame in video:
                ft = self.preprocess_image(
                    frame,
                    torch_dtype=torch_dtype,
                    device=device,
                    min_value=min_value,
                    max_value=max_value,
                )
                if ft.dim() == 4 and ft.size(0) == 1:
                    ft = ft.squeeze(0)
                if ft.dim() != 3:
                    raise ValueError(
                        f"preprocess_image must return 3D (C,H,W) or 4D (1,C,H,W) tensor, "
                        f"but got shape {tuple(ft.shape)}"
                    )
                frames.append(ft)

            vt = torch.stack(frames, dim=0)         # (T, C, H, W)
            vt = vt.permute(1, 0, 2, 3)             # (C, T, H, W)
            vt = vt.unsqueeze(0)                    # (1, C, T, H, W)
            return vt.to(device=device, dtype=torch_dtype)

        # 2) 批量样本：List[List[PIL.Image]]
        if isinstance(video, list) and len(video) > 0 and isinstance(video[0], (list, tuple)):
            batch_tensors = [
                self.preprocess_video(
                    sample,
                    torch_dtype=torch_dtype,
                    device=device,
                    min_value=min_value,
                    max_value=max_value,
                )
                for sample in video
            ]
            vt = torch.cat(batch_tensors, dim=0)
            return vt.to(device=device, dtype=torch_dtype)

        # 3) Tensor
        if isinstance(video, torch.Tensor):
            vt = video.to(device=device, dtype=torch_dtype)
            if vt.dim() == 5:
                return vt
            elif vt.dim() == 4:
                vt = vt.permute(1, 0, 2, 3).unsqueeze(0)
                return vt
            else:
                raise ValueError(
                    f"Unsupported video tensor shape {tuple(vt.shape)}. "
                    "Expected 4D (T, C, H, W) or 5D (B, C, T, H, W)."
                )

        raise TypeError(
            f"Unsupported video type {type(video)}. "
            "Expected List[PIL.Image], List[List[PIL.Image]] or torch.Tensor."
        )

    def load_lora(
        self,
        module: torch.nn.Module,
        lora_config: Union[ModelConfig, str] = None,
        alpha=1,
        hotload=False,
        state_dict=None,
    ):
        if state_dict is None:
            if isinstance(lora_config, str):
                lora = load_state_dict(lora_config, torch_dtype=self.torch_dtype, device=self.device)
            else:
                lora_config.download_if_necessary()
                lora = load_state_dict(lora_config.path, torch_dtype=self.torch_dtype, device=self.device)
        else:
            lora = state_dict

        if hotload:
            for name, module in module.named_modules():
                if isinstance(module, AutoWrappedLinear):
                    lora_a_name = f"{name}.lora_A.default.weight"
                    lora_b_name = f"{name}.lora_B.default.weight"
                    if lora_a_name in lora and lora_b_name in lora:
                        module.lora_A_weights.append(lora[lora_a_name] * alpha)
                        module.lora_B_weights.append(lora[lora_b_name])
        else:
            loader = GeneralLoRALoader(torch_dtype=self.torch_dtype, device=self.device)
            loader.load(module, lora, alpha=alpha)

    def _wan_debug_reset_timing(self):
        self._wan_debug_timing = {}

    def _wan_debug_add_time(self, name: str, duration: float):
        if os.environ.get("WAN_DEBUG_TIMING") == "1":
            self._wan_debug_timing[name] = self._wan_debug_timing.get(name, 0.0) + max(duration, 0.0)

    def _wan_debug_consume_time(self, name: str) -> float:
        return self._wan_debug_timing.pop(name, 0.0)

    def _wan_debug_log(self, message: str):
        if os.environ.get("WAN_DEBUG_TIMING") != "1":
            return
        log_path = os.environ.get("WAN_DEBUG_TIMING_LOG", "wan_debug_timing.log")
        try:
            with open(log_path, "a") as f:
                f.write(message + "\n")
        except Exception:
            pass
        print(message)

    def enable_vram_management(self, num_persistent_param_in_dit=None, vram_limit=None, vram_buffer=0.5):
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
                    torch.nn.Conv1d: AutoWrappedModule,
                    torch.nn.Embedding: AutoWrappedModule,
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

        if self.dit2 is not None:
            dtype = next(iter(self.dit2.parameters())).dtype
            device = "cpu" if vram_limit is not None else self.device
            enable_vram_management(
                self.dit2,
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
                module_map={torch.nn.Linear: AutoWrappedLinear},
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
            dtype = next(iter(self.vace.parameters())).dtype
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

    def _adjust_dit_input_dim_for_channel_concat(self, init_mode: Optional[str] = None):
        """Adjust DiT model input dimension for channel concatenation mode."""
        import math

        init_mode = init_mode or self.channel_concat_patch_init
        if init_mode not in {"copy", "zeros", "random"}:
            raise ValueError(f"Unsupported init_mode='{init_mode}' for channel concat patch embedding.")

        def adjust_single_dit(dit_model):
            if dit_model is None:
                return

            current_in_dim = dit_model.patch_embedding.in_channels
            orig_dtype = dit_model.patch_embedding.weight.dtype
            orig_device = dit_model.patch_embedding.weight.device
            new_in_dim = current_in_dim * 2

            new_patch_embedding = nn.Conv3d(
                new_in_dim,
                dit_model.patch_embedding.out_channels,
                kernel_size=dit_model.patch_embedding.kernel_size,
                stride=dit_model.patch_embedding.stride,
                padding=dit_model.patch_embedding.padding,
                dilation=dit_model.patch_embedding.dilation,
                groups=dit_model.patch_embedding.groups,
                bias=dit_model.patch_embedding.bias is not None,
                padding_mode=dit_model.patch_embedding.padding_mode,
            ).to(device=orig_device, dtype=orig_dtype)

            with torch.no_grad():
                orig_weight = dit_model.patch_embedding.weight.to(orig_dtype)
                new_patch_embedding.weight[:, :current_in_dim] = orig_weight

                if init_mode == "copy":
                    new_patch_embedding.weight[:, current_in_dim:] = orig_weight
                elif init_mode == "zeros":
                    new_patch_embedding.weight[:, current_in_dim:] = 0
                elif init_mode == "random":
                    pass
                else:
                    raise RuntimeError("Unexpected channel concat patch init mode")

                if dit_model.patch_embedding.bias is not None:
                    new_patch_embedding.bias.data = dit_model.patch_embedding.bias.data.to(orig_dtype)

                k = new_in_dim / float(current_in_dim)
                scale = math.sqrt(1.0 / k)
                new_patch_embedding.weight.mul_(scale)

            dit_model.patch_embedding = new_patch_embedding
            dit_model.in_dim = new_in_dim

        adjust_single_dit(self.dit)
        adjust_single_dit(self.dit2)

    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = "cuda",
        model_configs: list[ModelConfig] = [],
        tokenizer_config: ModelConfig = ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/*"),
        redirect_common_files: bool = True,
        source_concat_mode: str = "token",
        channel_concat_patch_init: Literal["copy", "zeros", "random"] = "copy",
        use_source_attention_mask: bool = False,
    ):
        # Redirect model path
        if redirect_common_files:
            redirect_dict = {
                "models_t5_umt5-xxl-enc-bf16.pth": "Wan-AI/Wan2.1-T2V-1.3B",
                "Wan2.1_VAE.pth": "Wan-AI/Wan2.1-T2V-1.3B",
                "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth": "Wan-AI/Wan2.1-I2V-14B-480P",
            }
            for model_config in model_configs:
                if model_config.origin_file_pattern is None or model_config.model_id is None:
                    continue
                if (
                    model_config.origin_file_pattern in redirect_dict
                    and model_config.model_id != redirect_dict[model_config.origin_file_pattern]
                ):
                    print(
                        f"To avoid repeatedly downloading model files, ({model_config.model_id}, {model_config.origin_file_pattern}) "
                        f"is redirected to ({redirect_dict[model_config.origin_file_pattern]}, {model_config.origin_file_pattern}). "
                        "You can use `redirect_common_files=False` to disable file redirection."
                    )
                    model_config.model_id = redirect_dict[model_config.origin_file_pattern]

        pipe = WanVideoPipeline(
            device=device,
            torch_dtype=torch_dtype,
            source_concat_mode=source_concat_mode,
            channel_concat_patch_init=channel_concat_patch_init,
            use_source_attention_mask=use_source_attention_mask,
        )

        model_manager = ModelManager()
        for model_config in model_configs:
            model_config.download_if_necessary()
            model_manager.load_model(
                model_config.path,
                device=model_config.offload_device or device,
                torch_dtype=model_config.offload_dtype or torch_dtype,
            )

        pipe.text_encoder = model_manager.fetch_model("wan_video_text_encoder")
        dit = model_manager.fetch_model("wan_video_dit", index=2)
        if isinstance(dit, list):
            pipe.dit, pipe.dit2 = dit
        else:
            pipe.dit = dit
            # lazy attach trainable MLP + semantic head
            if not hasattr(pipe.dit, "siglip_feat_mlp") or pipe.dit.siglip_feat_mlp is None:
                siglip_dim = 1152
                out_dim = int(pipe.dit.dim)
                hidden_dim = 1024
                pipe.dit.siglip_feat_mlp = SigLIPFeatMLP(in_dim=siglip_dim, out_dim=out_dim, hidden_dim=hidden_dim).to(
                    device=pipe.device, dtype=pipe.torch_dtype
                )

            if not hasattr(pipe.dit, "semantic_head") or pipe.dit.semantic_head is None:
                pipe.dit.semantic_head = SemanticDiffusionHead(dim=int(pipe.dit.dim), out_dim=siglip_dim).to(
                    device=pipe.device, dtype=pipe.torch_dtype
                )

        # Adjust DiT model input dimension for channel concatenation mode
        if source_concat_mode == "channel":
            pipe._adjust_dit_input_dim_for_channel_concat()

        pipe.vae = model_manager.fetch_model("wan_video_vae")
        pipe.image_encoder = model_manager.fetch_model("wan_video_image_encoder")
        pipe.motion_controller = model_manager.fetch_model("wan_video_motion_controller")

        vace = model_manager.fetch_model("wan_video_vace", index=2)
        if isinstance(vace, list):
            pipe.vace, pipe.vace2 = vace
        else:
            pipe.vace = vace

        pipe.animate_adapter = model_manager.fetch_model("wan_video_animate_adapter")

        if pipe.vae is not None:
            pipe.height_division_factor = pipe.vae.upsampling_factor * 2
            pipe.width_division_factor = pipe.vae.upsampling_factor * 2

        tokenizer_config.download_if_necessary()
        pipe.prompter.fetch_models(pipe.text_encoder)
        pipe.prompter.fetch_tokenizer(tokenizer_config.path)

        return pipe

    @torch.no_grad()
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
        source_video: Optional[list[Image.Image]] = None,
        input_video: Optional[list[Image.Image]] = None,
        denoising_strength: Optional[float] = 1.0,
        # Speech-to-video
        input_audio: Optional[np.array] = None,
        audio_embeds: Optional[torch.Tensor] = None,
        audio_sample_rate: Optional[int] = 16000,
        s2v_pose_video: Optional[list[Image.Image]] = None,
        s2v_pose_latents: Optional[torch.Tensor] = None,
        motion_video: Optional[list[Image.Image]] = None,
        # ControlNet
        control_video: Optional[list[Image.Image]] = None,
        reference_image: Optional[Image.Image] = None,
        # Camera control
        camera_control_direction: Optional[
            Literal["Left", "Right", "Up", "Down", "LeftUp", "LeftDown", "RightUp", "RightDown"]
        ] = None,
        camera_control_speed: Optional[float] = 1 / 54,
        camera_control_origin: Optional[
            tuple
        ] = (0, 0.532139961, 0.946026558, 0.5, 0.5, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0),
        # VACE
        vace_video: Optional[list[Image.Image]] = None,
        vace_video_mask: Optional[Image.Image] = None,
        vace_reference_image: Optional[list[Image.Image]] = None,
        vace_scale: Optional[float] = 1.0,
        # Animate
        animate_pose_video: Optional[list[Image.Image]] = None,
        animate_face_video: Optional[list[Image.Image]] = None,
        animate_inpaint_video: Optional[list[Image.Image]] = None,
        animate_mask_video: Optional[list[Image.Image]] = None,
        # Randomness
        seed: Optional[int] = None,
        rand_device: Optional[str] = "cpu",
        # Shape
        height: Optional[int] = 480,
        width: Optional[int] = 832,
        num_frames=81,
        batch_size: Optional[int] = 1,
        # Classifier-free guidance
        cfg_scale: Optional[float] = 5.0,
        cfg_merge: Optional[bool] = False,
        # NEW: 独立控制 text / source 的 CFG（不传就走旧逻辑）
        cfg_scale_text: Optional[float] = None,
        cfg_scale_source: Optional[float] = None,
        # Boundary
        switch_DiT_boundary: Optional[float] = 0.875,
        # Scheduler
        num_inference_steps: Optional[int] = 50,
        sigma_shift: Optional[float] = 5.0,
        # Speed control
        motion_bucket_id: Optional[int] = None,
        # VAE tiling
        tiled: Optional[bool] = True,
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
        timing_enabled = os.environ.get("WAN_DEBUG_TIMING") == "1"
        if timing_enabled:
            self._wan_debug_reset_timing()
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength, shift=sigma_shift)

        inputs_posi = {
            "prompt": prompt,
            "tea_cache_l1_thresh": tea_cache_l1_thresh,
            "tea_cache_model_id": tea_cache_model_id,
            "num_inference_steps": num_inference_steps,
        }
        inputs_nega = {
            "negative_prompt": negative_prompt,
            "tea_cache_l1_thresh": tea_cache_l1_thresh,
            "tea_cache_model_id": tea_cache_model_id,
            "num_inference_steps": num_inference_steps,
        }
        inputs_shared = {
            "input_image": input_image,
            "end_image": end_image,
            "input_video": input_video,
            "source_video": source_video,
            "denoising_strength": denoising_strength,
            "control_video": control_video,
            "reference_image": reference_image,
            "camera_control_direction": camera_control_direction,
            "camera_control_speed": camera_control_speed,
            "camera_control_origin": camera_control_origin,
            "vace_video": vace_video,
            "vace_video_mask": vace_video_mask,
            "vace_reference_image": vace_reference_image,
            "vace_scale": vace_scale,
            "seed": seed,
            "rand_device": rand_device,
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "batch_size": batch_size,
            "cfg_scale": cfg_scale,
            "cfg_merge": cfg_merge,
            "sigma_shift": sigma_shift,
            "motion_bucket_id": motion_bucket_id,
            "tiled": tiled,
            "tile_size": tile_size,
            "tile_stride": tile_stride,
            "sliding_window_size": sliding_window_size,
            "sliding_window_stride": sliding_window_stride,
            "input_audio": input_audio,
            "audio_sample_rate": audio_sample_rate,
            "s2v_pose_video": s2v_pose_video,
            "audio_embeds": audio_embeds,
            "s2v_pose_latents": s2v_pose_latents,
            "motion_video": motion_video,
            "animate_pose_video": animate_pose_video,
            "animate_face_video": animate_face_video,
            "animate_inpaint_video": animate_inpaint_video,
            "animate_mask_video": animate_mask_video,
        }

        stage_begin = time.time()
        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)
        if timing_enabled and torch.cuda.is_available():
            torch.cuda.synchronize()
        load_time = time.time() - stage_begin
        if timing_enabled:
            self._wan_debug_add_time("preprocess", load_time)

        # Denoise
        self.load_models_to_device(self.in_iteration_models)
        models = {name: getattr(self, name) for name in self.in_iteration_models}

        # joint semantic diffusion state (inference only)
        semantic_latents = None
        # use_joint_semantic = (not self.scheduler.training) and inputs_shared.get("use_joint_semantic_tokens", False)
        use_joint_semantic = True

        if inputs_shared.get("use_unified_sequence_parallel", False):
            use_joint_semantic = False
        if sliding_window_size is not None and sliding_window_stride is not None:
            use_joint_semantic = False
        
        siglip_dim = 1152
        if use_joint_semantic:
            dit_dim = int(models["dit"].dim)
            B = int(inputs_shared.get("batch_size", 1) or 1)
            N = int(inputs_shared.get("semantic_token_count") or inputs_shared.get("joint_semantic_token_count") or 65) #65 260
            semantic_latents = torch.randn((B, N, siglip_dim), device=self.device, dtype=self.torch_dtype)#####纯噪声

        total_begin = time.time()
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            if (
                timestep.item() < switch_DiT_boundary * self.scheduler.num_train_timesteps
                and self.dit2 is not None
                and not models["dit"] is self.dit2
            ):
                self.load_models_to_device(self.in_iteration_models_2)
                models["dit"] = self.dit2
                models["vace"] = self.vace2

            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)
            # semantic_tokens_noised = semantic_latents if (semantic_latents is not None) else None
            semantic_tokens_noised = models["dit"].siglip_feat_mlp(semantic_latents) if (semantic_latents is not None) else None

            def _call_model(**kw):
                out = self.model_fn(
                    **models,
                    **kw,
                    timestep=timestep,
                    source_concat_mode=self.source_concat_mode,
                    use_source_attention_mask=self.use_source_attention_mask,
                    semantic_tokens_noised=semantic_tokens_noised,
                    return_sem_pred=(semantic_tokens_noised is not None),
                    return_sem_loss=False,
                )
                if isinstance(out, (tuple, list)):
                    if len(out) >= 2:
                        return out[0], out[1]
                    return out[0], None
                return out, None

            use_two_cfg = (cfg_scale_text is not None or cfg_scale_source is not None) and not cfg_merge

            if use_two_cfg:
                base_source_latents = inputs_shared.get("source_latents", None)
                has_source = base_source_latents is not None
                has_text = inputs_posi.get("context", None) is not None

                eps_0T, sem_0T = _call_model(**inputs_shared, **inputs_posi, source_latents=None)
                eps_I0, sem_I0 = _call_model(**inputs_shared, **inputs_nega, source_latents=base_source_latents)

                if has_source and has_text:
                    eps_IT, sem_IT = _call_model(**inputs_shared, **inputs_posi, source_latents=base_source_latents)
                elif has_text:
                    eps_IT, sem_IT = _call_model(**inputs_shared, **inputs_posi, source_latents=None)
                else:
                    eps_IT, sem_IT = eps_I0, sem_I0

                s_I = cfg_scale_source or 0.0
                s_T = cfg_scale_text or 0.0

                noise_pred = eps_IT + s_I * (eps_IT - eps_0T) + s_T * (eps_IT - eps_I0)

                if semantic_latents is not None:
                    if (sem_IT is not None) and (sem_0T is not None) and (sem_I0 is not None):
                        sem_pred = sem_IT + s_I * (sem_IT - sem_0T) + s_T * (sem_IT - sem_I0)
                    else:
                        sem_pred = None
                else:
                    sem_pred = None

            else:
                eps_posi, sem_posi = _call_model(**inputs_shared, **inputs_posi)

                if cfg_scale != 1.0:
                    if cfg_merge:
                        eps_posi, eps_nega = eps_posi.chunk(2, dim=0)
                        if semantic_latents is not None and sem_posi is not None:
                            sem_posi, sem_nega = sem_posi.chunk(2, dim=0)
                        else:
                            sem_nega = None
                    else:
                        eps_nega, sem_nega = _call_model(**inputs_shared, **inputs_nega)

                    noise_pred = eps_nega + cfg_scale * (eps_posi - eps_nega)

                    if semantic_latents is not None:
                        if (sem_posi is not None) and (sem_nega is not None):
                            sem_pred = sem_nega + cfg_scale * (sem_posi - sem_nega)
                        else:
                            sem_pred = None
                    else:
                        sem_pred = None
                else:
                    noise_pred = eps_posi
                    sem_pred = sem_posi if semantic_latents is not None else None

            inputs_shared["latents"] = self.scheduler.step(
                noise_pred, self.scheduler.timesteps[progress_id], inputs_shared["latents"]
            )

            if semantic_latents is not None and sem_pred is not None:
                semantic_latents = self.scheduler.step(
                    sem_pred, self.scheduler.timesteps[progress_id], semantic_latents
                )

            if "first_frame_latents" in inputs_shared:
                inputs_shared["latents"][:, :, 0:1] = inputs_shared["first_frame_latents"]

        if timing_enabled and torch.cuda.is_available():
            torch.cuda.synchronize()
        forward_time = time.time() - total_begin
        _ = load_time + forward_time

        # VACE (TODO remove)
        if vace_reference_image is not None or (animate_pose_video is not None and animate_face_video is not None):
            if vace_reference_image is not None and isinstance(vace_reference_image, list):
                f = len(vace_reference_image)
            else:
                f = 1
            inputs_shared["latents"] = inputs_shared["latents"][:, :, f:]

        semantic_token_frames = inputs_shared.get("semantic_token_frames", 0)
        if semantic_token_frames:
            inputs_shared["latents"] = inputs_shared["latents"][:, :, :-semantic_token_frames]

        for unit in self.post_units:
            inputs_shared, _, _ = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)

        self.load_models_to_device(["vae"])
        video = self.vae.decode(
            inputs_shared["latents"], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride
        )
        video = self.vae_output_to_video(video)
        self.load_models_to_device([])
        return video


class WanVideoUnit_ShapeChecker(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("height", "width", "num_frames"))

    def process(self, pipe: WanVideoPipeline, height, width, num_frames):
        height, width, num_frames = pipe.check_resize_height_width(height, width, num_frames)
        return {"height": height, "width": width, "num_frames": num_frames}


class WanVideoUnit_NoiseInitializer(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("height", "width", "num_frames", "seed", "rand_device", "vace_reference_image", "batch_size", "input_video"))

    def process(
        self,
        pipe: WanVideoPipeline,
        height,
        width,
        num_frames,
        seed,
        rand_device,
        vace_reference_image,
        batch_size=1,
        input_video=None,
    ):
        batch_size = int(batch_size or 1)
        length = (num_frames - 1) // 4 + 1
        if vace_reference_image is not None:
            f = len(vace_reference_image) if isinstance(vace_reference_image, list) else 1
            length += f

        shape = (
            batch_size,
            pipe.vae.model.z_dim,
            length,
            height // pipe.vae.upsampling_factor,
            width // pipe.vae.upsampling_factor,
        )

        if isinstance(seed, (list, tuple)):
            if len(seed) != batch_size:
                raise ValueError("Length of `seed` must equal batch_size.")
            noises = []
            for seed_ in seed:
                noises.append(pipe.generate_noise((1,) + shape[1:], seed=seed_, rand_device=rand_device))
            noise = torch.cat(noises, dim=0)
        else:
            noise = pipe.generate_noise(shape, seed=seed, rand_device=rand_device)

        if vace_reference_image is not None:
            noise = torch.concat((noise[:, :, -f:], noise[:, :, :-f]), dim=2)

        if os.environ.get("WAN_DEBUG_SHAPE") == "1":
            rank = int(os.environ.get("RANK", "0"))
            if rank == 0:
                print(f"[SHAPE] NoiseInitializer: batch_size={batch_size}, noise.shape={tuple(noise.shape)}", flush=True)

        return {"noise": noise, "semantic_token_frames": 0}


class WanVideoUnit_InputVideoEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=(
                "input_video",
                "source_video",
                "noise",
                "tiled",
                "tile_size",
                "tile_stride",
                "height",
                "width",
                "batch_size",
                "num_frames",
            ),
            onload_model_names=("vae",),
        )

    def process(
        self,
        pipe,
        input_video,
        source_video,
        noise,
        tiled,
        tile_size,
        tile_stride,
        height,
        width,
        batch_size=1,
        num_frames=81,
    ):
        if input_video is None and source_video is None:
            return {"latents": noise, "semantic_tokens": None, "semantic_token_count": 0, "semantic_token_frames": 0}

        pipe.load_models_to_device(["vae"])

        def _normalize_video(video_frames, tag):
            if video_frames is None:
                return None
            if isinstance(video_frames, list) and len(video_frames) > 0 and all(v is None for v in video_frames):
                return None
            if not isinstance(video_frames, (list, tuple)) or len(video_frames) == 0:
                return None

            def _fix_length_and_resize(frames, idx=None):
                if len(frames) < num_frames:
                    if len(frames) == 0:
                        raise ValueError(f"{tag} video sample {idx} has 0 frames.")
                    last = frames[-1]
                    frames = list(frames) + [last] * (num_frames - len(frames))
                elif len(frames) > num_frames:
                    frames = list(frames)[:num_frames]
                return [frame.resize((width, height)) for frame in frames]

            if isinstance(video_frames[0], Image.Image):
                fixed = _fix_length_and_resize(video_frames)
                return [fixed]

            normalized = []
            for i, frames in enumerate(video_frames):
                if frames is None:
                    raise ValueError(f"{tag}: batch 内出现 None 样本，暂不支持混合输入。")
                normalized.append(_fix_length_and_resize(frames, idx=i))
            return normalized

        def _encode_video(video_frames, tag, normalized_frames=None):
            if normalized_frames is None:
                normalized_frames = _normalize_video(video_frames, tag)
            if normalized_frames is None:
                return None

            video_tensor = pipe.preprocess_video(
                normalized_frames,
                torch_dtype=pipe.torch_dtype,
                device=pipe.device,
            )
            return pipe.vae.encode(
                video_tensor,
                device=pipe.device,
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride,
            ).to(dtype=pipe.torch_dtype, device=pipe.device)

        target_normalized = _normalize_video(input_video, "target") if input_video is not None else None
        source_normalized = _normalize_video(source_video, "source") if source_video is not None else None
        target_latents = _encode_video(input_video, "target", normalized_frames=target_normalized) if input_video is not None else None
        source_latents = _encode_video(source_video, "source") if source_video is not None else None

        outputs = {}
        if source_latents is not None:
            outputs["source_latents"] = source_latents
        outputs["latents"] = noise
        outputs["siglip_feats"] = None
        outputs["semantic_tokens"] = None
        outputs["semantic_token_count"] = 0

        return outputs


class WanVideoUnit_PromptEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"prompt": "prompt", "positive": "positive"},
            input_params_nega={"prompt": "negative_prompt", "positive": "positive"},
            onload_model_names=("text_encoder",),
        )

    def process(self, pipe: WanVideoPipeline, prompt, positive) -> dict:
        pipe.load_models_to_device(self.onload_model_names)
        prompt_emb = pipe.prompter.encode_prompt(prompt, positive=positive, device=pipe.device)
        return {"context": prompt_emb}


class WanVideoUnit_CfgMerger(PipelineUnit):
    def __init__(self):
        super().__init__(take_over=True)
        self.concat_tensor_names = ["context", "clip_feature", "y", "reference_latents"]

    def process(self, pipe: WanVideoPipeline, inputs_shared, inputs_posi, inputs_nega):
        if not inputs_shared["cfg_merge"]:
            return inputs_shared, inputs_posi, inputs_nega
        for name in self.concat_tensor_names:
            tensor_posi = inputs_posi.get(name)
            tensor_nega = inputs_nega.get(name)
            tensor_shared = inputs_shared.get(name)
            if tensor_posi is not None and tensor_nega is not None:
                inputs_shared[name] = torch.concat((tensor_posi, tensor_nega), dim=0)
            elif tensor_shared is not None:
                inputs_shared[name] = torch.concat((tensor_shared, tensor_shared), dim=0)
        inputs_posi.clear()
        inputs_nega.clear()
        return inputs_shared, inputs_posi, inputs_nega


class SigLIPFeatMLP(nn.Module):
    """Trainable MLP: (B,N,D_siglip) -> (B,N,D_dit)."""

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 1024, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim, eps=eps)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.norm(x)
        x = self.fc2(self.act(self.fc1(x)))
        return x


class SemanticDiffusionHead(nn.Module):
    """Predict diffusion target (eps/v) for semantic tokens: (B,N,D)->(B,N,D)."""

    def __init__(self, dim: int, out_dim: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps)
        self.proj = nn.Linear(dim, out_dim, bias=True)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.normal_(self.proj.bias, std=1e-3)

    def forward(self, x):
        if x.dtype != self.norm.weight.dtype:
            x = x.to(self.norm.weight.dtype)
        return self.proj(self.norm(x))


def model_fn_wan_video(
    dit: WanModel,
    motion_controller: WanMotionControllerModel = None,
    vace: VaceWanModel = None,
    animate_adapter: WanAnimateAdapter = None,
    latents: torch.Tensor = None,
    timestep: torch.Tensor = None,
    context: torch.Tensor = None,
    clip_feature: Optional[torch.Tensor] = None,
    y: Optional[torch.Tensor] = None,
    source_latents: Optional[torch.Tensor] = None,
    reference_latents=None,
    vace_context=None,
    vace_scale=1.0,
    audio_embeds: Optional[torch.Tensor] = None,
    motion_latents: Optional[torch.Tensor] = None,
    s2v_pose_latents: Optional[torch.Tensor] = None,
    drop_motion_frames: bool = True,
    tea_cache=None,
    use_unified_sequence_parallel: bool = False,
    motion_bucket_id: Optional[torch.Tensor] = None,
    pose_latents=None,
    face_pixel_values=None,
    sliding_window_size: Optional[int] = None,
    sliding_window_stride: Optional[int] = None,
    cfg_merge: bool = False,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
    control_camera_latents_input=None,
    fuse_vae_embedding_in_latents: bool = False,
    source_concat_mode: str = "token",
    use_source_attention_mask: bool = False,
    # semantic diffusion (token-level)
    semantic_tokens_noised: Optional[torch.Tensor] = None,
    semantic_target: Optional[torch.Tensor] = None,
    return_sem_loss: bool = False,
    return_sem_pred: bool = False,
    **kwargs,
):
    if os.environ.get("WAN_DEBUG_SHAPE") == "1":
        rank = int(os.environ.get("RANK", "0"))
        if rank == 0:
            print(
                "[SHAPE] model_fn_wan_video: "
                f"latents={None if latents is None else tuple(latents.shape)}, "
                f"context={None if context is None else tuple(context.shape)}, "
                f"y={None if y is None else tuple(y.shape)}, "
                f"source_latents={None if source_latents is None else tuple(source_latents.shape)}, "
                f"semantic_tokens_noised={None if semantic_tokens_noised is None else tuple(semantic_tokens_noised.shape)}, "
                f"return_sem_loss={return_sem_loss}",
                flush=True,
            )

    if use_unified_sequence_parallel:
        import torch.distributed as dist
        from xfuser.core.distributed import (
            get_sequence_parallel_rank,
            get_sequence_parallel_world_size,
            get_sp_group,
        )

    # Timestep embed
    if dit.seperated_timestep and fuse_vae_embedding_in_latents:
        raise NotImplementedError("wan2.2 not support fuse vae embedding in latents")
        timestep = torch.concat(
            [
                torch.zeros((1, latents.shape[3] * latents.shape[4] // 4), dtype=latents.dtype, device=latents.device),
                torch.ones((latents.shape[2] - 1, latents.shape[3] * latents.shape[4] // 4), dtype=latents.dtype, device=latents.device) * timestep,
            ]
        ).flatten()
        t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep).unsqueeze(0))
        if use_unified_sequence_parallel and dist.is_initialized() and dist.get_world_size() > 1:
            t_chunks = torch.chunk(t, get_sequence_parallel_world_size(), dim=1)
            t_chunks = [
                torch.nn.functional.pad(chunk, (0, 0, 0, t_chunks[0].shape[1] - chunk.shape[1]), value=0)
                for chunk in t_chunks
            ]
            t = t_chunks[get_sequence_parallel_rank()]
        t_mod = dit.time_projection(t).unflatten(2, (6, dit.dim))
    else:
        t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
        t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))

    if motion_bucket_id is not None and motion_controller is not None:
        t_mod = t_mod + motion_controller(motion_bucket_id).unflatten(1, (6, dit.dim))

    context = dit.text_embedding(context)

    def _repeat_to_batch(x: Optional[torch.Tensor], target_b: int) -> Optional[torch.Tensor]:
        if x is None:
            return None
        if x.shape[0] == target_b:
            return x
        if target_b % x.shape[0] != 0:
            raise ValueError(f"Cannot repeat tensor batch {x.shape[0]} -> {target_b}")
        rep = target_b // x.shape[0]
        return torch.cat([x] * rep, dim=0)

    B_ctx = context.shape[0]
    latents = _repeat_to_batch(latents, B_ctx)
    timestep = _repeat_to_batch(timestep, B_ctx)
    if y is not None:
        y = _repeat_to_batch(y, B_ctx)
    if clip_feature is not None:
        clip_feature = _repeat_to_batch(clip_feature, B_ctx)
    if source_latents is not None:
        source_latents = _repeat_to_batch(source_latents, B_ctx)
    if reference_latents is not None and isinstance(reference_latents, torch.Tensor):
        reference_latents = _repeat_to_batch(reference_latents, B_ctx)
    if semantic_tokens_noised is not None:
        semantic_tokens_noised = _repeat_to_batch(semantic_tokens_noised, B_ctx)

    x = latents

    if y is not None and dit.require_vae_embedding:
        x = torch.cat([x, y], dim=1)
    if clip_feature is not None and dit.require_clip_embedding:
        clip_embdding = dit.img_emb(clip_feature)
        context = torch.cat([clip_embdding, context], dim=1)

    if source_latents is not None and source_concat_mode == "channel":
        source_latents = source_latents.to(dtype=x.dtype, device=x.device)
        x = torch.cat([x, source_latents], dim=1)

    x = dit.patchify(x, control_camera_latents_input)
    target_f, target_h, target_w = x.shape[2:]

    def _build_freqs(num_f, num_h, num_w, device, start_f=0, start_h=0, start_w=0):
        f_slice = dit.freqs[0][start_f : start_f + num_f]
        h_slice = dit.freqs[1][start_h : start_h + num_h]
        w_slice = dit.freqs[2][start_w : start_w + num_w]
        return torch.cat(
            [
                f_slice.view(num_f, 1, 1, -1).expand(num_f, num_h, num_w, -1),
                h_slice.view(1, num_h, 1, -1).expand(num_f, num_h, num_w, -1),
                w_slice.view(1, 1, num_w, -1).expand(num_f, num_h, num_w, -1),
            ],
            dim=-1,
        ).reshape(num_f * num_h * num_w, 1, -1).to(device)

    if pose_latents is not None and face_pixel_values is not None:
        x, motion_vec = animate_adapter.after_patch_embedding(x, pose_latents, face_pixel_values)

    target_tokens = rearrange(x, "b c f h w -> b (f h w) c").contiguous()
    target_freqs_full = _build_freqs(target_f, target_h, target_w, target_tokens.device)

    freqs_segments = []
    x_tokens_segments = []

    target_token_count = target_tokens.shape[1]
    source_token_count = 0
    reference_token_count = 0
    semantic_token_count = 0
    source_tokens = None
    reference_tokens = None

    if source_latents is not None and source_concat_mode == "token":
        source_latents = source_latents.to(dtype=latents.dtype, device=latents.device)
        source_patches = dit.patchify(source_latents)
        source_f, source_h, source_w = source_patches.shape[2:]
        source_tokens = rearrange(source_patches, "b c f h w -> b (f h w) c").contiguous()
        x_tokens_segments.append(source_tokens)
        freqs_segments.append(_build_freqs(source_f, source_h, source_w, source_tokens.device))
        source_token_count = source_tokens.shape[1]

    if reference_latents is not None:
        if len(reference_latents.shape) == 5:
            reference_latents = reference_latents[:, :, 0]
        reference_latents = reference_latents.to(dtype=latents.dtype, device=latents.device)
        reference_features = dit.ref_conv(reference_latents)
        reference_tokens = reference_features.flatten(2).transpose(1, 2).contiguous()
        x_tokens_segments.insert(0, reference_tokens)
        ref_f = 1
        ref_h, ref_w = reference_features.shape[2], reference_features.shape[3]
        freqs_segments.insert(0, _build_freqs(ref_f, ref_h, ref_w, reference_tokens.device))
        reference_token_count = reference_tokens.shape[1]

    if semantic_tokens_noised is not None:
        semantic_token_count = int(semantic_tokens_noised.shape[1])
        x_tokens_segments.append(semantic_tokens_noised.to(dtype=target_tokens.dtype, device=target_tokens.device))
        freqs_segments.append(
            torch.zeros(
                (semantic_token_count, 1, target_freqs_full.shape[-1]),
                device=target_tokens.device,
                dtype=target_freqs_full.dtype,
            )
        )

    x_tokens_segments.append(target_tokens)
    freqs_segments.append(target_freqs_full)

    x_tokens = torch.cat(x_tokens_segments, dim=1)
    freqs = torch.cat(freqs_segments, dim=0)

    B, total_tokens, D = x_tokens.shape
    cond_token_count = reference_token_count + source_token_count

    type_ids_chunks = []
    if cond_token_count > 0:
        type_ids_chunks.append(torch.zeros(B, cond_token_count, dtype=torch.long, device=x_tokens.device))
    if semantic_token_count > 0:
        type_ids_chunks.append(torch.ones(B, semantic_token_count, dtype=torch.long, device=x_tokens.device))
    type_ids_chunks.append(torch.full((B, target_token_count), 2, dtype=torch.long, device=x_tokens.device))
    type_ids = torch.cat(type_ids_chunks, dim=1)

    seg_embeds = dit.segment_embedding(type_ids)
    x_tokens = x_tokens + seg_embeds

    attn_mask = None
    if use_source_attention_mask and source_token_count > 0:
        total_tokens = x_tokens.shape[1]
        attn_mask = torch.zeros((total_tokens, total_tokens), dtype=torch.bool, device=x_tokens.device)
        src_start = reference_token_count
        src_end = src_start + source_token_count
        attn_mask[src_start:src_end, :] = True
        attn_mask[src_start:src_end, src_start:src_end] = False
        if semantic_token_count > 0:
            sem_start = src_end
            sem_end = sem_start + semantic_token_count
            attn_mask[src_start:src_end, sem_start:sem_end] = False
        if os.environ.get("WAN_DEBUG_ATTENTION_MASK") == "1":
            print(
                f"[WanDebug] attention_mask enabled: total={total_tokens}, "
                f"ref_tokens={reference_token_count}, source_tokens={source_token_count}, "
                f"semantic_tokens={semantic_token_count}, target_tokens={target_token_count}, "
                f"mask_ratio={attn_mask.float().mean().item():.4f}"
            )
        if use_unified_sequence_parallel:
            warnings.warn("Sequence parallel + source attention mask not yet supported; mask disabled.")
            attn_mask = None

    if t_mod.ndim == 4 and t_mod.shape[1] != x_tokens.shape[1]:
        token_gap = x_tokens.shape[1] - t_mod.shape[1]
        if token_gap > 0:
            pad = t_mod[:, :1, :, :].expand(-1, token_gap, -1, -1).contiguous()
            t_mod = torch.cat([pad, t_mod], dim=1)
        elif token_gap < 0:
            t_mod = t_mod[:, -x_tokens.shape[1] :, ...]

    if t.ndim == 3 and t.shape[1] != x_tokens.shape[1]:
        token_gap = x_tokens.shape[1] - t.shape[1]
        if token_gap > 0:
            pad = t[:, :1, :].expand(-1, token_gap, -1).contiguous()
            t = torch.cat([pad, t], dim=1)
        elif token_gap < 0:
            t = t[:, -x_tokens.shape[1] :, :]

    #####check tmod#####
    if os.environ.get("WAN_DEBUG_TMOD") == "1" and int(os.environ.get("RANK", "0")) == 0:
        try:
            t_val = float(timestep.flatten()[0].item()) if timestep is not None else None
            t_shape = tuple(t.shape) if t is not None else None
            tmod_shape = tuple(t_mod.shape) if t_mod is not None else None
            x_shape = tuple(x_tokens.shape) if x_tokens is not None else None
            t_delta = None
            tmod_delta = None
            if t is not None and t.ndim >= 2:
                t_ref = t[:, :1] if t.ndim >= 3 else t[:, :1]
                t_delta = float((t - t_ref).abs().max().item())
            if t_mod is not None and t_mod.ndim >= 3:
                tmod_ref = t_mod[:, :1] if t_mod.ndim >= 4 else t_mod[:, :1]
                tmod_delta = float((t_mod - tmod_ref).abs().max().item())
            print(
                f"[TMOD] t={t_val} t_shape={t_shape} t_mod_shape={tmod_shape} "
                f"x_tokens_shape={x_shape} t_delta={t_delta} t_mod_delta={tmod_delta}",
                flush=True,
            )
        except Exception as exc:
            print(f"[TMOD] debug failed: {exc}", flush=True)
    #####check tmod#####  

    x = x_tokens

    if os.environ.get("WAN_DEBUG_SHAPE") == "1" and int(os.environ.get("RANK", "0")) == 0:
        def _tok_info(toks):
            if toks is None:
                return "None"
            return f"len={toks.shape[1]} dim={toks.shape[2]}"

        print(
            "[SHAPE] tokens_before_dit: "
            f"ref={_tok_info(reference_tokens)}, "
            f"source={_tok_info(source_tokens)}, "
            f"semantic={_tok_info(semantic_tokens_noised)}, "
            f"target={_tok_info(target_tokens)}, "
            f"total=len={x_tokens.shape[1]} dim={x_tokens.shape[2]}",
            flush=True,
        )

    if os.environ.get("WAN_DEBUG_TOKENS") == "1":
        print(
            f"[WanDebug] concat_mode={source_concat_mode}, source_latents={'yes' if source_latents is not None else 'no'}, "
            f"ref_tokens={reference_token_count}, source_tokens={source_token_count}, "
            f"semantic_tokens={semantic_token_count}, target_tokens={target_token_count}, total_tokens={x.shape[1]}"
        )

    tea_cache_update = tea_cache.check(dit, x, t_mod) if tea_cache is not None else False

    if vace_context is not None:
        vace_hints = vace(
            x,
            vace_context,
            context,
            t_mod,
            freqs,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
        )

    if use_unified_sequence_parallel:
        if dist.is_initialized() and dist.get_world_size() > 1:
            chunks = torch.chunk(x, get_sequence_parallel_world_size(), dim=1)
            pad_shape = chunks[0].shape[1] - chunks[-1].shape[1]
            chunks = [
                torch.nn.functional.pad(chunk, (0, 0, 0, chunks[0].shape[1] - chunk.shape[1]), value=0)
                for chunk in chunks
            ]
            x = chunks[get_sequence_parallel_rank()]

    if tea_cache_update:
        x = tea_cache.update(x)
    else:
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward

        for block_id, block in enumerate(dit.blocks):
            if use_gradient_checkpointing_offload:
                with torch.autograd.graph.save_on_cpu():
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x,
                        context,
                        t_mod,
                        freqs,
                        use_reentrant=False,
                    )
            elif use_gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x,
                    context,
                    t_mod,
                    freqs,
                    use_reentrant=False,
                )
            else:
                x = block(x, context, t_mod, freqs)

            if vace_context is not None and block_id in vace.vace_layers_mapping:
                current_vace_hint = vace_hints[vace.vace_layers_mapping[block_id]]
                if use_unified_sequence_parallel and dist.is_initialized() and dist.get_world_size() > 1:
                    current_vace_hint = torch.chunk(current_vace_hint, get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()]
                    current_vace_hint = torch.nn.functional.pad(
                        current_vace_hint, (0, 0, 0, chunks[0].shape[1] - current_vace_hint.shape[1]), value=0
                    )
                x = x + current_vace_hint * vace_scale

            if pose_latents is not None and face_pixel_values is not None:
                x = animate_adapter.after_transformer_block(block_id, x, motion_vec)

        if tea_cache is not None:
            tea_cache.store(x)

    # semantic diffusion pred/loss
    loss_sem = None
    sem_pred = None

    can_sem = True
    if use_unified_sequence_parallel and ("dist" in locals()) and dist.is_initialized() and dist.get_world_size() > 1:
        can_sem = False

    if can_sem and semantic_token_count > 0:
        sem_start = cond_token_count
        sem_end = cond_token_count + semantic_token_count
        sem_hidden = x[:, sem_start:sem_end, :]

        sem_head = getattr(dit, "semantic_head", None)
        if sem_head is None:
            raise RuntimeError("dit.semantic_head is missing. Attach SemanticDiffusionHead (lazy attach in training_loss or from_pretrained).")

        sem_pred = sem_head(sem_hidden)

        if return_sem_loss and (semantic_target is not None):
            sem_tgt = semantic_target.to(device=sem_pred.device, dtype=sem_pred.dtype)
            loss_sem = torch.nn.functional.l1_loss(sem_pred.float(), sem_tgt.float()) #l1
            # loss_sem = torch.nn.functional.mse_loss(sem_pred.float(), sem_tgt.float())

    x = dit.head(x, t)

    if use_unified_sequence_parallel:
        if dist.is_initialized() and dist.get_world_size() > 1:
            x = get_sp_group().all_gather(x, dim=1)
            x = x[:, :-pad_shape] if pad_shape > 0 else x

    x = x[:, -target_token_count:]
    x = dit.unpatchify(x, (target_f, target_h, target_w))

    if loss_sem is None:
        loss_sem = x.new_zeros(())

    if return_sem_pred and return_sem_loss:
        return x, sem_pred, loss_sem
    if return_sem_pred:
        return x, sem_pred
    if return_sem_loss:
        return x, loss_sem
    return x
