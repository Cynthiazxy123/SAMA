# siglip_semantic.py
import math
import os
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


class SigLIPFirstFrameEncoder:
    def __init__(self, model_path_or_id, device="cpu", dtype=torch.bfloat16):
        self.model_path_or_id = model_path_or_id
        self.device = torch.device(device)
        self.dtype = dtype
        self._processor = None
        self._model = None

    def _lazy_init(self):
        if self._model is not None and self._processor is not None:
            return
        from transformers import AutoModel, AutoProcessor

        self._processor = AutoProcessor.from_pretrained(self.model_path_or_id)
        model = AutoModel.from_pretrained(self.model_path_or_id, torch_dtype=self.dtype)
        self._model = model.vision_model
        self._model.eval()
        self._model.to(self.device)
        del model

    def to(
        self,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        if device is not None:
            self.device = torch.device(device)
        if dtype is not None:
            self.dtype = dtype
        if self._model is not None:
            self._model.to(self.device, dtype=self.dtype)
        return self

    @torch.no_grad()
    def encode_first_frame(
        self,
        input_video: Union[List[Image.Image], List[List[Image.Image]]],
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """
        input_video:
          - single: List[PIL]           (T)
          - batch:  List[List[PIL]]     (B,T)
        return:
          semantic_tokens: [B, N, D_in]
        """
        self._lazy_init()
        if device is not None or dtype is not None:
            self.to(device=device, dtype=dtype)

        # pick first frame(s)
        if isinstance(input_video, list) and len(input_video) > 0 and isinstance(input_video[0], Image.Image):
            frames = [input_video[0]]
        elif isinstance(input_video, list) and len(input_video) > 0 and isinstance(input_video[0], (list, tuple)):
            frames = [v[0] for v in input_video]
        else:
            raise TypeError("input_video must be List[PIL] or List[List[PIL]].")

        proc = self._processor

        inputs = proc(images=frames, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device, dtype=self.dtype)
        # compatible with SiglipModel vs SiglipVisionModel
        out = self._model(pixel_values=pixel_values, return_dict=True)
        tokens = out.last_hidden_state  # [B, N, D_in]
        return tokens


class SigLIPSemanticProjector(nn.Module):
    """
    Encode the first frame with SigLIP and return raw patch tokens (B, N, D_in).
    """

    def __init__(
        self,
        model_path_or_id: str,
        out_channels: int = 16,
        hidden_dim: int = 1024,
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        if model_path_or_id is None:
            raise ValueError("`model_path_or_id` must be provided for SigLIPSemanticProjector.")
        self.encoder = SigLIPFirstFrameEncoder(model_path_or_id, device=device, dtype=dtype)
        self.out_channels = out_channels
        self.hidden_dim = hidden_dim

    def to(self, *args, **kwargs):  # type: ignore[override]
        device = kwargs.get("device", None)
        if device is None and len(args) > 0:
            device = args[0]
        dtype = kwargs.get("dtype", None)
        self.encoder.to(device=device, dtype=dtype)
        return super().to(*args, **kwargs)

    def forward(
        self,
        input_video: Union[List[Image.Image], List[List[Image.Image]]],
        target_spatial: Optional[tuple[int, int]] = None,  # kept for API compatibility; ignored
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        tokens = self.encoder.encode_first_frame(input_video, device=device, dtype=dtype)
        b, n, _ = tokens.shape

        # Drop the first token if it looks like a CLS token
        if int(math.sqrt(max(n - 1, 1))) ** 2 == n - 1:
            tokens = tokens[:, 1:]
            n = tokens.shape[1]

        target_device = device or self.encoder.device
        target_dtype = dtype or self.encoder.dtype
        return tokens.to(device=target_device, dtype=target_dtype)
