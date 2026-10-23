"""
SegFormer for multi-band input, e.g. 64-dim AlphaEarth satellite embeddings.

Ported from UCSD-E4E/ml-mangrove (DroneClassification/models/models.py: SegFormer,
SegformerDecodeCore) with one change: `in_channels` replaces the stage-1 patch-embedding conv
so the pretrained MiT encoder takes all bands natively (see SegFormer.__init__). Requires
transformers<5 (SegformerDecodeCore relies on the 4.x decode-head layout).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerDecodeHead, SegformerModel


class SegFormer(nn.Module):
    """
    SegFormer model for semantic segmentation from https://github.com/NVlabs/SegFormer.

    Available pretrained weights:
    - "nvidia/segformer-b0-finetuned-ade-512-512"
    - "nvidia/segformer-b1-finetuned-ade-512-512"
    - "nvidia/segformer-b2-finetuned-ade-512-512"
    - "nvidia/segformer-b3-finetuned-ade-512-512"
    - "nvidia/segformer-b4-finetuned-ade-512-512"
    - "nvidia/segformer-b5-finetuned-ade-640-640"

    - "nvidia/segformer-b0-finetuned-cityscapes-512-1024"
    - "nvidia/segformer-b0-finetuned-cityscapes-640-1280"
    - "nvidia/segformer-b0-finetuned-cityscapes-768-768"
    - "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
    - "nvidia/segformer-b1-finetuned-cityscapes-1024-1024"
    - "nvidia/segformer-b2-finetuned-cityscapes-1024-1024"
    - "nvidia/segformer-b3-finetuned-cityscapes-1024-1024"
    - "nvidia/segformer-b4-finetuned-cityscapes-1024-1024"
    - "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"
    
    """
    def __init__(self, num_classes=1, input_image_size=512, weights="nvidia/segformer-b2-finetuned-ade-512-512",
                 in_channels=3, stem_init="random", stem_stride=None):
        """
        Args:
            in_channels: input bands. 3 keeps the pretrained RGB stem (and the legacy behavior of
                truncating >3-band input to the first 3). Any other value replaces the stage-1
                patch-embedding conv so the encoder takes all bands natively, e.g. 64 for
                AlphaEarth / Google Satellite Embeddings. All other pretrained weights are kept.
            stem_init: init for a replaced stem. "random" (kaiming; recommended for non-RGB inputs
                such as embeddings) or "mean" (pretrained RGB kernel averaged over channels,
                tiled to in_channels and scaled by 3/in_channels).
            stem_stride: optional stride for the stage-1 patch embedding (pretrained: 4). Lower it
                (e.g. 2 or 1) for coarse inputs such as 10 m embeddings, where 4x downsampling
                leaves stage 1 at 40 m. Costs memory: stage-1 tokens scale with 1/stride^2.
        """
        super().__init__()
        self.num_classes = num_classes
        self.input_image_size = input_image_size
        self.weights = weights
        self.in_channels = in_channels

        # --- backbone: MiT encoder only ---
        self.backbone = SegformerModel.from_pretrained(weights)
        config = self.backbone.config
        if in_channels != 3 or stem_stride is not None:
            self._replace_stem(in_channels, stem_init, stem_stride)

        # --- create a decode_head from config and reuse its components ---
        hf_decode_head = SegformerDecodeHead(config)
        hf_decode_head.classifier = nn.Identity() # type: ignore
        self.decode_core = SegformerDecodeCore(hf_decode_head)

        # --- classifier head ---
        self.hidden_size = config.decoder_hidden_size
        self.classifier = nn.Sequential(
            nn.ConvTranspose2d(self.hidden_size, self.hidden_size // 2, kernel_size=4, stride=2, padding=1), 
            nn.BatchNorm2d(self.hidden_size // 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.hidden_size // 2, self.hidden_size // 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.hidden_size // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_size // 4, num_classes, kernel_size=3, padding=1),
        )

    def _get_backbone_features(self, x):
        # call backbone asking for hidden_states
        # HuggingFace SegformerModel returns hidden_states as a tuple of stage outputs when output_hidden_states=True
        outputs = self.backbone(x, output_hidden_states=True)
        if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            hs = list(outputs.hidden_states)
            # Many HF versions include embeddings; the stage outputs are often the last 4 entries.
            if len(hs) >= 4:
                hs = hs[-4:]
            return hs
        # Fallback: if outputs.last_hidden_state is already a fused map, try to adapt
        if hasattr(outputs, "last_hidden_state"):
            # some versions may return (B, seq_len, hidden) — attempt to reshape to (B, hidden, H, W)
            lh = outputs.last_hidden_state
            # If it looks like a feature map already (4D), return as single element list
            if lh.dim() == 4:
                return [lh]
        raise RuntimeError("Couldn't extract backbone hidden states. Inspect `outputs` returned by SegformerModel.")

    def _replace_stem(self, in_channels, stem_init, stem_stride):
        """Swap the stage-1 patch-embedding conv (the only conv that sees raw input channels).
        Located by search so it works across transformers versions (encoder.patch_embeddings[0]
        in 4.x, stages[0].patch_embeddings in 5.x)."""
        name, old = next((n, m) for n, m in self.backbone.named_modules()
                         if isinstance(m, nn.Conv2d) and m.in_channels == self.backbone.config.num_channels)
        stride = stem_stride if stem_stride is not None else old.stride
        new = nn.Conv2d(in_channels, old.out_channels, kernel_size=old.kernel_size, stride=stride,
                        padding=old.padding, bias=old.bias is not None)
        with torch.no_grad():
            if in_channels == old.in_channels:
                new.weight.copy_(old.weight)          # stride change only: keep pretrained kernel
            elif stem_init == "mean":
                w = old.weight.mean(dim=1, keepdim=True).repeat(1, in_channels, 1, 1)
                new.weight.copy_(w * (old.in_channels / in_channels))
            elif stem_init != "random":
                raise ValueError(f"stem_init must be 'random' or 'mean', got {stem_init!r}")
            if old.bias is not None:
                new.bias.copy_(old.bias)
        parent = self.backbone.get_submodule(name.rsplit(".", 1)[0])
        setattr(parent, name.rsplit(".", 1)[1], new)
        self.backbone.config.num_channels = in_channels

    def forward(self, image):
        # legacy: a 3-channel model truncates extra bands (e.g. RGBN -> RGB)
        if self.in_channels == 3 and image.shape[1] > 3:
            image = image[:, :3, :, :]
        elif image.shape[1] != self.in_channels:
            raise ValueError(f"expected {self.in_channels} input channels, got {image.shape[1]}")

        features = self._get_backbone_features(image)
        core = self.decode_core(features)  # (B, hidden_size, H/4, W/4)
        out = self.classifier(core)

        # upsample to input size
        if out.shape[-2:] != image.shape[-2:]:
            out = F.interpolate(out, size=image.shape[-2:], mode="bilinear", align_corners=False)
        return out


class SegformerDecodeCore(nn.Module):
    """
    Recreate SegFormer decode head up to (but not including) classifier.
    Uses safe reshape/permute operations to avoid .view() stride errors.
    """
    def __init__(self, decode_head):
        super().__init__()
        # reuse the modules from HF decode head
        self.linear_c = decode_head.linear_c       # ModuleList of SegformerMLP
        self.linear_fuse = decode_head.linear_fuse # Conv2d
        self.batch_norm = decode_head.batch_norm
        self.activation = decode_head.activation
        self.dropout = decode_head.dropout

    def forward(self, features, debug=False):
        # features: list/tuple of 4 tensors: [(B,C1,H1,W1), (B,C2,H2,W2), ...]
        # target spatial size: use features[0] (largest spatial)
        target_h, target_w = features[0].shape[-2], features[0].shape[-1]
        projected = []
        for i, proj_mlp in enumerate(self.linear_c):
            x = features[i]  # (B, Ci, Hi, Wi)
            if (x.shape[-2], x.shape[-1]) != (target_h, target_w):
                x = F.interpolate(x, size=(target_h, target_w), mode="bilinear", align_corners=False)
            
            # safe flatten+linear: (B, Ci, H*W) -> (B, H*W, Ci) -> proj -> (B, H*W, hidden) -> back
            b, c, h, w = x.shape
            x_flat = x.reshape(b, c, h*w).permute(0, 2, 1)   # (B, H*W, Ci)
            # Ensure contiguous before Linear just in case
            if not x_flat.is_contiguous():
                x_flat = x_flat.contiguous()
            x_proj = proj_mlp.proj(x_flat)                   # (B, H*W, hidden)
            x_proj = x_proj.permute(0, 2, 1).reshape(b, -1, h, w)  # (B, hidden, H, W)
            x_proj = x_proj.contiguous() if not x_proj.is_contiguous() else x_proj
            projected.append(x_proj)

            if debug:
                print(f"proj[{i}] -> shape {x_proj.shape}, contiguous={x_proj.is_contiguous()}")

        x = torch.cat(projected, dim=1)  # (B, sum(hidden), H, W)
        x = self.linear_fuse(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = x.contiguous() if not x.is_contiguous() else x
        if debug:
            print("fused ->", x.shape, "contig=", x.is_contiguous())
        return x
