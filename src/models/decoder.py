import math
import torch
import torch.nn as nn
from functools import partial

import torch.nn.functional as F

from src.models.utils.modules import Block
from src.models.utils.pos_embs import get_2d_sincos_pos_embed, get_3d_sincos_pos_embed
from src.utils.tensors import trunc_normal_
import time


class PixelDecoder3D(nn.Module):
    """
    Takes transformer tokens [B, N, D] with N = T * H_p * W_p,
    reshapes to [B, D, T, H_p, W_p], then upsamples spatially to [B, C_out, T, H, W].
    """
    def __init__(self, in_dim, out_channels, patch_size, base_dim=256):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(in_dim, base_dim, kernel_size=1)

        # we'll upsample only H,W; keep T as is
        #num_ups = int(torch.log2(torch.tensor(patch_size)).item())
        num_ups = int(math.log2(patch_size))

        blocks = []
        for _ in range(num_ups):
            blocks.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=(1, 2, 2), mode="nearest"),#, align_corners=False),
                    nn.Conv3d(base_dim, base_dim, kernel_size=3, padding=1),
                    nn.GELU(),
                    nn.Conv3d(base_dim, base_dim, kernel_size=3, padding=1),
                    nn.GELU(),
                )
            )
        self.upsampler = nn.Sequential(*blocks)
        self.out_conv = nn.Conv3d(base_dim, out_channels, kernel_size=1)

        print("proj device:", next(self.proj.parameters()).device)
        print("upsampler device:", next(self.upsampler[0][1].parameters()).device)
        print("out_conv device:", next(self.out_conv.parameters()).device)


    def forward(self, x, T, H_p, W_p):
        start = time.time()   # ⏱️ START TIMER
        # x: [B, N, D], N = T * H_p * W_p
        B, N, D = x.shape
        assert N == T * H_p * W_p

        x = x.view(B, T, H_p, W_p, D).permute(0, 4, 1, 2, 3).contiguous()  # [B, D, T, H_p, W_p]
        x = self.proj(x)                                                   # [B, C, T, H_p, W_p]
        x = self.upsampler(x)                                              # [B, C, T, H, W]
        x = self.out_conv(x)                                               # [B, out_ch, T, H, W]

        end = time.time()     # ⏱️ END TIMER
        print(f"[PixelDecoder3D] forward time: {(end - start)*1000:.2f} ms")
        return x

class PixelDecoder2D_learned(nn.Module):
    def __init__(self, in_dim, out_channels, patch_size):
        super().__init__()
        self.patch_size = patch_size
        self.to_pixel = nn.ConvTranspose2d(
            in_dim, out_channels,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x, T, H_p, W_p):
        # x: [B, N, D]
        B, N, D = x.shape
        x = x.view(B, T, H_p, W_p, D).permute(0, 1, 4, 2, 3)
        # x: [B, T, D, H_p, W_p]

        # merge batch and time
        x = x.reshape(B * T, D, H_p, W_p)

        # apply ConvTranspose2d
        x = self.to_pixel(x).contiguous()  # [B*T, C, H, W]

        # un-merge
        x = x.view(B, T, -1, x.shape[-2], x.shape[-1])
        x = x.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
        return x

class PixelDecoder2D_MultiStage(nn.Module):
    def __init__(self, in_dim, base_dim=96, out_channels=1):
        super().__init__()

        # force dependence on transformer features
        self.proj = nn.Sequential(
            nn.Conv2d(in_dim, base_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(base_dim, base_dim, kernel_size=1),
        )

        self.norm_proj = nn.GroupNorm(8, base_dim)

        # light multi-stage upsampler
        self.stages = nn.ModuleList([
            self._make_stage(base_dim),
            self._make_stage(base_dim),
            self._make_stage(base_dim),
            self._make_stage(base_dim),
        ])

        # coarse prediction at low res (for residual)
        self.coarse_head = nn.Conv2d(base_dim, out_channels, kernel_size=1)

        # final prediction at full res
        self.out_conv = nn.Conv2d(base_dim, out_channels, kernel_size=1)

        # NEW: high‑frequency refinement branch
        self.norm_refine = nn.GroupNorm(8, base_dim)
        self.refine = nn.Sequential(
            nn.Conv2d(base_dim, base_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(base_dim, out_channels, kernel_size=1),
        )

        # NEW: skip‑refine head for early high‑freq features
        '''self.skip_refine = nn.Sequential(
            nn.Conv2d(base_dim, base_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(base_dim, out_channels, kernel_size=1),
        )'''

    def _make_stage(self, dim):
        return nn.Sequential(
            nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2),
            nn.GELU(),
            #nn.GroupNorm(8, dim),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.GELU(),
            #nn.GroupNorm(8, dim),
        )

    def forward(self, x, T, H_p, W_p):
        B, N, D = x.shape
        x = x.view(B, T, H_p, W_p, D).permute(0, 1, 4, 2, 3)
        x = x.reshape(B * T, D, H_p, W_p)

        x = self.proj(x)
        x = self.norm_proj(x)

        # coarse map at low resolution
        coarse = self.coarse_head(x)  # [B*T, 1, H_p, W_p]

        for stage in self.stages:
            x = stage(x)

        # upsample coarse to full res and add as residual
        coarse_up = F.interpolate(coarse, size=x.shape[-2:], mode="bilinear", align_corners=False)
        features = x
        x = self.out_conv(x) + coarse_up  # residual prediction

        features = self.norm_refine(features)
        x = x + self.refine(features)   # <-- sharpen

        x = x.view(B, T, -1, x.shape[-2], x.shape[-1]).permute(0, 2, 1, 3, 4)
        return x
    '''def forward(self, x, T, H_p, W_p):
        B, N, D = x.shape
        x = x.view(B, T, H_p, W_p, D).permute(0, 1, 4, 2, 3)
        x = x.reshape(B * T, D, H_p, W_p)

        x = self.proj(x)
        x = self.norm_proj(x)

        # ---- LOW-RES SKIP (cheap, high-frequency) ----
        skip = x  # H_p × W_p

        coarse = self.coarse_head(x)

        for stage in self.stages:
            x = stage(x)

        features = x
        coarse_up = F.interpolate(coarse, size=features.shape[-2:], mode="bilinear", align_corners=False)
        x = self.out_conv(features) + coarse_up

        # ---- ONLY REFINE: skip refine ----
        skip_up = F.interpolate(skip, size=x.shape[-2:], mode="bilinear", align_corners=False)
        x = x + self.skip_refine(skip_up)

        x = x.view(B, T, -1, x.shape[-2], x.shape[-1]).permute(0, 2, 1, 3, 4)
        return x'''

class FloodMaskDecoder(nn.Module):
    def __init__(
        self,
        img_size=(256, 256),
        patch_size=14,
        num_frames=16,
        tubelet_size=2,
        embed_dim=768,
        decoder_embed_dim=1024,
        out_embed_dim=256,
        depth=6,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        drop_path_rate=0.0,
        attn_drop_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        uniform_power=False,
        use_silu=False,
        wide_silu=True,
        use_activation_checkpointing=False,
        use_rope=True,

        decoder_type="2d",
        only_flood=False,
        c=2,
        dropout=0.0,
    ):
        super().__init__()
        self.img_height = img_size[0]
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.is_video = self.num_frames > 1
        self.only_flood = only_flood
        
        self.grid_height = self.img_height // self.patch_size
        self.grid_depth = self.num_frames # we want depth to be 16 frames // tubelet_size
        self.use_activation_checkpointing = use_activation_checkpointing
        
        self._N = (self.num_frames) * self.grid_height ** 2
        #self.grid_height = img_size[0] // patch_size
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.uniform_power = uniform_power

        self.decoder_pos_emb = None
        if not use_rope:
            self.decoder_pos_emb = nn.Parameter(torch.zeros(1, self._N, decoder_embed_dim), requires_grad=False)
        self.use_rope = use_rope

        # Optional input projection (in case encoder dim != decoder dim you want)
        self.input_proj = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    use_rope=use_rope,
                    grid_size=self.grid_height,
                    grid_depth=self.grid_depth,
                    dim=decoder_embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    act_layer=nn.SiLU if use_silu else nn.GELU,
                    wide_silu=wide_silu,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

        self.decoder_norm = norm_layer(decoder_embed_dim)
        # Final head: map embedding -> flattened mask
        self.channels = c if not only_flood else 1
        self.out_embed_dim = self.channels * (self.patch_size ** 2)
        #self.head = nn.Linear(decoder_embed_dim, self.out_embed_dim, bias=True)

        # Choose decoder type
        self.decoder_type = decoder_type.lower()

        if self.decoder_type == "3d":
            self.pixel_decoder = PixelDecoder3D(
                in_dim=decoder_embed_dim,
                out_channels=self.channels,
                patch_size=self.patch_size,
                base_dim=256,
            )
        elif self.decoder_type == "2d":
            '''self.pixel_decoder = PixelDecoder2D_learned(
                in_dim=decoder_embed_dim,
                out_channels=self.channels,
                patch_size=self.patch_size,
            )'''
            self.pixel_decoder = PixelDecoder2D_MultiStage(
                in_dim=decoder_embed_dim,
                base_dim=128,
                out_channels=self.channels,
            )
        else:
            raise ValueError(f"Unknown decoder_type: {decoder_type}. Use '3d' or '2d'.")



        if self.decoder_pos_emb is not None:
            self._init_pos_embed(self.decoder_pos_emb.data)

        # ------ initialize weights
        self.init_std = init_std
        self.apply(self._init_weights)
        self._rescale_blocks()

    def _init_pos_embed(self, pos_embed):
        embed_dim = pos_embed.size(-1)
        grid_size = self.img_height // self.patch_size  # TODO: update; currently assumes square input
        if self.is_video:
            grid_depth = self.num_frames # we need the frames to be 16 remember! // self.tubelet_size
            sincos = get_3d_sincos_pos_embed(
                embed_dim, grid_size, grid_depth, cls_token=False, uniform_power=self.uniform_power
            )
        else:
            sincos = get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False)
        pos_embed.copy_(torch.from_numpy(sincos).float().unsqueeze(0))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _rescale_blocks(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.decoder_blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        """
        emb: [B, N, D]  (single embedding per frame)
        returns: [B, 1, H, W] logits
        """
        x = self.input_proj(emb)          # [B, N, D]

        B, N, PP = x.shape
        T = N // (self.grid_height * self.grid_height)

        assert (N == self._N) and T == 16, f"Expected T = 16, and N = {self._N}. But got: T = {T}, and N = {N}."
        #x = x.unsqueeze(1)                # [B, 1, D]  -> single "token"
        if not self.use_rope:
            #pos = self.decoder_pos_emb.repeat(1, T, 1)   # [1, T*H_p², D]
            #print(self.decoder_pos_emb.shape)
            x += self.decoder_pos_emb

        # Fwd prop
        for i, blk in enumerate(self.decoder_blocks):
            if self.use_activation_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    blk,
                    x,
                    None,
                    None,
                    use_reentrant=False,
                )
            else:
                x = blk(
                    x,
                    mask=None,
                    attn_mask=None,
                )

        #print("pre-head variance:", x.var(dim=1).mean())

        #x = x[:, -1]                       # [B, N*D]
        x = self.decoder_norm(x)                  # [B, N, P*P]

        x = self.pixel_decoder(x, self.num_frames, self.grid_height, self.grid_height)  # [B, C, T, H, W]

        return x

def vit_decoder(**kwargs):
    model = FloodMaskDecoder(
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model