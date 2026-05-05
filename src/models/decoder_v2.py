import math
import torch
import torch.nn as nn
from functools import partial


from src.models.utils.modules import Block
from src.models.utils.pos_embs import get_2d_sincos_pos_embed, get_3d_sincos_pos_embed
from src.utils.tensors import trunc_normal_


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
        self.head = nn.Sequential(
            nn.Linear(decoder_embed_dim, decoder_embed_dim),
            nn.GELU(),
            nn.Linear(decoder_embed_dim, self.out_embed_dim),
        )


        # Refinement head (Path A)
        self.refiner = nn.Sequential(
            nn.Conv2d(self.channels * self.num_frames, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, self.channels * self.num_frames, kernel_size=3, padding=1),
        )


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
            pos = self.decoder_pos_emb.repeat(B, T, 1)   # [1, T*H_p², D]
            x += pos

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
        x = self.head(self.decoder_norm(x))                  # [B, N, P*P]
        #grid_size = self.img_height // self.patch_size  # H_p

        x = x.view(
            B,
            T,
            self.grid_height,
            self.grid_height,
            self.channels,
            self.patch_size,
            self.patch_size,
        )  # [B, T, H_p, W_p, C, P, P]
        
        x = x.permute(0, 4, 1, 2, 5, 3, 6).contiguous()
        # [B, C, T, H_p, P, W_p, P]
        
        x = x.view(B, self.channels, T, self.img_height, self.img_height)

        #print("post-head variance:", x.var(dim=[3,4]).mean())

        # Merge frames into channels for refinement
        x_merged = x.view(B, self.channels * T, self.img_height, self.img_height)
        
        # Apply refinement CNN
        x_refined = self.refiner(x_merged)
        
        # Un-merge frames
        x = x_refined.view(B, self.channels, T, self.img_height, self.img_height)

        return x

def vit_decoder(**kwargs):
    model = FloodMaskDecoder(
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model