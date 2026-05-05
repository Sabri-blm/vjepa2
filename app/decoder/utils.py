# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import sys
import warnings

import torch
import yaml

import src.models.decoder as vit_dec
import src.models.vision_transformer as video_vit
from src.utils.checkpoint_loader import robust_checkpoint_loader
from src.utils.schedulers import CosineWDSchedule, LinearDecaySchedule, WarmupCosineSchedule
from src.utils.wrappers import MultiSeqWrapper, PredictorMultiSeqWrapper

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

MAX_RETRIES = 3


def build_eval_args(
    model_name,
    patch_size,
    tubelet_size,
    num_frames,
    logging_folder,
    checkpoint,
    write_tag,
    eval_cfg_paths,
    uniform_power=False,
    use_sdpa=False,
    clip_duration=None,
    use_silu=False,
    wide_silu=True,
    tag=None,
):
    """
    Helper function to parse the pre-training configs to construct the
    evaluation configs, return as a list of eval configs.
    """
    # By convention, the pre-training config should specify any required evals
    # in the 'evals' key
    if eval_cfg_paths is None:
        logger.info("No evaluations specified!")
        return

    eval_nodes = None
    eval_tasks_per_node = None
    args_eval = []
    for i, f in enumerate(eval_cfg_paths):
        with open(f, "r") as y_file:
            _args = yaml.load(y_file, Loader=yaml.FullLoader)
            _tag = _args.get("tag", "")
            _args["tag"] = f"{tag}-{_tag}"
            _nodes = _args.get("nodes", None)
            _tasks = _args.get("tasks_per_node", 8)
            eval_nodes = _nodes if eval_nodes is None else eval_nodes
            eval_tasks_per_node = _tasks if eval_tasks_per_node is None else eval_tasks_per_node
            if (eval_nodes != _nodes) or (eval_tasks_per_node != _tasks):
                warnings.warn("Configs for online evals must use same number of nodes for slurm-batch processing")

            # Model params
            _args["pretrain"] = {}
            _args["pretrain"]["model_name"] = model_name
            _args["pretrain"]["patch_size"] = patch_size
            _args["pretrain"]["tubelet_size"] = tubelet_size
            _args["pretrain"]["uniform_power"] = uniform_power
            _args["pretrain"]["use_sdpa"] = use_sdpa
            _args["pretrain"]["clip_duration"] = clip_duration
            _args["pretrain"]["use_silu"] = use_silu
            _args["pretrain"]["wide_silu"] = wide_silu

            # Data params
            _args["pretrain"]["frames_per_clip"] = num_frames

            # Misc
            _args["pretrain"]["folder"] = logging_folder
            _args["pretrain"]["checkpoint"] = checkpoint
            _args["pretrain"]["write_tag"] = write_tag

            args_eval += [_args]

    return eval_nodes, eval_tasks_per_node, args_eval

def load_pretrained(
    r_path,
    encoder=None,
    predictor=None,
    target_encoder=None,
    context_encoder_key="encoder",
    target_encoder_key="target_encoder",
    load_predictor=False,
    load_encoder=True,
):
    logger.info(f"Loading pretrained model from {r_path}")
    checkpoint = robust_checkpoint_loader(r_path, map_location=torch.device("cpu"))

    epoch = checkpoint["epoch"]

    if load_encoder and encoder:
        # -- loading encoder
        pretrained_dict = checkpoint[context_encoder_key]
        pretrained_dict = {k.replace("backbone.", ""): v for k, v in pretrained_dict.items()}
        msg = encoder.load_state_dict(pretrained_dict, strict=False)
        logger.info(f"loaded pretrained encoder from epoch {epoch} with msg: {msg}")

    if load_predictor:
        # -- loading predictor
        pretrained_dict = checkpoint["predictor"]
        pretrained_dict = {k.replace("backbone.", ""): v for k, v in pretrained_dict.items()}
        msg = predictor.load_state_dict(pretrained_dict, strict=False)
        logger.info(f"loaded pretrained predictor from epoch {epoch} with msg: {msg}")

    # -- loading target_encoder
    if load_encoder:
        if target_encoder is not None:
            print(list(checkpoint.keys()))
            pretrained_dict = checkpoint[target_encoder_key]
            pretrained_dict = {k.replace("backbone.", ""): v for k, v in pretrained_dict.items()}
            msg = target_encoder.load_state_dict(pretrained_dict, strict=False)
            logger.info(f"loaded pretrained target encoder from epoch {epoch} with msg: {msg}")

    del checkpoint

    return (
        encoder,
        predictor,
        target_encoder,
    )

def load_checkpoint(
    r_path,
    encoder,
    decoder,
    target_encoder,
    opt,
    scaler,
    is_anneal=False,
):
    logger.info(f"Loading checkpoint from {r_path}")
    checkpoint = robust_checkpoint_loader(r_path, map_location=torch.device("cpu"))

    epoch = 0
    if not is_anneal:
        epoch = checkpoint["epoch"]

    # -- loading encoder
    pretrained_dict = checkpoint["encoder"]
    msg = encoder.load_state_dict(pretrained_dict)
    logger.info(f"loaded pretrained encoder from epoch {epoch} with msg: {msg}")

    # -- loading decoder
    pretrained_dict = checkpoint["decoder"]
    msg = decoder.load_state_dict(pretrained_dict)
    logger.info(f"loaded pretrained decoder from epoch {epoch} with msg: {msg}")

    # -- loading target_encoder
    if target_encoder is not None:
        print(list(checkpoint.keys()))
        pretrained_dict = checkpoint["target_encoder"]
        msg = target_encoder.load_state_dict(pretrained_dict)
        logger.info(f"loaded pretrained target encoder from epoch {epoch} with msg: {msg}")

    # -- loading optimizer
    opt.load_state_dict(checkpoint["opt"])
    if scaler is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    logger.info(f"loaded optimizers from epoch {epoch}")
    logger.info(f"read-path: {r_path}")
    del checkpoint

    return (
        encoder,
        decoder,
        target_encoder,
        opt,
        scaler,
        epoch,
    )


def init_video_model(
    device,
    patch_size=16,
    max_num_frames=16,
    tubelet_size=2,
    model_name="vit_base",
    crop_size=224,
    depth=6,
    num_heads=None,
    decoder_embed_dim=384,
    uniform_power=False,
    use_silu=False,
    use_decoder_silu=False,
    wide_silu=False,
    use_activation_checkpointing=False,
    in_chans=3,
    use_sdpa=False,
    use_rope=True,
    c=2
):
    encoder = video_vit.__dict__[model_name](
        img_size=crop_size,
        patch_size=patch_size,
        num_frames=max_num_frames,
        tubelet_size=tubelet_size,
        uniform_power=uniform_power,
        use_sdpa=use_sdpa,
        use_silu=use_silu,
        wide_silu=wide_silu,
        use_activation_checkpointing=use_activation_checkpointing,
        use_rope=use_rope,
        in_chans=in_chans
    )
    #encoder = MultiSeqWrapper(encoder)
    decoder = vit_dec.__dict__["vit_decoder"](
        img_size=(crop_size, crop_size),
        patch_size=patch_size,
        embed_dim=encoder.embed_dim,
        decoder_embed_dim=decoder_embed_dim,
        depth=depth,
        num_heads=encoder.num_heads if num_heads is None else num_heads,
        use_silu=use_decoder_silu,
        wide_silu=wide_silu,
        use_activation_checkpointing=use_activation_checkpointing,
        num_frames=max_num_frames,
        tubelet_size=tubelet_size,
        use_rope=False,
        c=c
        #use_sdpa=use_sdpa,
    )
    #predictor = PredictorMultiSeqWrapper(predictor)
    encoder.to(device)
    decoder.to(device)
    logger.info(encoder)
    logger.info(decoder)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"device is: {device}")
    '''print("proj:", next(decoder.pixel_decoder.proj.parameters()).device)
    print("upsampler:", next(decoder.pixel_decoder.upsampler[0][1].parameters()).device)
    print("out_conv:", next(decoder.pixel_decoder.out_conv.parameters()).device)'''
    logger.info(f"Encoder number of parameters: {count_parameters(encoder)}")
    logger.info(f"decoder number of parameters: {count_parameters(decoder)}")

    return encoder, decoder


def init_opt(
    is_anneal,
    #encoder,
    decoder,
    iterations_per_epoch,
    start_lr,
    ref_lr,
    warmup,
    num_epochs,
    wd=1e-6,
    final_wd=1e-6,
    final_lr=0.0,
    mixed_precision=False,
    ipe_scale=1.25,
    betas=(0.9, 0.999),
    eps=1e-8,
    zero_init_bias_wd=True,
):
    param_groups = [
        {"params": (p for n, p in decoder.named_parameters() if ("bias" not in n) and (len(p.shape) != 1))},
        {
            "params": (p for n, p in decoder.named_parameters() if ("bias" in n) or (len(p.shape) == 1)),
            "WD_exclude": zero_init_bias_wd,
            "weight_decay": 0,
        },
    ]

    optimizer = torch.optim.AdamW(param_groups, betas=betas, eps=eps)
    if not is_anneal:
        scheduler = WarmupCosineSchedule(
            optimizer,
            warmup_steps=int(warmup * iterations_per_epoch),
            start_lr=start_lr,
            ref_lr=ref_lr,
            final_lr=final_lr,
            T_max=int(ipe_scale * num_epochs * iterations_per_epoch),
        )
    else:
        scheduler = LinearDecaySchedule(
            optimizer,
            ref_lr=ref_lr,
            final_lr=final_lr,
            T_max=int(ipe_scale * num_epochs * iterations_per_epoch),
        )
    wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=wd,
        final_wd=final_wd,
        T_max=int(ipe_scale * num_epochs * iterations_per_epoch),
    )
    # torch.cuda.GradScaler() is depricatedd in PyTorch2.2+
    #scaler = torch.cuda.amp.GradScaler() if mixed_precision else None
    scaler = torch.amp.GradScaler('cuda') if mixed_precision else None
    return optimizer, scaler, scheduler, wd_scheduler
