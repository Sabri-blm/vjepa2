# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

# -- FOR DISTRIBUTED TRAINING ENSURE ONLY 1 DEVICE VISIBLE PER PROCESS
try:
    # -- WARNING: IF DOING DISTRIBUTED TRAINING ON A NON-SLURM CLUSTER, MAKE
    # --          SURE TO UPDATE THIS TO GET LOCAL-RANK ON NODE, OR ENSURE
    # --          THAT YOUR JOBS ARE LAUNCHED WITH ONLY 1 DEVICE VISIBLE
    # --          TO EACH PROCESS
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["SLURM_LOCALID"]
except Exception:
    pass

import copy
import gc
import random
import time
from tqdm import tqdm

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
import torch.nn.functional as F
from torchmetrics.functional import structural_similarity_index_measure as ssim



from app.vjepa.transforms import make_transforms, GeoVideoTransformWithCrop
from app.decoder.utils import init_opt, init_video_model, load_checkpoint, load_pretrained
from src.datasets.data_manager import init_data
from src.datasets.geonpy_loader import make_geonpy_loader
from src.masks.multiseq_multiblock3d import MaskCollator
from src.masks.utils import apply_masks
from src.utils.distributed import init_distributed
from src.utils.logging import AverageMeter, CSVLogger, get_logger, gpu_timer

import wandb

wandb.require("service") # prevents multiprocessing deadlocks

# --
log_timings = True
log_freq = 10
CHECKPOINT_FREQ = 1
GARBAGE_COLLECT_ITR_FREQ = 50
# --
eval_freq = 10
# --

_GLOBAL_SEED = 0
random.seed(_GLOBAL_SEED)
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True


logger = get_logger(__name__, force=True)

import imageio.v2 as imageio
import matplotlib.pyplot as plt

def save_decoded_sample(decoded, epoch, step, save_dir="samples"):
    """
    decoded: [B, C, T, H, W]
    Saves 16 decoded frames for sample 0 using the 'Blues' colormap.
    """

    os.makedirs(save_dir, exist_ok=True)

    # pick sample 0
    dec = decoded[0].detach().cpu()   # [C, T, H, W]
    C, T, H, W = dec.shape
    assert T == 16 and H == W == 256, f"Expected 16 frames, got {T}"

    # directory for this epoch/step
    out_dir = os.path.join(save_dir, f"epoch_{epoch:03d}_step_{step:04d}")
    os.makedirs(out_dir, exist_ok=True)

    cmap = plt.get_cmap("Blues")

    for t in range(T):
        frame = dec[:, t].numpy()  # [C, H, W]

        # Collapse channels if needed
        if C == 1:
            img = frame[0]
        else:
            # Normalize each channel independently, then average
            img = (frame - frame.min()) / (frame.max() - frame.min() + 1e-8)
            img = img.mean(0)

        # Normalize to [0,1]
        img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)

        # Apply colormap → returns RGBA
        img_color = cmap(img_norm)[:, :, :3]  # drop alpha

        # Convert to uint8
        img_uint8 = (img_color * 255).astype(np.uint8)

        # Save
        imageio.imwrite(
            os.path.join(out_dir, f"decoded_t{t:02d}.png"),
            img_uint8
        )



def main(args, resume_preempt=False):
    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # -- META
    app = args.get("app")
    folder = args.get("folder")
    cfgs_meta = args.get("meta")
    load_model = cfgs_meta.get("load_checkpoint") or resume_preempt
    r_file = cfgs_meta.get("resume_checkpoint", None)
    p_file = cfgs_meta.get("pretrain_checkpoint", None)
    load_predictor = cfgs_meta.get("load_predictor", False)
    context_encoder_key = cfgs_meta.get("context_encoder_key", "encoder")
    target_encoder_key = cfgs_meta.get("target_encoder_key", "target_encoder")
    load_encoder = cfgs_meta.get("load_encoder", True)
    seed = cfgs_meta.get("seed", _GLOBAL_SEED)
    save_every_freq = cfgs_meta.get("save_every_freq", -1)
    skip_batches = cfgs_meta.get("skip_batches", -1)
    use_sdpa = cfgs_meta.get("use_sdpa", True)
    sync_gc = cfgs_meta.get("sync_gc", False)
    which_dtype = cfgs_meta.get("dtype")
    logger.info(f"{which_dtype=}")
    if which_dtype.lower() == "bfloat16":
        dtype = torch.bfloat16
        mixed_precision = True
    elif which_dtype.lower() == "float16":
        dtype = torch.float16
        mixed_precision = True
    else:
        dtype = torch.float32
        mixed_precision = False


    # -- MODEL
    cfgs_model = args.get("model")
    compile_model = cfgs_model.get("compile_model", False)
    use_activation_checkpointing = cfgs_model.get("use_activation_checkpointing", False)
    model_name = cfgs_model.get("model_name")
    decoder_depth = cfgs_model.get("decoder_depth")
    decoder_num_heads = cfgs_model.get("decoder_num_heads", None)
    decoder_embed_dim = cfgs_model.get("decoder_embed_dim")
    uniform_power = cfgs_model.get("uniform_power", False)
    use_rope = cfgs_model.get("use_rope", True)
    use_silu = cfgs_model.get("use_silu", False)
    use_decoder_silu = cfgs_model.get("use_decoder_silu", False)
    wide_silu = cfgs_model.get("wide_silu", True)

    # -- DATA
    cfgs_data = args.get("data")
    dataset_type = cfgs_data.get("dataset_type", "videodataset")
    dataset_paths = cfgs_data.get("datasets", [])
    datasets_weights = cfgs_data.get("datasets_weights", None)
    dataset_fpcs = cfgs_data.get("dataset_fpcs")
    num_clips = cfgs_data.get("num_clips", 1)
    max_num_frames = max(dataset_fpcs)

    if datasets_weights is not None:
        assert len(datasets_weights) == len(dataset_paths), "Must have one sampling weight specified for each dataset"
    batch_size = cfgs_data.get("batch_size")
    tubelet_size = cfgs_data.get("tubelet_size")
    fps = cfgs_data.get("fps")
    crop_size = cfgs_data.get("crop_size", 224)
    in_chans = cfgs_data.get("in_chans", 3)
    patch_size = cfgs_data.get("patch_size")
    pin_mem = cfgs_data.get("pin_mem", False)
    num_workers = cfgs_data.get("num_workers", 1)
    persistent_workers = cfgs_data.get("persistent_workers", True)

    # -- DATA AUGS
    cfgs_data_aug = args.get("data_aug")
    ar_range = cfgs_data_aug.get("random_resize_aspect_ratio", [3 / 4, 4 / 3])
    rr_scale = cfgs_data_aug.get("random_resize_scale", [0.3, 1.0])
    motion_shift = cfgs_data_aug.get("motion_shift", False)
    reprob = cfgs_data_aug.get("reprob", 0.0)
    use_aa = cfgs_data_aug.get("auto_augment", False)

    # -- LOSS
    cfgs_loss = args.get("loss")
    loss_exp = cfgs_loss.get("loss_exp")

    # -- OPTIMIZATION
    cfgs_opt = args.get("optimization")
    is_anneal = cfgs_opt.get("is_anneal", False)
    ipe = cfgs_opt.get("ipe", None)
    ipe_scale = cfgs_opt.get("ipe_scale", 1.0)
    wd = float(cfgs_opt.get("weight_decay"))
    final_wd = float(cfgs_opt.get("final_weight_decay"))
    num_epochs = cfgs_opt.get("epochs")
    warmup = cfgs_opt.get("warmup")
    start_lr = cfgs_opt.get("start_lr")
    lr = cfgs_opt.get("lr")
    final_lr = cfgs_opt.get("final_lr")
    betas = cfgs_opt.get("betas", (0.9, 0.999))
    eps = cfgs_opt.get("eps", 1.0e-8)
    # ----------------------------------------------------------------------- #
    # ----------------------------------------------------------------------- #

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    try:
        mp.set_start_method("spawn")
    except Exception:
        pass

    # -- init torch distributed backend
    world_size, rank = init_distributed()
    logger.info(f"Initialized (rank/world-size) {rank}/{world_size}")

    # -- set device
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)

    # -- log/checkpointing paths
    log_file = os.path.join(folder, f"log_r{rank}.csv")
    latest_file = "latest.pt"
    latest_path = os.path.join(folder, latest_file)
    load_path = None
    if load_model:
        if is_anneal:
            if os.path.exists(latest_path) and resume_anneal:
                load_path = latest_path
            else:
                load_path = anneal_ckpt
                resume_anneal = False
        else:
            load_path = r_file if r_file is not None else latest_path
        if not os.path.exists(load_path):
            load_path = None
            load_model = False

    # -- make csv_logger
    csv_logger = CSVLogger(
        log_file,
        ("%d", "epoch"),
        ("%d", "itr"),
        ("%.5f", "loss"),
        ("%d", "iter-time(ms)"),
        ("%d", "gpu-time(ms)"),
        ("%d", "dataload-time(ms)"),
    )

    # -- init model
    encoder, decoder = init_video_model(
        device=device,
        patch_size=patch_size,
        max_num_frames=max_num_frames,
        tubelet_size=tubelet_size,
        model_name=model_name,
        crop_size=crop_size,
        depth=decoder_depth,
        num_heads=decoder_num_heads,
        decoder_embed_dim=decoder_embed_dim,
        use_silu=use_silu,
        use_decoder_silu=use_decoder_silu,
        wide_silu=wide_silu,
        use_activation_checkpointing=use_activation_checkpointing,
        in_chans=in_chans,
        use_sdpa=use_sdpa,
        use_rope=use_rope,
        c=1
    )
    target_encoder = copy.deepcopy(encoder)

    if compile_model:
        logger.info("Compiling encoder, target_encoder, and decoder.")
        torch._dynamo.config.optimize_ddp = False
        encoder.compile()
        target_encoder.compile()
        decoder.compile()

    video_collator = torch.utils.data.default_collate

    transform = GeoVideoTransformWithCrop(
        ratio=ar_range,
        scale=rr_scale,
        crop_size=crop_size,
        motion_shift=False,
    )

    # -- init data-loaders/samplers
    (_ , unsupervised_loader, unsupervised_sampler) = init_data(
        data=dataset_type,
        root_path=dataset_paths,
        batch_size=batch_size,
        training=True,
        dataset_fpcs=dataset_fpcs,
        fps=fps,
        transform=transform,
        rank=rank,
        world_size=world_size,
        datasets_weights=datasets_weights,
        persistent_workers=persistent_workers,
        collator=video_collator,
        num_workers=num_workers,
        pin_mem=pin_mem,
        log_dir=None,
        num_clips=num_clips,
        # decoding=True if app == "decoder" else False
        decoding=True,
        only_flood=False,
        c=1
    )

    (val_dataset, val_loader, val_sampler) = make_geonpy_loader(
        root_path=dataset_paths,
        transform=transform,   # same transform or a deterministic one
        batch_size=batch_size,
        collator=video_collator,
        pin_mem=pin_mem,
        num_workers=num_workers,
        world_size=world_size,
        rank=rank,
        persistent_workers=persistent_workers,
        dataset_fpcs=dataset_fpcs,
        num_clips=num_clips,
        decoding=True,
        training=False,
        only_flood=False,
        c=1
    )

    try:
        _dlen = len(unsupervised_loader)
    except Exception:  # Different interface for webdataset
        _dlen = unsupervised_loader.num_batches
    if ipe is None:
        ipe = _dlen
    logger.info(f"iterations per epoch/dataset length: {ipe}/{_dlen}")

    # -- init optimizer and scheduler
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        is_anneal=is_anneal,
        decoder=decoder,
        wd=wd,
        final_wd=final_wd,
        start_lr=start_lr,
        ref_lr=lr,
        final_lr=final_lr,
        iterations_per_epoch=ipe,
        warmup=warmup,
        num_epochs=num_epochs,
        ipe_scale=ipe_scale,
        mixed_precision=mixed_precision,
        betas=betas,
        eps=eps,
    )
    #encoder = DistributedDataParallel(encoder, static_graph=True)
    decoder = DistributedDataParallel(decoder, static_graph=False, find_unused_parameters=False)
    target_encoder = DistributedDataParallel(target_encoder)
    target_encoder.eval()
    for p in target_encoder.parameters():
        p.requires_grad = False

    # -- looad pretrained weights
    _, _, target_encoder = load_pretrained(
        r_path=p_file,
        encoder=None,
        predictor=None,
        context_encoder_key=None,
        target_encoder_key=target_encoder_key,
        target_encoder=target_encoder,
        load_predictor=False,
        load_encoder=load_encoder,
    )

    start_epoch = 0
    # -- load training checkpoint
    if load_model and os.path.exists(latest_path):
        (
            encoder,
            decoder,
            target_encoder,
            optimizer,
            scaler,
            start_epoch,
        ) = load_checkpoint(
            r_path=load_path,
            encoder=encoder,
            decoder=decoder,
            target_encoder=target_encoder,
            opt=optimizer,
            scaler=scaler,
            is_anneal=is_anneal and not resume_anneal,
        )
        if not is_anneal or resume_anneal:
            for _ in range(start_epoch * ipe):
                scheduler.step()
                wd_scheduler.step()

    def save_checkpoint(epoch, path):
        if rank != 0:
            return
        save_dict = {
            "encoder": encoder.state_dict(),
            "decoder": decoder.state_dict(),
            "opt": optimizer.state_dict(),
            "scaler": None if scaler is None else scaler.state_dict(),
            "target_encoder": target_encoder.state_dict(),
            "epoch": epoch,
            "loss": loss_meter.avg,
            "batch_size": batch_size,
            "world_size": world_size,
            "lr": lr,
        }
        try:
            torch.save(save_dict, path)

            wandb.log({f"checkpoint_epoch_{epoch}_path": path})
            artifact = wandb.Artifact(f"checkpoint_epoch_{epoch}", type="model")
            artifact.add_file(path)
            wandb.log_artifact(artifact)

        except Exception as e:
            logger.info(f"Encountered exception when saving checkpoint: {e}")

    logger.info("Initializing loader...")
    unsupervised_sampler.set_epoch(start_epoch)
    loader = iter(unsupervised_loader)

    if skip_batches > 0:
        logger.info(f"Skip {skip_batches} batches")
        # -- update distributed-data-loader epoch

        for itr in range(skip_batches):
            if itr % 10 == 0:
                logger.info(f"Skip {itr}/{skip_batches} batches")
            try:
                _ = next(loader)
            except Exception:
                loader = iter(unsupervised_loader)
                _ = next(loader)

    if sync_gc:
        gc.disable()
        gc.collect()

    # -- TRAINING LOOP
    for epoch in tqdm(range(start_epoch, num_epochs), total=num_epochs, desc="Training decoder"):
        logger.info("Epoch %d" % (epoch + 1))

        loss_meter = AverageMeter()
        val_loss_meter = AverageMeter()
        iter_time_meter = AverageMeter()
        gpu_time_meter = AverageMeter()
        data_elapsed_time_meter = AverageMeter()

        sigma = 0.1

        for itr in range(ipe):
            itr_start_time = time.time()

            iter_retries = 0
            iter_successful = False
            while not iter_successful:
                try:
                    sample = next(loader)
                    iter_successful = True
                except StopIteration:
                    logger.info("Exhausted data loaders. Refreshing...")
                    unsupervised_sampler.set_epoch(epoch)
                    loader = iter(unsupervised_loader)
                except Exception as e:
                    NUM_RETRIES = 5
                    if iter_retries < NUM_RETRIES:
                        logger.warning(f"Encountered exception when loading data (num retries {iter_retries}):\n{e}")
                        iter_retries += 1
                        time.sleep(5)
                    else:
                        logger.warning(f"Exceeded max retries ({NUM_RETRIES}) when loading data. Skipping batch.")
                        raise e

            def load_clips(sample):
                clips = sample[0].to(device, non_blocking=True)  # [B C T H W]
                flood_maps = sample[1].to(device, non_blocking=True) # [B 1 T H W]
                #print("clips shape:", clips.shape)
                #print("flood_maps shape:", flood_maps.shape)

                return clips, flood_maps


            clips, flood_maps = load_clips(sample)
            data_elapsed_time_ms = (time.time() - itr_start_time) * 1000.0

            if sync_gc and (itr + 1) % GARBAGE_COLLECT_ITR_FREQ == 0:
                logger.info("Running garbage collection...")
                gc.collect()

            def train_step():
                _new_lr = scheduler.step()
                _new_wd = wd_scheduler.step()
                # --

                def forward_target(c):
                    batch_size = c.shape[0]
                    #print("ac:", c.shape)
                    with torch.no_grad():
                        c = c.permute(0, 2, 1, 3, 4).flatten(0, 1).unsqueeze(2).repeat(1, 1, 2, 1, 1) # [B*T, C, 2, H, W]
                        #print("bc:", c.shape)
                        h = target_encoder(c) # [B N D] with N == (256 // 16) ** 2 * T
                        ##print("ah:", h.shape)
                        h = h.view(batch_size, max_num_frames, -1, h.size(-1)).flatten(1, 2) # [B, T, N, D].flatten(1, 2) == {B, N_t, D}
                        #print("bh:", h.shape)
                        h = F.layer_norm(h, (h.size(-1),))
                        return h

                def forward_decoder(z):
                    _z = decoder(z) # output: a[B 1 H W]
                    return _z

                def loss_fn(flood_maps, z, w=0.5):
                    # weight non-zero flood pixels more
                    weights = 1 + w * (flood_maps > 0).float()
                    return (weights * torch.abs(z - flood_maps)).mean() # atorch.mean(torch.abs(z - flood_maps) ** loss_exp) / loss_exp

                # Step 1. Forward
                # torch.cuda.amp... is depricated in PyTorch 2.2+
                #with torch.cuda.amp.autocast(dtype=dtype, enabled=mixed_precision):
                with torch.amp.autocast('cuda', dtype=dtype, enabled=mixed_precision):
                    h = forward_target(clips)
                    z = forward_decoder(h)
                    #print(flood_maps.shape)
                    loss = loss_fn(flood_maps, z)  # jepa prediction loss

                # Step 2. Backward & step
                if mixed_precision:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                else:
                    loss.backward()
                if mixed_precision:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

                return (
                    float(loss),
                    _new_lr,
                    _new_wd,
                )

            (
                loss,
                _new_lr,
                _new_wd,
            ), gpu_etime_ms = gpu_timer(train_step)
            iter_elapsed_time_ms = (time.time() - itr_start_time) * 1000.0
            loss_meter.update(loss)
            iter_time_meter.update(iter_elapsed_time_ms)
            gpu_time_meter.update(gpu_etime_ms)
            data_elapsed_time_meter.update(data_elapsed_time_ms)

            val_loss = None
            if itr % eval_freq == 0 or itr == ipe - 1:
                # -- Validation
                def eval_step():
                    decoder.eval()
                    val_loss = 0.0
                    val_batches = 0
                    img_save = True
    
                    def forward_target(c):
                        v_batch_size = c.shape[0]
                        #print(c.shape)
                        with torch.no_grad():
                            c = c.permute(0, 2, 1, 3, 4).flatten(0, 1).unsqueeze(2).repeat(1, 1, 2, 1, 1) # [B*T, C, 2, H, W]
    
                            h = target_encoder(c) # [B N D] with N == (256 // 16) ** 2 * T
                            h = h.view(v_batch_size, max_num_frames, -1, h.size(-1)).flatten(1, 2) # [B, T, N, D].flatten(1, 2) == {B, N_t, D}
    
                            h = F.layer_norm(h, (h.size(-1),))
                            return h
    
                    def loss_fn(flood_maps, z, w=0.5):
                        weights = 1 + w * (flood_maps > 0).float()
                        return (weights * torch.abs(z - flood_maps)).mean() # h torch.mean(torch.abs(z - flood_maps) ** loss_exp) / loss_exp
    
                    with torch.no_grad():
                        for batch in val_loader:
                            clips, flood_maps = batch  # adjust if your collator returns differently
                            clips = clips.to(device, non_blocking=True)
                            flood_maps = flood_maps.to(device, non_blocking=True)
        
                            # forward pass (same as training but no predictor)
                            h = forward_target(clips)
                            z = decoder(h)
                            print("pred mean/std:", z.mean().item(), z.std().item())
                            print("gt   mean/std:", flood_maps.mean().item(), flood_maps.std().item())

                            #print(flood_maps.shape)

                            if rank==0 and img_save:
                                save_decoded_sample(z, epoch, itr)
                                img_save = False

                            loss = loss_fn(flood_maps, z)
        
                            val_loss += loss.item()
                            val_batches += 1
    
                    val_loss /= max(1, val_batches)
                    decoder.train()
    
                    return val_loss

                val_loss = eval_step()
                val_loss_meter.update(val_loss)


            # -- Logging
            def log_stats():
                csv_logger.log(epoch + 1, itr, loss, iter_elapsed_time_ms, gpu_etime_ms, data_elapsed_time_ms)
                if (itr % log_freq == 0) or (itr == ipe - 1) or np.isnan(loss) or np.isinf(loss):
                    logger.info(
                        "[%d, %5d] loss: %.3f, eval_loss: %.3f"
                        "[wd: %.2e] [lr: %.2e] "
                        "[mem: %.2e] "
                        "[iter: %.1f ms] "
                        "[gpu: %.1f ms] "
                        "[data: %.1f ms]"
                        % (
                            epoch + 1,
                            itr,
                            loss_meter.avg,
                            val_loss_meter.val,
                            _new_wd,
                            _new_lr,
                            torch.cuda.max_memory_allocated() / 1024.0**2,
                            iter_time_meter.avg,
                            gpu_time_meter.avg,
                            data_elapsed_time_meter.avg,
                        )
                    )
                    wandb.log({
                        "epoch": epoch + 1,
                        "iteration": itr,
                        "loss": loss_meter.avg,
                        "val_loss": val_loss_meter.val,
                        "lr": _new_lr,
                        "weight_decay": _new_wd,
                        "gpu_mem_mb": torch.cuda.max_memory_allocated() / 1024.0**2,
                        "iter_time_ms": iter_time_meter.avg,
                        "gpu_time_ms": gpu_time_meter.avg,
                        "data_time": data_elapsed_time_meter.avg,
                    })

            log_stats()

            assert not np.isnan(loss), "loss is nan"

        # -- Save Checkpoint
        logger.info("avg. loss %.3f" % loss_meter.avg)
        logger.info("avg. val_loss %.3f" % val_loss_meter.avg)
        # -- Save Last
        if epoch % CHECKPOINT_FREQ == 0 or epoch == (num_epochs - 1):
            save_checkpoint(epoch + 1, latest_path)
            if save_every_freq > 0 and epoch % save_every_freq == 0:
                save_every_file = f"e{epoch}.pt"
                save_every_path = os.path.join(folder, save_every_file)
                save_checkpoint(epoch + 1, save_every_path)

        # after loop
        it = getattr(loader, "_iterator", None)
        if it is not None:
            it._shutdown_workers()

        #wandb.finish()
