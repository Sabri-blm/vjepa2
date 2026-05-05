# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import multiprocessing as mp
import pprint
from pathlib import Path

import yaml
import torch

from app.scaffold import main as app_main
from src.utils.distributed import init_distributed
from src.utils.wandb_run_name import get_next_version

parser = argparse.ArgumentParser()
parser.add_argument("--fname", type=str, help="name of config file to load", default="configs.yaml")
parser.add_argument(
    "--devices",
    type=str,
    nargs="+",
    default=["cuda:0", "cuda:1", "cuda:2", "cuda:3", "cuda:4", "cuda:5", "cuda:6", "cuda:7"],
    help="which devices to use on local machine",
)
parser.add_argument(
    "--debugmode",
    type=bool,
    default=False,
    help="Setting this to true will not spin up new processes. "
    "The main code runs the main process, which makes it easier to \
    debug with checkpointing.",
)


def process_main(rank, fname, world_size, devices):
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = str(devices[rank].split(":")[-1])

    import logging

    from src.utils.logging import get_logger

    logger = get_logger(force=True)
    if rank == 0:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)

    logger.info(f"called-params {fname}")

    # Load config
    params = None
    with open(fname, "r") as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
        logger.info("loaded params...")

    # Log config
    if rank == 0:
        pprint.PrettyPrinter(indent=4).pprint(params)
        folder = params["folder"]
        params_path = os.path.join(folder, "params-pretrain.yaml")
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        with open(params_path, "w") as f:
            yaml.dump(params, f)

        base_name = params.get("exp_name")
        version = get_next_version(base_name)

        import wandb
        wandb.init(
            project="vjepa2-geospatial",
            name=f"{base_name}-v{version}",
            config=params
        )

    # Init distributed (access to comm between GPUS on same machine)
    world_size, rank = init_distributed(rank_and_world_size=(rank, world_size))
    logger.info(f"Running... (rank: {rank}/{world_size})")

    # Track run info for cleanup
    wandb_run = wandb.run if rank == 0 else None
    wandb_dir = wandb_run.dir if wandb_run is not None else None
    wandb_run_id = wandb_run.id if wandb_run is not None else None
    wandb_project = wandb_run.project if wandb_run is not None else None
    wandb_entity = wandb_run.entity if wandb_run is not None else None

    crashed = False

    try:
        # Launch the app with loaded config
        app_main(params["app"], args=params)

        # Barrier: wait for all ranks to finish
        torch.distributed.barrier()
    except KeyboardInterrupt:
        crashed = True
        logger.error(f"KeyboardInterrupt received on rank {rank}. Shutting down cleanly...")

        # -----------------------------
        # 1. DELETE W&B RUN (rank 0 only)
        # -----------------------------
        if rank == 0 and wandb_run is not None:
            logger.error("Ctrl+C detected — deleting W&B run...")

            try:
                api = wandb.Api()
                run = api.run(f"{wandb_entity}/{wandb_project}/{wandb_run_id}")
                run.delete()
                logger.error("Remote W&B run deleted.")
            except Exception as e2:
                logger.error(f"Failed to delete remote W&B run: {e2}")

            try:
                import shutil
                shutil.rmtree(wandb_dir, ignore_errors=True)
                logger.error("Local W&B run directory deleted.")
            except Exception as e3:
                logger.error(f"Failed to delete local W&B directory: {e3}")

        # -----------------------------
        # 3. KILL ALL CHILD PROCESSES
        # -----------------------------
        kill_children(logger)
        # -----------------------------
        # 4. EXIT CLEANLY
        # -----------------------------
        logger.error("Exiting due to Ctrl+C.")
        return  # IMPORTANT: do NOT re-raise

    except Exception as e:
        crashed = True
        logger.error(f"Training crashed on rank {rank}: {e}", exc_info=True)

        if rank == 0 and wandb_run is not None:
            logger.error("Deleting W&B run due to crash...")

            try:
                # Delete remote run
                api = wandb.Api()
                run = api.run(f"{wandb_entity}/{wandb_project}/{wandb_run_id}")
                run.delete()
                logger.error("Remote W&B run deleted.")
            except Exception as e2:
                logger.error(f"Failed to delete remote W&B run: {e2}")

            try:
                # Delete local run directory
                import shutil
                shutil.rmtree(wandb_dir, ignore_errors=True)
                logger.error("Local W&B run directory deleted.")
            except Exception as e3:
                logger.error(f"Failed to delete local W&B directory: {e3}")

        # Kill all child processes
        kill_children(logger)

        raise # re-raise to exit with error

    finally:
        # Destroy process group

        if rank == 0:
            if not crashed:
                wandb.finish()
            else:
                logger.error("W&B run was already cleaned up due to crash.")

        try:
            torch.distributed.destroy_process_group()
        except:
            pass

        kill_children(logger)

def kill_children(logger):
    import psutil, os
    try:
        parent = psutil.Process(os.getpid())
        for child in parent.children(recursive=True):
            try:
                child.kill()
            except Exception:
                pass
        logger.error("All subprocesses killed.")
    except Exception as e:
        logger.error(f"Failed to kill subprocesses: {e}")




if __name__ == "__main__":
    args = parser.parse_args()
    if args.debugmode:
        process_main(rank=0, fname=args.fname, world_size=1, devices=["cuda:0"])
    else:
        num_gpus = len(args.devices)
        mp.set_start_method("spawn")
        for rank in range(num_gpus):
            mp.Process(target=process_main, args=(rank, args.fname, num_gpus, args.devices)).start()
