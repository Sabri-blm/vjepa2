import yaml
from pathlib import Path

def get_next_version(base_name, wandb_dir="wandb"):
    wandb_path = Path(wandb_dir)
    if not wandb_path.exists():
        return 0

    count = 0
    for run_dir in sorted(wandb_path.glob("run-*")):
        cfg = run_dir / "files/config.yaml"
        if not cfg.exists():
            continue

        try:
            with open(cfg, "r") as f:
                params = yaml.safe_load(f)

            exp = params.get("exp_name", {}).get("value")
            if exp == base_name:
                count += 1

        except Exception:
            # corrupted or partial config.yaml
            continue

    return count
