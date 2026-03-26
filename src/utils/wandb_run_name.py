import json
from pathlib import Path

def get_next_version(base_name, wandb_dir="wandb"):
    wandb_path = Path(wandb_dir)
    if not wandb_path.exists():
        return 0

    count = 0
    for run_dir in wandb_path.glob("run-*"):
        meta = run_dir /  "files/wandb-metadata.json"
        if meta.exists():
            try:
                data = json.loads(meta.read_text())
                if data.get("runName") == base_name:
                    count += 1
            except Exception:
                pass

    return count
