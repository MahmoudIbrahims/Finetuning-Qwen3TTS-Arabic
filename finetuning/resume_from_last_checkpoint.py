"""
resume_from_last_checkpoint.py

Finds the highest-numbered checkpoint-epoch-N folder inside a previous
--output_model_path, and launches sft_12hz_multi_speakers.py using it as
--init_model_path -- so you don't have to find/type the checkpoint path
by hand every time you continue training.

USAGE:
    python resume_from_last_checkpoint.py \
        --previous_output_dir output_dialects \
        --new_output_dir output_dialects_continued \
        --train_jsonl train_with_codes.jsonl \
        --batch_size 16 \
        --lr 2e-6 \
        --num_epochs 5 \
        --sft_script sft_12hz_multi_speakers.py

Everything after --sft_script-related args (--train_jsonl, --batch_size,
--lr, --num_epochs) is forwarded as-is to the training script; only
--init_model_path is computed automatically for you.
"""

import argparse
import os
import re
import subprocess
import sys


def find_last_checkpoint(previous_output_dir: str) -> str:
    if not os.path.isdir(previous_output_dir):
        raise FileNotFoundError(f"'{previous_output_dir}' does not exist.")

    pattern = re.compile(r"^checkpoint-epoch-(\d+)$")
    candidates = []
    for name in os.listdir(previous_output_dir):
        m = pattern.match(name)
        if m and os.path.isdir(os.path.join(previous_output_dir, name)):
            candidates.append((int(m.group(1)), name))

    if not candidates:
        raise FileNotFoundError(
            f"No 'checkpoint-epoch-N' folders found inside '{previous_output_dir}'. "
            f"Found: {os.listdir(previous_output_dir)}"
        )

    candidates.sort(key=lambda x: x[0])
    last_epoch, last_name = candidates[-1]
    path = os.path.join(previous_output_dir, last_name)
    print(f"[info] Found {len(candidates)} checkpoint(s). "
          f"Using the latest: {path} (epoch {last_epoch})")
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--previous_output_dir", required=True,
                         help="The --output_model_path you used in the PREVIOUS "
                              "training run, e.g. output_dialects")
    parser.add_argument("--new_output_dir", required=True,
                         help="Where THIS continued run's checkpoints will be saved")
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--sft_script", default="sft_12hz_multi_speakers.py",
                         help="Path to the training script to launch")
    args = parser.parse_args()

    last_checkpoint = find_last_checkpoint(args.previous_output_dir)

    cmd = [
        sys.executable, args.sft_script,
        "--init_model_path", last_checkpoint,
        "--output_model_path", args.new_output_dir,
        "--train_jsonl", args.train_jsonl,
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr),
        "--num_epochs", str(args.num_epochs),
    ]

    print(f"[info] Launching: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
