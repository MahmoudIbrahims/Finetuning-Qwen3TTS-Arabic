"""
make_resumable_checkpoint.py

Fixes a checkpoint saved by the ORIGINAL sft_12hz_multi_speakers.py (which
dropped speaker_encoder weights and set tts_model_type to "custom_voice" --
fine for final deployment, but breaks resuming training). This script:

  1. Pulls the speaker_encoder weights back in from your original
     warm-started base model (the one that still has them).
  2. Resets tts_model_type back to "base" so the loader instantiates
     speaker_encoder instead of leaving it None.
  3. Keeps everything else (talker weights, codec_language_id, spk_id,
     spk_is_dialect) exactly as the checkpoint had it.

USAGE:
    python make_resumable_checkpoint.py \
        --checkpoint_path output_dialect/checkpoint-epoch-19 \
        --original_base_model_path ./Qwen3-TTS-0.6B-ar-dialects \
        --output_path output_dialect/checkpoint-epoch-19-resumable
"""

import argparse
import json
import os
import shutil

from safetensors.torch import load_file, save_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", required=True,
                         help="The broken-for-resuming checkpoint, e.g. "
                              "output_dialect/checkpoint-epoch-19")
    parser.add_argument("--original_base_model_path", required=True,
                         help="The warm-started base model that still has "
                              "speaker_encoder weights, e.g. "
                              "./Qwen3-TTS-0.6B-ar-dialects")
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    # 1. Copy everything from the checkpoint first (tokenizer files, etc.)
    print(f"[info] Copying {args.checkpoint_path} -> {args.output_path} ...")
    shutil.copytree(args.checkpoint_path, args.output_path, dirs_exist_ok=True)

    # 2. Merge speaker_encoder weights back in
    checkpoint_weights_path = os.path.join(args.checkpoint_path, "model.safetensors")
    base_weights_path = os.path.join(args.original_base_model_path, "model.safetensors")

    print(f"[info] Loading checkpoint weights from {checkpoint_weights_path} ...")
    checkpoint_state_dict = load_file(checkpoint_weights_path)

    print(f"[info] Loading speaker_encoder weights from {base_weights_path} ...")
    base_state_dict = load_file(base_weights_path)

    speaker_encoder_keys = [k for k in base_state_dict.keys() if k.startswith("speaker_encoder")]
    if not speaker_encoder_keys:
        raise ValueError(
            f"No 'speaker_encoder.*' keys found in {base_weights_path}. "
            f"Is this really the right original base model path?"
        )

    print(f"[info] Merging {len(speaker_encoder_keys)} speaker_encoder tensors ...")
    for key in speaker_encoder_keys:
        checkpoint_state_dict[key] = base_state_dict[key]

    merged_weights_path = os.path.join(args.output_path, "model.safetensors")
    save_file(checkpoint_state_dict, merged_weights_path)
    print(f"[info] Saved merged weights to {merged_weights_path}")

    # 3. Reset tts_model_type so the loader instantiates speaker_encoder
    config_path = os.path.join(args.output_path, "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = json.load(f)

    old_type = config_dict.get("tts_model_type")
    config_dict["tts_model_type"] = "base"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)
    print(f"[info] tts_model_type: '{old_type}' -> 'base'")

    print(f"[done] Resumable checkpoint ready at: {args.output_path}")
    print(f"[done] Use --init_model_path {args.output_path} to continue training.")


if __name__ == "__main__":
    main()