"""
prepare_arabic_dialects_from_base.py

Starts from the ORIGINAL Qwen3-TTS base model (which has NO Arabic support
at all) and adds THREE separate Arabic dialect language IDs directly:
  - arabic-sa  (Saudi)
  - arabic-uae (Emirati)
  - arabic-eg  (Egyptian)

Each one is warm-started independently as the MEAN of the 10 original
language embeddings (chinese, english, german, italian, portuguese,
spanish, japanese, korean, french, russian) -- the same warm-start recipe
used for the single generic "arabic" entry earlier, just applied 3 times
so each dialect starts from the same neutral point and can specialize
independently during training (instead of all 3 sharing one ID).

USAGE:
    pip install -U qwen-tts huggingface_hub safetensors

    python prepare_arabic_dialects_from_base.py \
        --base_model_path Qwen/Qwen3-TTS-12Hz-0.6B-Base \
        --output_model_path ./Qwen3-TTS-0.6B-ar-dialects \
        --dialects arabic-sa,arabic-uae,arabic-eg
"""

import argparse
import json
import os

import torch


def load_model(base_model_path):
    from qwen_tts import Qwen3TTSModel
    return Qwen3TTSModel.from_pretrained(base_model_path, device_map="cpu", dtype=torch.float32)


def get_talker_config(model):
    for path in ["config", "model.config"]:
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
            return getattr(obj, "talker_config")
        except AttributeError:
            continue
    raise AttributeError("Could not find talker_config. Run print(model) to locate it.")


def get_codec_embedding(model):
    candidates = [
        lambda m: m.talker.model.codec_embedding,
        lambda m: m.model.talker.model.codec_embedding,
        lambda m: m.talker.codec_embedding,
        lambda m: m.model.talker.codec_embedding,
    ]
    for fn in candidates:
        try:
            emb = fn(model)
            if isinstance(emb, torch.nn.Embedding):
                return emb
        except AttributeError:
            continue
    raise AttributeError("Could not find codec_embedding automatically. "
                          "Run print(model) and locate the nn.Embedding "
                          "with num_embeddings matching talker_config.vocab_size.")


def resize_codec_embedding(model, codec_emb, new_num_embeddings):
    old_weight = codec_emb.weight.data
    old_num, dim = old_weight.shape
    print(f"[info] Resizing codec_embedding from {old_num} to {new_num_embeddings} rows")
    new_emb = torch.nn.Embedding(new_num_embeddings, dim)
    new_emb.weight.data.zero_()
    new_emb.weight.data[:old_num] = old_weight
    for path, setter in [
        ("talker.model.codec_embedding", lambda m: setattr(m.talker.model, "codec_embedding", new_emb)),
        ("model.talker.model.codec_embedding", lambda m: setattr(m.model.talker.model, "codec_embedding", new_emb)),
    ]:
        try:
            setter(model)
            return new_emb
        except AttributeError:
            continue
    raise AttributeError("Could not re-attach resized codec_embedding onto the model.")


def add_dialects(model, dialect_names):
    talker_cfg = get_talker_config(model)
    lang_map = dict(getattr(talker_cfg, "codec_language_id"))
    print(f"[info] Base model's existing languages: {list(lang_map.keys())}")

    codec_emb = get_codec_embedding(model)
    num_embeddings = codec_emb.weight.shape[0]
    original_num_embeddings = num_embeddings

    # CHANGED: mean computed ONCE from the ORIGINAL 10 languages only, used
    # as the identical starting point for every new dialect -- they begin
    # neutral/identical and differentiate purely through training.
    original_ids = list(lang_map.values())
    with torch.no_grad():
        avg_vector = codec_emb.weight[original_ids].float().mean(dim=0)

    next_id = max(lang_map.values()) + 1

    for name in dialect_names:
        if any(k.lower() == name.lower() for k in lang_map):
            print(f"[skip] '{name}' already exists in codec_language_id.")
            continue

        new_id = next_id
        next_id += 1

        if new_id >= num_embeddings:
            codec_emb = resize_codec_embedding(model, codec_emb, new_id + 1)
            num_embeddings = new_id + 1

        with torch.no_grad():
            codec_emb.weight[new_id] = avg_vector.to(codec_emb.weight.dtype)

        lang_map[name] = new_id
        print(f"[info] '{name}' -> new ID {new_id} "
              f"(warm-started as mean of {len(original_ids)} original languages)")

    talker_cfg.codec_language_id = lang_map

    new_vocab_size = num_embeddings if num_embeddings != original_num_embeddings else None
    return model, lang_map, new_vocab_size


def save_weights_only(model_to_save, output_dir):
    from safetensors.torch import save_file
    os.makedirs(output_dir, exist_ok=True)
    state_dict = model_to_save.state_dict()
    state_dict = {k: v.contiguous() for k, v in state_dict.items()}
    save_file(state_dict, os.path.join(output_dir, "model.safetensors"))
    print(f"[info] Saved weights to {os.path.join(output_dir, 'model.safetensors')}")


def fetch_original_config_json(base_model_path):
    local_candidate = os.path.join(base_model_path, "config.json")
    if os.path.isfile(local_candidate):
        return local_candidate
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=base_model_path, filename="config.json")


def patch_and_write_config(base_model_path, output_dir, lang_map, new_vocab_size=None):
    """
    Loads the PRISTINE config.json (not the in-memory config object, to
    avoid transformers' save_pretrained bug that injects stray fields like
    'dtype' into small nested sub-configs) and patches only
    codec_language_id / vocab_size.
    """
    src_config_path = fetch_original_config_json(base_model_path)
    with open(src_config_path, "r", encoding="utf-8") as f:
        config_dict = json.load(f)

    config_dict["talker_config"]["codec_language_id"] = lang_map
    if new_vocab_size is not None:
        config_dict["talker_config"]["vocab_size"] = new_vocab_size

    out_config_path = os.path.join(output_dir, "config.json")
    with open(out_config_path, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)
    print(f"[info] Wrote patched config to {out_config_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", required=True,
                         help="e.g. Qwen/Qwen3-TTS-12Hz-0.6B-Base")
    parser.add_argument("--output_model_path", required=True)
    parser.add_argument("--dialects", default="arabic-sa,arabic-uae,arabic-eg",
                         help="Comma-separated dialect language names to add")
    args = parser.parse_args()

    dialect_names = [s.strip() for s in args.dialects.split(",")]

    print(f"[info] Loading base model from {args.base_model_path} ...")
    model = load_model(args.base_model_path)

    model, lang_map, new_vocab_size = add_dialects(model, dialect_names)

    os.makedirs(args.output_model_path, exist_ok=True)
    if not hasattr(model, "model"):
        raise AttributeError("model.model not found. Run print(dir(model)).")
    save_weights_only(model.model, args.output_model_path)
    patch_and_write_config(args.base_model_path, args.output_model_path, lang_map, new_vocab_size)

    # CHANGED: save_weights_only + patch_and_write_config only produce
    # model.safetensors + config.json. Copy everything else (tokenizer,
    # speech_tokenizer, processor configs) from the HF cache so the output
    # folder is a complete, loadable model directory.
    import shutil
    try:
        from huggingface_hub import snapshot_download
        cached_dir = snapshot_download(repo_id=args.base_model_path)
        for entry in os.listdir(cached_dir):
            src = os.path.join(cached_dir, entry)
            dst = os.path.join(args.output_model_path, entry)
            if os.path.exists(dst):
                continue
            if os.path.isdir(src):
                shutil.copytree(src, dst)
            else:
                shutil.copy(src, dst)
        print(f"[info] Copied remaining tokenizer/processor files from cache into {args.output_model_path}")
    except Exception as e:
        print(f"[warn] Could not auto-copy ancillary files: {e}. "
              f"Run copy_ancillary_files.py separately.")

    print(f"[done] Final language map: {lang_map}")
    print(f"[done] Use --init_model_path {args.output_model_path} in sft_12hz_multi_speakers.py")


if __name__ == "__main__":
    main()