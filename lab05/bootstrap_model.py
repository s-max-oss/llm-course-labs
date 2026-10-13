from pathlib import Path
import argparse, os, json, hashlib
from huggingface_hub import HfApi, snapshot_download

model = "HuggingFaceTB/SmolVLM-256M-Instruct"
parser=argparse.ArgumentParser(description="Download the exact lab05 base revision")
parser.add_argument("--revision",default="7e3e67edbbed1bf9888184d9df282b700a323964")
args=parser.parse_args()
files = ["config.json", "model.safetensors", "generation_config.json", "preprocessor_config.json",
         "processor_config.json", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json",
         "chat_template.json", "chat_template.jinja", "added_tokens.json"]
for endpoint in ["https://hf-mirror.com", "https://huggingface.co"]:
    try:
        api = HfApi(endpoint=endpoint)
        revision = api.model_info(model, revision=args.revision, timeout=20).sha
        directory = snapshot_download(model, revision=revision, endpoint=endpoint,
                                      local_dir="model", allow_patterns=files, max_workers=2)
        actual = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in Path(directory).iterdir() if p.is_file()}
        Path("model_provenance.json").write_text(json.dumps({"model_id": model, "revision": revision,
            "endpoint": endpoint, "files": actual}, indent=2)+"\n", encoding="utf-8")
        print("MODEL_READY", revision, actual, flush=True)
        break
    except Exception as exc:
        print(type(exc).__name__, str(exc)[:400], flush=True)
else:
    raise SystemExit("Model download failed on both documented endpoints")
