# ZeGA: Zero-Gradient Gated Adaptation — Synapse Cartridges (`v1`)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24168085.svg)](https://doi.org/10.5281/zenodo.24168085)

**Sub-megabyte `.safetensors` Synapse Cartridges (`aphygo.synapse_cartridge.v1`) compiled without gradient optimization, loaded natively into `nn.Linear` projections via a 272-line PyTorch loader (`0` runtime hooks, `100.0000%` SHA-256 bit-identical base weights).**

- **Author:** Johan Thulin (`johan.thulin@aphygo.com`), **APHYGO AB**, Helsingborg, Sweden
- **Technical Preprint (CERN Zenodo DOI):** <https://doi.org/10.5281/zenodo.24168085> ([`PDF`](05_PAPER/ZeGA_Synapse_Cartridges_arXiv_Preprint.pdf))
- **Component License Map:** [`LICENSE_SCOPE.md`](LICENSE_SCOPE.md) *(multi-license repository — see table below)*
- **Cryptographic Cartridge Manifest:** [`CARTRIDGE_MANIFEST.json`](CARTRIDGE_MANIFEST.json)
- **GitHub Repository:** <https://github.com/Aphygo/zega-synapse-cartridges>
- **Hugging Face Cartridges:**
  - **Ministral-3-8B Swedish (`Apache-2.0`):** <https://huggingface.co/Aphygo/zega-ministral3-8b-swedish-v1>
  - **CLIP-ViT-B/32 Polarization (`MIT`):** <https://huggingface.co/Aphygo/zega-clip-vit-b32-polarization-v1>

---

## 📦 Repository Structure & Per-Component Licenses

> **Important:** This repository contains code, model-specific weight cartridges, benchmark data, and a scientific manuscript under **separate per-component licenses** (see [`LICENSE_SCOPE.md`](LICENSE_SCOPE.md)):

| Directory | Contents | License |
| :--- | :--- | :--- |
| [`01_LOADER/`](01_LOADER/) | `cartridge_loader.py` (272 lines), `verify_and_benchmark.py`, `live_studio.py` | **`Apache-2.0`** ([`LICENSE`](01_LOADER/LICENSE), [`NOTICE`](01_LOADER/NOTICE)) |
| [`03_CARTRIDGES/ministral/`](03_CARTRIDGES/ministral/) | `ministral3_8b_swedish_boost_v1.safetensors` (`960.95 KiB`, for `mistralai/Ministral-3-8B-Instruct-2512-BF16` @ `f6fae979`) | **`Apache-2.0`** ([`LICENSE`](03_CARTRIDGES/ministral/LICENSE), [`NOTICE`](03_CARTRIDGES/ministral/NOTICE)) |
| [`03_CARTRIDGES/clip/`](03_CARTRIDGES/clip/) | `clip_vit_b32_{4row,2row,1row}_*_v1.safetensors` (`336.80 KiB` quartet, for `openai/clip-vit-base-patch32` @ `3d74acf9`) | **`MIT`** ([`LICENSE`](03_CARTRIDGES/clip/LICENSE) + [`OPENAI_CLIP_LICENSE`](03_CARTRIDGES/clip/OPENAI_CLIP_LICENSE)) |
| [`04_BENCHMARKS/`](04_BENCHMARKS/) | `benchmarks_frozen.json`, `before_after_verified_results.json` | **`CC BY 4.0`** ([`LICENSE_CC_BY_4.0.md`](04_BENCHMARKS/LICENSE_CC_BY_4.0.md)) |
| [`05_PAPER/`](05_PAPER/) | `ZeGA_Synapse_Cartridges_arXiv_Preprint.pdf` & `figures/*.png` | **`CC BY 4.0`** *(manuscript & figures only)* |

*Note on Diffusion & Compiler Releases:*
- **Pillar III (`Juggernaut XL / SDXL`) Cartridge Weights:** Excluded from the `v1` public weight bundle while upstream checkpoint distribution terms are finalized.
- **ZeGA Weight-Space Compiler (`02_COMPILER`):** Held private at Day-1 launch and scheduled for source-available release under unmodified **PolyForm Noncommercial License 1.0.0**. For commercial compiler licensing inquiries, contact **`johan.thulin@aphygo.com`**.

---

## 🚀 Quickstart (`Ministral-3-8B` Swedish Cartridge)

```python
import sys, torch
sys.path.insert(0, "01_LOADER")
from transformers import AutoModelForImageTextToText, AutoTokenizer
from cartridge_loader import load_cartridge

model_id = "mistralai/Ministral-3-8B-Instruct-2512-BF16"  # rev: f6fae9795746f63c9be8344932f01275f3c63734
tokenizer = AutoTokenizer.from_pretrained(model_id, fix_mistral_regex=True)
model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda").eval()

cartridge = load_cartridge("03_CARTRIDGES/ministral/ministral3_8b_swedish_boost_v1.safetensors")
cartridge.attach(model)    # ~8.4 ms hot-swap; 0 hooks, 100% native nn.Linear!
# cartridge.detach(model)  # ~2.5 ms bit-identical restore
```
