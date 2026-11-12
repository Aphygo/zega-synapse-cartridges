---
license: apache-2.0
library_name: safetensors
pipeline_tag: image-text-to-text
language:
  - sv
  - en
  - fr
base_model: mistralai/Ministral-3-8B-Instruct-2512-BF16
base_model_relation: adapter
tags:
  - zega
  - weight-space-programming
  - synapse-cartridge
  - zero-gradient
  - swedish
  - ministral
---

# 🇸🇪 Ministral-3-8B — ZeGA Swedish Synapse Cartridge (`960.95 KiB`, Zero-Gradient)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24168084.svg)](https://doi.org/10.5281/zenodo.24168084)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Code](https://img.shields.io/badge/GitHub-zega--synapse--cartridges-181717?logo=github)](https://github.com/Aphygo/zega-synapse-cartridges)

Upgrade **`mistralai/Ministral-3-8B-Instruct-2512-BF16`** on idiomatic Swedish grammar (`de/dem`, `var/vart`, neuter gender agreement), collocations, and executive translation via a detachable **`960.95 KiB`** `.safetensors` **ZeGA Synapse Cartridge** (`aphygo.synapse_cartridge.v1`) — compiled **without gradient optimization** (`0` training steps, `0` system-prompt tokens, `0` runtime hooks, and `100.0000%` SHA-256 bit-identical base weights).

- **Author:** Johan Thulin (`johan.thulin@aphygo.com`), **APHYGO AB**, Helsingborg, Sweden
- **Technical Preprint (CERN Zenodo DOI):** <https://doi.org/10.5281/zenodo.24168084>
- **Loader, Verification Suite & Benchmarks:** <https://github.com/Aphygo/zega-synapse-cartridges>

---

## 🚀 Quickstart (8 Lines of Python)

```python
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForImageTextToText, AutoTokenizer

# cartridge_loader.py (Apache-2.0) is included right in this repo and on GitHub:
from cartridge_loader import load_cartridge

model_id = "mistralai/Ministral-3-8B-Instruct-2512-BF16"  # confirmed rev: f6fae9795746f63c9be8344932f01275f3c63734
tokenizer = AutoTokenizer.from_pretrained(model_id, fix_mistral_regex=True)
model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda").eval()

cart_path = hf_hub_download("Aphygo/zega-ministral3-8b-swedish-v1", "ministral3_8b_swedish_boost_v1.safetensors")
cartridge = load_cartridge(cart_path)
cartridge.attach(model)    # ~8.40 ms hot-swap onto native nn.Linear (+8 ch L32, +32 ch L33); 0 hooks!
# cartridge.detach(model)  # ~2.55 ms bit-identical base-model restoration
```

---

## 📊 Paired Evaluation (`2,640` Deterministic Test Items, `0` Calibration Overlaps)

| Benchmark / Audit Suite | $n$ | Unmodified `Ministral-3-8B` | **`+ZeGA Swedish Cartridge`** | Delta / Exact McNemar $p$ |
| :--- | :---: | :---: | :---: | :---: |
| **`SwedFreeGen-60`** *(Multi-sentence `[INST]` generation)* | `60` | `70.00%` *(42/60)* | **`95.00%`** *(57/60)* | **`+25.00 pp`** ($p = 2.75 \times 10^{-4}$) |
| **`SwedMaster-300`** *(6 Swedish linguistic domains)* | `300` | `73.67%` *(221/300)* | **`84.00%`** *(252/300)* | **`+10.33 pp`** ($p = 1.64 \times 10^{-6}$) |
| **`Symmetric-150`** *(Linear Steering collapses to `54.67%`)* | `150` | `59.33%` | **`72.67%`** | **`+13.33 pp`** *(two-pole gated)* |
| **ScandEval `ScaLA-sv`** *(First `1,000` of Test Split)* | `1,000` | `71.30%` | **`75.40%`** | **`+4.10 pp`** ($p = 0.00864$) |
| **SuperLim v2 `DaLAJ`-Derived Minimal Pairs** | `500` | `90.80%` | **`90.00%`** | `-0.80 pp` ($p = 0.3438$, noise) |
| **Meta `Belebele` (`SV` / `EN` / `FR`)** | `700` | `88.0% / 94.0% / 90.0%` | **`88.0% / 94.0% / 90.0%`** | **`0` per-item flag flips** ($p = 1.0$) |
| **`Anchor-40` / `Python-20` / `Math-20`** | `80` | `40/40` · `20/20` · `7/20` | **`40/40` · `20/20` · `7/20`** | `40/40` char-identical Code+Math ($\overline{D}_{\text{KL}} = 5.66 \times 10^{-5}$) |

### Real Greedy Output Examples (`do_sample=False`)
- **Executive Translation (`FG01`):**
  - ❌ **Base:** *"I går träffade styrelsen **nyckelstakeholders** … skapade **någon egentlig värde** för kunder…"*
  - ✅ **`+ZeGA Cartridge`:** *"I går möttes styrelsen med **centrala intressenter** … skapade **något egentligt värde** för kunder…"*
- **Direction (`vart`) vs. Static Location (`var`) in One Sentence (`FG04`):**
  - ❌ **Base:** *"…frågade **var** de är på väg nästa kvartal … visste ingen **var** backup-servrarna var belägna eller om förslaget erbjöd **någon konkret stöd**."*
  - ✅ **`+ZeGA Cartridge`:** *"…frågade **vart** de är på väg nästa kvartal … visste ingen **var** backup-servrarna var belägna eller om förslaget erbjöd **något konkret stöd**."*

---

## 🔐 Artifact Manifest & License

- **File:** `ministral3_8b_swedish_boost_v1.safetensors` (`984,016` bytes / `960.95 KiB`)
- **File SHA-256:** `c5bf9f32df2ee3787cdebb92e48ccf486bd467e6c053eefc6d228d41e42d584e`
- **Tensor Payload SHA-256:** `f1ce543a5fc389bc32995d509e89814743a728230334879e1ba759237c628f05`
- **Target Base Model:** `mistralai/Ministral-3-8B-Instruct-2512-BF16` (confirmed compilation revision `f6fae9795746f63c9be8344932f01275f3c63734`)
- **License:** APHYGO AB's cartridge weight contributions and bundled `cartridge_loader.py` are licensed under **Apache-2.0** (see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE)). No base model weights from Mistral AI are redistributed in this file; users obtain the base checkpoint separately under Mistral AI's Apache-2.0 terms.
- **Commercial Compiler Inquiries:** `johan.thulin@aphygo.com` (**APHYGO AB**, Helsingborg, Sweden)

## 📚 Citation
```bibtex
@misc{thulin2026zega,
  title   = {ZeGA: Zero-Gradient Gated Adaptation via Weight-Space Synapse Cartridges},
  author  = {Thulin, Johan},
  year    = {2026},
  month   = nov,
  doi     = {10.5281/zenodo.24168084},
  url     = {https://doi.org/10.5281/zenodo.24168084}
}
```
