---
license: mit
library_name: safetensors
pipeline_tag: zero-shot-image-classification
language:
  - en
base_model: openai/clip-vit-base-patch32
base_model_relation: adapter
tags:
  - zega
  - weight-space-programming
  - synapse-cartridge
  - zero-gradient
  - clip
  - compositional-binding
  - vision-language
---

# 🎨 CLIP-ViT-B/32 — ZeGA Family II Polarization Cartridges (`336.80 KiB`, Zero-Gradient)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24168084.svg)](https://doi.org/10.5281/zenodo.24168084)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code](https://img.shields.io/badge/GitHub-zega--synapse--cartridges-181717?logo=github)](https://github.com/Aphygo/zega-synapse-cartridges)

Synthesize nonlinear bilinear gate–value selectivity inside standard **non-gated** 2-layer Vision Transformer MLPs (`QuickGELU`, lacking `gate_proj`) **without gradient optimization**. Compiled in **`26.6 seconds`** via **Theorem 3** (*Stationary-Minimum & Fixed-Center Symmetric Quartet Polarization*), this **`336.80 KiB`** Synapse Cartridge (`+56/3,072` channels in visual Layer 10 of `openai/clip-vit-base-patch32`) lifts Winoground-style compositional Pair Group Score from `0.83%` to **`25.83%`** on `CompBind-240` and from `2.22%` to **`34.44%`** on `ClauseSwap-180` (`0%` caption/template overlap) while preserving `CIFAR-100` zero-shot classification and `1.0000000` `CelebA` embedding cosine (`0.40 ms` hot-swap).

- **Author:** Johan Thulin (`johan.thulin@aphygo.com`), **APHYGO AB**, Helsingborg, Sweden
- **Technical Preprint (CERN Zenodo DOI):** <https://doi.org/10.5281/zenodo.24168084>
- **Native 272-Line Loader (`cartridge_loader.py`) & Benchmarks:** <https://github.com/Aphygo/zega-synapse-cartridges>

![Pillar II CLIP-ViT-B/32 Polarization Results](fig_pillar2_clip_vit_polarization.png)
*(Figure from preprint licensed under `CC BY 4.0`; cartridge weights in this repository are licensed under `MIT`.)*

---

## 🚀 Quickstart (Python)

First grab `cartridge_loader.py` from [`Aphygo/zega-synapse-cartridges`](https://github.com/Aphygo/zega-synapse-cartridges/tree/main/01_LOADER), then attach the 4-row symmetric quartet in `< 1 ms`:

```python
import torch
from huggingface_hub import hf_hub_download
from transformers import CLIPModel, CLIPProcessor
from cartridge_loader import load_cartridge

model_id = "openai/clip-vit-base-patch32"  # confirmed rev: 3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268
model = CLIPModel.from_pretrained(model_id).eval()
processor = CLIPProcessor.from_pretrained(model_id)

cart_path = hf_hub_download("Aphygo/zega-clip-vit-b32-polarization-v1", "clip_vit_b32_4row_symmetric_v1.safetensors")
cartridge = load_cartridge(cart_path)
cartridge.attach(model)    # ~0.40 ms native nn.Linear width expansion (+56 rows in visual L10); 0 hooks!
# cartridge.detach(model)  # instant 100% SHA-256 bit-identical base restoration
```

---

## 📊 Controlled Ablation of Theorem 3 (`290` Unique Unseen-Seed Image Pairs + `700` Natural Photos)

| Condition | Layer 10 Rows | `CompBind-240` Grp ($p$) / I2T | `ClauseSwap-180` Grp ($p$) / I2T | `SpatialRel-160` Grp ($p$) / I2T | `CIFAR-100` Agree / KL | `CelebA` Probe / Cos |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Base `CLIP-ViT-B/32` | `+0` | `0.83%` / `52.92%` | `2.22%` / `52.78%` | `0.00%` / `50.62%` | Base (`65.8%`) / `0` | Base / `1.0000000` |
| `+1`-Row Unsymmetrized Control | `+14` | `0.83%` (`1.0`) / `51.67%` | `3.33%` (`1.0`) / `53.89%` | `0.00%` (`1.0`) / `48.75%` | `499/500` / `8.08e-7` | `200/200` / `1.0000000` |
| `+2`-Row Stationary Pair *(Thm 3a)* | `+28` | `9.17%` (`6.3e-3`) / `63.75%` | `12.22%` (`0.022`) / `62.22%` | `1.25%` (`1.0`) / `57.50%` | **`500/500`** / `9.74e-8` | `200/200` / `1.0000000` |
| **`+4`-Row Symmetric Quartet *(Thm 3b)*** | **`+56`** | **`25.83%`** (**`1.5e-8`**) / **`89.2%`** | **`34.44%`** (**`3.0e-8`**) / **`81.7%`** | **`6.25%`** (`0.062`) / **`65.0%`** | `499/500` / **`1.99e-9`** | **`200/200`** / **`1.0000000`** |

---

## 🔐 Cartridge Files & Licenses (`MIT`)

| Cartridge File | Role | Size | SHA-256 |
| :--- | :--- | :--- | :--- |
| `clip_vit_b32_4row_symmetric_v1.safetensors` | 4-Row Fixed-Center Symmetric Quartet (Theorem 3b, Primary) | `344,880` B (`336.80 KiB`) | `a8351af23cadb7989f99d2e4feaed091b00c97b2dc2006903ab3b9a42acf4f53` |
| `clip_vit_b32_2row_stationary_v1.safetensors` | 2-Row Stationary-Minimum Pair (Theorem 3a, Ablation) | `172,736` B (`168.69 KiB`) | `af2d307cbb39ea51c8ac7976c961dd4456546cd9e911a3c29b54b6d548335cb3` |
| `clip_vit_b32_1row_additive_v1.safetensors` | 1-Row Unsymmetrized Control (Ablation) | `86,664` B (`84.63 KiB`) | `17dc1612c9e1b0c8d36ed52aab2ee244bba947d6e54402628dbf1e6f17d30f81` |

- **APHYGO AB Cartridge Contributions:** Licensed under the **MIT License** (`Copyright (c) 2026 APHYGO AB`; see [`LICENSE`](LICENSE)).
- **Upstream Base Model (`openai/clip-vit-base-patch32`, rev. `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`):** Original CLIP model and code by OpenAI (`Copyright (c) 2021 OpenAI`) under the **MIT License**. OpenAI's original unmodified MIT license and copyright notice are preserved verbatim in [`OPENAI_CLIP_LICENSE`](OPENAI_CLIP_LICENSE).
- **Figure Note:** The embedded summary plot `fig_pillar2_clip_vit_polarization.png` is reproduced from the technical preprint under **CC BY 4.0**.
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
