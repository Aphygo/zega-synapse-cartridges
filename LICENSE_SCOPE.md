# LICENSE_SCOPE.md — Per-Component License Map (`ZeGA v1`)

**Copyright 2026 APHYGO AB, Helsingborg, Sweden (`johan.thulin@aphygo.com`)**

This repository is a **multi-component release** and does **not** use a single blanket license across code, cartridge weights, benchmark data, and documentation. Each component directory is governed strictly by the license specified below:

| Component / Path | Material Covered | Applicable License | Notes & Upstream Provenance |
| :--- | :--- | :--- | :--- |
| [`01_LOADER/`](01_LOADER/) | `cartridge_loader.py`, `verify_and_benchmark.py`, `live_studio.py`, `run_studio.bat`, `requirements.txt` | **`Apache-2.0`** | Unmodified Apache License 2.0 in [`01_LOADER/LICENSE`](01_LOADER/LICENSE) + [`01_LOADER/NOTICE`](01_LOADER/NOTICE). |
| [`03_CARTRIDGES/ministral/`](03_CARTRIDGES/ministral/) | `ministral3_8b_swedish_boost_v1.safetensors` (`960.95 KiB`, SHA-256 `c5bf9f32df2ee378…`) | **`Apache-2.0`** *(APHYGO contribution)* | Built against `mistralai/Ministral-3-8B-Instruct-2512-BF16` (rev. `f6fae9795746f63c9be8344932f01275f3c63734`, Apache-2.0). No base model weights included. |
| [`03_CARTRIDGES/clip/`](03_CARTRIDGES/clip/) | `clip_vit_b32_{4row_symmetric,2row_stationary,1row_additive}_v1.safetensors` | **`MIT`** *(APHYGO contribution)* | Built against `openai/clip-vit-base-patch32` (rev. `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`, MIT). Includes OpenAI's unmodified MIT notice in [`03_CARTRIDGES/clip/OPENAI_CLIP_LICENSE`](03_CARTRIDGES/clip/OPENAI_CLIP_LICENSE). |
| [`04_BENCHMARKS/`](04_BENCHMARKS/) | `benchmarks_frozen.json`, `before_after_verified_results.json` | **`CC BY 4.0`** | APHYGO-authored Swedish & zero-collateral benchmark suites (`SwedMaster-300`, `SwedFreeGen-60`, `Anchor-40`, `Python-20`, `Math-20`). Separate data license in [`04_BENCHMARKS/LICENSE_CC_BY_4.0.md`](04_BENCHMARKS/LICENSE_CC_BY_4.0.md). |
| [`05_PAPER/`](05_PAPER/) | `ZeGA_Synapse_Cartridges_arXiv_Preprint.pdf` & `figures/*.png` | **`CC BY 4.0`** | Applies strictly to manuscript text and original figures, **not** to software code or cartridge weights. |
| **Pillar III Diffusion Cartridges (`Juggernaut XL / SDXL`)** | *Not included in `v1` public weight release* | *Withheld in `v1`* | Weight files (`sdxl_sapphire_*_v1.safetensors`) are excluded from `v1` distribution pending final upstream hosting-chain license confirmation. |
| **ZeGA Weight-Space Compiler (`02_COMPILER`)** | *Not included in Day-1 public release* | *Proprietary at Day 1 -> `PolyForm Noncommercial 1.0.0`* | Held private at initial launch; scheduled for separate source-available release under unmodified `PolyForm Noncommercial 1.0.0` (commercial inquiries: `johan.thulin@aphygo.com`). |
