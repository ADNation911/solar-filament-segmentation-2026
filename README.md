# Apex Sovereign: Solar Filament Segmentation (Global Rank 2 — 0.64 PQ)

Official solution repository for **Team Tempest** (`tensura911`) for the **Solar Filament Segmentation Challenge 2026** hosted on Kaggle by the Earth-Space AI Research (ESAIR) Lab and sponsored by the National Science Foundation (NSF) and National Solar Observatory (NSO).

- **Team Members:** Aaditya Verma ([@tensura911](https://www.kaggle.com/tensura911), aadityaverma0018@gmail.com) & Mansha Nirapure ([@manshanirapure](https://www.kaggle.com/manshanirapure), nirapuremansha@gmail.com)
- **Kaggle Leaderboard Rank:** **Global Rank 2** (Top-Tier Solution)
- **Official Public Leaderboard Score (Panoptic Quality):** **`0.64`** (`0.7421` Continuous Micro PQ / `0.7782` Macro PQ)
- **Primary Submission File:** `submission.csv` (`1,413` disjoint filament instances across all `180/180` test observations)

---

## 1. Repository Architecture

```text
code/
├── src/
│   ├── __init__.py          # Package exports
│   ├── preprocessing.py     # Otsu solar limb circle detection, radial limb darkening correction (LDC), Z-score normalization
│   ├── models.py            # Multi-backbone U-Net builders, Soft Dice / Asymmetric Tversky / Lovász hinge losses, D4 TTA & logit ensembling
│   ├── postprocess.py       # Decoupled R=6 topological spine grouping, 18-feature GBDT component filter, LP Venn-SDF boundary calibrator
│   └── evaluation.py        # COCO RLE encoding/decoding, official Panoptic Quality (PQ = SQ x RQ), 2D non-overlapping contract verifier
├── reproduce_submission.py  # End-to-end verification and inference driver (--mode [verify|infer])
├── pipeline_walkthrough.ipynb # Self-contained, executed Jupyter Notebook demonstrating every pipeline stage
├── requirements.txt         # Pinned Python dependencies
└── submission.csv           # Official Global Rank 1 (0.64 PQ) competition submission (1,413 instances)
```

---

## 2. Environment Setup

Install all pinned dependencies:

```bash
pip install -r requirements.txt
```

Core dependencies:
- Python $\ge$ 3.10
- PyTorch $\ge$ 2.0.0, `segmentation-models-pytorch` $\ge$ 0.3.3, `timm` $\ge$ 0.9.12
- OpenCV (`opencv-python` $\ge$ 4.8.0), `scipy` $\ge$ 1.10.0
- `pycocotools` $\ge$ 2.0.7
- `scikit-learn` $\ge$ 1.3.0, `pandas` $\ge$ 2.0.0, `numpy` $\ge$ 1.24.0

---

## 3. Solution Overview

1. **Radial Limb Darkening Correction & Stratified Validation (`src.preprocessing`):**
   - Automated Otsu solar disk center/radius estimation and quadratic limb darkening correction ($I_{\text{corr}}(r)$) flatten background chromospheric roll-off across all 6 GONG observing stations before robust on-disk Z-score standardization.
2. **Multi-Backbone Deep Neural Ensemble (`src.models`):**
   - Combines **ResNet-34 U-Net** (Dice + BCE), **EfficientNet-B3 U-Net** (Dice), **EfficientNet-B3 Tversky U-Net** ($\alpha=0.3, \beta=0.7$), and **DenseNet-121 U-Net** (submodular Lovász hinge loss) with full 8-way dihedral ($D_4$) test-time augmentation fused in uncompressed log-odds (logit) space.
3. **Decoupled Topological Spine Grouping ($R=6$) & 18-Feature GBDT Filter (`src.postprocess`):**
   - Bridges 3–6 pixel chromospheric absorption dips via an elliptical structuring element $K_R$ ($13 \times 13$ px) at $T_{\text{group}}=0.40$ while preserving crisp boundaries at $T_{\text{trim}}=0.480$, followed by an 18-feature Histogram GBDT classifier ($\tau=0.38$) that prunes false-alarm sunspot pores and chromospheric fibrils.
4. **Linear Programming (LP) Venn-Region Signed Distance Field (SDF) Calibration (`src.postprocess`):**
   - Resolves multi-annotator boundary disagreement on 2-pass and 3-pass GONG observations via exact 8-region Venn decomposition, Sequential Linear Programming (SLP) excess maximization, and Euclidean distance transform level sets while strictly enforcing disjoint 2D instance masks.

---

## 4. Reproduction & Validation

To execute the functional self-tests across all `src/` modules and run the complete geometric and non-overlapping RLE contract audit on `submission.csv`:

```bash
python reproduce_submission.py submission.csv
```

To run end-to-end inference on a directory of raw $2048 \times 2048$ GONG H-$\alpha$ images:

```bash
python reproduce_submission.py --mode infer --image-dir /path/to/test/images --output submission_reproduced.csv
```

Or open the executed walkthrough notebook:

```bash
jupyter notebook pipeline_walkthrough.ipynb
```

The audit verifies:
- Exactly `1,413` filament instances across all `180 / 180` test stems (`100.0%` coverage).
- Zero nulls, zero duplicate `filament_id` values, zero empty RLE masks, and `0` 2D pixel overlap collisions.
