# Self-Supervised Image Representation Learning using Masked Autoencoders (MAE)

<div align="center">

![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch)
![ViT](https://img.shields.io/badge/ViT-Vision%20Transformer-blue?style=flat-square)
![MAE](https://img.shields.io/badge/MAE-Self--Supervised-green?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-Demo-FF4B4B?style=flat-square&logo=streamlit)
![Platform](https://img.shields.io/badge/Platform-Kaggle%20T4%20x2-20BEFF?style=flat-square&logo=kaggle)

**Asymmetric Vision Transformer encoder-decoder that reconstructs 75% masked image patches from scratch**

[📝 Medium Blog](#) · [💼 LinkedIn Post](#) · [🤗 Live Demo](#)

</div>

---

## 📌 Overview

This project implements a **Masked Autoencoder (MAE)** for self-supervised visual representation learning. The model randomly masks **75% of image patches** and learns to reconstruct them using an asymmetric ViT encoder-decoder architecture — all implemented from scratch using base PyTorch layers.

```
224×224 Image  →  Split into 16×16 patches (196 total)
                          ↓
              Randomly mask 75% (147 patches)
                          ↓
         Encoder sees only 25% visible patches (49)
                          ↓
    Decoder reconstructs all 196 patches from latent + mask tokens
                          ↓
              Pixel-level reconstruction output
```

---

## ✨ Features

- ✅ Full MAE pipeline from scratch (base PyTorch only — no pretrained weights)
- ✅ Asymmetric design: large ViT-Base encoder, lightweight ViT-Small decoder
- ✅ 75% random patch masking with proper positional ordering
- ✅ MSE loss computed only on masked patches
- ✅ Mixed precision training (torch.cuda.amp) for Kaggle T4×2
- ✅ PSNR and SSIM quantitative evaluation
- ✅ Visualization: masked input | reconstruction | ground truth
- ✅ Streamlit / Gradio app with masking ratio slider

---

## 🗂️ Repository Structure

```
📦 mae-image-reconstruction/
├── 📁 model/
│   ├── patch_embed.py             # Patchification & positional embeddings
│   ├── encoder.py                 # ViT-Base encoder (visible patches only)
│   ├── decoder.py                 # ViT-Small decoder (full patch sequence)
│   └── mae.py                     # Full MAE model
├── 📁 training/
│   └── train.py                   # MSE training loop (AdamW + cosine LR)
├── 📁 evaluation/
│   └── metrics.py                 # PSNR, SSIM scoring
├── 📁 visualization/
│   └── visualize.py               # Masked input / reconstruction / GT
├── 📁 app/
│   └── app.py                     # Streamlit app with masking ratio control
├── 📓 mae_reconstruction.ipynb    # Full pipeline notebook
├── 📄 requirements.txt
└── 📄 README.md
```

---

## 🧠 Model Architecture

### Encoder — ViT-Base (B/16)
| Config | Value |
|---|---|
| Patch Size | 16 × 16 |
| Image Size | 224 × 224 |
| Hidden Dimension | 768 |
| Transformer Layers | 12 |
| Attention Heads | 12 |
| Parameters | ~86M |

Encoder receives **only the 25% visible patches** (49 out of 196). Mask tokens are never input to the encoder — this is the key asymmetry that makes MAE efficient.

### Decoder — ViT-Small (S/16)
| Config | Value |
|---|---|
| Hidden Dimension | 384 |
| Transformer Layers | 12 |
| Attention Heads | 6 |
| Parameters | ~22M |

Decoder receives encoder latent tokens + learnable mask tokens for all 147 missing patches, then reconstructs the full pixel-level image.

---

## 📊 Dataset

**TinyImageNet**
- Source: [Kaggle — akash2sharma/tiny-imagenet](https://www.kaggle.com/datasets/akash2sharma/tiny-imagenet)
- Content: 100,000 images across 200 classes at 64×64 (resized to 224×224)
- Platform: Kaggle GPU T4 x2

---

## ⚙️ Training Configuration

| Setting | Value |
|---|---|
| Loss | MSE on masked patches only |
| Optimizer | AdamW |
| LR Scheduler | Cosine |
| Batch Size | 32–64 |
| Precision | Mixed (torch.cuda.amp) |
| Masking Ratio | 75% |

---

## 📈 Evaluation

| Metric | Description |
|---|---|
| PSNR | Peak Signal-to-Noise Ratio — higher = better reconstruction |
| SSIM | Structural Similarity Index — measures perceptual quality |

Minimum 5 qualitative reconstruction examples displayed: masked input → model output → ground truth.

---

## 🚀 Quick Start

```bash
git clone https://github.com/yourusername/mae-image-reconstruction.git
cd mae-image-reconstruction
pip install -r requirements.txt
python training/train.py
streamlit run app/app.py
```

---

## ✅ Tasks Completed

- [x] Patchification (196 patches of 16×16) and 75% random masking
- [x] ViT-Base encoder (visible patches only) + ViT-Small decoder
- [x] MSE loss on masked patches only
- [x] AdamW + cosine LR scheduler
- [x] Mixed precision training
- [x] PSNR and SSIM evaluation
- [x] Visualization module (5+ reconstruction examples)
- [x] Streamlit / Gradio app with masking ratio selector

---

## 🔗 Links

| Resource | Link |
|---|---|
| Medium Blog | [Read on Medium](#) |
| LinkedIn Post | [View on LinkedIn](#) |
| Live Demo | [Streamlit / Gradio](#) |
| Dataset | [TinyImageNet on Kaggle](https://www.kaggle.com/datasets/akash2sharma/tiny-imagenet) |

---

*FAST-NUCES | Generative AI | Spring 2026 | Roll No. 22F-8816*
