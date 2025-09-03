# 🖌️ Anime Sketch Colorization (pix2pix)

![License](https://img.shields.io/badge/License-Apache--2.0-blue)
![Notebook](https://img.shields.io/badge/Notebook-Jupyter-orange)
![GAN](https://img.shields.io/badge/Model-pix2pix%20(U--Net%20%2B%20PatchGAN)-brightgreen)

A clean, reproducible implementation of **image-to-image translation** for **anime sketch → color image** using the **pix2pix** framework (Generator: **U-Net**, Discriminator: **PatchGAN**).  
Training and experiments live in a single notebook: `pix2pix.ipynb`.

---

## 📦 Dataset

We use the **anime-sketch-colorization-pair** dataset from Kaggle.

- 👉 **Dataset:** [anime-sketch-colorization-pair](https://www.kaggle.com/ktaebum/anime-sketch-colorization-pair)  
- 👉 **Your Kaggle Notebook:** [pix2pix (Sumit Sah)](https://www.kaggle.com/code/sumitsahjr/pix2pix)

**Format:** Each sample is a **1024×512** PNG composed of two **512×512** halves:  
left = **colored image**, right = **sketch**.  
During preprocessing, each 1024×512 image is **split** into two 512×512 images `(color, sketch)`.

**Counts:**
- Train: **14,200** pairs
- Val: **3,545** pairs  
*(Dataset ≈ 6.5 GB; recommended to use Kaggle/Colab for faster access.)*

---

## 🧱 Architecture

### Generator — U-Net
Encoder–decoder with skip connections between layer *i* and layer *(n − i)*.

- **Encoder:** `C64 – C128 – C256 – C512 – C512 – C512 – C512 – C512`  
- **Decoder:** `CD512 – CD1024 – CD1024 – C1024 – C1024 – C512 – C256 – C128`  
- Final layer: **Conv (3 ch)** + **tanh**  
- Blocks:  
  - Encoder block: Conv → (BatchNorm except first) → LeakyReLU  
  - Decoder block: Transposed Conv → (Dropout in first 3) → ReLU

**Generator diagram:**  
<img width="313" alt="Generator_U-net_result" src="https://github.com/sumit-jr/Anime-Sketch-Colorization-Pair/assets/81641001/7a501b02-cd56-49b5-bc1a-6d32037bdc79">

---

### Discriminator — PatchGAN
Classifies **image patches** as real/fake.

- **PatchGAN:** `C64 – C128 – C256 – C512`  
- Each block: Conv → BatchNorm → LeakyReLU  
- Output: **16×16×1** map (per-patch real/fake)

**Discriminator diagram:**  
<img width="644" alt="discriminator U-net model" src="https://github.com/sumit-jr/Anime-Sketch-Colorization-Pair/assets/81641001/274dab73-2562-4f95-ab6e-b1e9f8d83811">

---

## 📂 Repository Structure

```text
├── pix2pix.ipynb     # Main training/experiment notebook
├── README.md         # This documentation
└── LICENSE           # Apache-2.0
```

## ⚙️ Environment & Setup

💡 **Easiest path:** Open the notebook in Kaggle (dataset available) or Google Colab (mount from Kaggle or Drive).

### Local (optional)

#### Clone
```bash
git clone https://github.com/sumit-jr/Anime-Sketch-Colorization-Pair.git
cd Anime-Sketch-Colorization-Pair
```

### Create environment (example with conda)
```bash
conda create -n pix2pix python=3.10 -y
conda activate pix2pix
```
### Install libraries (choose your DL stack as used in the notebook)
```bash
# If using TensorFlow
pip install tensorflow numpy matplotlib pillow tqdm

# If using PyTorch (optional alternative)
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# pip install numpy matplotlib pillow tqdm
```
### Launch
```bash
jupyter notebook pix2pix.ipynb

```

In the notebook, point the dataset path to where you downloaded/unzipped the Kaggle dataset and run the preprocessing cell that splits 1024×512 pairs into `(color, sketch)`.

---

## 🚀 Training

Open `pix2pix.ipynb` and run cells in order:

1. **Config & Imports**  
2. **Load & Split Data** (from Kaggle pairs to `(color, sketch)` tensors)  
3. **Build Models** (U-Net generator & PatchGAN discriminator)  
4. **Adversarial Losses & Optimizers**  
5. **Train Loop** (with sample visualizations per epoch)  

### Training snapshots

**Epoch 1**  
<img width="1252" alt="GL&DL_at_1" src="https://github.com/sumit-jr/Anime-Sketch-Colorization-Pair/assets/81641001/b0c360bb-d37b-43b1-bf5c-51d1774fd044">

**Epoch 25**  
<img width="1262" alt="GL&DL_at_25" src="https://github.com/sumit-jr/Anime-Sketch-Colorization-Pair/assets/81641001/afa8683d-f220-497b-8379-08575682a98e">

**Epoch 49**  
<img width="1258" alt="GL&DL_at_49" src="https://github.com/sumit-jr/Anime-Sketch-Colorization-Pair/assets/81641001/2a979bff-7267-4cd0-8405-72729f5c89d8">

---

## 📉 Training Curves

**Generator loss**  
<img width="651" alt="Generator_loss" src="https://github.com/sumit-jr/Anime-Sketch-Colorization-Pair/assets/81641001/888b1a54-7ce1-486f-aeed-2f453ec1b83f">

**Discriminator loss**  
<img width="659" alt="discriminator_loss" src="https://github.com/sumit-jr/Anime-Sketch-Colorization-Pair/assets/81641001/8fc3f606-ecc2-4c93-be58-562652d70864">

## ✅ Results

<img width="901" alt="result_after_training" src="https://github.com/sumit-jr/Anime-Sketch-Colorization-Pair/assets/81641001/01d3784c-2e25-4320-b708-11e10cd274b7">

---

## 🤝 Acknowledgements

- **pix2pix** (Image-to-Image Translation with Conditional Adversarial Networks)  
  Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, Alexei A. Efros, CVPR 2017.

- **Dataset:** anime-sketch-colorization-pair  

- **Kaggle notebook:** pix2pix (Sumit Sah)

---

## 📜 License

This project is released under the **Apache-2.0 License**. See LICENSE for details.

