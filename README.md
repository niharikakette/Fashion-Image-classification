# Fashion_Image_Classification


---

# 👕 FashionMNIST VAE – Variational Autoencoder with PyTorch

This project implements a **Variational Autoencoder (VAE)** using **PyTorch**, trained on the **FashionMNIST** dataset. It learns to encode and decode fashion images in an unsupervised manner, enabling it to reconstruct inputs from a compressed latent representation.

---

## 🧠 Model Overview

The VAE model is composed of:
- **Encoder**: Fully connected layers that map input to latent space
- **Latent Variables**: Mean (μ) and standard deviation (σ) for reparameterization
- **Decoder**: Fully connected layers to reconstruct the image from the latent vector

---

## 🗃️ Dataset

- **FashionMNIST** (automatically downloaded using `torchvision.datasets`)
- Contains grayscale images (28x28) of 10 fashion categories:
  ```
  ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
  ```

---

## 📦 Libraries Used

| Library       | Purpose                                        |
|---------------|------------------------------------------------|
| `torch`       | Building and training the neural network       |
| `torchvision` | Accessing datasets and transforms              |
| `numpy`       | Numerical operations                           |
| `matplotlib`  | Data visualization                             |
| `PIL`         | Image preprocessing                            |
| `random`      | Image sampling and visualization               |
| `sklearn`     | Metrics (optional)                             |
| `google.colab`| Google Drive integration (optional)            |

---

## 🛠️ How It Works

### 🔄 Workflow

1. **Data Loading**:  
   FashionMNIST is downloaded and transformed to tensors, normalized to [-1, 1].

2. **Training**:  
   The model is trained using a combination of:
   - **Reconstruction Loss** (MSE)
   - **KL Divergence** Loss

3. **Evaluation**:  
   The model reconstructs test images from the learned latent space.

---

## 📉 Training Results

| Epoch | Reconstruction Loss | KL Loss | Accuracy (%) |
|-------|----------------------|---------|---------------|
| 1     | 64354.42             | 1144.68 | 0.25          |
| 5     | 60354.80             | 886.98  | 0.51          |
| 10    | 59945.69             | 887.77  | 0.57          |

> Accuracy is computed by comparing reconstructed pixels with original inputs (threshold > 0.5).

---

## 📊 Visualizations

### 🔍 Sample Input vs Reconstruction

```python
fig, ax = plt.subplots(1, 2)
ax[0].imshow(original_image, cmap='bone_r')
ax[0].set_title("Original Image")
ax[1].imshow(reconstructed_image, cmap='bone_r')
ax[1].set_title("Reconstructed Image")
```

Example:

![Example reconstruction](https://via.placeholder.com/300x100?text=Original+vs+Reconstructed)

---

## ▶️ How to Run

1. Clone or download this repository
2. Open the notebook in **Google Colab** or Jupyter
3. Run all cells sequentially
4. (Optional) Mount Google Drive to save models or access datasets

---

## ✅ Features

- Custom-built VAE using PyTorch
- Reconstruction visualizations
- Training from scratch on FashionMNIST
- Simple modular code for experimentation

---

## 📁 Directory Structure

```
.
├── vae_fashionmnist.ipynb  # Main training & evaluation notebook
└── README.md               # Project overview
```

---

## 🙏 Acknowledgements

- [PyTorch](https://pytorch.org/)
- [FashionMNIST Dataset](https://github.com/zalandoresearch/fashion-mnist)
- Google Colab for training infrastructure

---
