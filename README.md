# Pix2Pix Framework for Image-to-Image Translation

This repository implements a **Pix2Pix framework**, a conditional Generative Adversarial Network (cGAN) designed for image-to-image translation tasks. It includes components for training, evaluation, and image generation, with support for visualization, checkpointing, and custom configurations.

---

## Table of Contents
1. [Overview](#overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
    - [Training](#training)
    - [Evaluation](#evaluation)
    - [Image Generation](#image-generation)
5. [Project Structure](#project-structure)
6. [Configuration](#configuration)
7. [Examples](#examples)
8. [Contributing](#contributing)
9. [License](#license)

---

## Overview

Pix2Pix is a supervised learning model that learns a mapping from input images (e.g., edges) to output images (e.g., realistic photos). This implementation is based on PyTorch and allows easy customization for different datasets and hyperparameters.

---

## Features

- **Conditional GAN with Pix2Pix architecture**.
- **Trainable generator and discriminator**.
- Supports **custom schedulers**, **loss functions**, and **multi-GPU training**.
- **Visualization tools**: Generate side-by-side comparisons of edges, generated images, and real images.
- Checkpointing for **training resumption**.
- **Evaluation metrics**: Computes L1 loss and allows manual inspection of generated results.

---

## Installation

### Prerequisites
- Python 3.8 or higher
- PyTorch 1.11 or higher
- NVIDIA GPU with CUDA support (for GPU acceleration)

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/developsomethingcool/Pix2Pix-GANs
   cd pix2pix-framework

2. Install dependencies:
   ```bash
   pip install -r requirements.txt


3. Prepare your dataset (see Configuration):

## Usage

### Training

To train the Pix2Pix model:
```bash
python main.py --task train --edge_dir <path_to_edge_images> --real_image_dir <path_to_real_images>

### Evaluation

To evaluate the model on a validation set:
```bash
python main.py --task eval --edge_dir <path_to_edge_images> --real_image_dir <path_to_real_images>

### Image Generation

To generate images using the trained generator:
```bash
python main.py --task gen --edge_dir <path_to_edge_images>


### Project Structure

pix2pix-framework/
├── data/                   # Dataloader utilities
├── models/                 # Generator and discriminator architectures
├── training/               # Training and evaluation logic
│   ├── trainer.py          # Training loop
│   ├── evaluator.py        # Evaluation loop
├── utils/                  # Helper functions
│   ├── utils.py            # Utilities for checkpointing, visualization, etc.
├── main.py                 # Entry point for training, evaluation, and generation
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation

## Configuration

### Dataset

- Prepare two folders:
  - `edges/`: Contains edge maps or input images.
  - `real_images/`: Contains the target real images.

- Ensure that file names in both directories match (e.g., `edges/img1.png` and `real_images/img1.png`).

### Hyperparameters

Modify parameters directly in `main.py` or pass them as command-line arguments:

- `--num_epochs`: Number of training epochs.
- `--batch_size`: Batch size for training and evaluation.
- `--lr`: Learning rate for optimizers.
- `--lambda_l1`: Weight of the L1 loss component.

## Examples

### Visualization

During training, side-by-side comparisons of edges, generated images, and real images are saved in the `visualization_results` directory.

### Generated Images

After training, generate images using the command in the [Image Generation](#image-generation) section. Results will be saved in the `generated_images` directory.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [Pix2Pix Paper](https://phillipi.github.io/pix2pix/)
- [PyTorch Framework](https://pytorch.org/)
