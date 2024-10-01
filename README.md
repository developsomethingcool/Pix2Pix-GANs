# Pix2Pix-GANs

Pix2Pix GAN Implementation in PyTorch

This repository contains an implementation of the Pix2Pix Generative Adversarial Network (GAN) using PyTorch. The Pix2Pix model is used for image-to-image translation tasks, such as converting edge maps to real images.

Project Structure

project_root/
│
├── data/
│   ├── __init__.py
│   ├── dataset.py            # Custom dataset classes
│   └── dataloader.py         # Data loading utilities
│
├── models/
│   ├── __init__.py
│   ├── generator.py          # Generator model (U-Net)
│   ├── discriminator.py      # Discriminator model (PatchGAN)
│
├── utils/
│   ├── __init__.py
│   ├── utils.py              # Utility functions (e.g., saving, loading, generating images)
│
├── training/
│   ├── __init__.py
│   ├── trainer.py            # Training loop
│   └── evaluator.py          # Evaluation functions
│
├── main.py                   # Main script to run training/evaluation/generation
└── requirements.txt          # List of dependencies