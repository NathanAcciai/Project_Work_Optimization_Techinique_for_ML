# Memory-efficiency of Modern Optimization Algorithms for Deep Learning

## Overview

This project investigates the impact of modern optimization algorithms on the training of deep neural networks, with a particular focus on the trade-off between **predictive performance**, **computational cost**, and **GPU memory consumption**.

The goal is to experimentally compare several first-order optimization methods commonly used in deep learning, evaluating how different optimizers affect convergence speed, final accuracy, runtime, and memory efficiency.

The experiments are performed on image classification tasks using standard benchmark datasets and architectures.

---

## Objectives

The main objectives of this project are:

- Study the theoretical properties and practical behavior of modern optimization algorithms.
- Implement and integrate different optimizers in PyTorch.
- Compare their optimization dynamics under a controlled experimental protocol.
- Analyze the relationship between:
  - model performance,
  - training efficiency,
  - computational resources,
  - memory requirements.

---

## Optimizers

The following optimization algorithms are considered:

### SGD + Cosine Scheduler
Classical stochastic gradient descent with cosine learning rate scheduling.  
Used as a baseline method due to its simplicity and strong generalization properties.

### Adam
Adaptive Moment Estimation optimizer that combines momentum and adaptive learning rates.

Reference:
> D. P. Kingma and J. Ba, "Adam: A Method for Stochastic Optimization", ICLR 2015.

### AdamW
A variant of Adam with decoupled weight decay, improving regularization behavior.

Reference:
> I. Loshchilov and F. Hutter, "Decoupled Weight Decay Regularization", ICLR 2019.

### Adafactor
Memory-efficient adaptive optimizer that reduces optimizer-state memory through factored second-moment estimation.

Reference:
> N. Shazeer and M. Stern, "Adafactor: Adaptive Learning Rates with Sublinear Memory Cost", ICML 2018.

### Lion
A recently proposed optimizer based on sign-based momentum updates, designed to improve optimization efficiency.

Reference:
> C. Chen et al., "Symbolic Discovery of Optimization Algorithms", NeurIPS 2023.

### Adam-mini
A memory-efficient variant of Adam that reduces optimizer states while maintaining competitive performance.

Reference:
> Adam-mini: Use Fewer Learning Rate States To Gain More (2024).

---

## Experimental Setup

All experiments and VRAM measurements were conducted on a single **NVIDIA GeForce RTX 5090 GPU (32 GB VRAM)** using the PyTorch framework with CUDA acceleration.

### Datasets

The following image classification benchmarks are considered:

- **CIFAR-10** (10 classes)
- **CIFAR-100** (100 classes)
- **Tiny ImageNet** (200 classes, 64x64 resolution)

---

### Neural Network Architectures

The optimizers are evaluated on:

- **ResNet-18**
- **Vision Transformer Tiny (ViT-Tiny patch16 224)**
- **ConvNeXt-Large** (Pretrained)
- **Vision Transformer Large (ViT-Large patch16 224)** (Pretrained)

---

## Experimental Protocol

Each experiment is defined as a unique combination of:

Different batch sizes are tested to analyze the impact on memory consumption and scalability:

until reaching the maximum GPU memory capacity.

All optimizers are evaluated under the same training conditions:

- identical dataset split,
- same architecture,
- same training budget,
- same evaluation metrics.

---

## Hyperparameter Selection

Before the final experiments, preliminary runs are performed to identify suitable hyperparameters, mainly:

- learning rate,
- weight decay,
- scheduler configuration.

Particular attention is given to SGD, which is generally more sensitive to learning-rate selection.

### 1. Configuration for ResNet-18 and ViT-Tiny (CIFAR-10 / CIFAR-100)

**General Training Setup:**
*   **Batch Size:** [256, 512, 1024]
*   **Total Epochs:** 300
*   **Scheduler:** Cosine Annealing (T_max = 300)
*   **Min Learning Rate ($\eta_{min}$):** 1e-5
*   **Warm-Up:** Linear (5 epochs)

**Dataset-Specific Settings:**
*   **CIFAR-10 (10 classes):**
    *   Patience ResNet-18: 15
    *   Patience ViT-Tiny: 30
    *   Label Smoothing (ViT): 0.1
*   **CIFAR-100 (100 classes):**
    *   Patience ResNet-18: 20
    *   Patience ViT-Tiny: 35
    *   Label Smoothing (ViT): 0.15
*   **ViT-Specific Setup:** Patch size: 4, Drop Path Rate: 0.1

**Optimizer-Specific Hyperparameters:**
*   **SGD:** Momentum 0.9 (Nesterov=True)
    *   *ResNet-18:* LR 5e-2, Weight Decay 5e-4
    *   *ViT-Tiny:* LR 3e-4, Weight Decay 5e-4
*   **Adam:** Betas (0.9, 0.999), Eps 1e-8
    *   *ResNet-18:* LR 1e-3, Weight Decay 0.0
    *   *ViT-Tiny:* LR 5e-4, Weight Decay 0.0
*   **AdamW:** Betas (0.9, 0.999), Eps 1e-8
    *   *ResNet-18:* LR 1e-3, Weight Decay 1e-4
    *   *ViT-Tiny:* LR 8e-4, Weight Decay 0.05
*   **Adafactor:** Beta1 0.9, Decay Rate -0.8
    *   *ResNet-18:* LR 1e-3, Weight Decay 1e-4
    *   *ViT-Tiny:* LR 5e-4, Weight Decay 0.0
*   **Lion:** Betas (0.9, 0.99)
    *   *ResNet-18:* LR 3e-4, Weight Decay 1e-4
    *   *ViT-Tiny:* LR 3e-4, Weight Decay 0.05
*   **Adam-mini:** Betas (0.9, 0.999), Eps 1e-8
    *   *ResNet-18:* LR 1e-3, Weight Decay 1e-4, 1 Head
    *   *ViT-Tiny:* LR 7e-4, Weight Decay 0.02, 3 Heads

### 2. Configuration for ConvNeXt-Large and ViT-Large (Tiny ImageNet)

**General Training Setup:**
*   **Batch Size:** 256
*   **Total Epochs:** 100
*   **Scheduler:** Cosine Annealing (T_max = 100)
*   **Min Learning Rate ($\eta_{min}$):** 1e-6
*   **Warm-Up:** Linear (5 epochs)
*   **Early Stopping Patience:** ConvNeXt-Large: 10 | ViT-Large: 15
*   **ViT-Specific Setup:** Label Smoothing: 0.1, Drop Path Rate: 0.1

**Optimizer-Specific Hyperparameters:**
*   **SGD:** Momentum 0.9 (Nesterov=True)
    *   *ConvNeXt:* LR 1e-3, Weight Decay 1e-2
    *   *ViT:* LR 5e-4, Weight Decay 1e-4
*   **Adam / AdamW:** Betas (0.9, 0.999), Eps 1e-8
    *   *ConvNeXt:* LR 1e-4, Weight Decay 0.0 (Adam) / 0.05 (AdamW)
    *   *ViT:* LR 5e-5, Weight Decay 0.0 (Adam) / 0.05 (AdamW)
*   **Adafactor:** Beta1 0.9, Decay Rate -0.8
    *   *ConvNeXt:* LR 1e-4, Weight Decay 1e-2
    *   *ViT:* LR 5e-5, Weight Decay 1e-2
*   **Lion:** Betas (0.9, 0.99)
    *   *ConvNeXt:* LR 1e-5, Weight Decay 0.1
    *   *ViT:* LR 5e-6, Weight Decay 0.1
*   **Adam-mini:** Betas (0.9, 0.999), Eps 1e-8
    *   *ViT:* LR 5e-5, Weight Decay 0.05, 16 Heads (Note: Omitted for ConvNeXt as it lacks Multi-Head Attention structures).

---

## Metrics

For each experimental configuration, the following metrics are collected:

### Accuracy
- Training accuracy
- Test accuracy

Used to evaluate predictive performance and generalization capability.

### Loss
Training loss curves are monitored to analyze:

- convergence speed,
- optimization stability,
- final objective value.

### Runtime

Measured as total training time to evaluate computational efficiency.

### GPU Memory Usage

Peak GPU memory consumption is recorded to compare the memory footprint of each optimizer.

This metric is particularly relevant because adaptive optimizers typically require additional memory for storing optimizer states.