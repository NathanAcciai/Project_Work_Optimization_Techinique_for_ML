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

TO DO 

### Datasets

The following image classification benchmarks are considered:

- CIFAR-10
- CIFAR-100

---

### Neural Network Architectures

The optimizers are evaluated on:

- ResNet-18
- Vision Transformer Tiny (ViT-Tiny)

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

---
