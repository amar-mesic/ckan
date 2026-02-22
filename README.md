# Extending Kolmogorov-Arnold Networks (KANs) to Images: The CKAN

## Overview
This repository explores the extension of the novel Kolmogorov-Arnold Network (KAN) architecture into the realm of Computer Vision (CV). We introduce and implement the **Convolutional KAN (CKAN)**, adapting traditional Convolutional Neural Networks (CNNs) by replacing scalar weights with learnable activation functions.

This project serves as a proof-of-concept for integrating KANs into CNNs and benchmarks their performance against standard CNNs on image classification tasks using MNIST, Fashion-MNIST, and CIFAR-10.

## Theoretical Background

### MLPs vs. KANs
Traditional Multi-Layer Perceptrons (MLPs) rely on the **Universal Approximation Theorem**, utilizing fixed activation functions on nodes and learnable weights on edges. 

In contrast, KANs are based on the **Kolmogorov-Arnold Representation Theorem**. The key innovation is that KANs *learn* their activation functions, which are placed on the edges rather than the nodes. 

For a smooth function $f:[0,1]^{n}\rightarrow\mathbb{R}$, the theorem is expressed as:
$$f(x)=f(x_{1},\cdot\cdot\cdot,x_{n})=\sum_{q=1}^{2n+1}\Phi_{q}(\sum_{p=1}^{n}\phi_{q,p}(x_{p}))$$

### B-Splines and Grid Extension
To make KANs practically implementable, each 1D activation function is parametrized as a B-spline curve. A B-spline is a linear combination of control points (the learnable parameters) and basis functions. 

A powerful optimization used in this implementation is **Grid Extension**. This technique allows the network to adapt the function's resolution from coarse to fine during training by increasing the granularity of the B-spline grids, significantly speeding up convergence and accuracy.

## The Convolutional KAN (CKAN) Layer
In a traditional 2D convolutional layer, a kernel matrix containing scalar weights slides over an image. In our CKAN implementation, we replace these scalar weights with our learnable B-spline functions. 

A standard $3\times3$ CKAN kernel is represented as:
$$K=[\begin{matrix}\phi_{1,1}&\phi_{1,2}&\phi_{1,3}\\ \phi_{2,1}&\phi_{2,2}&\phi_{2,3}\\ \phi_{3,1}&\phi_{3,2}&\phi_{3,3}\end{matrix}]$$

## Experiments and Results

We tested four architecture combinations (CNN + MLP, CNN + KAN, CKAN + MLP, and CKAN + KAN) across three datasets. We controlled the architectures to ensure a comparable number of learnable parameters for fair evaluation.

### Performance Summary
*Note: Models were trained for 10 epochs using an Adam optimizer (LR=0.001) with an 80/20 train/test split.*

| Dataset | Model | Trainable Params | Accuracy | Inference Time (s) |
| :--- | :--- | :--- | :--- | :--- |
| **MNIST** | CNN + MLP | 34,858 | 99.46% | 2.8 |
| | CKAN + KAN | 60,688 | 99.06% | 56.5 |
| **Fashion-MNIST** | CNN + MLP | 34,858 | 92.21% | 1.4 |
| | CKAN + KAN | 60,688 | 89.77% | 56.5 |
| **CIFAR-10** | CNN + MLP | 38,506 | 81.40% | 2.6 |
| | CKAN + KAN | 60,688 | 64.47% | 38.0 |

### Key Takeaways
1. **Complexity vs. Performance:** While CKANs perform comparably to CNNs on simpler datasets (MNIST), the performance gap widens on more complex datasets like CIFAR-10. 
2. **Computational Overhead:** CKANs currently have slower training and inference times compared to heavily optimized PyTorch CNN classes. 
3. **Grid Extension is Highly Effective:** Implementing grid extension dramatically improved our test set accuracy on MNIST from 86.60% to 95.80% in early simplified testing.

## Repository Contents

* `Experiments/`: Contains notebooks and scripts for running benchmarks.
    * `KAN performance on MNIST`
    * `CKAN performance on MNIST`
* `KAN/`: Core architecture implementations.
    * `KAN.py`
    * `KANLayer.py`
* `my_spline.py`: Custom spline implementations.
* `PyKAN Utilities`: Integrated utilities from the original PyKAN implementation.
    * `spline.py`
    * `utils.py`

## Getting Started

To see a basic KAN perform on a toy dataset, check out our introductory notebook: 
👉 [`kan_basic_amar.ipynb`](kan_basic_amar.ipynb)

## Authors
* Amar Mešić
* Parviz Ahmadov
* Reinout Vos 
*(Note: Based on a research project at TU Delft)*
