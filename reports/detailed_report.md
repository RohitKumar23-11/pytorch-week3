# Detailed Report

## 1. Introduction
This project implemented two foundational deep learning architectures **from scratch using PyTorch primitives**:
- **ResNet-18** (Residual Neural Network) on **CIFAR-10** image classification.
- **Transformer encoder-decoder** on a **toy sequence-to-sequence translation task**.

No high-level modules like `torchvision.models.resnet18` or `nn.Transformer` were used. All architectures were built from `nn.Conv2d`, `nn.Linear`, `nn.LayerNorm`, etc.

---

## 2. ResNet-18 on CIFAR-10

### Approach
- Implemented **BasicBlock** and **ResNet** modules following He et al. (2015).
- Adapted to CIFAR-10 (32×32 images) using an initial 3×3 conv instead of 7×7.
- Residual connections with identity/projection shortcuts.
- Applied **random crop** and **horizontal flip** augmentation.
- Optimized with SGD + momentum, learning rate decay, and weight decay.

### Results
- **Validation accuracy** stabilized above **80%**, meeting the acceptance target.
- Confusion matrix showed **strong diagonal dominance**, with most errors between visually similar classes.
- Grad-CAM highlighted class-discriminative regions (e.g., airplane wings, animal faces).

### Key Challenges
- Getting the learning rate schedule right: too high led to divergence, too low stalled progress.
- Preventing overfitting: solved using weight decay and light augmentation.
- Debugging shortcut connections: mismatched dimensions required careful downsampling.

---

## 3. Transformer for Toy Translation

### Approach
- Implemented:
  - **Token & positional embeddings** (sinusoidal encoding).
  - **Multi-head attention** with scaled dot-product.
  - **Feed-forward networks** and **residual+layer norm**.
  - **Causal masks** to prevent future token peeking.
- Dataset: toy parallel corpus with synthetic source-target pairs.
- Objective: minimize cross-entropy loss, maximize BLEU.

### Results
- Validation loss decreased steadily; achieved **BLEU ≥15**.
- Attention heads showed interpretable alignment between source and target tokens.
- Mask visualization confirmed correct blocking of future tokens.

### Key Challenges
- Masking logic was tricky: off-by-one errors led to information leakage.
- Balancing model depth/heads with small dataset to avoid overfitting.
- Implementing positional encodings correctly (sin/cos formula).

---

## 4. Learnings
- Implementing **ResNets** clarified how identity mappings ease optimization and why deeper networks benefit from skip connections.
- Writing a **Transformer from scratch** made the role of self-attention explicit, showing how parallelism differs from recurrent models.
- Visualization (Grad-CAM, attention heatmaps) was critical for interpretability.
- Training stability depended heavily on correct initialization, normalization, and learning rate schedules.

---

## 5. References
- He, K., Zhang, X., Ren, S., & Sun, J. (2015). *Deep Residual Learning for Image Recognition*. CVPR. [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)
- Vaswani, A., et al. (2017). *Attention Is All You Need*. NeurIPS. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [scikit-learn Confusion Matrix Docs](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html)
- Stack Overflow and PyTorch Forums for debugging tensor shapes.

---

## 6. Conclusion
Both implementations met acceptance criteria:
- **ResNet-18** achieved >80% test accuracy on CIFAR-10.
- **Transformer** produced interpretable attention and BLEU ≥15.
The project deepened understanding of **residual learning, attention mechanisms, and model interpretability**.
