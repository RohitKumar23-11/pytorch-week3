# pytorch-week3 🚀

Deep Learning Architectures from Scratch in PyTorch  
**Objective:** Implement **ResNet-18** (for image classification) and a **Transformer encoder-decoder** (for sequence-to-sequence translation) **from primitives** (`torch.nn.Conv2d`, `torch.nn.Linear`, etc.), train them on public datasets, and generate visual artifacts to demonstrate understanding of modern architectures.

---

## 📂 Project Structure

```

pytorch-week3/
│
├── code/
│   ├── resnet\_cifar10.py        # ResNet-18 for CIFAR-10
│   ├── transformer\_toy.py       # Minimal Transformer for toy MT
│
├── runs/
│   ├── cls/                     # ResNet outputs
│   │   ├── curves\_cls.png
│   │   ├── confusion\_matrix.png
│   │   ├── preds\_grid.png
│   │   ├── miscls\_grid.png
│   │   ├── gradcam\_0.png ...
│   │
│   └── mt/                      # Transformer outputs
│       ├── curves\_mt.png
│       ├── attention\_layer1\_head1.png ...
│       ├── masks\_demo.png
│       ├── decodes\_table.png
│       ├── bleu\_report.png
│
├── report/
│   ├── onepage\_visual.md        # One-page summary with figures
│   └── detailed\_report.md       # Longer writeup (sources, insights, notes)
│
└── README.md                    # This file

````

---

## 🖼️ Models

### 1️⃣ ResNet-18 (CIFAR-10)

- Implemented from scratch (no `torchvision.models`)  
- Residual blocks with identity/projection shortcuts  
- Adapted for CIFAR-10 (32×32, removed initial maxpool, stride adjustments)  
- Global average pooling + fully connected classifier  

**Training Setup:**
- Dataset: CIFAR-10 (50k train, 10k test)  
- Augmentation: RandomCrop(32, padding=4), RandomHorizontalFlip  
- Optimizer: SGD with momentum=0.9, weight decay=5e-4  
- Criterion: CrossEntropyLoss  
- Target: ≥80% test accuracy  

**Generated Figures (in `runs/cls/`):**
- `curves_cls.png`: Training/validation loss & accuracy curves  
- `confusion_matrix.png`: Normalized confusion matrix  
- `preds_grid.png`: Grid of correct predictions  
- `miscls_grid.png`: Grid of misclassified predictions  
- `gradcam_*.png`: Grad-CAM heatmaps  

---

### 2️⃣ Transformer Encoder-Decoder (Toy MT)

- Implemented from scratch (no `nn.Transformer`)  
- Components:  
  - Token embeddings + sinusoidal positional encodings  
  - Multi-head scaled dot-product attention  
  - Feed-forward layers with ReLU  
  - Layer normalization + residual connections  
  - Encoder stack & decoder stack with causal masks  

**Training Setup:**
- Dataset: Toy parallel corpus (small synthetic translation task)  
- Preprocessing: padding, start/end tokens, attention masks  
- Optimizer: Adam with β1=0.9, β2=0.98, Noam LR scheduler  
- Criterion: CrossEntropyLoss with ignore_index=pad  
- Target: BLEU ≥15  

**Generated Figures (in `runs/mt/`):**
- `curves_mt.png`: Training/validation loss (perplexity optional)  
- `attention_layer{L}_head{H}.png`: Multi-head attention heatmaps  
- `masks_demo.png`: Visualization of source/target masks  
- `decodes_table.png`: Table comparing decoded vs ground truth translations  
- `bleu_report.png`: Corpus BLEU score summary  

---

## 🏃 Running the Code

### 🔹 ResNet-18 (CIFAR-10)
```bash
cd code
python resnet_cifar10.py --epochs 20 --batch_size 128 --lr 0.1
````

This will:

* Train ResNet-18 on CIFAR-10
* Save the best model checkpoint (`resnet18_cifar10.pth`)
* Generate plots into `runs/cls/`

---

### 🔹 Transformer (Toy MT)

```bash
cd code
python transformer_toy.py --epochs 30 --batch_size 64 --lr 0.0005
```

This will:

* Train encoder-decoder on toy parallel dataset
* Save checkpoints & attention heatmaps
* Generate visual artifacts into `runs/mt/`

---

## 📊 Results

### ✅ ResNet-18

* Validation Accuracy: **\~82%** on CIFAR-10
* Strong diagonal dominance in confusion matrix
* Grad-CAM shows clear focus on object regions

### ✅ Transformer

* Validation BLEU: **\~17–18**
* Attention heatmaps show clear alignment patterns
* Decoded sentences mostly coherent vs ground truth

---

## 📑 Deliverables

* **Code:** clean, from primitives (no `torchvision.models`, no `nn.Transformer`)
* **Figures:** stored in `runs/cls/` and `runs/mt/`
* **Reports:**

  * `report/onepage_visual.md`: one-page visual summary
  * `report/detailed_report.md`: key learnings, challenges, references

---

## 📚 References

* He et al. (2015), *Deep Residual Learning for Image Recognition* – [arXiv](https://arxiv.org/abs/1512.03385)
* Vaswani et al. (2017), *Attention Is All You Need* – [arXiv](https://arxiv.org/abs/1706.03762)
* PyTorch Tutorials: [https://pytorch.org/tutorials](https://pytorch.org/tutorials)
* scikit-learn Confusion Matrix: [docs](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html)
* Grad-CAM: Selvaraju et al. (2017), *Visual Explanations from Deep Networks via Gradient-based Localization* – [arXiv](https://arxiv.org/abs/1610.02391)
* BLEU: Papineni et al. (2002), *BLEU: a Method for Automatic Evaluation of Machine Translation* – [ACL Anthology](https://aclanthology.org/P02-1040/)

---

## ✨ Acceptance Criteria

* [x] ResNet-18 reaches ≥80% test accuracy on CIFAR-10
* [x] Confusion matrix shows clear diagonal dominance
* [x] Grad-CAM visualizations highlight discriminative regions
* [x] Transformer reaches ≥15 BLEU score on toy dataset
* [x] Attention heatmaps are interpretable
* [x] Repo contains one-page visual report

---

## 👩‍💻 Author

**Rohit Kumar**


```
