# One-Page Visual Summary

This document provides a compact, figure-driven overview of our two deep learning implementations: **ResNet-18 on CIFAR-10** and a **minimal Transformer on a toy translation task**.

---

## 🖼️ ResNet-18 (CIFAR-10)

- **Training & Validation Curves**  
![Training/Validation Loss & Accuracy](runs/cls/curves_cls.png) 
![Training/Validation Loss & Accuracy]([runs/cls/curves_cls.png](https://github.com/RohitKumar23-11/pytorch-week3/blob/main/runs/cls/curves_cls.png))

- **Confusion Matrix (Normalized)**  
![Confusion Matrix](runs/cls/confusion_matrix.png)

- **Correct Predictions Grid**  
![Correct Predictions](runs/cls/preds_grid.png)

- **Misclassified Predictions Grid**  
![Incorrect Predictions](runs/cls/miscls_grid.png)

- **Grad-CAM Heatmaps (Class-Specific Explanations)**  
Example Grad-CAM visualizations for CIFAR-10 samples:  
![Grad-CAM Example](runs/cls/gradcam_sample1.png)  
![Grad-CAM Example](runs/cls/gradcam_sample2.png)

---

## 🌐 Transformer (Toy Translation)

- **Loss Curve (Training & Validation)**  
![Loss/Perplexity](runs/mt/curves_mt.png)

- **Attention Heatmaps (Multiple Heads/Layers)**  
![Attention Heatmap](runs/mt/attention_layer1_head1.png)  
![Attention Heatmap](runs/mt/attention_layer2_head2.png)

- **Mask Visualization**  
![Mask Visualization](runs/mt/masks_demo.png)

- **Decoded Outputs vs Ground Truth**  
![Decoded Table](runs/mt/decodes_table.png)

- **Corpus BLEU Score Summary**  
![BLEU Report](runs/mt/bleu_report.png)

---

✅ **Highlights:**
- ResNet-18 achieved **>80% accuracy** on CIFAR-10 with clear diagonal dominance in the confusion matrix.  
- Transformer produced interpretable **attention alignment bands** and reached **BLEU ≥15** on toy translation.
