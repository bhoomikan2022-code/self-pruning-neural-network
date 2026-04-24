# Self-Pruning Neural Network Report

## 1. Introduction

Modern neural networks often contain a large number of parameters, many of which may contribute very little to final predictions. Large models increase memory usage, computational cost, and inference time. Model pruning is a common compression technique where less important weights are removed after training.

The objective of this project was to design a **self-pruning neural network** that learns to suppress unnecessary connections during training itself, instead of relying on a separate post-training pruning step.

---

## 2. Problem Statement

Build a neural network for image classification on the CIFAR-10 dataset with a built-in pruning mechanism.

Each weight is associated with a learnable gate parameter. During training:

- Important connections remain active
- Less useful connections are suppressed
- The model learns a compact representation automatically

---

## 3. Methodology

## 3.1 Learnable Gates

A custom layer named **PrunableLinear** was implemented.

For every trainable weight:

- A corresponding gate score is learned
- Gate score is passed through sigmoid activation
- Final effective weight becomes:

Effective Weight = Weight × Gate

Gate values remain between 0 and 1.

If a gate becomes small, the corresponding connection contribution is reduced.

---

## 3.2 Temperature Scaling

To obtain sharper gate behavior, temperature-scaled sigmoid was used:

Gate = sigmoid(score / T)

Where:

- T < 1 creates sharper transitions
- Lower temperature helps stronger pruning pressure

---

## 3.3 Final Architecture

Input Image (32×32 RGB)  
↓  
Conv2D (3 → 32) + ReLU + MaxPool  
↓  
Conv2D (32 → 64) + ReLU + MaxPool  
↓  
Flatten  
↓  
PrunableLinear  
↓  
PrunableLinear  
↓  
Output Layer (10 Classes)

Convolutional layers were used to preserve spatial image information and improve feature extraction.

---

## 4. Loss Function

Training objective combined classification accuracy and sparsity regularization.

Total Loss = CrossEntropy Loss + λ × Sparsity Loss

Where sparsity loss is the sum of all gate activations.

This encourages:

- correct classification
- reduced effective network complexity

---

## 5. Dataset

CIFAR-10 dataset containing 10 image classes:

- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

---

## 6. Data Preprocessing

The following transformations were applied:

- Random Crop
- Random Horizontal Flip
- Tensor Conversion
- Normalization

These augmentations improve generalization.

---

## 7. Experiments

Two versions were tested:

| Model | Accuracy |
|------|----------|
| Fully Connected Baseline | 27.3% |
| CNN + Prunable Head | 49.75% |

---

## 8. Observations

1. Replacing a fully connected network with CNN feature extraction significantly improved accuracy.

2. The learnable gate mechanism successfully integrated pruning into training.

3. Hard-threshold sparsity remained low, indicating that gates mostly performed soft suppression rather than exact zero pruning.

4. Temperature-scaled gating improved optimization stability compared to the initial baseline model.

---

## 9. Challenges Faced

- Balancing classification performance and pruning pressure
- Selecting suitable lambda regularization strength
- Achieving measurable hard sparsity while preserving accuracy
- Limited training time and compute budget

---

## 10. Future Improvements

- Hard Concrete gates for exact zero pruning
- Structured channel pruning
- Dynamic temperature annealing
- Latency benchmarking after pruning
- Training on larger datasets

---

## 11. Conclusion

This project successfully implemented a self-pruning neural network in PyTorch. The model learned to suppress unnecessary connections during training while achieving strong classification improvement after introducing a CNN backbone.

The final system demonstrates how pruning can be integrated directly into optimization, making neural networks more efficient and adaptive.