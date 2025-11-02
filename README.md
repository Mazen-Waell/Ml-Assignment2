# Assignment 2 — From Linear Models to Deep Networks

**Overview**
This project explores the progression from linear models to deep neural networks for handwritten digit recognition using the MNIST dataset.
It demonstrates a complete understanding of data preprocessing, model implementation, performance analysis, and comparison between various architectures.

part A     for Ahmed Gamal       --> Linear Models (Logistic & Softmax Regression)       

part B     for Mohamed Mostafa   --> Neural Network Implementation (Feedforward Network) 

part C, D  for Mazen Wael        --> Analysis, Model Comparison, and Advanced Techniques

**Part A — Linear Classification Models (Ahmed Gamal)**

A1. Data Preparation:

Downloaded and preprocessed the MNIST dataset.
Normalized pixel values to [0, 1].
Flattened images to 784 features for linear models.
Split dataset: 60% train, 20% validation, 20% test.
Built PyTorch DataLoader objects for efficient batching.

**A2. Logistic Regression**

Implemented binary classifier (0 vs 1) from scratch using PyTorch tensors.
Used sigmoid activation and binary cross-entropy loss.
Trained with gradient descent (learning rate = 0.01).
Delivered accuracy curves and confusion matrix.

**A3. Softmax Regression**

Extended model to multi-class classification (10 digits).
Used softmax activation with cross-entropy loss.
Compared custom implementation with PyTorch’s built-in softmax.
Produced per-class accuracy and performance plots.

**Part B — Neural Network Implementation (Mohamed Mostafa)**

**B1. Custom Neural Network**

Built a feedforward network using PyTorch.
Architecture: Input (784) → Hidden1 → Hidden2 → Output (10)
Used ReLU activations and Xavier/He initialization.
Designed for flexible layer/neurons modification.

**B2. Training Infrastructure**

Developed custom training loop supporting batching and backpropagation.
Optimizer: SGD (lr=0.01), Loss: Cross-Entropy, Batch Size: 64.
Implemented progress logging and validation split.

**B3. Visualization**

Generated plots for:
Training & validation loss
Training & validation accuracy
Learning curves with convergence analysis

**Part C — Comprehensive Analysis (Mazen Wael)**

**C1. Hyperparameter Tuning:**

Tested different learning rates: [0.001, 0.01, 0.1, 1.0]
Tested various batch sizes: [16, 32, 64, 128]
Evaluated multiple architectures (2–5 layers, 64–512 neurons/layer).
Produced detailed tables and convergence graphs.

**C2. Model Comparison**

Compared logistic regression, softmax regression, and best neural network.
Analyzed training time, accuracy, and computational cost.
Included confusion matrices and discussion on model limitations.

**Part D — Advanced Techniques (Mazen Wael)**

**D1. Convolutional Neural Networks (CNNs)**

Implemented a simple CNN for MNIST digit recognition.
Compared CNN results vs fully connected NN.
Discussed spatial feature advantages.

**D2. Regularization**

Added Dropout layers (rates: 0.1–0.7).
Applied Batch Normalization to improve convergence.
Studied the combined effect on overfitting and generalization.

**Final Notes**

All work implemented from scratch without external code.
GPU acceleration used for faster training.
Followed academic integrity guidelines and submission rules.
