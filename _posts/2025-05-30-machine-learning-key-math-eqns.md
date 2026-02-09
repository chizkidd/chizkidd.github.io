---
layout: post
comments: true
title: "Essential Machine Learning Equations: A Reference Guide"
excerpt: A reference guide to fundamental machine learning equations with correct implementations and clear explanations of how they connect to each other.
date: 2025-05-30
mathjax: true
---

## Why This Guide Exists

I created this as a practical reference for the mathematical foundations of machine learning. It's not comprehensive (no guide could be), but it covers equations I find myself returning to regularly. Each section includes working Python implementations that I've tested or used at some point.

This started from a [tweet by @goyal__pramod](https://x.com/goyal__pramod/status/1923064911501914216) and grew as I collected formulas I actually use.

---

## Table of Contents

* [Information Theory](#information-theory)
  * [The Foundation: Entropy](#the-foundation-entropy)
  * [Cross-Entropy: Encoding with the Wrong Distribution](#cross-entropy-encoding-with-the-wrong-distribution)
  * [KL Divergence: The Extra Cost](#kl-divergence-the-extra-cost)
  * [Bayes' Theorem](#bayes-theorem)
* [Linear Algebra](#linear-algebra)
  * [Affine Transformations (Not "Linear"!)](#affine-transformations-not-linear)
  * [Eigenvalues and Eigenvectors](#eigenvalues-and-eigenvectors)
  * [The Normal Equation (Closed-Form Linear Regression)](#the-normal-equation-closed-form-linear-regression)
  * [Singular Value Decomposition (SVD)](#singular-value-decomposition-svd)
* [Optimization](#optimization)
  * [Gradient Descent](#gradient-descent)
* [Neural Network Fundamentals](#neural-network-fundamentals)
  * [The Multi-Layer Perceptron (Forward Pass)](#the-multi-layer-perceptron-forward-pass)
  * [Backpropagation](#backpropagation)
* [Loss Functions](#loss-functions)
  * [Mean Squared Error (MSE)](#mean-squared-error-mse)
  * [Binary Cross-Entropy Loss](#binary-cross-entropy-loss)
* [Advanced Architectures](#advanced-architectures)
  * [Softmax Function](#softmax-function)
  * [Convolution Operation](#convolution-operation)
  * [Attention Mechanism](#attention-mechanism)
  * [Diffusion Models (Reverse Process)](#diffusion-models-reverse-process)
* [What's Missing](#whats-missing)
* [References](#references)

---

## Information Theory

Information theory provides the mathematical foundation for understanding uncertainty and measuring differences between probability distributions. These concepts are deeply interconnected.

### The Foundation: Entropy

**Equation:** 

$$
H(P) = -\sum_{x} P(x) \log_2 P(x)
$$

**What it means:** Entropy measures the average number of bits needed to encode events from distribution $P$. Higher entropy means more uncertainty.

**Why it matters:** Entropy is the baseline for all other information-theoretic measures. A fair coin flip has 1 bit of entropy; you need exactly 1 bit to encode the outcome. A biased coin (90/10) has less entropy because outcomes are more predictable.

**Implementation:**

```python
import numpy as np

def entropy(p):
    """
    Calculate entropy of a probability distribution.
    
    Parameters:
    p: Probability distribution array (must sum to 1)
    
    Returns:
    Entropy in bits (using log base 2)
    """
    # Only compute log for non-zero probabilities
    p_nonzero = p[p > 0]
    return -np.sum(p_nonzero * np.log2(p_nonzero))

# Fair coin: maximum uncertainty for binary outcome
fair_coin = np.array([0.5, 0.5])
print(f"Fair coin entropy: {entropy(fair_coin):.4f} bits")  # 1.0000

# Biased coin: less uncertainty
biased_coin = np.array([0.9, 0.1])
print(f"Biased coin entropy: {entropy(biased_coin):.4f} bits")  # 0.4690
```

### Cross-Entropy: Encoding with the Wrong Distribution

**Equation:**

$$
H(P, Q) = -\sum_{x} P(x) \log_2 Q(x)
$$

**What it means:** Cross-entropy measures the average number of bits needed to encode events from distribution $P$ when using a code optimized for distribution $Q$ instead.

**Why it matters:** This is the connection to machine learning. $P$ is your true data distribution, $Q$ is your model's predictions. Cross-entropy tells you how inefficient your model's encoding is.

**Implementation:**

```python
def cross_entropy(p, q):
    """
    Calculate cross-entropy H(P, Q).
    
    Parameters:
    p: True distribution
    q: Predicted distribution
    
    Returns:
    Cross-entropy in bits
    """
    # Only where p > 0 matters (can't have log(0))
    return -np.sum(p * np.log2(q + 1e-10))  # Small epsilon for numerical stability

# True distribution vs predicted distribution
p = np.array([0.8, 0.2])
q_good = np.array([0.75, 0.25])  # Close to truth
q_bad = np.array([0.3, 0.7])     # Far from truth

print(f"H(P, Q_good): {cross_entropy(p, q_good):.4f} bits")  # ~0.74
print(f"H(P, Q_bad): {cross_entropy(p, q_bad):.4f} bits")   # ~1.84
```

### KL Divergence: The Extra Cost

**Equation:**

$$
D_{KL}(P \| Q) = \sum_{x} P(x) \log_2 \left( \frac{P(x)}{Q(x)} \right)
$$

**What it means:** KL divergence is the *extra* bits needed when using $Q$ instead of $P$. It's literally the difference:

$$
D_{KL}(P \| Q) = H(P, Q) - H(P)
$$

**Why it matters:** This measures how much your model $Q$ diverges from reality $P$. Unlike cross-entropy, it's relative to the irreducible entropy in the data. Note: KL divergence is asymmetric: $D_{KL}(P \| Q) \neq D_{KL}(Q \| P)$.

**Implementation:**

```python
def kl_divergence(p, q):
    """
    Calculate KL divergence D_KL(P || Q).
    
    Parameters:
    p: True distribution
    q: Predicted distribution
    
    Returns:
    KL divergence in bits (always >= 0)
    """
    p_nonzero = p > 0
    return np.sum(p[p_nonzero] * np.log2(p[p_nonzero] / (q[p_nonzero] + 1e-10)))

# Verify the relationship: KL = Cross-Entropy - Entropy
p = np.array([0.8, 0.2])
q = np.array([0.5, 0.5])

kl = kl_divergence(p, q)
ce = cross_entropy(p, q)
h = entropy(p)

print(f"KL divergence: {kl:.4f}")
print(f"Cross-entropy: {ce:.4f}")
print(f"Entropy: {h:.4f}")
print(f"CE - H = {ce - h:.4f} (should equal KL)")
```

### Bayes' Theorem

**Equation:** 

$$
P(A|B) = \frac{P(B|A) P(A)}{P(B)}
$$

**What it means:** Update beliefs about $A$ given evidence $B$. The prior $P(A)$ becomes the posterior $P(A|B)$ after observing $B$.

**Why it matters:** This is how you should update probabilities when you get new information. Most famously used in spam filters and medical diagnosis.

**Implementation:**

```python
def bayes_theorem(prior, likelihood, evidence):
    """
    Calculate P(A|B) using Bayes' theorem.
    
    Parameters:
    prior: P(A)
    likelihood: P(B|A)
    evidence: P(B)
    
    Returns:
    posterior: P(A|B)
    """
    return (likelihood * prior) / evidence

# Medical test example
p_disease = 0.01           # 1% of population has disease
p_pos_given_disease = 0.99 # Test sensitivity
p_pos_given_healthy = 0.02 # False positive rate

# Calculate P(positive test)
p_positive = (p_pos_given_disease * p_disease + 
              p_pos_given_healthy * (1 - p_disease))

# What's P(disease | positive test)?
p_disease_given_pos = bayes_theorem(p_disease, p_pos_given_disease, p_positive)

print(f"P(disease | positive) = {p_disease_given_pos:.4f}")  # 0.3333
print("Only 33% chance of disease despite 99% accurate test!")
```

---

## Linear Algebra

### Affine Transformations (Not "Linear"!)

**Equation:**

$$
y = Ax + b
$$

**What it means:** This is actually an *affine* transformation, not a linear one. A truly linear transformation has no bias term ($b=0$). The bias term shifts the output, which breaks the definition of linearity: $f(x + y) = f(x) + f(y)$.

**Why it matters:** Every layer in a neural network is an affine transformation. The bias term is crucial because it lets your network learn patterns that don't pass through the origin.

**Implementation:**

```python
import numpy as np

def affine_transform(x, A, b):
    """
    Apply affine transformation y = Ax + b.
    
    Parameters:
    x: Input vector (n,)
    A: Weight matrix (m, n)
    b: Bias vector (m,)
    
    Returns:
    y: Transformed vector (m,)
    """
    return A @ x + b

# Example: 2D to 2D transformation
A = np.array([[2, -1], [1, 1]])
x = np.array([3, 2])
b = np.array([1, -1])

y = affine_transform(x, A, b)
print(f"Input: {x}")
print(f"Output: {y}")  # [5, 4]
```

### Eigenvalues and Eigenvectors

**Equation:**

$$
Av = \lambda v
$$

**What it means:** Eigenvectors are special directions where the matrix $A$ only scales the vector (by eigenvalue $\lambda$), without rotating it.

**Why it matters:** Eigenvectors reveal the principal directions of variation in a given dataset. Principal Component Analysis (PCA) finds eigenvectors of the covariance matrix. Eigenvectors point in the direction of maximum variance of the data, while eigenvalues correspond to the amount of variance captured along each corresponding eigenvector's direction. These vectors are essential in pattern recognition within high dimensional datasets.

**Implementation:**

```python
from numpy.linalg import eig

# Covariance matrix example
C = np.array([[4, 2], [2, 3]])

eigenvalues, eigenvectors = eig(C)

print("Eigenvalues:", eigenvalues)
print("Eigenvectors:\n", eigenvectors)

# Verify: Cv = λv for first eigenvector
v1 = eigenvectors[:, 0]
lambda1 = eigenvalues[0]
print(f"\nVerification:")
print(f"Cv = {C @ v1}")
print(f"λv = {lambda1 * v1}")
```

### The Normal Equation (Closed-Form Linear Regression)

**Equation:**

$$
\hat{\beta} = (X^T X)^{-1} X^T y
$$

**What it means:** This gives the exact solution to linear regression in one step, without iteration. It's the derivative of the MSE loss set to zero and solved algebraically.

**Why it matters:** This is the analytical solution that gradient descent approximates. The term $(X^T X)^{-1} X^T$ is called the Moore-Penrose pseudoinverse.  In practice, gradient descent or regularization (Ridge: add $\lambda I$ to $X^T X$) is preferred for large datasets (computing the inverse is expensive, $O(n^3)$), but this equation reveals the underlying linear algebra structure. For small datasets (< 10,000 samples) with few features, this is often faster and gives the exact solution.

**When to use it:** Small datasets (< 10,000 samples) with few features where $(X^T X)$ is invertible. For larger problems or when $X^T X$ is singular, use gradient descent or regularized versions (Ridge: add $\lambda I$ to $X^T X$).

**Implementation:**

```python
import numpy as np

def normal_equation(X, y):
    """
    Solve linear regression using the normal equation.
    
    Parameters:
    X: Design matrix (m, n)
    y: Target values (m,)
    
    Returns:
    beta: Optimal parameters (n,)
    """
    return np.linalg.inv(X.T @ X) @ X.T @ y

# Example: fit y = 2x + 3
np.random.seed(42)
X = np.column_stack([np.ones(100), np.linspace(0, 10, 100)])
y = 2 * X[:, 1] + 3 + np.random.randn(100) * 0.5

beta = normal_equation(X, y)
print(f"Normal equation solution: β = {beta}")  # Should be close to [3, 2]

# Compare to gradient descent
from scipy.optimize import minimize
def mse(beta, X, y):
    return np.mean((X @ beta - y)**2)

result = minimize(mse, x0=[0, 0], args=(X, y), method='BFGS')
print(f"Optimization solution:   β = {result.x}")

# They should match!
print(f"Difference: {np.linalg.norm(beta - result.x):.10f}")
```

### Singular Value Decomposition (SVD)

**Equation:**

$$
A = U \Sigma V^T
$$

**What it means:** Any matrix can be decomposed into three matrices: two rotation matrices ($U$, $V^T$) and a scaling matrix ($\Sigma$). Think of it as "every transformation is rotate, scale, rotate."

**Why it matters:** SVD is everywhere: dimensionality reduction (truncated SVD), recommendation systems (matrix factorization), image compression, and the foundation of PCA. It's also how you compute the pseudoinverse in the normal equation when $X^T X$ is not invertible.

**Implementation:**

```python
from numpy.linalg import svd

# User-item matrix (e.g., movie ratings)
A = np.array([
    [5, 3, 0, 1],
    [4, 0, 0, 1],
    [1, 1, 0, 5],
    [1, 0, 0, 4],
    [0, 1, 5, 4],
])

U, S, Vt = svd(A, full_matrices=False)

print("Original shape:", A.shape)
print("U shape:", U.shape)  # Users × concepts
print("S shape:", S.shape)  # Singular values
print("Vt shape:", Vt.shape)  # Concepts × items

# Low-rank approximation (keep top 2 components)
k = 2
A_approx = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
print(f"\nReconstruction error: {np.linalg.norm(A - A_approx):.4f}")
```

---

## Optimization

### Gradient Descent

**Equation:**

$$
\theta_{t+1} = \theta_t - \eta \nabla_{\theta} L(\theta_t)
$$

**What it means:** Move parameters $\theta$ in the direction that decreases loss $L$ most rapidly. The learning rate $\eta$ controls step size.

**Why it matters:** This is how neural networks learn. The gradient $\nabla_{\theta} L$ points uphill; we go downhill by subtracting it.

**Implementation:**

```python
def gradient_descent(X, y, lr=0.01, epochs=1000):
    """
    Fit linear regression using gradient descent.
    
    Parameters:
    X: Design matrix (m, n) with intercept column
    y: Target values (m,)
    lr: Learning rate
    epochs: Number of iterations
    
    Returns:
    theta: Learned parameters (n,)
    """
    m, n = X.shape
    theta = np.zeros(n)
    
    for epoch in range(epochs):
        # Gradient of MSE loss
        predictions = X @ theta
        errors = predictions - y
        gradient = (1/m) * X.T @ errors
        
        # Update step
        theta = theta - lr * gradient
        
        if epoch % 100 == 0:
            loss = np.mean(errors**2)
            print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    return theta

# Example: fit y = 2x + 3
X = np.column_stack([np.ones(100), np.linspace(0, 10, 100)])
y = 2 * X[:, 1] + 3 + np.random.randn(100) * 0.5

theta = gradient_descent(X, y, lr=0.01, epochs=500)
print(f"\nLearned parameters: {theta}")  # Should be close to [3, 2]
```

---

## Neural Network Fundamentals

### The Multi-Layer Perceptron (Forward Pass)

**Equations:**

$$
z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)}
$$

$$
a^{(l)} = \sigma(z^{(l)})
$$

**What it means:** Each layer computes a weighted sum of the previous layer's activations ($z$), then applies a nonlinear activation function ($\sigma$) to get the new activations ($a$).

**Why it matters:** This is the basic building block of deep learning. Without the nonlinearity $\sigma$, stacking layers would be pointless, basically multiple affine transformations will compose into a single affine transformation.

**Implementation:**

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def relu(z):
    return np.maximum(0, z)

class MLP:
    def __init__(self, layer_sizes):
        """
        Initialize multi-layer perceptron.
        
        Parameters:
        layer_sizes: List of layer dimensions [input, hidden1, ..., output]
        """
        self.weights = []
        self.biases = []
        
        for i in range(len(layer_sizes) - 1):
            W = np.random.randn(layer_sizes[i+1], layer_sizes[i]) * 0.1
            b = np.zeros(layer_sizes[i+1])
            self.weights.append(W)
            self.biases.append(b)
    
    def forward(self, x):
        """Forward pass through network."""
        a = x
        activations = [a]
        
        for W, b in zip(self.weights[:-1], self.biases[:-1]):
            z = W @ a + b
            a = relu(z)  # ReLU for hidden layers
            activations.append(a)
        
        # Output layer (no activation for regression)
        W, b = self.weights[-1], self.biases[-1]
        z = W @ a + b
        activations.append(z)
        
        return activations

# Example: 2 → 4 → 1 network
mlp = MLP([2, 4, 1])
x = np.array([1.0, 2.0])
activations = mlp.forward(x)
print(f"Input: {activations[0]}")
print(f"Hidden: {activations[1]}")
print(f"Output: {activations[2]}")
```

### Backpropagation

**Equation:**

$$
\frac{\partial L}{\partial W^{(l)}} = \delta^{(l)} (a^{(l-1)})^T
$$

$$
\delta^{(l)} = (W^{(l+1)})^T \delta^{(l+1)} \odot \sigma'(z^{(l)})
$$

**What it means:** Backpropagation computes gradients by working backward through the network, using the chain rule. The $\delta$ terms are the "errors" at each layer.

**Why it matters:** This is *not* an optimization algorithm, it's a way to efficiently compute gradients. You still need an optimizer (like gradient descent) to update the weights. Backpropagation just tells you which direction to move.

**Implementation:**

```python
import torch
import torch.nn as nn

# Define a simple network
model = nn.Sequential(
    nn.Linear(2, 4),
    nn.ReLU(),
    nn.Linear(4, 1)
)

# Forward pass
x = torch.tensor([[1.0, 2.0]])
y_true = torch.tensor([[3.0]])

y_pred = model(x)
loss = nn.MSELoss()(y_pred, y_true)

print(f"Prediction: {y_pred.item():.4f}")
print(f"Loss: {loss.item():.4f}")

# Backward pass (backpropagation computes gradients)
loss.backward()

# Inspect gradients
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name} gradient norm: {param.grad.norm().item():.4f}")
```

---

## Loss Functions

### Mean Squared Error (MSE)

**Equation:**

$$
L_{MSE} = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2
$$

**What it means:** Average squared difference between predictions and targets. Squaring penalizes large errors more than small ones.

**Why it matters:** MSE is the maximum likelihood estimator when errors are normally distributed. It's also convex, which makes optimization easier. Works well for regression but is sensitive to outliers.

**Implementation:**

```python
def mse_loss(y_true, y_pred):
    """Mean squared error loss."""
    return np.mean((y_true - y_pred)**2)

def mse_gradient(y_true, y_pred):
    """Gradient of MSE with respect to predictions."""
    return 2 * (y_pred - y_true) / len(y_true)

y_true = np.array([1.0, 2.0, 3.0])
y_pred = np.array([1.1, 2.3, 2.8])

loss = mse_loss(y_true, y_pred)
grad = mse_gradient(y_true, y_pred)

print(f"MSE Loss: {loss:.4f}")
print(f"Gradient: {grad}")
```

### Binary Cross-Entropy Loss

**Equation:**

$$
L_{BCE} = -\frac{1}{n} \sum_{i=1}^n [y_i \log(\hat{y}_i) + (1-y_i) \log(1-\hat{y}_i)]
$$

**What it means:** For binary classification, this measures how well predicted probabilities $\hat{y}_i$ match the true labels $y_i \in \{0, 1\}$.

**Why it matters:** This is exactly the cross-entropy between the true distribution and predicted distribution for binary outcomes. When $y=1$, we only care about $\log(\hat{y})$; when $y=0$, we only care about $\log(1-\hat{y})$.

**Implementation:**

```python
def binary_cross_entropy(y_true, y_pred, epsilon=1e-10):
    """
    Binary cross-entropy loss.
    
    Parameters:
    y_true: True labels (0 or 1)
    y_pred: Predicted probabilities (0 to 1)
    epsilon: Small value for numerical stability
    
    Returns:
    loss: BCE loss value
    """
    # Clip predictions to avoid log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

y_true = np.array([1, 0, 1, 1, 0])
y_pred = np.array([0.9, 0.1, 0.8, 0.7, 0.2])

loss = binary_cross_entropy(y_true, y_pred)
print(f"BCE Loss: {loss:.4f}")  # Lower is better
```

---

## Advanced Architectures

### Softmax Function

**Equation:**

$$
\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}
$$

**What it means:** Converts a vector of real numbers into a probability distribution. The exponential ensures positive values, and normalization ensures they sum to 1.

**Why it matters:** This is how neural networks output class probabilities. Combined with cross-entropy loss, the gradients have a beautiful form that makes training efficient.

**Implementation:**

```python
def softmax(z):
    """
    Numerically stable softmax.
    
    Parameters:
    z: Input logits
    
    Returns:
    probabilities: Normalized probabilities
    """
    # Subtract max for numerical stability
    z_shifted = z - np.max(z)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z)

# Class scores for 3 classes
logits = np.array([2.0, 1.0, 0.1])
probs = softmax(logits)

print(f"Logits: {logits}")
print(f"Probabilities: {probs}")
print(f"Sum: {np.sum(probs):.6f}")  # Should be 1.0
```

### Convolution Operation

**Equation:**

$$
(f * g)(t) = \int_{-\infty}^{\infty} f(\tau) g(t - \tau) \, d\tau
$$

**Discrete version for images:**

$$
(I * K)(i, j) = \sum_m \sum_n I(i-m, j-n) K(m, n)
$$

**What it means:** Slide a filter (kernel) across the input, computing dot products at each position. This detects local patterns.

**Why it matters:** CNNs use convolutions because they're translation invariant, which means that a cat detector works whether the cat is in the corner or center of the image.

**Implementation:**

```python
import torch
import torch.nn.functional as F

# Create a simple edge detection kernel
edge_kernel = torch.tensor([
    [[-1., -1., -1.],
     [ 0.,  0.,  0.],
     [ 1.,  1.,  1.]]
])  # Shape: (1, 3, 3)

# Create a dummy image
image = torch.randn(1, 1, 28, 28)  # (batch, channels, height, width)

# Apply convolution
output = F.conv2d(image, edge_kernel.unsqueeze(0))

print(f"Input shape: {image.shape}")
print(f"Kernel shape: {edge_kernel.shape}")
print(f"Output shape: {output.shape}")  # (1, 1, 26, 26) - smaller due to no padding
```

### Attention Mechanism

**Equation:**

$$
\text{Attention}(Q, K, V) = \text{softmax}\left( \frac{QK^T}{\sqrt{d_k}} \right) V
$$

**What it means:** Compute similarity between queries $Q$ and keys $K$, use these as weights to combine values $V$. The $\sqrt{d_k}$ scaling prevents dot products from getting too large.

**Why it matters:** This is the core of transformers. Unlike RNNs, attention can directly connect any two positions in a sequence, making it much more powerful for long-range dependencies.

**Implementation:**

```python
import torch
import torch.nn.functional as F

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Compute scaled dot-product attention.
    
    Parameters:
    Q: Queries (batch, seq_len, d_k)
    K: Keys (batch, seq_len, d_k)
    V: Values (batch, seq_len, d_v)
    mask: Optional mask (batch, seq_len, seq_len)
    
    Returns:
    output: Attention output (batch, seq_len, d_v)
    attention_weights: Attention weights (batch, seq_len, seq_len)
    """
    d_k = Q.size(-1)
    
    # Compute attention scores
    scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
    
    # Apply mask if provided (for causal attention)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    # Apply softmax
    attention_weights = F.softmax(scores, dim=-1)
    
    # Compute weighted sum of values
    output = torch.matmul(attention_weights, V)
    
    return output, attention_weights

# Example: 3 words, 4-dimensional embeddings
Q = torch.randn(1, 3, 4)  # Queries
K = torch.randn(1, 3, 4)  # Keys  
V = torch.randn(1, 3, 4)  # Values

output, weights = scaled_dot_product_attention(Q, K, V)

print(f"Output shape: {output.shape}")
print(f"Attention weights:\n{weights[0]}")  # See which positions attend to which
```

### Diffusion Models (Reverse Process)

**Forward process (noise addition):**

$$
q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} x_{t-1}, \beta_t I)
$$

**Reverse process (denoising):**

$$
p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))
$$

**What it means:** Diffusion models learn to reverse a gradual noising process. The forward process adds Gaussian noise over many steps; the learned reverse process removes it.

**Why it matters:** This is how models like DALL-E and Stable Diffusion generate images. The forward process is simple (just add noise), but learning the reverse is what makes generation possible.

**Implementation:**

```python
import torch

def forward_diffusion(x_0, t, beta_schedule):
    """
    Add noise to x_0 at timestep t.
    
    Parameters:
    x_0: Clean data
    t: Timestep (how much noise to add)
    beta_schedule: Noise schedule (variance at each step)
    
    Returns:
    x_t: Noised data
    noise: The noise that was added
    """
    beta_t = beta_schedule[t]
    alpha_t = 1 - beta_t
    alpha_bar_t = torch.prod(torch.tensor([1 - beta_schedule[i] for i in range(t+1)]))
    
    # Sample noise
    noise = torch.randn_like(x_0)
    
    # Add noise according to schedule
    x_t = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1 - alpha_bar_t) * noise
    
    return x_t, noise

# Example: noise a 1D signal
x_0 = torch.tensor([1.0, 2.0, 3.0])
T = 100
beta_schedule = torch.linspace(0.0001, 0.02, T)

# Noise at different timesteps
for t in [0, 25, 50, 99]:
    x_t, noise = forward_diffusion(x_0, t, beta_schedule)
    print(f"t={t:3d}: {x_t.numpy()}")
```

---

## What's Missing

This guide deliberately omits several important topics to keep it focused:

- **Kernel methods**: SVMs, kernel PCA, Gaussian processes
- **Reinforcement learning**: Bellman equations, policy gradients, Q-learning
- **Statistical learning theory**: VC dimension, PAC learning, generalization bounds
- **Probabilistic graphical models**: Belief propagation, junction tree algorithm
- **Optimization theory**: Convex optimization, Lagrange multipliers, KKT conditions
- **Regularization**: L1/L2 penalties, dropout, batch normalization mathematics

Each of these deserves its own deep dive.

---

## References

**Books:**
- *Pattern Recognition and Machine Learning* by Christopher Bishop (thorough, Bayesian perspective)
- *Deep Learning* by Goodfellow, Bengio, and Courville (comprehensive, modern)
- *Information Theory, Inference, and Learning Algorithms* by David MacKay (beautiful connections)

**Papers:**
- Attention Is All You Need (Vaswani et al., 2017) - Transformers
- Denoising Diffusion Probabilistic Models (Ho et al., 2020) - Diffusion models
- ImageNet Classification with Deep CNNs (Krizhevsky et al., 2012) - AlexNet

**Courses:**
- [Stanford CS229: Machine Learning](https://cs229.stanford.edu/)
- [NYU Deep Learning](https://atcold.github.io/pytorch-Deep-Learning/)
- [Fast.ai Practical Deep Learning](https://course.fast.ai/)