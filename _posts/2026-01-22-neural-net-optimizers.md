---
layout: post
comments: true
title: "A Complete Guide to Neural Network Optimizers"
excerpt: A highly technical yet understandable exploration of various neural network optimization algorithms.
date: 2026-01-22
mathjax: true
---


Training neural networks is fundamentally an optimization problem: we're searching for the best set of weights that minimize our loss function. While the concept sounds straightforward, the path from random initialization to a well-trained model is rarely a smooth descent. The landscape of loss functions in high-dimensional spaces is filled with valleys, plateaus, and saddle points that can trap or slow down naive optimization approaches.

This is where optimization algorithms come in. Over the years, researchers have developed increasingly sophisticated methods to navigate these challenging landscapes more efficiently. Each optimizer builds upon the limitations of its predecessors, introducing new mechanisms to accelerate convergence, handle sparse gradients, or adapt to different learning scenarios.

In this guide, we'll explore seven key optimization techniques—SGD, Momentum, Nesterov Momentum, AdaGrad, RMSProp, Adam, and AdamW—examining how each one works, what problems it solves, and when you might want to use it.

---

## Quick Reference: Optimizer Comparison

| Optimizer | Key Feature | Solves Issue in | Pros | Cons |
|-----------|-------------|-----------------|------|------|
| SGD | Simple gradient descent | N/A | Easy to implement | Oscillation, fixed learning rate |
| Momentum | Gradient accumulation | SGD | Reduces oscillations | No anticipation of future trends |
| Nesterov | Lookahead gradients | Momentum | Better convergence | Slightly higher computation |
| AdaGrad | Adaptive learning rates | Nesterov | Handles sparse gradients | Learning rate decays too fast |
| RMSProp | Smoothed adaptive learning rates | AdaGrad | Stabilizes learning rates | Sensitive to hyperparameters |
| Adam | Momentum + RMSProp | RMSProp | Combines best features | May converge to suboptimal minima |
| AdamW | Decoupled weight decay | Adam | Better generalization | Requires tuning decay parameter |


---


## 1. Stochastic Gradient Descent (SGD)

**How It Works:** Updates weights by calculating gradients using a small batch of data.

$$w_t = w_{t-1} - \eta \nabla f(w_{t-1})$$

**Pros:**
- Simple and computationally efficient
- Works well with large datasets

**Cons:**
- Can oscillate or converge slowly, especially in narrow valleys or near saddle points
- Learning rate (η) is fixed, leading to potential overshooting or slow convergence

---

## 2. Momentum

**How It Works:** Accumulates gradients to build momentum in directions with consistent gradients.

$$v_t = \beta v_{t-1} - \eta \nabla f(w_{t-1})$$<br>
$$w_t = w_{t-1} + v_t$$

**Pros:**
- Speeds up convergence in shallow but consistent directions (e.g., valleys)
- Reduces oscillations compared to SGD

**Cons:**
- Still overshoots if the learning rate is too high
- Cannot predict future gradient directions

**Improvement Over SGD:** Addresses oscillation and slow convergence by incorporating past gradients.

---

## 3. Nesterov Momentum

**How It Works:** Looks ahead by computing gradients at the projected position.

$$v_t = \beta v_{t-1} - \eta \nabla f(w_{t-1} + \beta v_{t-1})$$<br>
$$w_t = w_{t-1} + v_t$$

**Pros:**
- More precise updates by considering where the momentum is leading
- Accelerates convergence further compared to vanilla momentum

**Cons:**
- Slightly more computationally expensive due to gradient computation at the lookahead point

**Improvement Over Momentum:** Anticipates future gradient directions, resulting in better convergence.

---

## 4. AdaGrad

**How It Works:** Adjusts the learning rate for each parameter based on the magnitude of past gradients.

$$g_t = \nabla f(w_{t-1})$$<br>
$$w_t = w_{t-1} - \frac{\eta}{\sqrt{G_t + \epsilon}} g_t, \quad G_t = \sum_{i=1}^t g_i^2$$

**Pros:**
- Works well for sparse gradients (e.g., NLP tasks)
- Automatically adapts learning rates for each parameter

**Cons:**
- Learning rate diminishes too quickly due to cumulative gradient sum, leading to potential underfitting

**Improvement Over Nesterov Momentum:** Introduces adaptive learning rates to handle sparse gradients.

---

## 5. RMSProp

**How It Works:** Modifies AdaGrad by using an exponentially weighted moving average of past squared gradients instead of a cumulative sum.

$$v_t = \beta v_{t-1} + (1 - \beta)(\nabla f(w_{t-1}))^2$$<br>
$$w_t = w_{t-1} - \frac{\eta}{\sqrt{v_t + \epsilon}} \nabla f(w_{t-1})$$

**Pros:**
- Prevents the learning rate from diminishing too quickly
- Suitable for non-stationary objectives

**Cons:**
- Sensitive to hyperparameter choices (e.g., β)

**Improvement Over AdaGrad:** Stabilizes learning rates by introducing an exponentially weighted average of squared gradients.

---

## 6. Adam (Adaptive Moment Estimation)

**How It Works:** Combines Momentum (first moment) and RMSProp (second moment).

- Update rules:<br>
$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) \nabla f(w_{t-1})$$<br>
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2)(\nabla f(w_{t-1}))^2$$

- Bias corrections:<br>
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

- Update step:<br>
$$w_t = w_{t-1} - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

**Pros:**
- Combines the benefits of Momentum and RMSProp
- Automatically adjusts learning rates for each parameter
- Bias correction ensures stability in early training

**Cons:**
- May converge to suboptimal solutions in some scenarios (e.g., small datasets or high regularization)
- Hyperparameter tuning can be challenging

**Improvement Over RMSProp:** Adds momentum and bias correction to handle noisy gradients and early instability.

---

## 7. AdamW

**How It Works:**

Decouples weight decay from the gradient update to improve generalization.

$$w_t = w_{t-1} - \eta \bigg( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda w_{t-1} \bigg)$$

**Pros:**
- Better generalization compared to Adam
- Retains benefits of adaptive learning rates

**Cons:**
- Still requires careful hyperparameter tuning

**Improvement Over Adam:** Decouples weight decay from gradient updates, improving generalization performance.

---

## Detailed Technical Comparison

| Method | Working Mechanism | Pros | Cons | Improvement Over Prior Method |
|--------|-------------------|------|------|-------------------------------|
| **SGD** | Updates weights using gradients calculated on mini-batches. $w_t = w_{t-1} - \eta\nabla f(w_{t-1})$ | Simple, computationally efficient | Oscillates, slow convergence, fixed learning rate | N/A |
| **Momentum** | Accumulates gradients to build momentum for smoother updates. $v_t = \beta v_{t-1} - \eta\nabla f(w_{t-1})$, $w_t = w_{t-1} + v_t$ | Speeds up convergence, reduces oscillations | May overshoot, lacks anticipation of future gradients | Reduces oscillations and improves convergence speed |
| **Nesterov** | Looks ahead to compute gradients at a projected future position. $v_t = \beta v_{t-1} - \eta\nabla f(w_{t-1} + \beta v_{t-1})$, $w_t = w_{t-1} + v_t$ | More precise updates, faster convergence | Slightly more computationally expensive | Anticipates future gradient directions |
| **AdaGrad** | Adjusts learning rates based on accumulated squared gradients. $w_t = w_{t-1} - \frac{\eta}{\sqrt{G_t + \epsilon}}g_t$, $G_t = \sum g_i^2$ | Adapts learning rates, good for sparse gradients | Learning rate diminishes too quickly, potential underfitting | Introduces adaptive learning rates for sparse features |
| **RMSProp** | Uses exponentially weighted moving averages of squared gradients. $v_t = \beta v_{t-1} + (1-\beta)g_t^2$, $w_t = w_{t-1} - \frac{\eta}{\sqrt{v_t + \epsilon}}g_t$ | Prevents learning rate decay, handles non-stationary objectives | Sensitive to hyperparameters (e.g., β) | Stabilizes learning rates using moving averages |
| **Adam** | Combines Momentum (1st moment) and RMSProp (2nd moment) with bias correction. | Fast convergence, handles noisy gradients | May converge to suboptimal minima in some cases | Combines momentum and adaptive learning rates |
| **AdamW** | Decouples weight decay from gradient updates. $w_t = w_{t-1} - \eta[\frac{\hat{m}\_t}{\sqrt{\hat{v}\_t} + \epsilon} + \lambda w_{t-1}]$ | Better generalization, retains Adam's benefits | Requires tuning of decay parameter | Improves generalization by decoupling weight decay |

---

## Hyperparameter Reference

| Method | Hyperparameter | Meaning | Typical Values | Tuning Suggestions |
|--------|----------------|---------|----------------|-------------------|
| **SGD** | Learning rate (η) | Step size for updating weights | 0.01 to 0.1 | Start with a smaller value and adjust based on convergence |
| **Momentum** | Momentum coefficient (β) | Controls the contribution of past gradients to the current update | 0.9 | Keep fixed at 0.9 or tune slightly |
| **Nesterov** | Momentum coefficient (β) | Same as Momentum, with anticipation of future gradients | 0.9 | Same as Momentum |
| **AdaGrad** | Learning rate (η) | Base learning rate scaled by the inverse square root of accumulated squared gradients | 0.01 | Lower than SGD learning rates to avoid overshooting |
| **RMSProp** | Learning rate (η) | Similar to AdaGrad, with smoothing via an exponential moving average | 0.001 to 0.01 | Tune for stability based on loss |
| | Decay rate (β) | Smoothing parameter for the moving average of squared gradients | 0.9 | Commonly fixed at 0.9 |
| **Adam** | Learning rate (η) | Base learning rate for parameter updates | 0.001 | Often works well without much tuning |
| | β₁ | Decay rate for the first moment (mean of gradients) | 0.9 | Usually fixed |
| | β₂ | Decay rate for the second moment (variance of gradients) | 0.999 | Keep fixed or tune slightly for sensitivity |
| | ε | Small value to avoid division by zero | 10⁻⁷ or smaller | Rarely changed |
| **AdamW** | Learning rate (η) | Same as Adam | 0.001 | Same as Adam |
| | β₁, β₂, ε | Same as Adam | 0.9, 0.999, 10⁻⁷ | Same as Adam |
| | Weight decay (λ) | Regularization parameter to control overfitting by penalizing large weights | 10⁻⁴ to 10⁻² | Start small and increase if overfitting is observed |

---

## Conclusion

Choosing the right optimizer can dramatically impact your model's training efficiency and final performance. While there's no universal "best" optimizer, understanding the strengths and weaknesses of each approach helps you make informed decisions for your specific use case.

For most modern deep learning applications, **Adam** and **AdamW** have emerged as go-to choices due to their robust performance across diverse tasks with minimal hyperparameter tuning. Adam's combination of momentum and adaptive learning rates makes it particularly effective for handling noisy gradients and training deep networks, while AdamW's improved weight decay mechanism often leads to better generalization.

However, don't overlook the classics. **SGD with Momentum** remains highly competitive, especially for computer vision tasks, and often achieves better final test accuracy when combined with proper learning rate scheduling. For problems with sparse gradients, such as natural language processing with large vocabularies, **AdaGrad** or **RMSProp** might be more appropriate.

The key takeaway is that optimizer selection should be guided by your problem's characteristics: dataset size, gradient sparsity, computational budget, and generalization requirements. Start with a well-established baseline (Adam is usually a safe bet), monitor your training dynamics, and don't hesitate to experiment with alternatives if you're not seeing the convergence behavior you expect.

As the field continues to evolve, new optimizers and variants will undoubtedly emerge. But the fundamental principles underlying these seven methods—managing learning rates, leveraging momentum, and adapting to gradient statistics—will remain central to training neural networks effectively.
