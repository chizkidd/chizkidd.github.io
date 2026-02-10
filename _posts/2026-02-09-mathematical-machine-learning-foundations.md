---
layout: post
comments: true
title: "Architectural and Mathematical Foundations of Machine Learning: A Rigorous Synthesis of Theory, Geometry, and Implementation"
excerpt: A rigorous exploration of machine learning mathematics, from information theory and linear algebra to optimization dynamics and generative models, with correct implementations and theoretical connections.
date: 2026-02-09
mathjax: true
---

---
>**Abstract:** The maturation of machine learning from a subfield of heuristic-driven statistics into a cornerstone of modern computational science has necessitated a re-evaluation of its pedagogical foundations. Modern practitioners often rely on high-level libraries that abstract away the underlying mathematics, but as evidenced by critical reviews from the research community, this abstraction often leads to a superficial understanding of model dynamics, failure modes, and optimization bottlenecks.[^1] A robust understanding of machine learning is not merely a collection of isolated equations but a synthesis of linear algebra, information theory, multivariate calculus, and probabilistic estimation. This report provides an exhaustive analysis of these mathematical pillars, correcting common technical misconceptions and bridging the gap between theoretical derivation and numerically stable implementation.

---

## The Geometry of Representation: Linear and Affine Transformations

In the discourse of deep learning, the term "linear layer" is frequently used as a shorthand for the fundamental operation of weight-input multiplication followed by a bias shift. However, a rigorous geometric analysis reveals that the operations defining neural networks are more accurately described as affine transformations.[^1] This distinction is not merely semantic; it defines the topology of the latent space and the constraints of the optimization landscape.

### Defining the Affine Space

A linear transformation between two vector spaces must satisfy two core properties: additivity and homogeneity. Geometrically, this requires that the transformation fixes the origin; the zero vector in the input space must map to the zero vector in the output space. In a standard neural network layer, the operation is defined as $y = Ax + b$. While the term $Ax$ represents a linear transformation (scaling, rotating, or shearing the input $x$), the addition of the bias vector $b$ shifts the resulting vector away from the origin.[^2]

This shift renders the transformation affine. An affine transformation is the composition of a linear mapping followed by a translation. In high-dimensional spaces, the bias term $b$ is what allows a hyperplane to exist in any position within the space, rather than being forced to pass through the coordinate center.[^2] Without this capability, the expressive power of a neural network would be severely diminished, as it would be unable to model datasets where the decision boundary does not intersect the origin.

| Feature | Linear Transformation ($Ax$) | Affine Transformation ($Ax+b$) |
|---------|------------------------------|--------------------------------|
| Origin Preservation | Maps zero vector to zero vector ($f(0) = 0$) | Shifts the origin by the vector $b$ <br>($f(0) = b$) |
| Algebraic Properties | Satisfies $f(x+y) = f(x) + f(y)$ and $f(cx) = cf(x)$ | Violates both unless $b = 0$[^2] |
| Geometric Action | Rotation, scaling, reflection, shearing | Rotation/Scaling followed by Translation |
| Machine Learning Role | Feature interaction and dimensionality change | Decision boundary positioning and normalization |

### Spectral Decomposition and the Warping of Space

Beyond simple layer operations, the internal structure of data matrices is analyzed through spectral decomposition. Eigendecomposition and Singular Value Decomposition (SVD) provide the mathematical tools to understand how a model "views" the variance of its input. An eigenvector $v$ of a square matrix $A$ is a characteristic direction that, under the transformation $A$, is only scaled by a factor $\lambda$, termed the eigenvalue: $Av = \lambda v$.[^3] Geometrically, if we align our coordinate system with these eigenvectors, the matrix $A$ simply acts as a scaling factor along those axes.[^4]

However, eigendecomposition is limited to square matrices and often lacks orthogonality in the basis vectors unless the matrix is symmetric.[^5] Singular Value Decomposition (SVD) generalizes this concept to any $m \times n$ matrix $A$, decomposing it into $A = U\Sigma V^T$. This decomposition reveals a three-step geometric process:

1. **Input Rotation**: The matrix $V^T$ rotates the input space to align with the principal axes of the data.[^6]
2. **Stretching**: The diagonal matrix $\Sigma$ scales the data along these axes according to the singular values $\sigma_i$.[^4]
3. **Output Rotation**: The matrix $U$ rotates the scaled data into the output coordinate system.[^6]

In the context of dimensionality reduction, SVD allows for the optimal projection of data onto a lower-dimensional subspace. By retaining only the largest $k$ singular values in $\Sigma$ and setting the rest to zero, we minimize the reconstruction error in terms of the Frobenius norm, a principle that underlies both Principal Component Analysis (PCA) and modern matrix completion algorithms.[^3]

## Information Theory as the Metric of Learning

While linear algebra defines the transformations, information theory provides the objective functions used to measure the "success" of those transformations. In machine learning, the goal is often to minimize the distance between a predicted probability distribution and the true data distribution.[^7]

### Surprisal and the Derivation of Entropy

The fundamental unit of information theory is "surprisal" or self-information. Intuitively, an event that is certain carries no information, whereas a rare event provides significant insight when it occurs. This is quantified by the negative logarithm of the probability $p$ of an event: $I(x) = -\log p(x)$.[^8] The logarithmic form is essential because it ensures that information is additive for independent events: $I(x,y) = I(x) + I(y)$.[^7]

Entropy $H(P)$ is the expected value of surprisal across an entire distribution $P$. It represents the average amount of uncertainty or "average surprise" inherent in the distribution[^7]:

$$H(P) = -\sum_{x} P(x) \log P(x)$$

A uniform distribution maximizes entropy, as it represents a state where every outcome is equally likely, providing the highest level of average uncertainty. In decision trees, for example, entropy is used to measure the "purity" of a node; a node with low entropy contains samples mostly from a single class, indicating high certainty in the prediction.[^10]

### Cross-Entropy: The Cost of Misaligned Models

When we train a model, we generate a predicted distribution $Q$ intended to approximate the true distribution $P$. Cross-entropy $H(P,Q)$ measures the average surprisal we experience if we encode data from $P$ using the "codebook" optimized for $Q$.[^8] Mathematically:

$$H(P, Q) = -\sum_{x} P(x) \log Q(x)$$

Cross-entropy is a staple loss function in classification tasks. It is inherently asymmetric ($H(P,Q) \neq H(Q,P)$), a property that reflects the physical reality of communication: the cost of using a wrong model depends on which direction the error occurs.[^7] Specifically, if the model $Q$ predicts a zero probability for an event that actually occurs in $P$, the cross-entropy becomes infinite, reflecting "infinite surprise" and forcing the model to never be "certainly wrong".[^7]

### Kullback-Leibler (KL) Divergence

KL Divergence $D_{KL}(P \| Q)$ isolates the "extra" surprisal caused by the model's inaccuracy. It is defined as the difference between cross-entropy and the inherent entropy of the data[^8]:

$$D_{KL}(P \| Q) = H(P, Q) - H(P) = \sum_{x} P(x) \log \frac{P(x)}{Q(x)}$$

Since the entropy of the data $H(P)$ is constant with respect to the model parameters, minimizing cross-entropy is functionally identical to minimizing KL divergence.[^7] This relationship is the backbone of Maximum Likelihood Estimation (MLE), as minimizing the divergence between the data and the model is equivalent to finding the parameters that make the observed data most probable.[^7]

| Metric | Formula | Intuition | Application |
|--------|---------|-----------|-------------|
| Entropy | $H(P) = -\sum P(x) \log P(x)$ | Average uncertainty of a single source | Data compression, decision tree splitting |
| Cross-Entropy | $H(P,Q) = -\sum P(x) \log Q(x)$ | Total cost of using model $Q$ for data $P$ | Loss function for classifiers[^8] |
| KL Divergence | $D_{KL}(P \| Q) = \sum P(x) \log \frac{P(x)}{Q(x)}$ | "Distance" or extra cost between distributions | Variational inference, GANs, RL regularization |

## Optimization Dynamics: Jacobians, Hessians, and the Curvature of Loss

Optimization in machine learning is essentially a navigation problem through a high-dimensional landscape. While the gradient provides the direction of the slope, higher-order derivatives provide the context of that slope: its sensitivity and its curvature.[^15]

### The Jacobian: First-Order Sensitivity

For a vector-valued function $f: \mathbb{R}^n \to \mathbb{R}^m$, the Jacobian matrix $J$ contains all first-order partial derivatives. Each element $J_{ij} = \frac{\partial f_i}{\partial x_j}$ represents how the $i$-th output changes with respect to the $j$-th input.[^17] In neural network training, the Jacobian is the fundamental object of backpropagation.

A common misunderstanding in technical literature is the classification of backpropagation itself. As noted in expert feedback, backpropagation is not an optimization algorithm like Gradient Descent; rather, it is a computationally efficient method for calculating the Jacobian through the application of the chain rule.[^1] The efficiency of backpropagation stems from its ability to reuse intermediate partial derivatives, avoiding the combinatorial explosion that would occur if each path through the network were differentiated independently.[^17]

### The Hessian and the Topology of Generalization

While the Jacobian tells us where to move, the Hessian matrix $H$ (the second derivative) tells us the shape of the area we are moving through. The Hessian is a square matrix of second-order partial derivatives: $H_{ij} = \frac{\partial^2 L}{\partial \theta_i \partial \theta_j}$.[^16] The eigenvalues of the Hessian at a local minimum define the "sharpness" or "flatness" of that minimum.[^19]

The "Flat Minimum" hypothesis suggests that minima with low curvature (low Hessian eigenvalues) are more likely to generalize to unseen data.[^21] The intuition is that a flat minimum represents a region of parameter space where small perturbations in the weights (caused by noise in the data or finite precision) do not significantly increase the loss. In contrast, a "sharp" minimum is highly sensitive; a slight shift in the data distribution might move the "true" minimum slightly, causing the loss for the sharp-minimum parameters to skyrocket.[^21]

| Hessian Eigenvalue Status | Geometric Interpretation | Generalization Outcome |
|---------------------------|--------------------------|------------------------|
| Large Eigenvalues | Sharp, steep "valley" | High sensitivity, prone to overfitting[^19] |
| Small Eigenvalues | Broad, flat "plateau" | Robust to noise, better generalization[^20] |
| Negative Eigenvalues | Surface curves downward (Max/Saddle) | Unstable, gradient descent will move away |
| Zero Eigenvalues | Function is locally linear | Inconclusive; often indicates overparameterization |

Advanced optimization algorithms, such as Sharpness-Aware Minimization (SAM), explicitly incorporate the Hessian's information by seeking parameter values whose entire neighborhood has low loss, rather than just a single point.[^22] This shift from point-wise optimization to neighborhood optimization marks a significant trend in improving the robustness of Large Language Models (LLMs).[^19]

## Statistical Frameworks: MLE, MAP, and the Bayesian Paradigm

The process of "learning" from data is fundamentally an exercise in statistical estimation. Machine learning models typically operate under one of two paradigms: Frequentist or Bayesian.[^24]

### Maximum Likelihood Estimation (MLE)

MLE is the Frequentist's primary tool. It assumes that there is a fixed, "true" parameter $\theta$ and seeks the value that makes the observed data $D$ most probable:

$$\hat{\theta}_{MLE} = \arg\max_{\theta} P(D|\theta)$$

In practice, we maximize the log-likelihood to transform product-based probabilities into summation-based losses, which are easier to differentiate and less prone to numerical underflow.[^26] MLE is effective for large datasets where the data itself provides enough signal to overcome initial uncertainty, but it is notoriously prone to overfitting in high-dimensional settings with sparse data.[^26]

### Maximum A Posteriori (MAP) and Regularization

MAP estimation adopts a Bayesian stance, treating the parameter $\theta$ as a random variable with its own prior distribution $P(\theta)$. Using Bayes' Theorem:

$$P(\theta|D) = \frac{P(D|\theta)P(\theta)}{P(D)}$$

$$\hat{\theta}_{MAP} = \arg\max_{\theta} P(\theta|D) = \arg\max_{\theta} [P(D|\theta)P(\theta)]$$

The inclusion of the prior $P(\theta)$ acts as a "regularizer." For instance, assuming a Gaussian prior centered at zero is mathematically equivalent to $L_2$ regularization (Weight Decay), while a Laplacian prior yields $L_1$ regularization (Sparsity).[^26] MAP provides a bridge between pure data-driven learning and the incorporation of domain knowledge, acting as the "experienced analyst" who balances new evidence against historical trends.[^28]

| Aspect | Maximum Likelihood Estimation (MLE) | Maximum A Posteriori (MAP) |
|--------|-------------------------------------|----------------------------|
| Philosophy | Frequentist: $\theta$ is fixed but unknown | Bayesian: $\theta$ is a random variable |
| Prior Used? | No[^24] | Yes ($P(\theta)$)[^25] |
| Regularization | None (unless explicit) | Implicit via the prior distribution[^26] |
| Data Sensitivity | High; prone to overfitting on small sets | Lower; the prior stabilizes estimates[^28] |
| Convergence | Converges to MAP as data size $\to \infty$ | Incorporates prior knowledge for small $n$ |

## Architecture Dynamics: Softmax, Attention, and Implicit Mappings

Modern neural architectures rely on specific functional forms to control the flow of information and the stability of gradients. Two of the most critical are the Softmax activation and the Attention mechanism.

### The Jacobian of Softmax and the Backpropagation Fusion

The Softmax function $\sigma(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$ is the standard output for multi-class classification. A common technical oversight in tutorial literature is failing to explain the Jacobian of Softmax. Because each output $\sigma_i$ depends on every input $z_j$ (due to the shared denominator), the derivative is not a simple vector but a matrix.[^13]

- **Diagonal elements**: $\frac{\partial \sigma_i}{\partial z_i} = \sigma_i(1 - \sigma_i)$
- **Off-diagonal elements**: $\frac{\partial \sigma_i}{\partial z_j} = -\sigma_i \sigma_j$

However, when Softmax is combined with the Categorical Cross-Entropy loss, the gradient of the entire block with respect to the input $z$ simplifies to $\sigma - y$, where $y$ is the one-hot encoded ground truth.[^13] This simplicity is why the combination is ubiquitous in deep learning frameworks; it provides a clean, linear error signal $(\sigma - y)$ that directly represents the model's confidence error.[^18]

### The Attention Mechanism: A Retrieval Framework

The Attention mechanism, particularly in the Transformer architecture, revolutionized sequential modeling by replacing fixed-length memory with a dynamic retrieval system. This system is defined by three vectors: Query ($Q$), Key ($K$), and Value ($V$).[^30]

The intuition is analogous to a library search:

- **Query ($Q$)**: The search term or information you are currently looking for.[^30]
- **Key ($K$)**: The metadata or "index" on the spine of every book.[^31]
- **Value ($V$)**: The actual content or "knowledge" inside the book.[^32]

The attention weight is computed by measuring the compatibility (dot product) between $Q$ and $K$. After scaling and applying Softmax, these weights determine how much of each $V$ is aggregated into the final representation.[^31] The "Scaled" Dot-Product Attention includes a factor of $\frac{1}{\sqrt{d_k}}$ to prevent the dot products from growing so large that the Softmax function enters a region of near-zero gradients, which would stall learning.[^32]

### Kernel Machines and the Mapping Paradox

Support Vector Machines (SVMs) and kernel-based methods provide an alternative to deep learning's explicit feature engineering. The "Kernel Trick" allows a model to operate in an implicitly high-dimensional space without ever actually computing the coordinates in that space.[^35]

By reformulating the optimization problem into its "Dual Form," the objective depends only on the dot products between inputs: $\langle x_i, x_j \rangle$.[^37] Replacing this dot product with a kernel function $K(x_i, x_j)$ effectively maps the data into a high-dimensional feature space where it may be linearly separable.[^35] For example, the Radial Basis Function (RBF) kernel corresponds to an infinite-dimensional feature space, yet it can be computed with a simple exponential function in the original input space.[^36]

| Kernel Type | Function $K(x,y)$ | Geometric Space |
|-------------|-------------------|-----------------|
| Linear | $x^T y$ | Original input space |
| Polynomial | $(x^T y + c)^d$ | Finite-dimensional feature combinations |
| Gaussian RBF | $\exp(-\gamma \|x-y\|^2)$ | Infinite-dimensional space[^38] |
| Sigmoid | $\tanh(\alpha x^T y + c)$ | Relates SVMs to Neural Networks |

## Generative Modeling: Variational Inference and Diffusion

The frontier of machine learning math is currently dominated by generative models, which require estimating the underlying probability density of high-dimensional data.

### Variational Autoencoders (VAEs) and the ELBO

VAEs treat generation as a latent variable problem: we assume data $x$ is generated from a hidden code $z$. The true posterior $P(z \mid x)$ is intractable, so we approximate it with $Q(z \mid x)$ (the encoder).[^39] To train this, we maximize the Evidence Lower Bound (ELBO)[^12]:

$$\text{ELBO} = \mathbb{E}_{Q(z|x)}[\log P(x|z)] - D_{KL}(Q(z|x) \| P(z))$$

The first term is the **Reconstruction Error**, ensuring the decoder can recreate the input from the code. The second is the **KL Regularizer**, which forces the latent codes to follow a standard Gaussian distribution.[^41] This ensures the latent space is well-behaved, allowing us to sample new points and generate realistic data.

### Diffusion Models: Score-Based Generative Dynamics

Diffusion models represent a paradigm shift. Rather than learning a mapping or a lower bound, they learn to reverse a stochastic process.[^43] The forward process gradually destroys data by adding Gaussian noise until the sample is pure noise.[^45]

The model is trained to predict the noise $\epsilon$ that was added at any given step $t$. By knowing how to remove the noise, the model can iteratively "denoise" a random sample into a high-quality data point.[^45] Mathematically, this is governed by the Stochastic Differential Equation (SDE):

$$dx = f(x, t)dt + g(t)dw$$

The reverse process involves the "score function" $\nabla_x \log p(x)$, which points in the direction of increasing data density.[^44] Modern diffusion models essentially learn this score function, providing a robust mathematical way to sample from complex, high-dimensional manifolds.[^43]

## Numerical Pragmatism: The Gap Between Math and Machine

One of the most persistent failures in machine learning development is the "theoretical success, numerical failure" trap. Mathematical equations assume infinite precision, but hardware operates on finite-precision floating-point numbers.[^49]

### The Softmax Instability

The Softmax function is mathematically robust but numerically fragile. Large logits cause the exponential function to overflow into `inf`, while large negative logits cause underflow to `0`, resulting in `NaN` gradients.[^49] The standard solution is the **Translation Invariance Trick**: subtracting the maximum value from all logits before exponentiating. This ensures that the largest exponent is $e^0 = 1$, preventing overflow and guaranteeing numerical stability.[^51]

### The LogSumExp Trick

In the calculation of cross-entropy, we often encounter the log of a sum of exponentials. A naive implementation would calculate the exponentials, sum them, and then take the log, which is prone to overflow. The stable approach uses the LogSumExp identity:

$$\log \sum_i e^{x_i} = \alpha + \log \sum_i e^{x_i - \alpha}$$

where $\alpha = \max_i x_i$.[^49] This ensures that intermediate computations stay within the representable range of floating-point numbers.

### Correct Implementation of Entropy

Critiques of earlier implementations highlighted that naive entropy calculations using `np.log(p, where=p > 0)` can be dangerous if the output is not properly initialized, as it leaves the results at $p=0$ locations as uninitialized garbage values.[^1] A robust implementation must explicitly handle the limit $\lim_{p \to 0} p \log p = 0$ to ensure consistency and correctness across the entire domain.[^1]

| Mathematical Operation | Potential Numerical Failure | Robust Implementation Strategy |
|------------------------|----------------------------|-------------------------------|
| Softmax | Overflow ($e^{1000} = \infty$) | Subtract maximum logit before $\exp$[^51] |
| Cross-Entropy | Underflow ($\log(0) = -\infty$) | Use LogSoftmax and LogSumExp fusion[^49] |
| Information Entropy | Uninitialized memory at $p=0$ | Use `np.where` with default initialization to zero[^1] |
| Hessian Calculation | High memory cost/Instability | Use Hessian-Vector Products (HVP)[^19] |
| SVD | Convergence failure on singular matrices | Use Moore-Penrose pseudo-inverse via SVD[^5] |

## Advanced Theoretical Integration: Kernels, Attention, and Manifold Learning

The synthesis of these concepts reveals the deeper structure of modern machine learning. For instance, the Attention mechanism can be viewed as a data-dependent kernel where the weights are dynamically computed for each input pair.[^31] Similarly, the success of Diffusion models is intrinsically linked to the spectral properties of the data manifold, the model learns to project noise back onto the low-dimensional manifold where data resides.[^44]

The distinction between "Sharp" and "Flat" minima provides a bridge between the optimization dynamics of the Hessian and the statistical requirements of generalization. A flat minimum is not just a point of low loss; it is a region of high local entropy in the parameter space, suggesting that the solution is not a lucky "overfit" but a robust feature of the data distribution.[^20]

## Synthesis and Recommendations for Practitioners

The evolution of machine learning mathematics demonstrates that technical robustness is achieved only through the rigorous application of foundational principles. The fundamental, mathematical, derivative approach used here addresses the community's concerns regarding "LLM slop" and technical vapidity.[^1] By explicitly connecting surprisal to KL divergence, and the Jacobian of Softmax to the cross-entropy gradient, we move from rote memorization to functional understanding.

For practitioners looking to improve their models, the focus should be on three critical areas:

1. **Numerical Integrity**: Always use fused loss functions and log-domain calculations to avoid the silent corruption of gradients.[^49]

2. **Geometric Awareness**: Recognize that model operations are affine, and prioritize architectures that allow for flexible decision boundary placement.[^2]

3. **Curvature Monitoring**: In high-stakes applications, move beyond monitoring simple training loss. Analyzing the Hessian spectrum or the local flatness of the solution provides the only reliable indicator of how a model will perform on unseen, real-world data.[^21]

The future of machine learning lies in this intersection of physics-inspired dynamics (Diffusion), information theory (Entropy), and the geometry of high-dimensional spaces (SVD/Kernels). As models continue to scale, the mathematical "shortcuts" of the past will increasingly fail, leaving only those who understand the foundational rigor of the field capable of driving its next major breakthroughs.[^1]

---

## References

[^1]: Important machine learning equations. Hacker News, accessed February 9, 2026, https://news.ycombinator.com/item?id=45050931

[^2]: Fundamentals Part 2: Hessians and Jacobians - Ian Quah, accessed February 9, 2026, https://ianq.ai/Hessian-Jacobian/

[^3]: Eigen Intuitions: Understanding Eigenvectors and Eigenvalues - Towards Data Science, accessed February 9, 2026, https://towardsdatascience.com/eigen-intuitions-understanding-eigenvectors-and-eigenvalues-630e9ef1f719/

[^4]: Introduction: The geometry of linear transformations - Department of Mathematics @ University of Toronto, accessed February 9, 2026, https://www.math.utoronto.ca/mpugh/Teaching/MAT267_19/Geometric_description_of_SVD.pdf

[^5]: Intuitively, what is the difference between Eigendecomposition and Singular Value Decomposition? - Mathematics Stack Exchange, accessed February 9, 2026, https://math.stackexchange.com/questions/320220/intuitively-what-is-the-difference-between-eigendecomposition-and-singular-valu

[^6]: Geometrical interpretations of SVD - Math Stack Exchange, accessed February 9, 2026, https://math.stackexchange.com/questions/1450097/geometrical-interpretations-of-svd

[^7]: Entropy, Cross-Entropy, and KL Divergence: Mathematical Foundations and Applications - by Sidharth SS. Medium, accessed February 9, 2026, https://medium.com/@sidharth.ss/entropy-cross-entropy-and-kl-divergence-mathematical-foundations-and-applications-6a6f23da5ef1

[^8]: Cross-entropy and KL divergence - Eli Bendersky's website, accessed February 9, 2026, https://eli.thegreenplace.net/2025/cross-entropy-and-kl-divergence/

[^10]: [D] A Short Introduction to Entropy, Cross-Entropy and KL-Divergence : r/MachineLearning, accessed February 9, 2026, https://www.reddit.com/r/MachineLearning/comments/7vhmp7/d_a_short_introduction_to_entropy_crossentropy/

[^12]: The evidence lower bound (ELBO) - Matthew N. Bernstein, accessed February 9, 2026, https://mbernste.github.io/posts/elbo/

[^13]: Derivative of the Softmax Function and the Categorical Cross-Entropy Loss - Medium, accessed February 9, 2026, https://medium.com/data-science/derivative-of-the-softmax-function-and-the-categorical-cross-entropy-loss-ffceefc081d1

[^15]: Why are the Hessian and Jacobian matrices important for quant? - Reddit, accessed February 9, 2026, https://www.reddit.com/r/quant/comments/1muhmro/why_are_the_hessian_and_jacobian_matrices/

[^16]: Hessian Matrix: A Guide to Second-Order Derivatives in Optimization and Beyond, accessed February 9, 2026, https://www.datacamp.com/tutorial/hessian-matrix

[^17]: Jacobian and Hessian Matrices - GeeksforGeeks, accessed February 9, 2026, https://www.geeksforgeeks.org/engineering-mathematics/jacobian-and-hessian-matrices/

[^18]: Back-propagation with Cross-Entropy and Softmax - MLDawn Academy, accessed February 9, 2026, https://www.mldawn.com/back-propagation-with-cross-entropy-and-softmax/

[^19]: A Scalable Measure of Loss Landscape Curvature for Analyzing the Training Dynamics of LLMs - arXiv, accessed February 9, 2026, https://arxiv.org/html/2601.16979

[^20]: Flat Minima and Generalization - Emergent Mind, accessed February 9, 2026, https://www.emergentmind.com/topics/flat-minima-and-generalization

[^21]: The Generalization Mystery: Sharp vs Flat Minima - inFERENCe, accessed February 9, 2026, https://www.inference.vc/sharp-vs-flat-minima-are-still-a-mystery-to-me/

[^22]: Connection between Flatness and Generalization - Tuan-Anh Bui, accessed February 9, 2026, https://tuananhbui89.github.io/blog/2024/sharpness/

[^24]: 7.5: Maximum A Posteriori Estimation, accessed February 9, 2026, https://web.stanford.edu/class/archive/cs/cs109/cs109.1218/files/student_drive/7.5.pdf

[^25]: MLE vs MAP: the connection between Maximum Likelihood and Maximum A Posteriori Estimation - Agustinus Kristiadi, accessed February 9, 2026, https://agustinus.kristia.de/blog/mle-vs-map/

[^26]: MLE vs MAP - GeeksforGeeks, accessed February 9, 2026, https://www.geeksforgeeks.org/data-science/mle-vs-map/

[^28]: The Intuition behind Maximum Likelihood Estimation (MLE)and Maximum A Posteriori Estimation (MAP) - by Bohsun Chen. Medium, accessed February 9, 2026, https://medium.com/@devcharlie2698619/the-intuition-behind-maximum-likelihood-estimation-mle-and-maximum-a-posteriori-estimation-map-b8ba1ba1078f

[^30]: Understanding Attention in Transformers: A Visual Guide - by Nitin Mittapally. Medium, accessed February 9, 2026, https://medium.com/@nitinmittapally/understanding-attention-in-transformers-a-visual-guide-df416bfe495a

[^31]: Query, Key, Value: The Foundation of Transformer Attention - Michael Brenndoerfer, accessed February 9, 2026, https://mbrenndoerfer.com/writing/query-key-value-attention-mechanism

[^32]: How GPT works: A Metaphoric Explanation of Key, Value, Query in Attention, using a Tale of Potion - by Lili Jiang. TDS Archive, Medium, accessed February 9, 2026, https://medium.com/data-science/how-gpt-works-a-metaphoric-explanation-of-key-value-query-in-attention-using-a-tale-of-potion-8c66ace1f470

[^35]: Kernel Trick Under The Hood. Understanding how SVMs handle… | by Nguyen Ha Thai Son | Data Science Collective, accessed February 9, 2026, https://medium.com/data-science-collective/kernel-trick-under-the-hood-246ca9b36bae

[^36]: Machine learning - How to intuitively explain what a kernel is? - Stats StackExchange, accessed February 9, 2026, https://stats.stackexchange.com/questions/152897/how-to-intuitively-explain-what-a-kernel-is

[^37]: Support Vector Machines (and the Kernel Trick) - Columbia University, accessed February 9, 2026, http://www.columbia.edu/~mh2078/MachineLearningORFE/SVMs_MasterSlides.pdf

[^38]: Mastering SVM Kernel Tricks: A Comprehensive Guide to Dual Problems and Kernel Functions - by Sanghavi harsh. Medium, accessed February 9, 2026, https://medium.com/@sanghaviharsh666/mastering-svm-kernel-tricks-a-comprehensive-guide-to-dual-problems-and-kernel-functions-612bfff2061e

[^39]: Variational autoencoder implemented in PyTorch. Derives the ELBO, Log-Derivative trick, Reparameterization trick. - GitHub, accessed February 9, 2026, https://github.com/tonyduan/variational-autoencoders

[^41]: Evidence lower bound - Wikipedia, accessed February 9, 2026, https://en.wikipedia.org/wiki/Evidence_lower_bound

[^43]: Understanding Diffusion Objectives as the ELBO with Simple Data Augmentation, accessed February 9, 2026, https://openreview.net/forum?id=NnMEadcdyD

[^44]: Diffusion Models and (Many) Differential Equations - Katie Keegan. Emory University, accessed February 9, 2026, https://katiekeegan.org/2025/08/11/diffeqs.html

[^45]: Notes on Diffusion Model: Intuition - Flaneur2020, accessed February 9, 2026, https://flaneur2020.github.io/posts/2024-07-22-diffusion-model/

[^49]: Implementing Softmax From Scratch: Avoiding the Numerical Stability Trap - MarkTechPost, accessed February 9, 2026, https://www.marktechpost.com/2026/01/06/implementing-softmax-from-scratch-avoiding-the-numerical-stability-trap/

[^51]: Numerically Stable Softmax and Cross Entropy - Jay Mody, accessed February 9, 2026, https://jaykmody.com/blog/stable-softmax/