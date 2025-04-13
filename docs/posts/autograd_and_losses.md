---
title: Empirical Risk and Cross-Entropy in MicroTorch
description: Dive deep into loss functions, expected risk in the MicroTorch.
authors:
  - nick
date:
  created: 2025-04-13
comments: true
categories:
  - Deep Learning
  - Machine Learning
  - Neural Networks
tags:
  - Empirical Risk
  - Cross Entropy Loss
  - Softmax
  - Classification
  - Binary Classification
  - Loss Function Design
  - Deep Learning Fundamentals
  - Backpropagation
  - Training Loop
---


In the previous chapter we prepared the [MicroTorch - Deep Learning from Scratch](./autograd_essential.md). Now, it's time to dive into creating the loss functions that will guide our model during training. In this session, we're going to focus on building two fundamental loss functions: `Binary Cross-Entropy (BCE)` and `Cross-Entropy (CE)`, using the Microtorch framework. These functions are essential for training models, especially for classification tasks, and I'll walk you through how to implement them from scratch.


![Autograd cover](../assets/autograd/losses.png)
/// caption
Medieval loss discovery
///


<!-- more -->

### [Check the Jupyter Notebook](https://github.com/nickovchinnikov/datasatanism/blob/master/code/13.Tensor.ipynb)

## Loss Functions

Loss functions, also known as cost functions or objective functions, are used to measure how well the model's predictions match the true labels. The loss function computes a scalar value that represents the error between the model's predictions and the target labels. **During training, the goal is to minimize this value using optimization techniques, thereby improving the model's performance.**


<iframe width="927" height="521" src="https://www.youtube.com/embed/s7blWKlV3uM" title="Building Loss Functions in MicroTorch: Expected Risk, Empirical Risk, and Binary Cross-Entropy" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>


In our framework, we define a base class `Loss` that serves as a template for various types of loss functions. We also provide a few commonly used loss functions like **Cross-Entropy Loss** and **Binary Cross-Entropy Loss**.

For more details you can [check my post about the cross-entropy loss](./cross_entropy_loss.md)


### Expected Risk and Reduction of Loss

In machine learning, the **expected risk** measures the **average loss of your model on the true data distribution** — not just the training set.

Mathematically, it's defined as:

$$R(f) = \mathbb{E}_{(x, y) \sim P_{\text{data}}} \left[ \mathcal{L}(f(x), y) \right]$$


Where:

- $R(f)$ is the expected risk (true error)

- $\mathcal{L}$ is your loss function

- $f(x)$ is the model's prediction

- and $(x, y)$ are samples drawn from the data distribution


Let's break it down and first we focus on: 

$$\mathbb{E}_{(x, y) \sim P_{\text{data}}}$$

This is saying:

> **Take the expected value (average) over all possible input-output pairs $(x, y)$** that are drawn from the true data distribution $P_{\text{data}}$.

What's Going On: $\mathbb{E}$ is the expectation operator — it computes the *average* value of whatever comes after it. $(x, y) \sim P_{\text{data}}$ means - we are sampling input-output pairs from a **probability distribution** $P_{\text{data}}$. This is the *true* underlying distribution that your data comes from in the real world. Think of it as: _"draw all possible data points the universe can produce."_ (Not just the ones in your training set.)

**Putting it all together:**

$$R(f) = \mathbb{E}_{(x, y) \sim P_{\text{data}}} \left[ \mathcal{L}(f(x), y) \right]$$

Means:

> **"On average, over the entire true data distribution, how much error does my model make?"**

That's the **expected risk** — it tells us how good our model is **in the real world**, not just on our training samples.

Since we don't know the *full data distribution*, we approximate this using the **empirical risk** - the average loss over a finite training set:

$$\hat{R}(f) = \frac{1}{N} \sum_{i=1}^N \mathcal{L}(f(x_i), y_i)$$

In code, this corresponds to:

```python
loss.mean()
```

If you're summing the individual losses:

$$\mathcal{L}_{\text{sum}} = \sum_{i=1}^N \mathcal{L}(f(x_i), y_i)$$

You're computing the **total loss across the batch**. This is not normalized, so it's not equivalent to empirical risk — you're missing the $\frac{1}{N}$ factor.

In code:

```python
loss.sum()
```

This is where **loss reduction** comes into play.

After computing the loss for each element in a batch, we typically **reduce** it to a single scalar — this is known as **loss reduction**. You can choose:


- Using **sum** makes the total loss grow with batch size, which may affect learning rate sensitivity.

- **Mean** normalizes the loss, making it more stable across batches.

- Keeping **none** gives full control, useful for debugging or custom aggregation.


The `reduction_loss` function handles this reduction process. Here's the implementation:

```python
def reduction_loss(
    loss: Tensor, reduction: Literal["mean", "sum", "none"] = "mean"
) -> Tensor:
    r"""
    Reduction loss function.
    Apply the specified reduction method to the loss.

    Args:
        loss (Tensor): The computed loss tensor.
        reduction (str): The reduction method to apply, can be "mean", "sum", or "none".

    Returns:
        Tensor: The reduced loss value.
    """

    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss
```


**When Would You Use `mean`?**

- **Standard Training Setups**: The most common choice, `mean` normalizes the loss across batches, providing consistent scaling regardless of batch size, ensuring training stability.

- **Unbiased Estimator of Expected Risk**: By averaging the losses, you get an unbiased estimate of the **expected risk**, which is the true error of the model when evaluated on the full data distribution (not just the training samples). It helps avoid the model being biased by large batches or a small batch size.

- **Stability**: Using `mean` ensures that learning rate settings remain consistent across different batch sizes, making optimization more stable.


**When Would You Use `sum`?**

- **Gradient Accumulation**: If you're manually accumulating gradients over multiple small batches before an optimizer step, summing the losses ensures the gradients accumulate properly across all batches.

- **Loss Weighting**: If you're applying a global scaling factor later in the training process and want to control the weight globally, using `sum` allows you to work with the total loss rather than averaging over the batch.

- **Scale Consistency**: Be cautious when the batch size varies — using `sum` introduces scale inconsistency. The loss (and gradients!) will change if your batch size is different from one iteration to the next, which might affect training stability.


**When Would You Use `none`?**

- **Per-Sample Losses**: If you need the loss per individual sample (e.g., when you're applying a custom weighting or masking strategy), `none` allows you to work with each sample's loss independently.

- **Debugging**: When debugging or analyzing your model's behavior on specific examples, having access to the per-sample loss can be extremely helpful in pinpointing issues with certain data points.

- **Custom Aggregation Logic**: If you're implementing custom loss reduction strategies, or if you want to compute a non-standard aggregate of the per-sample losses, keeping the losses separate with `none` offers the flexibility needed to apply custom logic.


| Reduction | Math | Approximates | Use Case |
|----------|------|--------------|----------|
| `mean` | \( \frac{1}{N} \sum \mathcal{L}_i \) | Expected Risk \( \mathbb{E}[\mathcal{L}] \) | Standard training |
| `sum` | \( \sum \mathcal{L}_i \) | \( N \cdot \text{Empirical Risk} \) | Gradient accumulation, custom loss scaling |
| `none` | \( [\mathcal{L}_1, ..., \mathcal{L}_N] \) | Per-sample view | Custom reductions, masking, debugging |



### Base Loss Class

The `Loss` class serves as a base class for all loss functions. It defines a `compute_loss` method that subclasses must implement, and a `forward` method that **applies the loss computation and reduction**:


```python
class Loss(Module):
    r"""
    Base class for loss functions.

    This class provides a common interface for all loss functions.
    """

    def __init__(self, reduction: Literal["mean", "sum", "none"] = "mean"):
        r"""
        Initialize the loss function with the specified reduction method.

        Args:
            reduction (str): The reduction method to apply to the loss. Options are "mean", "sum", or "none".
        """

        self.reduction = reduction

    def compute_loss(self, pred: Tensor, target: Tensor) -> Tensor:
        r"""
        Compute the loss function. This method must be implemented in subclasses.

        Args:
            pred (Tensor): The predicted values (output of the model).
            target (Tensor): The true target values (ground truth).

        Returns:
            Tensor: The computed loss value.
        """

        raise NotImplementedError

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        r"""
        Forward pass.
        Apply the loss computation and reduction.

        Args:
            pred (Tensor): The predicted values (output of the model).
            target (Tensor): The true target values (ground truth).

        Returns:
            Tensor: The reduced loss value after applying the specified reduction.

        Raises:
            ValueError: If `pred` and `target` do not have the same shape.
        """

        if pred.shape != target.shape:
            raise ValueError(
                f"Input and target must have the same shape, but got {pred.shape} and {target.shape}"
            )

        loss = self.compute_loss(pred, target)
        return reduction_loss(loss, self.reduction)

```


### Binary Cross-Entropy Loss

Binary Cross-Entropy Loss is used for binary classification tasks, where each output is a probability value between 0 and 1. It measures the dissimilarity between the true labels and predicted probabilities for binary classification. The formula is:

$$L(y, \hat{y}) = -(y \log(\hat{y}) + (1 - y) \log(1 - \hat{y}))$$

Where $y$ is the true label (0 or 1) and $\hat{y}$ is the predicted probability (between 0 and 1).

The `BCELoss` class implements this loss:

```python
class BCELoss(Loss):
    r"""
    Binary Cross Entropy (BCE) Loss function.
    
    This loss function is used for binary classification tasks. It measures 
    the difference between the predicted probabilities and the actual binary 
    values (0 or 1).

    Args:
        eps (float): Small constant to avoid numerical instability when
            taking logarithms of values close to 0 or 1.

    Inherits from:
        Loss: The base loss class.
    """

    def __init__(self, eps: float = 1e-9):
        """
        Initialize BCE Loss with mean reduction by default.
        """

        super().__init__(reduction="mean")
        self.eps = eps

    def compute_loss(self, prediction: Tensor, target: Tensor) -> Tensor:
        r"""
        Compute Binary Cross Entropy Loss.

        The Binary Cross-Entropy loss is defined as:

            L = -(target * log(prediction) + (1 - target) * log(1 - prediction))

        where `prediction` is the predicted probability, and `target` is the 
        true label (0 or 1).

        Args:
            prediction (Tensor): The predicted probabilities (values between 0 and 1).
            target (Tensor): The true labels (0 or 1).

        Returns:
            Tensor: The computed Binary Cross-Entropy loss for each sample.
        """

        # Clip predictions to avoid log(0) or log(1)
        pred = prediction.clip(self.eps, 1 - self.eps)
        # Compute the BCE loss using the formula: -(y*log(p) + (1-y)*log(1-p))
        loss = -(target * pred.log() + (1 - target) * (1 - pred).log())
        return loss

```


### Cross-Entropy Loss and Softmax Together


<iframe width="927" height="521" src="https://www.youtube.com/embed/ftob7lsheX4" title="" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>


When we are dealing with multiple classes instead of just two, we need to **scale up** the entropy from `BinaryCrossEntropyLoss` to `CrossEntropyLoss`. And the `Sigmoid` function is not the best choice for this task. `Sigmoid` outputs probabilities for each class independently, which is not good for multi-class classification. Instead, we need to assign probabilities across multiple classes, ensuring they **sum to 1**. A much better approach is to use the `Softmax` function, which converts raw model outputs (logits) into a probability distribution over all classes. This allows our model to make more accurate predictions by selecting the class with the highest probability.

The numerically stable `Softmax` calculation:

$$\text{Softmax}(x)_i = \frac{\exp(x_i - \max(x))}{\sum \exp(x_j - \max(x))}$$


In **multiclass classification**, the combination of **Softmax + Cross-Entropy Loss** has a unique property that simplifies the backward pass.

**The Softmax function** is defined as: $S_i = \frac{e^{z_i}}{\sum_{j} e^{z_j}}$ and its derivative forms a **Jacobian matrix**:

$$\frac{\partial S_i}{\partial z_j} =
\begin{cases}
S_i (1 - S_i) & \text{if } i = j \\
- S_i S_j & \text{if } i \neq j
\end{cases}$$


This **Jacobian matrix** is $N \times N$ (where $N$ is the number of classes), which makes direct backpropagation inefficient.

But, the Cross-Entropy Loss $L = -\sum_{i} y_i \log(S_i)$, and its gradient **after softmax** is simply:

$$\frac{\partial L}{\partial z} = S - y$$


The Softmax Jacobian **cancels out** with the Cross-Entropy derivative, so we **avoid computing the full Jacobian**. Instead, Softmax **directly passes** the gradient from Cross-Entropy, making backpropagation **simpler and more efficient**!


??? note "Why does the derivative of Cross-Entropy take the form $\frac{\partial L}{\partial z_i} = S_i - y_i$?"

    The Cross-Entropy Loss function is $L = -\sum_{i} y_i \log(S_i)$, where $y_i$ is the one-hot encoded true label ($y_i = 1$ for the correct class, 0 otherwise). $S_i$ is the softmax output (predicted probability for class $i$).

    Now, let's compute the derivative of $L$ with respect to $S_i$:

    $$\frac{\partial L}{\partial S_i} = -\frac{y_i}{S_i}$$

    However, the goal is to compute the gradient with respect to $z_i$ (the input logits), not $S_i$. This is where the `Softmax` derivative comes in. Softmax is defined as:

    $$S_i = \frac{e^{z_i}}{\sum_{j} e^{z_j}}$$


    The derivative of $S_i$ with respect to $z_j$ gives a **Jacobian matrix**:

    $$
    \frac{\partial S_i}{\partial z_j} =
    \begin{cases}
    S_i (1 - S_i) & \text{if } i = j \quad \text{(diagonal terms)}\\
    - S_i S_j & \text{if } i \neq j \quad \text{(off-diagonal terms)}
    \end{cases}
    $$

    This means that if we want to find how the loss $L$ changes with respect to $z_i$, we need to apply the **chain rule**:

    $$\frac{\partial L}{\partial z_i} = \sum_{j} \frac{\partial L}{\partial S_j} \frac{\partial S_j}{\partial z_i}$$

    Substituting:

    $$\frac{\partial L}{\partial S_j} = -\frac{y_j}{S_j}$$

    and

    $$
    \frac{\partial S_j}{\partial z_i} =
    \begin{cases}
    S_j (1 - S_j) & \text{if } i = j \\
    - S_j S_i & \text{if } i \neq j
    \end{cases}
    $$

    Let's expand:

    $$
    \frac{\partial L}{\partial z_i} = \sum_{j} -\frac{y_j}{S_j} \cdot \frac{\partial S_j}{\partial z_i}
    $$

    Breaking it into cases:

    1. **Diagonal term ($i = j$)**:

    $$
    -\frac{y_i}{S_i} \cdot S_i (1 - S_i) = - y_i (1 - S_i)
    $$

    2. **Off-diagonal terms ($i \neq j$)**:

    $$
    -\frac{y_j}{S_j} \cdot (- S_j S_i) = y_j S_i
    $$

    Summing over all $j$, we get:

    $$
    \frac{\partial L}{\partial z_i} = - y_i (1 - S_i) + \sum_{j \neq i} y_j S_i
    $$

    Since $y$ is a one-hot vector, only one $y_j = 1$, and all others are 0, meaning:

    $$
    \frac{\partial L}{\partial z_i} = S_i - y_i
    $$

    **Intuition Behind Cancellation**

    Instead of explicitly computing the full Softmax Jacobian, the multiplication of the Cross-Entropy derivative and the Softmax Jacobian **simplifies directly to $S - y$**.

    - This happens because the off-diagonal terms in the Jacobian sum *cancel out in the chain rule application.*
    - The result is **a simple gradient computation** without the need for the full Jacobian matrix.

    This is why, in backpropagation, the Softmax layer doesn't need to explicitly compute its Jacobian. Instead, we can directly use:

    $$\frac{\partial L}{\partial z} = S - y$$

    to efficiently update the parameters in neural network training.


Now we are ready to make implementation of the `CrossEntropyLoss` function with the `Softmax`:


```python
class CrossEntropyLoss(Loss):
    def __init__(self, eps: float = 1e-9):
        """
        Cross-Entropy Loss function for multi-class classification.
        
        This loss combines softmax activation and negative log-likelihood loss
        for multi-class classification problems.
        
        Args:
            eps (float): Small constant to avoid numerical instability when
                taking logarithms of values close to 0.
        """
        super().__init__(reduction="mean")
        self.eps = eps

    def compute_loss(self, prediction: Tensor, target: Tensor) -> Tensor:
        """
        Computes the Cross-Entropy Loss.

        Args:
            prediction (Tensor): The raw model outputs (logits).
            target (Tensor): The ground truth labels (one-hot encoded or class indices).

        Returns:
            Tensor: The computed cross-entropy loss.
        """

        # For numerical stability, subtract max value (doesn't change softmax result)
        shifted = prediction - prediction.max(axis=-1, keepdims=True)

        # Compute softmax probabilities: exp(x_i) / sum(exp(x_j))
        exp_values = shifted.exp()
        probabilities = exp_values / (exp_values.sum(axis=-1, keepdims=True) + self.eps)

        # Compute Cross-Entropy
        loss = -(target * (probabilities + self.eps).log()).sum(axis=-1)

        return loss

```

**But there is another way.** The **LogSumExp (LSE)** function is defined as:  

$$\operatorname{LSE}(z) = \log \sum_{j} e^{z_j}$$

For numerical stability, it's often rewritten as:  

$$\operatorname{LSE}(z) = \max(z) + \log \sum_{j} e^{z_j - \max(z)}$$

Now we need to revisit the softmax function. It's defined as:

$$S_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

If we take the logarithm of both sides:

$$\log S_i = \log \left( \frac{e^{z_i}}{\sum_j e^{z_j}} \right)$$

Applying **log rules** - $\log e^{z_i} = z_i$ and $\log \frac{a}{b} = \log a - \log b$:

$$\log S_i = z_i - \log \sum_j e^{z_j}$$

This is **exactly** the key equation used in the **LogSumExp (LSE)** version:

$$\log S_i = z_i - \operatorname{LSE}(z)$$

So instead of computing **Softmax first**, then taking the log, we **directly compute log-softmax in one step**.

To implement **LogSumExp** in the `Tensor` class, we simply define it using existing operations like `sum`, `squeeze`, `max`, `exp`, and `log`. Since these functions already have their `backward` steps implemented, we don't need to manually define backpropagation for `logsumexp`.  

During the backward pass, each operation computes its gradient step-by-step, propagating derivatives automatically. **Mathematically, computing the gradient separately for each step or treating the entire `logsumexp` function as a single operation gives the same result.**


```python
def logsumexp(
    self,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdims: bool = False
):
    x_max = self.max(axis, keepdims)
    shifted_exp = (self - x_max).exp()
    sum_exp = shifted_exp.sum(axis, keepdims)
    logsumexp = sum_exp.log() + x_max.squeeze(axis)

    return Tensor(logsumexp, requires_grad=self.requires_grad)

```


And now we are ready to implement the `CrossEntropyLoss` with `logsumexp`:


```python
class CrossEntropyLoss(Loss):
    r"""
    Cross-Entropy Loss function with Log-Softmax.

    This loss is used for multi-class classification. Instead of computing
    Softmax explicitly, it applies Log-Softmax internally for numerical stability.

    Args:
        eps (float): Small constant to prevent log(0) issues.

    Inherits from:
        Loss: The base loss class.
    """

    def __init__(self, eps: float = 1e-9):
        super().__init__(reduction="mean")
        self.eps = eps

    def compute_loss(self, logits: Tensor, target: Tensor) -> Tensor:
        r"""
        Compute Cross-Entropy Loss using Log-Softmax.

        Instead of computing softmax explicitly, we use the identity:

            log(Softmax(x)) = x - logsumexp(x)

        This improves numerical stability and simplifies backpropagation.

        Args:
            logits (Tensor): The raw output logits (not probabilities).
            target (Tensor): The true labels (one-hot encoded).

        Returns:
            Tensor: The computed Cross-Entropy loss.
        """

        # Compute log-softmax in a numerically stable way
        log_softmax = logits - logits.logsumexp(axis=-1, keepdims=True)

        # Compute cross-entropy loss (negative log likelihood)
        loss = -(target * log_softmax).sum(axis=-1)

        return loss

```

The `compute_loss` method calculates the Cross-Entropy loss for each data point and loss is summed across the last axis (`axis=-1`) to account for all classes.



## Summary

In this post, we dive into the concept of **empirical risk** and its relationship to loss functions in deep learning. We implement `CrossEntropyLoss` and `BCELoss` from scratch using the MicroTorch framework and explore the math behind expected risk, reduction strategies (`mean`, `sum`, `none`), and why the combination of *Softmax + CrossEntropy* simplifies backpropagation.
