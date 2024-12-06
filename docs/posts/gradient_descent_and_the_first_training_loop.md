---
title: Essential Math for the Deep Learning
description: Build your first traing loop with pure mathematical concepts
authors:
  - nick
date:
  created: 2024-12-10
comments: true
categories:
  - Mathematics
  - Programming
  - Machine Learning
  - Data Science
  - Neural Networks
tags:
  - Gradient Descent
  - Optimization Algorithms
  - Machine Learning Basics
  - Python Visualization
  - Matplotlib
---

You may ask me - you deceived us in the previous chapter you shown us only the derivative, no gradient! Yes, you are right, and today we'll build the real gradient descent and build our first training loop!

<!-- more -->

First, let's come back for a moment to the gradient. I have a question - why the gradient is the steepest ascent? You can check my [full explanation here](./why_does_the_gradient_point_to_the_steepest_ascent.md)


## What is gradient?

Gradient is a vector of partial derivatives of a function. It's a vector of derivatives of a function with respect to its variables.
For example, if we have a function $f(x, y)$, then the gradient of this function is a vector of its partial derivatives with respect to $x$ and $y$.

$$\nabla f(x, y) = \left(\frac{\partial f}{\partial x}, \frac{\partial f}{\partial y}\right)$$

In the previous video we talked about the derivative but for the gradient it's a pretty similar but scaled to the multi dims space. **Gradient is the direction of the steepest ascent in every dimention.**


## Gradient descent

On the gradient field plot you see the gradient vector at each point. The gradient vector points in the direction of the steepest ascent. If we start at some point on the field and move in the opposite direction of the gradient vector, we will move downhill. This is the idea of gradient descent - follow the direction of the gradient to reach the minimum of the function.


### Recall of the 2D case

The best intuitive way is to reduce everything to the *2d case -> input x and output y*. [More information here](./gradient_descent_downhill_to_the_minima.md)

```python
def cdiff(func, x0, h=1e-3):
    """Centered difference approximation of the derivative"""
    return (func(x0 + h) - func(x0 - h)) / (2 * h)


def gradient_descent(func, x0, learning_rate, num_iterations, normalize=False):
    r"""
    Gradient descent algorithm to find the minimum of a function.

    Args:
    func: function to minimize
    x0: starting x coordinate
    learning_rate: learning rate
    num_iterations: number of iterations
    normalize: normalize the gradient
    """

    result = [x0]

    for _ in range(num_iterations):
        # Use centered difference approximation to compute the gradient
        grad = cdiff(func, x0)

        # Update the x coordinate in the opposite direction of the gradient
        x1 = x0 - grad * learning_rate

        result.append(x1)
        x0 = x1

    return result


# Let's define a function to minimize and plot the result
def func(x):
    return x**2


# Hyperparameters
x0 = 10
learning_rate = 0.1
num_iterations = 100


def plot_gradient_descent(x0, learning_rate, num_iterations):
    # Generate x values for plotting
    x = np.linspace(-10, 10, 200)
    y = func(x)

    # Run gradient descent
    x_path = gradient_descent(func, x0, learning_rate, num_iterations)

    # Create the figure and axis
    _, ax = plt.subplots(figsize=(12, 6))

    # Plot gradient descent path
    x_path = np.array(x_path)
    y_path = func(x_path)

    ax.plot(x_path, y_path, 'g.-', markersize=10, label='Gradient Descent Path')

    # Plot arrows to show direction
    for i in range(len(x_path) - 1):
        ax.annotate(
            '',
            xy=(x_path[i+1], y_path[i+1]),
            xytext=(x_path[i], y_path[i]),
            arrowprops=dict(facecolor='red', shrink=0.01, width=2),
        )

    ax.plot(x, y, 'b-', label='$f(x) = x^2$')

    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.set_title('Gradient Descent Path')
    ax.legend()

    plt.show()

# Create interactive plot
interact(plot_gradient_descent, 
    x0=FloatSlider(min=-10, max=10, step=1, value=9, description='Starting x'),
    learning_rate=FloatSlider(min=0.01, max=1, step=0.1, value=0.1, description='Learning Rate'),
    num_iterations=IntSlider(min=1, max=20, step=1, value=3, description='Number of Iterations'))

```

![Gradient in 2D](../assets/gradient_descent_and_the_first_training_loop/descent_in_2d.png){ align=center }
/// caption
Gradient descent in 2D
///


## Problem: Numerical instability

Using the **Centered Difference** approximation of the derivative can lead us to the numerical instability. The function optimization is a very precise job, we can't use metods that add instability and unpredictability to our result. I found one exact case that can demonstrate this.

Let's compare the numerical derivative of the oscillating function $f(x) = \sin(\frac{1}{x})$ with its exact derivative, we first need to determine the exact derivative analytically.


### Exact Derivative

The exact derivative of $f(x) = \sin(\frac{1}{x})$ can be computed using the chain rule. The derivative is:

$$
f'(x) = \cos(\frac{1}{x}) \cdot \left(-\frac{1}{x^2}\right) = -\frac{\cos(\frac{1}{x})}{x^2}
$$

This derivative also oscillates as $x \rightarrow 0$ due to the $\cos(\frac{1}{x})$ term.


### Implementation

Let's implement a Python script that computes both the numerical and exact derivatives for $f(x) = \sin(\frac{1}{x})$ as $x$ approaches 0, and then compare the results.

Here are the LaTeX formulas for the given oscillating function and its exact derivative:

1. **Oscillating Function**:
The function $f(x)$ is defined as:

$$
f(x) = 
\begin{cases} 
\sin\left(\frac{1}{x}\right) & \text{if } x \neq 0 \\
0 & \text{if } x = 0
\end{cases}
$$

This function oscillates rapidly as $x \to 0$.

2. **Derivative**:
The derivative $f'(x)$ is:

$$
f'(x) = 
\begin{cases} 
-\frac{\cos\left(\frac{1}{x}\right)}{x^2} & \text{if } x \neq 0 \\
0 & \text{if } x = 0
\end{cases}
$$


### Function and derivative plot

We can build a function with the derivative

```python
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider, IntSlider


def oscillating(x):
    """Vector-valued function that oscillates rapidly as x approaches 0."""
    result = np.zeros_like(x)  # Initialize an array of zeros with the same shape as x
    non_zero_mask = x != 0  # Mask for elements where x is not equal to 0
    result[non_zero_mask] = np.sin(1 / x[non_zero_mask])  # Apply sin(1/x) where x != 0
    return result


def d_oscillating(x):
    """Vector-valued exact derivative of the oscillating function."""
    result = np.zeros_like(x)  # Initialize an array of zeros with the same shape as x
    non_zero_mask = x != 0  # Mask for elements where x is not equal to 0
    result[non_zero_mask] = -np.cos(1 / x[non_zero_mask]) / (x[non_zero_mask] ** 2)  # Derivative where x != 0
    return result
    

def plot_oscillating(min=-0.1, max=0.1, steps=500):
    # Generate values of x around 0, excluding 0 to avoid division by zero
    x_values = np.linspace(min, max, steps)

    y_values = oscillating(x_values)
    dy_values = d_oscillating(x_values)

    # Create plots
    plt.figure(figsize=(14, 6))

    # Plot oscillating function
    plt.subplot(1, 2, 1)
    plt.plot(x_values, y_values, label='Oscillating Function', color='blue')
    plt.title('Oscillating Function $f(x) = sin(1/x)$')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid(True)
    plt.legend()

    # Plot exact derivative
    plt.subplot(1, 2, 2)
    plt.plot(x_values, dy_values, label='Exact Derivative', color='orange')
    plt.title("Exact Derivative $f'(x) = -\\frac{cos(1/x)}{x^2}$")
    plt.xlabel('x')
    plt.ylabel("f'(x)")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

interact(
    plot_oscillating,
    min=FloatSlider(min=-0.1, max=-0.01, step=0.01, value=-0.1, description='Min'),
    max=FloatSlider(min=0.01, max=0.1, step=0.01, value=0.1, description='Max'),
    steps=IntSlider(min=100, max=1000, step=100, value=500, description='Steps')
)

```

#### Plot

![Oscillating Function VS Exact Derivative](../assets/gradient_descent_and_the_first_training_loop/oscilation_vs_derivative.png){ align=center }
/// caption
Oscillating Function VS Exact Derivative
///



The derivative of this function is not well-defined at $x = 0$, but for practical purposes, we can treat it as $0$.

These expressions describe an oscillating function that becomes increasingly unpredictable as $x$ approaches zero, and its derivative reflects the complex behavior in the same region.


```python
import numpy as np


def cdiff(func, x, h=1e-3):
    """Centered difference approximation of the derivative."""
    return (func(x + h) - func(x - h)) / (2 * h)


def oscillating(x):
    """Function that oscillates rapidly as x approaches 0."""
    return np.sin(1/x) if x != 0 else 0


def d_oscillating(x):
    """Exact derivative of the oscillating function."""
    if x != 0:
        return -np.cos(1/x) / (x**2)
    else:
        return 0  # Not defined, but can be treated as 0 for comparison


# Test values close to 0
x_values = [0.1, 0.01, 0.001, 0.0001, 0.00001]

# Compute and compare the derivatives
print(f"{'x':>12} | {'Numerical Derivative':>20} | {'Exact Derivative':>20} | {'Abs Error':>20}")
print("-" * 80)
for x in x_values:
    numerical_derivative = cdiff(oscillating, x)
    exact_deriv = d_oscillating(x)
    
    # Calculate the relative error
    if exact_deriv != 0:  # Avoid division by zero for relative error
        abs_error = np.abs((numerical_derivative - exact_deriv))
    else:
        abs_error = np.nan  # Not defined for comparison

    print(f"{x:>12} | {numerical_derivative:>20.6f} | {exact_deriv:>20.6f} | {abs_error:>20.6f}")

```


#### Output

|       x       | Numerical Derivative | Exact Derivative      | Abs Error           |
|---------------|-----------------------|-----------------------|---------------------|
| 0.1           | 83.721363            | 83.907153             | 0.185790           |
| 0.01          | 555.383031           | -8623.188723          | 9178.571754        |
| 0.001         | -233.885903          | -562379.076291        | 562145.190388      |
| 0.0001        | -884.627816          | 95215536.825901       | 95216421.453717    |
| 1e-05         | -736.979383          | 9993608074.376921     | 9993608811.356304  |

**Rapid Oscillation**: The exact derivative $-\frac{\cos(\frac{1}{x})}{x^2}$ oscillates rapidly, especially as $x$ approaches zero. The $\cos(\frac{1}{x})$ term will produce oscillations between `-1` and `1`, causing the derivative to take on large values in both positive and negative directions. The centered difference method may still provide accurate approximations for large value of $x$ like `0.1` when we have:

* **Numerical Derivative: 83.721363 VS Exact Derivative 83.907153**

**Numerical Stability** started for small $x$ values, from the `0.01` we have:

* **Numerical Derivative: 555.383031 VS Exact Derivative -8623.188723**

As $x$ approaches zero, it started exhibiting instability due to the oscillatory nature of the function and its derivative.
This instability can be critical for the deep learning application, because we need to optimize function that works with high-dimentional data. Remember the [MNIST dataset challange from my first video](./dive_into_learning_from_data.md)? 784 pixels for such simple data! Nowadays we work with text data, video, image in high resolution and it require as much precision as possible.


### 3D case

But what about more dims? The derivative idea is the same for any dims, just apply the derivative for every dimention and follow the negative direction of the gradient to minimise a function.

$$\nabla f(x, y) = \left(\frac{\partial f}{\partial x}, \frac{\partial f}{\partial y}\right)$$

Every component of the gradient vector shows us the direction of the steepest ascent. You need to follow to the negative direction, to build a gradient descent.


```python
import numpy as np


def cdiff(func, x, h=1e-3):
    """Centered difference approximation of the derivative."""
    return (func(x + h) - func(x - h)) / (2 * h)


def oscillating(x):
    """Function that oscillates rapidly as x approaches 0."""
    return np.sin(1/x) if x != 0 else 0


def d_oscillating(x):
    """Exact derivative of the oscillating function."""
    if x != 0:
        return -np.cos(1/x) / (x**2)
    else:
        return 0  # Not defined, but can be treated as 0 for comparison


# Test values close to 0
x_values = [0.1, 0.01, 0.001, 0.0001, 0.00001]

# Compute and compare the derivatives
print(f"{'x':>12} | {'Numerical Derivative':>20} | {'Exact Derivative':>20} | {'Abs Error':>20}")
print("-" * 80)
for x in x_values:
    numerical_derivative = cdiff(oscillating, x)
    exact_deriv = d_oscillating(x)
    
    # Calculate the relative error
    if exact_deriv != 0:  # Avoid division by zero for relative error
        abs_error = np.abs((numerical_derivative - exact_deriv))
    else:
        abs_error = np.nan  # Not defined for comparison

    print(f"{x:>12} | {numerical_derivative:>20.6f} | {exact_deriv:>20.6f} | {abs_error:>20.6f}")

```

           x | Numerical Derivative |     Exact Derivative |            Abs Error
--------------------------------------------------------------------------------
         0.1 |            83.721363 |            83.907153 |             0.185790
        0.01 |           555.383031 |         -8623.188723 |          9178.571754
       0.001 |          -233.885903 |       -562379.076291 |        562145.190388
      0.0001 |          -884.627816 |      95215536.825901 |      95216421.453717
       1e-05 |          -736.979383 |    9993608074.376921 |    9993608811.356304


```python
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider, IntSlider


def oscillating(x):
    """Vector-valued function that oscillates rapidly as x approaches 0."""
    result = np.zeros_like(x)  # Initialize an array of zeros with the same shape as x
    non_zero_mask = x != 0  # Mask for elements where x is not equal to 0
    result[non_zero_mask] = np.sin(1 / x[non_zero_mask])  # Apply sin(1/x) where x != 0
    return result


def d_oscillating(x):
    """Vector-valued exact derivative of the oscillating function."""
    result = np.zeros_like(x)  # Initialize an array of zeros with the same shape as x
    non_zero_mask = x != 0  # Mask for elements where x is not equal to 0
    result[non_zero_mask] = -np.cos(1 / x[non_zero_mask]) / (x[non_zero_mask] ** 2)  # Derivative where x != 0
    return result
    

def plot_oscillating(min=-0.1, max=0.1, steps=500):
    # Generate values of x around 0, excluding 0 to avoid division by zero
    x_values = np.linspace(min, max, steps)

    y_values = oscillating(x_values)
    dy_values = d_oscillating(x_values)

    # Create plots
    plt.figure(figsize=(14, 6))

    # Plot oscillating function
    plt.subplot(1, 2, 1)
    plt.plot(x_values, y_values, label='Oscillating Function', color='blue')
    plt.title('Oscillating Function $f(x) = sin(1/x)$')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid(True)
    plt.legend()

    # Plot exact derivative
    plt.subplot(1, 2, 2)
    plt.plot(x_values, dy_values, label='Exact Derivative', color='orange')
    plt.title("Exact Derivative $f'(x) = -\\frac{cos(1/x)}{x^2}$")
    plt.xlabel('x')
    plt.ylabel("f'(x)")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

interact(
    plot_oscillating,
    min=FloatSlider(min=-0.1, max=-0.01, step=0.01, value=-0.1, description='Min'),
    max=FloatSlider(min=0.01, max=0.1, step=0.01, value=0.1, description='Max'),
    steps=IntSlider(min=100, max=1000, step=100, value=500, description='Steps')
)

```

### Observations

1. **Close Agreement**: As shown, the numerical derivative closely matches the exact derivative for values of $ x $ that are not too close to zero. This indicates that the centered difference approximation is performing well in these regions.

2. **Rapid Oscillation**: The exact derivative $ -\frac{\cos(1/x)}{x^2} $ oscillates rapidly, especially as $ x $ approaches zero. The $ \cos(1/x) $ term will produce oscillations between -1 and 1, causing the derivative to take on large values in both positive and negative directions. The centered difference method may still provide accurate approximations for larger values of $ x $.

3. **Numerical Stability**: At very small $ x $ values, the numerical derivative may still be quite accurate, but as $ x $ approaches zero, it may start exhibiting instability due to the oscillatory nature of the function and its derivative.

4. **Relative Error**: The relative error in this case remains very small for non-zero values, indicating the effectiveness of the numerical method. However, as $ x $ approaches zero, both the numerical and exact derivatives will oscillate wildly, making comparisons less meaningful.

### Conclusion

This example illustrates how numerical differentiation can effectively approximate the derivative of oscillating functions when the values of $ x $ are not too close to the point of interest. However, it also highlights the potential for instability when dealing with functions that exhibit rapid changes, especially as they approach non-differentiable points or areas of high curvature. In practice, for functions with such behaviors, analytical methods are often preferred.
