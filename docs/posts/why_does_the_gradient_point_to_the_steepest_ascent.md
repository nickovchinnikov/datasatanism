---
title: Why Does the Gradient Point to the Steepest Ascent?
description: Explore why the gradient vector indicates the direction of steepest ascent for functions, and learn how to visualize and apply this concept in optimization algorithms like gradient descent.
authors:
  - nick
date:
  created: 2024-12-04
comments: true
categories:
  - Mathematics
  - Programming
  - Machine Learning
  - Data Science
tags:
  - Gradient Descent
  - Optimization Algorithms
  - Machine Learning Basics
  - Python Visualization
  - Matplotlib
---

The gradient, \( \nabla f(\textbf{x}) \), tells us the direction in which a function increases the fastest. But why?

![Gradient direction in 3D from Min => Max](../assets/why_does_the_gradient_point_to_the_steepest_ascent/Vector_Field_of_a_Function.png){ align=center }
/// caption
Gradient direction in 3D from Min => Max
///

<!-- more -->

### [Check the jupyter notebook](https://github.com/nickovchinnikov/datasatanism/blob/master/code/2.GradientIsTheSteepestAscent.ipynb)

For a given unit vector \( \vec{v} \), the **directional derivative** measures this, and it’s defined as:

$$\nabla_{\vec{v}} f = \nabla f(\textbf{x}) \cdot \vec{v}$$

Using the dot product formula, we can rewrite this as:

$$\nabla_{\vec{v}} f = |\nabla f(\textbf{x})||\vec{v}|\cos \theta$$

Since \( \vec{v} \) is a unit vector (\( |\vec{v}| = 1 \)), this simplifies to:

$$\nabla_{\vec{v}} f = |\nabla f(\textbf{x})| \cos \theta$$

Here, \( \theta \) is the angle between \( \nabla f(\textbf{x}) \) and \( \vec{v} \). The key insight? The **cosine of the angle**, \( \cos \theta \), determines how large the directional derivative is:

- **When \( \theta = 0^\circ \):** \( \cos(0) = 1 \), so \( \nabla_{\vec{v}} f \) reaches its maximum value:
  
$$\nabla_{\vec{v}} f = |\nabla f(\textbf{x})|$$

- **For any other angle:** \( \cos \theta < 1 \), so the directional derivative is smaller.

Thus, the gradient \( \nabla f(\textbf{x}) \) points in the **steepest ascent** direction because that's where \( \cos \theta = 1 \) the function increases the fastest when you move directly in the direction of the gradient.

## Gradient Descent: Finding the Steepest Descent

To minimize a function, we use the **negative gradient**, following **the steepest descent**. Let’s visualize this concept with an interactive 3D plot of the gradient field.

### Visualizing the Gradient Field

The function \( f(x, y) = x^2 + y^2 \) is a classic example. Its gradient points outward, showing how the function rises steeply as you move away from the origin. Here’s an interactive plot to explore the gradient field:

```python
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, IntSlider, FloatSlider


def f2d(x, y):
    r"""
    3d Paraboloid function $f(x, y) = x^2 + y^2$.

    Args:
    x: x coordinate
    y: y coordinate

    Returns:
    Value of the function at point (x, y)
    """
    return x**2 + y**2


def grad_f2d(x, y):
    """
    Gradient of the function $f(x, y) = x^2 + y^2$.

    Args:
    x: x coordinate
    y: y coordinate

    Returns:
    Gradient of the function at point (x, y)
    """
    return np.array([2*x, 2*y])


def plot_gradient_field(density=10, arrow_scale=20):
    r"""
    Plot the gradient field of the function $f(x, y) = x^2 + y^2$.

    Args:
    density: density of the grid
    arrow_scale: scale of the arrows
    """
    # Create the x/y grid
    x = np.linspace(-5, 5, density)
    y = np.linspace(-5, 5, density)

    # Create the meshgrid X/Y
    X, Y = np.meshgrid(x, y)
    # Compute the function values
    Z = f2d(X, Y)

    # Create the figure and subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # 3D surface plot
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('3D Surface Plot')

    # 2D contour plot with gradient field
    ax2.contourf(X, Y, Z, levels=20, cmap='viridis')
    
    # Compute the gradient
    U, V = grad_f2d(X, Y)
    # Plot the gradient field
    ax2.quiver(X, Y, U, V, scale=arrow_scale, scale_units='inches', color='w', alpha=0.7)
    
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('Gradient Field')

    plt.tight_layout()
    plt.close(fig)
    return fig

# Create interactive plot
interact(plot_gradient_field, 
         density=IntSlider(min=5, max=30, step=1, value=10, description='Grid Density'),
         arrow_scale=FloatSlider(min=1, max=100, step=1, value=20, description='Arrow Scale'))

```

![Gradient in 3D](../assets/why_does_the_gradient_point_to_the_steepest_ascent/gradient_3d.png){ align=center }
/// caption
Gradient field direction in 3D towards the maximum
///

### Gradient in Action

When exploring this plot, notice how the arrows point directly away from the origin—the direction of the steepest ascent. By following these arrows in reverse (negative gradient), you can descend to the **minimum**.
