---
title: MicroTorch - Deep Learning from Scratch!
description: Learn how to build a PyTorch-like, autograd-powered Deep Learning Framework with automatic differentiation.
authors:
  - nick
date:
  created: 2025-04-03
comments: true
categories:
  - Deep Learning
  - Machine Learning
  - Neural Networks
tags:
  - Autograd
  - Automatic Differentiation
  - Backpropagation
  - Computational Graphs
  - Tensor Operations
  - Framework Development
---


Implementing deep learning algorithms involves managing data flow in two directions: `forward` and `backward`. While the `forward` pass is typically straightforward, handling the `backward` pass can be more challenging. As discussed in previous posts, implementing backpropagation requires a strong grasp of calculus, and even minor mistakes can lead to significant issues.

Fortunately, modern frameworks like PyTorch simplify this process with **autograd**, an automatic differentiation system that dynamically computes gradients during training. This eliminates the need for manually deriving and coding gradient calculations, making development more efficient and less error-prone.

Now, let's build the backbone of such an algorithm - `Tensor` class!

![Autograd cover](../assets/autograd/cover.jpg)
/// caption
Build an autograd!
///


<!-- more -->


### [Check the Jupyter Notebook](https://github.com/nickovchinnikov/datasatanism/blob/master/code/13.Tensor.ipynb)

In the previous chapters, I built everything from the ground up. Now, we will create a `Tensor` object that abstracts the implementation of the backward steps for our building blocks.  

Instead of manually coding the backward pass, we'll design a class that constructs a computation graph, tracking every operation within it. Once the graph is established, we will run the backward pass to compute gradients for all operations automatically.  

Let's start by defining the `Tensor` class and initializing its basic methods.


<iframe width="1707" height="765" src="https://www.youtube.com/embed/7GHa_Cla5wU" title="Building PyTorch from Scratch: Create a Tensor Class with AutoDiff" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>


### Imports and Type Definitions

Imports necessary libraries and defines type aliases for better readability.


```python
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union
import numpy as np

# Scalar is a type alias for either an int or float.
Scalar = Union[int, float]

# Data is a type alias for any valid input that can be converted into a Tensor 
# (e.g., scalars, lists, NumPy arrays, or other Tensor objects).
Data = Union[Scalar, list, np.ndarray, "Tensor"]

```

**Example:**

```python
scalar_value: Scalar = 5.0

data_list: Data = [1, 2, 3]
data_np: Data = np.array([1, 2, 3])

```


### Computational graph

A **computation graph** is a directed graph where nodes represent operations (like addition or multiplication), and edges represent the flow of data (tensors). In the context of your `Tensor` class, each tensor is a node, and operations between tensors create edges between them.

When you perform an operation, such as `Tensor A + Tensor B`, a new tensor is created, which records its dependencies on `A` and `B`. These dependencies are tracked in the `Tensor` object via the `dependencies` list. During the backward pass, the graph is traversed in reverse order to compute gradients for each tensor, starting from the final result back to the input tensors.

By storing these dependencies in `Leaf` objects, the graph allows automatic differentiation, meaning gradients are computed for all involved tensors without manually specifying the backpropagation steps.

`Leaf` is a simple class used for storing the relationship between a `Tensor` (as value) and a function (`grad_fn`), which is responsible for computing the gradient for that `Tensor`. The `frozen=True` parameter makes the instance of the class immutable, meaning once created, its attributes cannot be changed.


```python
@dataclass(frozen=True)
class Leaf:
    value: "Tensor"
    grad_fn: Callable[[np.ndarray], np.ndarray]

```


### Tensor Class

The Tensor class is the core of this implementation. It encapsulates data and provides functionality for automatic differentiation. The code bellow defines a simple `Tensor` class backbone. 

```python
class Tensor:
    r"""
    A class representing a multi-dimensional array (Tensor) with automatic differentiation support.
    """

    def __init__(
        self,
        data: Data,
        requires_grad: bool = False,
        dependencies: Optional[List[Leaf]] = None,
        dtype=np.float32
    ) -> None:
        r"""
        Initializes a Tensor object.

        Args:
            data (Data): The input data, which can be a scalar, list, NumPy array, or another Tensor.
            requires_grad (bool, optional): If True, enables gradient tracking. Defaults to False.
            dependencies (Optional[List[Leaf]], optional): Dependencies in the computation graph. Defaults to None.
            dtype (dtype, optional): The data type of the Tensor. Defaults to np.float32.
        """

        # data: The input data, which can be a scalar, list, NumPy array, or another Tensor.
        self._data = Tensor.build_ndarray(data, dtype)
        # dtype: The data type of the Tensor (default is np.float32)
        self.dtype = self._data.dtype
        # requires_grad: If True, the Tensor will track operations for gradient computation.
        self.requires_grad = requires_grad
        # dependencies: A list of Leaf objects representing dependencies in the computation graph.
        self.dependencies: List[Leaf] = dependencies or []
        self.grad: np.ndarray = None

        # zero_grad(): Initializes the gradient to zero if requires_grad is True.
        if self.requires_grad:
            self.zero_grad()

        ############################
        # Properties of the Tensor # 
        ############################

        # ndim: Returns the number of dimensions of the Tensor
        @property
        def ndim(self) -> int:
            r"""
            Returns the number of dimensions of the Tensor.

            Returns:
                int: Number of dimensions.
            """

            return self._data.ndim

        # shape: Returns the shape of the Tensor
        @property
        def shape(self) -> Tuple[int, ...]:
            r"""
            Returns the shape of the Tensor.

            Returns:
                Tuple[int, ...]: Shape of the Tensor.
            """

            return self._data.shape

        # size: Returns the total number of elements in the Tensor
        @property
        def size(self) -> int:
            r"""
            Returns the total number of elements in the Tensor.

            Returns:
                int: Total number of elements.
            """

            return self._data.size

        # data: Gets or sets the underlying NumPy array
        @property
        def data(self) -> np.ndarray:
            r"""
            Gets the underlying NumPy array.

            Returns:
                np.ndarray: The data stored in the Tensor.
            """

            return self._data
        
        @data.setter
        def data(self, new_data: Data) -> None:
            r"""
            Sets new data for the Tensor and resets gradients if required.

            Args:
                new_data (Data): The new data to be assigned to the Tensor.
            """

            self._data = Tensor.build_ndarray(new_data, self.dtype)
            if self.requires_grad:
                self.zero_grad()

        # String Representation: Provides a string representation of the Tensor
        def __repr__(self) -> str:
            r"""
            Returns a string representation of the Tensor.

            Returns:
                str: A string describing the Tensor.
            """

            return f"Tensor({self.data}, requires_grad={self.requires_grad}, shape={self.shape})"

        # Gradient Management - resets the gradient to zero
        def zero_grad(self) -> None:
            r"""
            Resets the gradient of the Tensor to zero.
            """

            if self.grad is None:
                self.grad = np.zeros_like(self.data, dtype=self.dtype)
            else:
                self.grad.fill(0.0)

        ##################
        # Static Methods #
        ##################

        # build_ndarray: Converts input data into a NumPy array.
        @staticmethod
        def build_ndarray(data: Data, dtype=np.float32) -> np.ndarray:
            r"""
            Converts input data into a NumPy array.

            Args:
                data (Data): The input data which could be a Tensor, NumPy array, or a list.
                dtype (dtype, optional): The target data type. Defaults to np.float32.

            Returns:
                np.ndarray: The converted NumPy array.
            """

            if isinstance(data, Tensor):
                return np.array(data.data, dtype=dtype)
            if isinstance(data, np.ndarray):
                return data.astype(dtype)
            return np.array(data, dtype=dtype)

```

**Example:**

```python
t = Tensor([1, 2, 3], requires_grad=True)
t.zero_grad() # Resets the gradient to zero
print(t)  # Output: Tensor([1 2 3], requires_grad=True, shape=(3,))

t = Tensor([[1, 2], [3, 4]])
print(t.shape)  # Output: (2, 2)

```


### First operation - Transpose

<iframe width="1707" height="765" src="https://www.youtube.com/embed/xUkEKzq7XeQ?list=PLWUV973D6J8imrTO4yJk3aI0NKJZgzFeG" title="Building PyTorch from Scratch: Tensor Operations, Transpose &amp; Backward Method Explained" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

The backbone of the `Tensor` is basically useless - it just a mechanizm above the `numpy.array`. But now we can track the dependencies in the list and compute the gradient for the whole list of the dependencies!

I can show you the example of the foundamental tensor operation - the `transpose` method. The `transpose` operation reorders the dimensions of a tensor. If no axes are specified, automatically reverses the dimensions (`[::-1]`). For example, given a 3D tensor (shape `[2, 3, 4]`):

```python
X = np.random.randn(2, 3, 4)
X.T.shape  # Output: (4, 3, 2)

```

The dimensions are flipped:
`(2, 3, 4) → (4, 3, 2)`.

However, if specific axes are provided, permutes the dimensions accordingly.

```python
X.transpose((1, 0, 2))  # Changes order of dimensions

```

In forward pass, `np.transpose(self.data, axes=axes)` swaps the tensor dimensions. In backward pass, we must apply the inverse permutation to propagate gradients correctly. If `axes=None`, the gradient reverses dimensions back with the transpose operation (default case). If `axes` is provided (i.e., custom permutation), we must invert that permutation. `np.argsort(axes)` finds the inverse order to revert the transpose. For example, if we permute `(0,1,2) → (1,2,0)`, we need the inverse mapping to undo it:

```python
axes = (1, 2, 0)   # Forward permutation
inv_axes = np.argsort(axes)  # Output: (2, 0, 1)  → This restores original order

```

**Implementation:**

```python
def transpose(self, axes: Tuple[int, ...] = None) -> "Tensor":
    # Perform the transpose operation
    output = np.transpose(self.data, axes=axes)

    # Handle dependencies for autograd
    dependencies: List[Leaf] = []

    if self.requires_grad:
        def _bkwd(grad: np.ndarray) -> np.ndarray:
            # Compute the inverse permutation of axes for the backward function
            if axes is None:
                # Implicitly reverses dimensions
                return np.transpose(grad)  
            else:
                # Compute the inverse permutation of axes
                inv_axes = tuple(np.argsort(axes))
                # Transpose the gradient back using the inverse permutation
                return np.transpose(grad, axes=inv_axes)

        dependencies.append(
            Leaf(value=self, grad_fn=_bkwd)
        )

    # Return the new tensor with the transposed data
    return Tensor(
        output,
        requires_grad=self.requires_grad,
        dependencies=dependencies
    )

```


### The `backward` Method

The `transpose` method is the first operation that the `Tensor` class supports. To fully showcase the power of our simple implementation, let's implement the `backward` method.

The `backward` method implements **reverse-mode automatic differentiation** using the **chain rule of calculus**.

**Chain Rule:**

$$
\frac{dz}{dx} = \frac{dz}{dy} \cdot \frac{dy}{dx}
$$

If we have a function composition:  

$$
f(x) = g(h(x))
$$

Then, by the chain rule:

$$
f'(x) = g'(h(x)) \cdot h'(x)
$$

In the context of our `Tensor` class, this method is responsible for **propagating gradients backward** through a computation graph, ensuring that each node in the graph correctly accumulates its contribution to the final gradient.


```python
# Backward Propagation: Computes gradients using backpropagation
def backward(self, grad: Optional[np.ndarray] = None) -> None:
    # Step 1: Checking If Gradient Tracking is Enabled
    if not self.requires_grad:
        raise RuntimeError(
            "Cannot call backward() on a tensor that does not require gradients. "
            "If you need gradients, ensure that requires_grad=True when creating the tensor."
        )

    # Step 2: Initializing the Gradient If Not Provided
    if grad is None:
        if self.shape == ():
            # The gradient of a scalar itself is 1
            grad = np.array(1.0)
        else:
            # If the tensor is not a scalar, `grad` must be provided explicitly.
            raise ValueError("Grad must be provided if tensor has shape")

    # Step 3: Accumulating the Gradient
    self.grad = self.grad + grad

    # The Chain Rule in Action
    for dependency in self.dependencies:
        # Step 4: Applying the Chain Rule
        # Propagates the gradient through the computation graph using `grad_fn`
        backward_grad = dependency.grad_fn(grad)
        # Step 5: Recursively Propagating Gradients
        dependency.value.backward(backward_grad)

```


**Step 3: Accumulating the Gradient**

```python
self.grad = self.grad + grad
```

**Mathematically, this represents:** $\text{self.grad} \gets \text{self.grad} + \text{grad}$


**The Chain Rule in Action.** This part implements the **chain rule**:

```python
for dependency in self.dependencies:
    backward_grad = dependency.grad_fn(grad)
    dependency.value.backward(backward_grad)
```


**Step 4: Applying the Chain Rule**

For each dependency (i.e., an operation that contributed to this tensor), we compute the gradient contribution using the chain rule:

$$\frac{dz}{dx} = \frac{dz}{dy} \cdot \frac{dy}{dx}$$

where $dz/dy$ is `grad` (gradient of the current tensor with respect to its output), $dy/dx$ is `dependency.grad_fn` (gradient of the dependency with respect to its input).


```python
backward_grad = dependency.grad_fn(grad)
```

- This calls the stored gradient function (`grad_fn`) for this dependency.
- It effectively computes: $\text{backward_grad} = \frac{dz}{dy} \cdot \frac{dy}{dx}$


**Step 5: Recursively Propagating Gradients**

```python
dependency.value.backward(backward_grad)
```

- This recursively calls `backward` on the dependency.
- It ensures that gradients are **propagated through the entire computation graph**.

**Why Does This Work?**

When we compute gradients **backward**, we need to apply the **chain rule** in reverse order from the output back to the inputs.

- **Forward Pass:** Builds a **directed acyclic graph (DAG)** where each tensor stores dependencies (operations that produced it).
- **Backward Pass:** Uses **recursive calls**, which implicitly use a **stack**, ensuring the **last dependency is processed first**.

This is crucial because the last computed tensor (final output, e.g., loss) is at the top of the graph. Gradients flow backward through dependencies **(from output to input).** Recursive calls unwind the computation graph in the correct order **(LIFO - Last In, First Out).**


### Parameter Class: Foundation for Neural Network Parameters

<iframe width="1707" height="765" src="https://www.youtube.com/embed/b16qKLmp2ro?list=PLWUV973D6J8imrTO4yJk3aI0NKJZgzFeG" title="Building PyTorch: A Hands-On Guide to the Core Foundations of a Training Framework" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>


The `Parameter` class handles the initialization and management of model parameters like weights and biases in neural networks. This class simplifies defining and managing weights and biases, ensuring efficient model optimization. It supports multiple initialization methods like ["xavier", "he", "normal", "uniform"](./weights_init.md) to set the right starting values, preventing issues like vanishing or exploding gradients. The class also ensures parameters are ready for optimization by setting `requires_grad=True` for backpropagation and includes a `gain` parameter to fine-tune initialization. 


```python
from typing import Any, Literal, Optional, Tuple

import numpy as np

from au2grad.tensor import Tensor

type InitMethod = Literal["xavier", "he", "normal", "uniform"]


class Parameter(Tensor):
    r"""
    Foundation for models parameters.
    """

    def __init__(
        self,
        *shape: int,
        data: Optional[np.ndarray] = None,
        init_method: InitMethod = "normal",
        gain: float = 1.0,
        alpha: float = 0.01,
    ) -> None:
        r"""
        Initialize the parameter.

        Args:
            shape (tuple of int): The shape of the parameter.
            data (np.ndarray, optional): The data of the parameter. If not \
                provided, the parameter is initialized using the initialization \
                method.
            init_method (str): The initialization method. Defaults to 'normal'. \
                Possible values are 'xavier', 'he', 'normal', 'uniform'.
            gain (float): The gain for the initialization method. Defaults to 1.0.
            alpha (float): Slope for Leaky ReLU in "he_leaky" initialization.
        """

        if data is None:
            data = self._initialize(shape, init_method, gain, alpha)

        super().__init__(data=data, requires_grad=True)

    def _initialize(
        self, shape: Tuple[int, ...], method: InitMethod | Any, gain: float, alpha: float
    ) -> np.ndarray:
        r"""
        Initialize the parameter data.
        """

        weights = np.random.randn(*shape)

        if init_method == "xavier":
            std = gain * np.sqrt(1.0 / shape[0])
            return std * weights
        if init_method == "he":
            std = gain * np.sqrt(2.0 / shape[0])
            return std * weights
        if init_method == "he_leaky":
            std = gain * np.sqrt(2.0 / (1 + alpha**2) * (1 / shape[0]))
            return std * weights
        if init_method == "normal":
            return gain * weights
        if init_method == "uniform":
            return gain * np.random.uniform(-1, 1, size=shape)

        raise ValueError(f"Unknown initialization method: {method}")

```


### Module Class: Base for All Neural Network Modules

The `Module` class serves as the foundation for building neural network components, like layers and models. It defines essential methods like `forward`, which must be implemented in subclasses to process inputs and generate outputs. The class also provides functionality to switch between training (`train`) and evaluation (`eval`) modes, ensuring that all submodules are properly updated. 

The `parameters` method recursively collects all parameters from the module and its submodules, while `zero_grad` resets gradients for all parameters. The `params_count` method returns the total number of parameters in the module. 

In neural network development, the `Module` class simplifies handling the structure, state, and parameters of layers and models, making it easier to implement and train complex architectures.


```python
from typing import Any

class Module:
    r"""
    Base class for all modules.
    """

    def __call__(self, *args: Any) -> Tensor:
        return self.forward(*args)

    def forward(self, *input: Any) -> Tensor:
        r"""
        Forward method to be implemented in children class

        Args:
            input (Tensor or different object): Inputs

        Returns:
            Tensor: Outputs
        """
        raise NotImplementedError()

    def parameters(self) -> List[Parameter]:
        r"""
        Returns:
            List[Parameter]: Iterator of parameters
        """

        params = []
        for _, item in self.__dict__.items():
            if isinstance(item, Parameter):
                params.append(item)
            elif isinstance(item, Module):
                params.extend(item.parameters())
        return params

    def zero_grad(self) -> None:
        r"""
        Zero the gradients of all parameters
        """

        for param in self.parameters():
            param.zero_grad()

    def params_count(self) -> int:
        r"""
        Returns:
            int: Number of parameters
        """

        num_parameters = sum(p.data.size for p in self.parameters())
        return num_parameters

```


## Sequential Class: Chaining Modules in Order

The `Sequential` class provides a simple way to stack multiple `Module` instances in a defined order. It automates the forward pass by passing the input tensor through each module sequentially, making it useful for building feedforward networks.  

The `parameters` method collects all parameters from the contained modules, ensuring easy access for optimization. The `forward` method iterates through the sequence, applying each module to the input.  

By structuring models in a linear fashion, `Sequential` simplifies neural network construction, reducing boilerplate and improving code clarity.


```python
class Sequential(Module):
    def __init__(self, *modules: Module):
        self.modules = modules

    def parameters(self) -> List[Parameter]:
        r"""
        Returns a list of all parameters in the sequential module and its submodules.
        """
        params = []
        for module in self.modules:
            params.extend(module.parameters())
        return params

    def forward(self, x):
        r"""
        Passes the input through all modules in sequence.
        """
        for module in self.modules:
            x = module(x)
        return x

```


## Linear Layer: Matrix-Matrix Dot Product

<iframe width="1707" height="765" src="https://www.youtube.com/embed/F9GH3nF4nkM?list=PLWUV973D6J8imrTO4yJk3aI0NKJZgzFeG" title="Building PyTorch: Crafting Linear Layers and Parameter Counting in MicroTorch" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>


The mathematics behind the linear layer rely on matrix-matrix multiplication instead of vector operations. This allows efficient computation when processing multiple input samples simultaneously.

At layer $i$, the transformation is defined as:  

$$\tag{linear step}
\label{eq:linear_step}
A_i(\mathbf{X}) = \mathbf{X} \mathbf{W}_i + \mathbf{B}_i$$

Where $\mathbf{X}$ is the input matrix (batch of samples), $\mathbf{W}_i$ represents the weight matrix, and $\mathbf{B}_i$ is the bias matrix, typically broadcasted across the batch. The activation function $\sigma$ introduces non-linearity after this transformation.

For a single layer:  

$$F_i(\mathbf{X}) = \sigma(A_i(\mathbf{X}))$$

where $A_i(\mathbf{X})$ is the linear transformation at layer $i$.  

A deep neural network applies these transformations layer by layer, leading to the final output:  

$$F(\mathbf{X}) = \sigma(A_L(\sigma(A_{L-1}(\dots \sigma(A_1(\mathbf{X})) \dots )))$$

Using **functional composition**, this process is compactly written as:  

$$\tag{deep neural net}
\label{eq:deep_nn}
F(\mathbf{X}) = A_L \circ \sigma \circ A_{L-1} \circ \dots \circ \sigma \circ A_1 (\mathbf{X})$$

The **forward pass** computes these transformations, storing intermediate values for the **backward pass**. We can implement the `Linear` layer's `forward` method directly based on these equations.

The `tensor` implementation must support all necessary operations since it tracks dependencies within the gradient graph and accumulates gradients for the `backward` pass. This ensures automatic differentiation works seamlessly.

For the `Linear` layer, we only need to implement the `forward` step, as all gradient computations are handled within the `tensor` itself. However, inside `tensor`, we must implement `backward` for every operation used in the `Linear` layer’s `forward` step to enable proper gradient propagation during backpropagation.

As the step number one, let's implement the **matrix dot product** inside the tensor class.


### Dot product

Matrix multiplication follows the **chain rule** during backpropagation. Let's break it down step by step.

<iframe width="1707" height="765" src="https://www.youtube.com/embed/UQIGmdXZd_U?list=PLWUV973D6J8imrTO4yJk3aI0NKJZgzFeG" title="Building PyTorch: Mastering Matrix Multiplication and Linear Layers in MicroTorch" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

**1. Forward Pass (MatMul Operation)**

Given two tensors **A** and **B**, matrix multiplication is:

$$Z = A \times B$$

Where $A$ has shape $(m, n)$, $B$ has shape $(n, p)$ and the result $Z$ has shape $(m, p)$. You can find more details here: [Matrix Multiplication in Detail](./matmul_broadcasting.md#exploring-matrix-multiplication-in-detail)

This is implemented in the forward pass:

```python
# Matrix multiplication
output = a.data @ b.data

```

**2. Backward Pass (Gradients Computation)**

For backpropagation, we need to compute $\frac{\partial L}{\partial A}$ and $\frac{\partial L}{\partial B}$ using the chain rule.

**Gradient w.r.t. A**
The gradient of the loss $L$ with respect to $A$ is given by:

$$\frac{\partial L}{\partial A} = \frac{\partial L}{\partial Z} \times B^T$$

Where: $\frac{\partial L}{\partial Z}$ is the **incoming gradient** (represented as `grad`) and $B^T$ is the **transpose of B**.

This is implemented as:

```python
if a.requires_grad:
    def _bkwd(grad: np.ndarray) -> np.ndarray:
        if b.ndim > 1:
            return grad @ b.data.swapaxes(-1, -2)  # grad * B^T
        return np.outer(grad, b.data.T).squeeze()  # Handles 1D case
```

- If $B$ is 2D, we use `b.data.swapaxes(-1, -2)` to compute $B^T$.
- If $B$ is 1D, we use `np.outer(grad, b.data.T)` to ensure correct shape.


**Gradient w.r.t. B**

The gradient of the loss $L$ with respect to $B$ is given by:

$$\frac{\partial L}{\partial B} = A^T \times \frac{\partial L}{\partial Z}$$

Where $A^T$ is the **transpose of A**.

This is implemented as:

```python
if b.requires_grad:
    def _bkwd(grad: np.ndarray) -> np.ndarray:
        if a.ndim > 1:
            return a.data.swapaxes(-1, -2) @ grad  # A^T * grad
        return np.outer(a.data.T, grad).squeeze()  # Handles 1D case
```

- If $A$ is 2D, we use `a.data.swapaxes(-1, -2)` to compute $A^T$.
- If $A$ is 1D, we use `np.outer(a.data.T, grad)`.


**3. Why Do We Use `swapaxes(-1, -2)` Instead of `.T`?**

`swapaxes(-1, -2)` is a **general approach** for transposing the last two dimensions. This ensures compatibility with **both 2D matrices and higher-dimensional tensors** (e.g., batches of matrices).

- `.T` works **only for 2D matrices**, affecting all axes in higher dimensions.
- `swapaxes(-1, -2)` **preserves batch and other leading dimensions**, modifying only the last two.

Example:

| Shape of Tensor | `.T` Output | `swapaxes(-1, -2)` Output |
|----------------|------------|---------------------------|
| `(m, n)` | `(n, m)` | `(n, m)` |
| `(batch, m, n)` | `(n, m, batch)` (incorrect) | `(batch, n, m)` (correct) |
| `(batch, time, m, n)` | `(n, m, time, batch)` (incorrect) | `(batch, time, n, m)` (correct) |


**4. How Does This Work in Backpropagation?**

- **During backpropagation**, when a gradient **flows back** through the `matmul` operation, it needs to be **properly propagated** to both `A` and `B`.
- The **gradient computation follows the chain rule** and ensures that the gradients for both matrices are computed **correctly**.


**5. Summary**

Matrix multiplication follows the chain rule. The backward pass computes gradients for both $A$ and $B$ using transposes. Uses `swapaxes(-1, -2)` to generalize for higher-dimensional cases.

| Tensor  | Gradient Formula | Code Implementation |
|---------|-----------------|----------------------|
| $A$ | $\frac{\partial L}{\partial A} = \frac{\partial L}{\partial Z} \times B^T$ | `grad @ b.data.swapaxes(-1, -2)` |
| $B$ | $\frac{\partial L}{\partial B} = A^T \times \frac{\partial L}{\partial Z}$ | `a.data.swapaxes(-1, -2) @ grad` |


**Implementation**

To perform matrix-matrix multiplication, we first implement the static method `matmul` in the `Tensor` class. This method computes the dot product of two matrices $A$ and $B$, tracks dependencies, and sets up gradient functions for backpropagation.

```python
@staticmethod
def matmul(a: "Tensor", b: "Tensor") -> "Tensor":
    r"""
    Static method to perform matrix multiplication of two tensors.

    Args:
        a (Tensor): First matrix.
        b (Tensor): Second matrix.

    Returns:
        Tensor: Resulting tensor with tracked dependencies.
    """
    
    output = a.data @ b.data
    requires_grad = a.requires_grad or b.requires_grad
    dependencies: List[Leaf] = []

    if a.requires_grad:
        def _bkwd(grad: np.ndarray) -> np.ndarray:
            r"""
            Backward gradient function for matmul with respect to a.
            """

            if b.ndim > 1:
                return grad @ b.data.swapaxes(-1, -2)
            return np.outer(grad, b.data.T).squeeze()

        dependencies.append(
            Leaf(
                value=a,
                grad_fn=_bkwd
            )
        )

    if b.requires_grad:
        def _bkwd(grad: np.ndarray) -> np.ndarray:
            r"""
            Backward gradient function for matmul with respect to b.
            """

            if a.ndim > 1:
                return a.data.swapaxes(-1, -2) @ grad
            return np.outer(a.data.T, grad).squeeze()
        
        dependencies.append(
            Leaf(
                value=b,
                grad_fn=_bkwd
            )
        )

    return Tensor(output, requires_grad, dependencies)
```


### Ensuring Data Consistency with `data_gate`  

When performing matrix multiplication or other tensor operations, we must ensure that the data types are compatible. For example, attempting to multiply a `Tensor` with a `numpy.ndarray` directly may lead to unexpected behavior. To prevent such issues, we can create a `data_gate` method that automatically converts inputs to the `Tensor` type if they are not already.  


```python
@staticmethod
def data_gate(data_object: Data) -> "Tensor":
    r"""
    Ensures the input is a Tensor.
    
    This method checks if the provided object is already a Tensor. 
    If not, it converts it into a Tensor before proceeding with operations.
    
    Args:
        data_object (Data): The input data, which can be a Tensor or a compatible type.
    
    Returns:
        Tensor: The input converted to a Tensor if necessary.
    """
    if isinstance(data_object, Tensor):
        return data_object  # Return as-is if already a Tensor
    return Tensor(data_object)  # Convert to Tensor if not

```

This function acts as a safeguard, ensuring that all operations are performed with the correct data type. Simple but effective, preventing potential errors when working with mixed data types.


### Matmul operator `@` 

Next, we define the dot method as the standard interface for matrix dot products. To enable the `@` operator for matrix multiplication, we overload the `__matmul__` method.

```python
def dot(self, other: Data) -> "Tensor":
    r"""
    Perform matrix dot product with another tensor or data.

    Args:
        other (Data): The other operand.

    Returns:
        Tensor: Result of the dot product.
    """

    return Tensor.matmul(self, Tensor.data_gate(other))


def __matmul__(self, other: Data) -> "Tensor":
    r"""
    Overload the `@` operator for matrix multiplication.

    Args:
        other (Data): The other operand.

    Returns:
        Tensor: Result of the matrix multiplication.
    """

    return self.dot(other)

```

This implementation ensures that the `Tensor` class handles both forward and backward computations for matrix multiplication, integrating smoothly into the automatic differentiation framework.


## Linear Layer Implementation  

The `Linear` layer applies a linear transformation to the input tensor, mapping it from `in_features` to `out_features`. It consists of learnable weight parameters and an optional bias.  

During initialization, the weights are assigned based on the specified `init_method`, and the bias is included if enabled. The `forward` method performs matrix multiplication between the input tensor and the weight matrix. If the bias is present, it is reshaped accordingly and added to the output.  

To support both 2D and 3D inputs, the implementation ensures that matrix dimensions align properly before performing operations. The result is a transformed tensor, ready for further processing in the neural network.


```python
class Linear(Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        init_method: InitMethod = "xavier",
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(out_features, in_features, init_method=init_method)
        self.bias = Parameter(out_features, init_method=init_method) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        # Check dimensions of input tensors
        assert x.ndim in (2, 3), f"Input must be 2D or 3D Tensor! x.ndim={x.ndim}"

        # Check if the last dimension of input matches in_features
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"Last dimension of input: {x.shape[-1]} does not match in_features: {self.in_features}"
            )

        # Compute matrix multiplication: x @ weight^T
        output = x @ self.weight.T
        
        # Add the bias directly. Broadcasting will handle it!
        if self.bias is not None:
            output = output + self.bias

        return output

```

But here, we have the `+` operation: `output = output + self.bias`, which is not implemented inside the `Tensor` class. To make the `Linear` implementation work, we need to handling this operation correctly.


## Broadcasting in backward mode

<iframe width="1707" height="765" src="https://www.youtube.com/embed/pdZij4qj2WQ?list=PLWUV973D6J8imrTO4yJk3aI0NKJZgzFeG" title="Building PyTorch: Adding Broadcasting and Addition Operations to MicroTorch" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

Gradients must be correctly propagated across dimensions that may differ between tensors. *Broadcasting* allows tensors of different shapes to interact, but when computing gradients, we need to handle these differences in dimensions.

Operations like `+`, `-`, or `*` are straightforward to implement in the `forward` pass. However, during the `backward` pass, we must account for broadcasting rules when computing gradients. To handle this, we need to introduce an additional method in the `Tensor` class.

**Broadcasting** is a method used by most scientific computing libraries like PyTorch or NumPy to handle operations between arrays of different shapes. **Broadcasting Rules** - when performing an operation, compare the dimensions from right to left side. If the dimensions do not match, the shape with a size of 1 is stretched to match the other shape.

In our example, if we were to broadcast:

- **Matrix `A`** with shape `(3, 1)` (our `X * W` result)
- **Matrix `B`** with shape `(1, 4)` (our bias `B` expanded to match `X * W` for broadcasting)

For the addition, we:

1. Stretch `A` to match `B` by duplicating the column four times.
2. Stretch `B` to match `A` by duplicating the row three times.

Thus, both matrices would be aligned to have dimensions of `3x4`, allowing for element-wise addition.

Let's see this in code using NumPy:

```python
import numpy as np

# Define array A with shape (3, 1)
A = np.array([
    [1],
    [2],
    [3],
])
print(f"Array A shape: {A.shape}")

# Define array B with shape (1, 4)
B = np.array([
    [1, 2, 3, 4],
])
print(f"Array B shape: {B.shape}")

# Perform broadcasting addition
result = A + B

print("A + B result: ")
print(result)

print(f"Result of A + B shape: {result.shape}")

```

*Output:*

```
Array A shape: (3, 1)
Array B shape: (1, 4)
A + B result: 
[[2 3 4 5]
 [3 4 5 6]
 [4 5 6 7]]
Result of A + B shape: (3, 4)

```

Matrix multiplication example:

```python
# Broadcasting the same for the matrix multiplication
matmul = A @ B
print(f"Matmul A @ B shape: {matmul.shape}")

print("Matmul result: ")
print(matmul)

```

*Output:*

```
Matmul A @ B shape: (3, 4)
Matmul result: 
[[ 1  2  3  4]
 [ 2  4  6  8]
 [ 3  6  9 12]]

```


The `bkwd_broadcast` method ensures gradients are correctly summed across broadcasted dimensions in `backward` mode. When tensors of different shapes interact, broadcasting aligns them by repeating elements. The method handles *gradient propagation* by summing over the extra dimensions created by broadcasting, ensuring consistency and preventing errors in gradient calculations. This is crucial for element-wise operations with mismatched tensor shapes, maintaining correct backpropagation.


In **Scenario 1**, `b` has shape `(1,)`, meaning it was **expanded to match both dimensions** of `a`. The backward pass gives `grad_c` shape `(2,2)`, but `b` originally had no explicit dimensions. We **sum over all extra axes `(0,1)`** (`keepdims=False`) to return to shape `(1,)`.


```python
a = np.array([[1, 2], 
              [3, 4]])  # Shape: (2, 2)

b = np.array([10])      # Shape: (1,)  (Broadcasted across both axis)

c = a + b
print(f"c: {c}")

grad_c = np.ones_like(c)
print(f"grad_c: {grad_c}")

# Since `a` was not broadcasted, the gradient just passes through
grad_a = grad_c
print(f"grad_a: {grad_a}")

# Since `b` was expanded to match both dimensions
# We **sum over all extra axes `(0,1)`** (`keepdims=False`)
# to return to shape `(1,)`.
grad_b = grad_c.sum(axis=(0, 1), keepdims=False)
print(f"grad_b: {grad_b}")

```

*Output:*

```
c: [[11 12]
    [13 14]]

grad_c: [[1 1]
         [1 1]]

grad_a: [[1 1]
         [1 1]]

grad_b: 4

```


In **Scenario 2**, `b` has shape `(2,1)`, meaning it was broadcasted along axis `1` to match `a`'s shape `(2,2)`. During the backward pass, `grad_c` has shape `(2,2)`, so we **sum over axis 1** (`keepdims=True`) to restore `b`'s original shape `(2,1)`.

```python
a = np.array([[1, 2], 
              [3, 4]])      # Shape: (2, 2)

b = np.array([[10], 
              [20]])        # Shape: (2, 1)  (Broadcasted across axis 1)

# element-wise addition
c = a + b                   # Shape: (2, 2) (Broadcasting rules)
print(f"c: {c}")

# generate the initial gradient
# Shape: (2, 2)
grad_c = np.ones_like(c)
print(f"grad_c: {grad_c}")

# Since `a` was not broadcasted, the gradient just passes through
grad_a = grad_c
print(f"grad_a: {grad_a}")

# Since `b` was **broadcasted along axis 1**, we must **sum** over 
# that axis to reduce it back to `b`'s original shape `(2,1)`
grad_b = grad_c.sum(axis=1, keepdims=True)
print(f"grad_b: {grad_b}")

```

*Output:*

```
c: [[11 12]
    [23 24]]

grad_c: [[1 1]
         [1 1]]

grad_a: [[1 1]
         [1 1]]

grad_b: [[2]
         [2]]

```

We need to compute **gradients for A and B** in the gradient tree.

Since `a` was not broadcasted, the gradient just passes through:

```python
grad_a = grad_c  # Same shape as a (2, 2)
```

Since `b` was **broadcasted along axis 1**, we must **sum** over that axis to reduce it back to `b`'s original shape `(2,1)`.  

```python
grad_b = grad_c.sum(axis=1, keepdims=True)
```


The `bkwd_broadcast` function ensures that gradients are correctly summed over broadcasted dimensions during backpropagation. When an operation involves tensors of different shapes, broadcasting aligns them by expanding dimensions as needed. If extra dimensions were added during this process, they must be summed over in the backward pass to maintain consistency with the original tensor shape. In this case, since `B` was originally `(2,1)`, no additional dimensions were introduced (`ndim_added = 0`), so this step is skipped.  

To correctly compute the gradient for `B`, we must sum over the broadcasted axis. Since `B` was expanded along axis `1` to match the shape of `A`, its corresponding gradient `grad_Z` retains this extra information across all columns. To revert the gradient to `B`’s original shape `(2,1)`, we sum over axis `1`, ensuring that the total contribution from each row is preserved while eliminating the artificially expanded dimension.


```python
@staticmethod
def bkwd_broadcast(tensor: "Tensor"):
    r"""
    Backward closure function to sum across broadcasted dimensions.
   
    When performing operations between tensors of different shapes, broadcasting is used
    to align their shapes. This function ensures that the gradients are correctly summed
    over the broadcasted dimensions during the backward pass.
    
    Args:
        tensor (Tensor): The tensor involved in the operation, used to handle its shape
                         during backward gradient computation.
    Returns:
        _bkwd (function): A function that computes the gradient, summing over broadcasted
                          dimensions to match the original tensor's shape.
    """

    def _bkwd(grad: np.ndarray) -> np.ndarray:
        # Handle scalar tensor case:
        # Original tensor was a scalar: sum all gradients
        if tensor.ndim == 0:
            return np.sum(grad)

        # Handle scalar grad case
        if grad.ndim == 0:
            return grad

        # Calculate the number of dimensions *added* to the tensor to achieve
        # the grad shape. This is where broadcasting might have "prepended"
        # dimensions.
        ndim_added = max(0, grad.ndim - tensor.ndim)

        if ndim_added > 0:
            grad = grad.sum(axis=tuple(range(ndim_added)), keepdims=False)

        # Sum over dimensions where tensor was broadcasted (size 1)
        reduce_axes = tuple(
            dim for dim in range(tensor.ndim)
            if tensor.shape[dim] == 1 and grad.shape[dim] > 1
        )

        if reduce_axes:
            grad = grad.sum(axis=reduce_axes, keepdims=True)

        # Ensure the final shape matches the tensor shape exactly
        if grad.shape != tensor.shape:
            grad = grad.reshape(tensor.shape)

        return grad

    return _bkwd

```


**Handle Scalar Tensor Case**

If the original tensor is a scalar (`tensor.ndim == 0`), it means that during the forward pass, this scalar was broadcasted to match the shape of another tensor. To compute the gradient for a scalar tensor, we need to sum up all the gradients from the larger tensor (e.g., matrix or vector) because the scalar contributes to every element of the result.

```python
if tensor.ndim == 0:
    return np.sum(grad)

```


**Handle Scalar Gradient Case**

If the gradient itself is a scalar (`grad.ndim == 0`), no broadcasting occurred during the forward pass. In this case, the gradient can be returned as-is because there are no dimensions to reduce.

```python
if grad.ndim == 0:
    return grad

```


**Calculate Dimensions Added by Broadcasting**

During broadcasting, NumPy may prepend dimensions to the smaller tensor to align its shape with the larger tensor. For example: Forward shapes: `(3,) + (5, 3) -> (5, 3)` - a new dimension is prepended to the first tensor. And we calculate in `ndim_added` how many such dimensions were added to the original tensor to match the gradient's shape.

```python
ndim_added = max(0, grad.ndim - tensor.ndim)

```


**Scenario 1 - Sum Over Added Dimensions:** These are collapsed using `keepdims=False` because they don't exist in the original tensor.

```python
if ndim_added > 0:
    grad = grad.sum(axis=tuple(range(ndim_added)), keepdims=False)

```


**Scenario 2 - Sum Over Broadcasted Dimensions:** These are summed while retaining their size as `1` using `keepdims=True` to preserve the original tensor's structure.

```python
reduce_axes = tuple(
    dim for dim in range(tensor.ndim)
    if tensor.shape[dim] == 1 and grad.shape[dim] > 1
)

if reduce_axes:
    grad = grad.sum(axis=reduce_axes, keepdims=True)

```

**Ensure Final Shape Matches:** This is a safeguard to ensure that the gradient's shape exactly matches the original tensor's shape. While the previous steps should handle most cases, this ensures correctness in edge cases.

```python
if grad.shape != tensor.shape:
    grad = grad.reshape(tensor.shape)

```


### `add`, `sub` and their friends

The first one is the `add` method, which will handle element-wise addition of two `Tensor` objects.

The **addition** operation computes the element-wise sum of two tensors.

$$f(a, b) = a + b$$

The derivative of $a + b$ with respect to $a$ and $b$ is 1:

$$\frac{d}{da} (a + b) = 1$$

$$\frac{d}{db} (a + b) = 1$$


```python
@staticmethod
def add(a: "Tensor", b: "Tensor") -> "Tensor":
    r"""
    Add two tensors and return a new tensor containing the result.
    
    This method performs element-wise addition of two tensors, handling broadcasting 
    if necessary. If either tensor requires gradients, the resulting tensor will also 
    track gradients and backpropagate them correctly.

    Args:
        a (Tensor): The first tensor to be added.
        b (Tensor): The second tensor to be added.

    Returns:
        Tensor: A new tensor that contains the element-wise sum of a and b.
    """
    
    # Perform element-wise addition of the data of tensors a and b
    output = a.data + b.data
    
    # Determine if the result requires gradients (if any input tensor requires it)
    requires_grad = a.requires_grad or b.requires_grad
    
    # List to store dependencies (grad functions) for backpropagation
    dependencies: List[Leaf] = []

    # If tensor a requires gradients, add its gradient function to dependencies
    # Apply bkwd_broadcast to the tensor a
    if a.requires_grad:
        dependencies.append(
            Leaf(value=a, grad_fn=Tensor._bkwd_broadcast(a))
        )

    # If tensor b requires gradients, add its gradient function to dependencies
    # Apply bkwd_broadcast to the tensor b
    if b.requires_grad:
        dependencies.append(
            Leaf(value=b, grad_fn=Tensor._bkwd_broadcast(b))
        )

    # Return a new tensor with the result, gradient flag, and dependencies
    return Tensor(output, requires_grad, dependencies)

```


Now, we are ready to overload the `+` and `-` operations! By implementing these operator overloads, we make tensor arithmetic more intuitive and user-friendly. 

Additionally, we implement **in-place addition (`+=`) and subtraction (`-=`)** to modify tensors directly without creating new ones. However, note that **in-place operations do not track gradients** for automatic differentiation.

To simplify subtraction, we introduce the `__neg__` method (`-` operator), which multiplies the tensor by `-1`. This allows us to redefine subtraction as **adding the negated tensor**, replacing `a - b` with `a + (-b)`, keeping the logic clean and consistent.


<iframe width="1707" height="765" src="https://www.youtube.com/embed/kPRDyKfLYlA?list=PLWUV973D6J8imrTO4yJk3aI0NKJZgzFeG" title="Building PyTorch: Overloading Operators for Subtraction and Multiplication in MicroTorch" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>


```python
def __add__(self, other: Data) -> "Tensor":
    """
    Overload the `+` operator to perform element-wise tensor addition.

    Args:
        other (Data): Another tensor or scalar to add.

    Returns:
        Tensor: The result of element-wise addition.
    """

    return Tensor.add(self, Tensor.data_gate(other))

def __radd__(self, other: Data) -> "Tensor":
    """
    Overload the right-hand `+` operator (other + self).

    Args:
        other (Data): Another tensor or scalar to add.

    Returns:
        Tensor: The result of element-wise addition.
    """

    return Tensor.add(Tensor.data_gate(other), self)

def __iadd__(self, other: Data) -> "Tensor":
    """
    Overload the `+=` operator for in-place addition.
    WARNING: In-place operations do not track gradients!

    Args:
        other (Data): Another tensor or scalar to add in-place.

    Returns:
        Tensor: The updated tensor after in-place addition.
    """

    self.data = self.data + Tensor.build_ndarray(other)
    return self

def __neg__(self) -> "Tensor":
    """
    Overload the unary `-` operator to negate a tensor.
    This allows defining subtraction as addition with negation.

    Returns:
        Tensor: The negated tensor (-self).
    """

    output = -self.data
    dependencies: List[Leaf] = []

    # Define the backward function: gradient negation
    if self.requires_grad:
        dependencies.append(
            Leaf(value=self, grad_fn=lambda grad: -grad)
        )

    return Tensor(output, self.requires_grad, dependencies)

def __sub__(self, other: Data) -> "Tensor":
    """
    Overload the `-` operator for element-wise subtraction.
    Uses addition with negation: a - b → a + (-b).

    Args:
        other (Data): Another tensor or scalar to subtract.

    Returns:
        Tensor: The result of element-wise subtraction.
    """

    return self + (-Tensor.data_gate(other))

def __rsub__(self, other: Data) -> "Tensor":
    """
    Overload the right-hand `-` operator (other - self).
    Uses addition with negation: b - a → b + (-a).

    Args:
        other (Data): Another tensor or scalar.

    Returns:
        Tensor: The result of element-wise subtraction.
    """

    return Tensor.data_gate(other) + (-self)

def __isub__(self, other: Data) -> "Tensor":
    """
    Overload the `-=` operator for in-place subtraction.
    WARNING: In-place operations do not track gradients!

    Args:
        other (Data): Another tensor or scalar to subtract in-place.

    Returns:
        Tensor: The updated tensor after in-place subtraction.
    """

    self.data = self.data - Tensor.build_ndarray(other)
    return self
```


### `mul`

The **multiplication** operation computes the element-wise product of two tensors.

$$f(a, b) = a \cdot b$$

The derivative of $a \cdot b$ with respect to $a$ and $b$ is:

$$\frac{d}{da} (a \cdot b) = b$$

$$\frac{d}{db} (a \cdot b) = a$$


```python
@staticmethod
def mul(a: "Tensor", b: "Tensor") -> "Tensor":
    """
    Performs element-wise multiplication between two tensors and returns the result.
    Handles tensors that require gradients by defining the backward pass for backpropagation.

    Args:
        a (Tensor): First tensor to be multiplied.
        b (Tensor): Second tensor to be multiplied.

    Returns:
        Tensor: A new tensor containing the result of the element-wise multiplication.
    """
    # Ensure both tensors contain their data correctly, handling any potential gates
    a = Tensor.data_gate(a)
    b = Tensor.data_gate(b)

    # Perform element-wise multiplication on the tensor data
    output = a.data * b.data

    # Determine if the resulting tensor should require gradients
    requires_grad = a.requires_grad or b.requires_grad
    dependencies: List[Leaf] = []

    # Define the backward pass function for multiplication
    def _backward(a: Tensor, b: Tensor):
        """
        Backward closure function for Mul operation.
        Computes the gradient of the multiplication operation.
        """
        def _bkwd(grad: np.ndarray) -> np.ndarray:
            """
            The gradient of the multiplication operation.
            The gradient of a * b is grad * b for a and grad * a for b.
            """
            # Multiply the gradient by tensor b's data for the gradient w.r.t a
            grad = grad * b.data
            # Ensure the gradient is properly reshaped using broadcasting
            return Tensor._bkwd_broadcast(a)(grad)

        return _bkwd

    # If tensor a requires gradients, add the backward function to the dependencies
    if a.requires_grad:
        dependencies.append(
            Leaf(
                value=a,
                grad_fn=_backward(a, b)  # Link tensor a's backward pass
            )
        )

    # If tensor b requires gradients, add the backward function to the dependencies
    if b.requires_grad:
        dependencies.append(
            Leaf(
                value=b,
                grad_fn=_backward(b, a)  # Link tensor b's backward pass
            )
        )

    # Return the result as a new tensor, with the appropriate gradient information
    return Tensor(output, requires_grad, dependencies)

```


Now, we are ready to overload the multiplication operators. This allows us to use `*` for element-wise multiplication of tensors, additionally, we implement in-place multiplication (`*=`), which modifies the tensor directly but does not support gradient tracking.  


```python
def __mul__(self, other: Data) -> "Tensor":
    r"""
    Overloads the `*` operator for element-wise multiplication.
    
    This method ensures that Tensor multiplication can be performed seamlessly 
    with both other Tensors and scalar values.
    
    Args:
        other (Data): The other operand, which can be a Tensor or a compatible scalar.

    Returns:
        Tensor: A new Tensor representing the element-wise product.
    """

    return Tensor.mul(self, Tensor.data_gate(other))

def __rmul__(self, other: Data) -> "Tensor":
    r"""
    Overloads the right-hand `*` operator.

    This ensures that multiplication works correctly when a scalar or another
    compatible type appears on the left side of the `*` operator.

    Args:
        other (Data): The left-hand operand, which can be a scalar or Tensor.

    Returns:
        Tensor: A new Tensor representing the element-wise product.
    """

    return Tensor.mul(Tensor.data_gate(other), self)

def __imul__(self, other: Data) -> "Tensor":
    r"""
    Overloads the `*=` operator for in-place multiplication.

    This modifies the Tensor’s data directly, which improves efficiency.
    However, in-place operations do not support automatic differentiation
    (i.e., gradients will not be tracked).

    Args:
        other (Data): The operand to multiply with.

    Returns:
        Tensor: The modified Tensor after in-place multiplication.
    """

    self.data = self.data * Tensor.build_ndarray(other)
    return self
```


## Logs, Exponents, and Activation Functions

<iframe width="1707" height="765" src="https://www.youtube.com/embed/IM_W-RGMTVc?list=PLWUV973D6J8imrTO4yJk3aI0NKJZgzFeG" title="Building PyTorch: Enriching MicroTorch with Logs, Exponents, and Activation Functions" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>


### `log`

The **logarithmic** operation computes the natural logarithm of each element in the tensor.

$$f(x) = \log(x)$$

The derivative of $\log(x)$ is:

$$\frac{d}{dx} \log(x) = \frac{1}{x}$$


```python
def log(self) -> "Tensor":
    r"""
    Computes the natural logarithm of all elements in the tensor.

    The logarithm is applied element-wise to the tensor's data. This function assumes that 
    the data values are positive, as the logarithm of non-positive values is undefined.

    Returns:
        Tensor: A new tensor containing the element-wise natural logarithm of the input tensor.
    
    The natural logarithm of a value `x` is calculated as:
        log(x) = ln(x)
    
    The derivative of log(x) with respect to x is:
        d/dx log(x) = 1/x
    """

    # Perform logarithmic operation on the data
    output = np.log(self.data)
    
    # Initialize an empty list for dependencies (used for backpropagation)
    dependencies: List[Leaf] = []

    def _bkwd(grad: np.ndarray) -> np.ndarray:
        r"""
        Backward pass for the logarithm operation.
        
        The derivative of the logarithm is 1/x, so we compute the gradient as:
            grad(x) = grad(x) / x
            
        Args:
            grad (np.ndarray): The gradient propagated from the next layer.
        
        Returns:
            np.ndarray: The gradient to propagate backward.
        """

        # The derivative of log(x) is 1/x, so we divide the gradient by the data (x)
        return grad / self.data

    # If the tensor requires gradients (i.e., it's part of the computation graph), 
    # we store the backward function in the dependencies.
    if self.requires_grad:
        dependencies.append(
            Leaf(
                value=self,
                grad_fn=_bkwd
            )
        )

    # Return a new Tensor containing the result of the log operation and the necessary dependencies.
    return Tensor(output, self.requires_grad, dependencies)

```


### `tanh`

The **tanh** operation computes the hyperbolic tangent of each element in the tensor.

$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$

The derivative of $\tanh(x)$ is:

$$\frac{d}{dx} \tanh(x) = 1 - \tanh^2(x)$$


```python
def tanh(self) -> "Tensor":
    r"""
    Computes the hyperbolic tangent (tanh) of all elements in the tensor.

    The hyperbolic tangent function is applied element-wise to the tensor's data. The tanh 
    function maps the input values to the range (-1, 1).

    Returns:
        Tensor: A new tensor containing the element-wise hyperbolic tangent of the input tensor.

    The hyperbolic tangent function is defined as:
        tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))

    The derivative of tanh(x) with respect to x is:
        d/dx tanh(x) = 1 - tanh(x)^2
    """

    # Perform hyperbolic tangent operation on the data
    output = np.tanh(self.data)
    
    # Initialize an empty list for dependencies (used for backpropagation)
    dependencies: List[Leaf] = []

    def _bkwd(grad: np.ndarray) -> np.ndarray:
        """
        Backward pass for the tanh operation.
        
        The derivative of tanh(x) is 1 - tanh(x)^2, so we compute the gradient as:
            grad(x) = grad(x) * (1 - tanh(x)^2)
        
        Args:
            grad (np.ndarray): The gradient propagated from the next layer.
        
        Returns:
            np.ndarray: The gradient to propagate backward.
        """
        
        # The derivative of tanh(x) is 1 - tanh(x)^2, so we multiply the gradient by this value
        return grad * (1 - output**2)

    # If the tensor requires gradients (i.e., it's part of the computation graph), 
    # we store the backward function in the dependencies.
    if self.requires_grad:
        dependencies.append(
            Leaf(
                value=self,
                grad_fn=_bkwd
            )
        )

    # Return a new Tensor containing the result of the tanh operation and the necessary dependencies.
    return Tensor(output, self.requires_grad, dependencies)

```


### `pow`

The **power** operation raises each element of the tensor to the specified power.

$$f(x) = x^p$$

The derivative of $x^p$ is:

$$\frac{d}{dx} x^p = p \cdot x^{p-1}$$

```python
def pow(self, pow: Scalar) -> "Tensor":
    r"""
    Computes the element-wise power of the tensor's data.

    The operation applies the power function element-wise, raising each element in the tensor 
    to the given power `pow`.

    Args:
        pow (Scalar): The exponent to which each element in the tensor should be raised.

    Returns:
        Tensor: A new tensor where each element is raised to the specified power.

    The power function is defined as:
        y = x^pow, where `x` is the input tensor's element and `pow` is the given exponent.

    The derivative of x^pow with respect to x is:
        d/dx (x^pow) = pow * x^(pow - 1)
    """

    # Perform element-wise power operation (raise each element to the given power)
    output = self.data**pow
    
    # Initialize an empty list for dependencies (used for backpropagation)
    dependencies: List[Leaf] = []

    def _bkwd(grad: np.ndarray) -> np.ndarray:
        """
        Backward pass for the power operation.
        
        The derivative of x^pow with respect to x is:
            d/dx (x^pow) = pow * x^(pow - 1)
        
        We multiply the gradient by the derivative to propagate the gradient backward.
        
        Args:
            grad (np.ndarray): The gradient propagated from the next layer.
        
        Returns:
            np.ndarray: The gradient to propagate backward.
        """
        # The derivative of x^pow is pow * x^(pow - 1), so we multiply the gradient by this value
        return grad * (pow * (self.data**(pow - 1)))

    # If the tensor requires gradients (i.e., it's part of the computation graph), 
    # we store the backward function in the dependencies.
    if self.requires_grad:
        dependencies.append(
            Leaf(
                value=self,
                grad_fn=_bkwd
            )
        )

    # Return a new Tensor containing the result of the power operation and the necessary dependencies.
    return Tensor(output, self.requires_grad, dependencies)

```

To enable exponentiation using the `**` operator, we overload the `__pow__` method. This allows us to perform element-wise power operations naturally, like:


```python
tensor ** 2  # Equivalent to tensor.pow(2)
```

**This's more intuitive syntax:** `tensor ** 2` instead of `tensor.pow(2)`.

Internally, `__pow__` simply calls the `pow` method, which handles the power operation while ensuring proper gradient computation.


```python
def __pow__(self, pow: Scalar) -> "Tensor":
    """
    Overload the `**` operator for element-wise exponentiation.

    Args:
        pow (Scalar): The exponent to raise the tensor to.

    Returns:
        Tensor: A new tensor with each element raised to the given power.
    """

    return self.pow(pow)
```


### division  

The last set of operator overloads covers division. These methods ensure that the `/` operator works naturally with Tensors and scalars, handling both standard and in-place division. Since division can be expressed as multiplication by the reciprocal, we reuse the `**` operator to compute the inverse.  

```python
def __truediv__(self, other: Data) -> "Tensor":
    r"""
    Overloads the `/` operator for element-wise division.

    Instead of direct division, this method multiplies the Tensor by the
    reciprocal of `other`, ensuring compatibility with automatic differentiation.

    Args:
        other (Data): The divisor, which can be a Tensor or a scalar.

    Returns:
        Tensor: A new Tensor representing the division result.
    """

    other = Tensor.data_gate(other)
    return self * (other**-1)

def __rtruediv__(self, other: Data) -> "Tensor":
    r"""
    Overloads the right-hand `/` operator.

    This ensures that division works correctly when a scalar or another
    compatible type appears on the left side of the `/` operator.

    Args:
        other (Data): The numerator, which can be a scalar or Tensor.

    Returns:
        Tensor: A new Tensor representing the division result.
    """

    other = Tensor.data_gate(other)
    return other * (self**-1)

def __itruediv__(self, other: Data) -> "Tensor":
    r"""
    Overloads the `/=` operator for in-place division.

    This modifies the Tensor’s data directly, improving efficiency.
    However, in-place operations do not support automatic differentiation
    (i.e., gradients will not be tracked).

    Args:
        other (Data): The divisor for in-place division.

    Returns:
        Tensor: The modified Tensor after in-place division.
    """

    self.data = self.data / Tensor.build_ndarray(other)
    return self
```


### `exp`

The **exponential** operation computes the exponent (base $e$) of each element in the tensor.

$$\exp(x) = e^x$$

The derivative of $e^x$ is:

$$\frac{d}{dx} e^x = e^x$$

```python
def exp(self) -> "Tensor":
    r"""
    Computes the element-wise exponential of the tensor's data.

    The operation applies the exponential function element-wise, raising the constant e 
    (Euler's number) to the power of each element in the tensor.

    Returns:
        Tensor: A new tensor where each element is the exponential of the corresponding element 
        in the input tensor.

    The exponential function is defined as:
        y = e^x, where `e` is Euler's number (approximately 2.71828), and `x` is the input tensor's element.

    The derivative of e^x with respect to x is:
        d/dx (e^x) = e^x
    """

    # Perform element-wise exponential operation (raise e to the power of each element)
    output = np.exp(self.data)
    
    # Initialize an empty list for dependencies (used for backpropagation)
    dependencies: List[Leaf] = []

    def _bkwd(grad: np.ndarray) -> np.ndarray:
        """
        Backward pass for the exponential operation.
        
        The derivative of e^x with respect to x is:
            d/dx (e^x) = e^x
        
        We multiply the gradient by e^x to propagate the gradient backward.
        
        Args:
            grad (np.ndarray): The gradient propagated from the next layer.
        
        Returns:
            np.ndarray: The gradient to propagate backward.
        """
        # The derivative of e^x is e^x, so we multiply the gradient by the output value
        return grad * output

    # If the tensor requires gradients (i.e., it's part of the computation graph), 
    # we store the backward function in the dependencies.
    if self.requires_grad:
        dependencies.append(
            Leaf(
                value=self,
                grad_fn=_bkwd
            )
        )

    # Return a new Tensor containing the result of the exponential operation and the necessary dependencies.
    return Tensor(output, self.requires_grad, dependencies)

```


## Simple activation functions

Because we prepared the derivative computations inside the `Tensor` class we don't need anything but the `forward` implemenatation for the activation functions! It would be a great example of the autogradient power.


```python
class Tanh(Module):
    def forward(self, input: Tensor) -> Tensor:
        return input.tanh()

class Sigmoid(Module):
    def forward(self, input: Tensor) -> Tensor:
        self.output = 1 / (1 + Tensor.exp(-input))
        return self.output

```

The `backward` step is fully on the `Tensor` class side. It's beautiful, isn't it?


## More Ops

In this section, we extend the functionality of the `Tensor` class by adding more fundamental operations. These operations are essential for building neural networks. For each operation, we provide the reasoning behind it, the derivative (for backpropagation), and the implementation of the forward and backward passes.


<iframe width="1707" height="765" src="https://www.youtube.com/embed/9EaSwTdoTag?list=PLWUV973D6J8imrTO4yJk3aI0NKJZgzFeG" title="Building PyTorch: Enhancing MicroTorch with Squeeze, View, and Clip Operations" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>


### Squeeze and Unsqueeze Operations

The `squeeze` and `unsqueeze` methods simplify reshaping operations, making it easier to adjust tensor shapes. These methods modify tensor dimensions by removing or adding singleton dimensions while preserving gradient tracking.

The `squeeze` method removes dimensions of size 1 from a specified axis, and its backward function ensures gradients are expanded back to their original shape during backpropagation. If `axis` is specified we only remove certain singleton dimensions, so during the backward pass, we must restore those specific dimensions. If `axis=None` we reshape the gradient to the original tensor's shape.


```python
def squeeze(self, axis: Union[int, Tuple[int], None] = None) -> "Tensor":
    r"""
    Removes dimensions of size 1 from the specified axis.

    Args:
        dim (int or Tuple[int]): The axis or axes to squeeze. Defaults to None

    Returns:
        Tensor: The squeezed tensor with tracked dependencies.
    """

    output = np.squeeze(self.data, axis=dim)
    dependencies: List[Leaf] = []

    if self.requires_grad:
        def _bkwd(grad: np.ndarray) -> np.ndarray:
            r"""
            Backward function for squeeze operation.

            Args:
                grad (np.ndarray): Incoming gradient.

            Returns:
                np.ndarray: Expanded gradient to match original dimensions.
            """

            if axis is None:
                # Reshape the gradient to the original tensor's shape
                return grad.reshape(self.shape)
            return np.expand_dims(grad, axis=axis)

        dependencies.append(
            Leaf(value=self, grad_fn=_bkwd)
        )

    return Tensor(output, self.requires_grad, dependencies)

```


The `unsqueeze` method adds a singleton dimension at the specified axis. The backward function removes this dimension during gradient propagation.

```python
def unsqueeze(self, dim: int) -> "Tensor":
    r"""
    Adds a singleton dimension at the specified axis.

    Args:
        dim (Tuple[int]): The axis at which to insert a new dimension

    Returns:
        Tensor: The unsqueezed tensor with tracked dependencies.
    """

    output = np.expand_dims(self.data, axis=dim)
    dependencies: List[Leaf] = []

    def _bkwd(grad: np.ndarray) -> np.ndarray:
        r"""
        Backward function for unsqueeze operation.

        Args:
            grad (np.ndarray): Incoming gradient.

        Returns:
            np.ndarray: Squeezed gradient to remove added dimension.
        """

        return np.squeeze(grad, axis=dim)  # Correctly squeeze the axis

    if self.requires_grad:
        dependencies.append(
            Leaf(value=self, grad_fn=_bkwd)
        )

    return Tensor(output, self.requires_grad, dependencies)

```

These methods ensure proper gradient tracking while reshaping tensors, making them useful in various neural network operations.


### `view`

The `view` method allows a `tensor` to be reshaped **without copying the underlying data**, provided the new shape is compatible with the original memory layout. This operation is efficient because it does not involve new memory allocation


```python
def view(self, shape: Tuple[int, ...]) -> "Tensor":
    r"""
    Reshape the tensor without changing its underlying data.

    This method returns a new `Tensor` object that shares the same data but is represented 
    with a different shape. The operation is efficient as it does not involve memory reallocation.

    Args:
        shape (Tuple[int, ...]): The desired shape of the tensor.

    Returns:
        Tensor: A new tensor with the same data but a different shape.

    Example:
        >>> x = Tensor(np.array([[1, 2, 3], [4, 5, 6]]), requires_grad=True)
        >>> y = x.view((3, 2))
        >>> print(y)
        Tensor([[1, 2],
                [3, 4],
                [5, 6]], requires_grad=True, shape=(3,2))

    Notes:
        - The returned tensor shares the same data as the original tensor.
        - If `requires_grad=True`, the backward function ensures gradients are reshaped correctly.
    """

    # This will be a new view object if possible; otherwise, it will be a copy
    output = self.data.reshape(shape)
    dependencies: List[Leaf] = []

    if self.requires_grad:
        def _bkwd(grad: np.ndarray) -> np.ndarray:
            r"""
            Backward pass for tensor view operation.

            This function ensures that the incoming gradient is reshaped 
            back to the original shape of the tensor.

            Args:
                grad (np.ndarray): The incoming gradient.

            Returns:
                np.ndarray: The gradient reshaped to match the original tensor.
            """

            return grad.reshape(self.shape)

        dependencies.append(Leaf(value=self, grad_fn=_bkwd))

    return Tensor(output, self.requires_grad, dependencies, dtype=self.dtype)
```


**Example Usage:**

```python
# Create a 2x3 tensor
x = Tensor(np.array([[1, 2, 3], [4, 5, 6]]), requires_grad=True)

# Reshape into a 3x2 tensor
y = x.view((3, 2))

# Print reshaped tensor
print(y)
# Output:
# Tensor([[1, 2],
#         [3, 4],
#         [5, 6]], requires_grad=True, shape=(3,2))

# Backward pass
y.backward(Tensor(np.ones((3, 2))))  # Gradient of ones

# Check gradients
print(x.grad)
# Output should be reshaped correctly to match original shape:
# [[1. 1. 1.]
#  [1. 1. 1.]]
```

**Example with backward:**

```python
a = Tensor([[1, 2],
            [3, 4]], requires_grad=True)

# a.view((4, 1)).unsqueeze(0).squeeze(0).backward(np.ones_like(a.data))
# a.view((4, 1)).unsqueeze(1).squeeze(1).unsqueeze(0).squeeze(0).backward(np.ones_like(a.data))

b = a.view((1, 4)).unsqueeze(1).squeeze(1).unsqueeze(0).squeeze(0)
b.backward(np.ones_like(b.data))
```


### `clip`

The `clip()` method allows you to restrict the values of a tensor to a specific range. This is particularly useful in deep learning, where you may want to avoid exploding or vanishing gradients by limiting the range of tensor values during backpropagation. The `clip()` method ensures that all values of the tensor remain within a given minimum and maximum value.

This method clips (limits) the values of a tensor to a specified range, `[min_value, max_value]`. Any values below `min_value` are set to `min_value`, and any values above `max_value` are set to `max_value`. If either bound is not specified (i.e., None), no limit is applied on that side.

Gradient Behavior: when `requires_grad=True`, this method tracks the gradient of the clipping operation. For values that fall within the clipping range, the gradient is passed through unchanged. For values outside the clipping range, the gradient is set to zero because the clipping operation is not differentiable at the boundaries.

```python
def clip(self, min_value: Optional[float] = None, max_value: Optional[float] = None) -> "Tensor":
    r"""
    Clip the tensor's values to the range [min_value, max_value].

    Args:
        min_value (Optional[float]): The minimum value to clip to. If None, no lower bound is applied.
        max_value (Optional[float]): The maximum value to clip to. If None, no upper bound is applied.

    Returns:
        Tensor: A new tensor with values clipped to the specified range.
    """

    # Perform clipping on the data
    output = np.clip(self.data, min_value, max_value)

    # Track dependencies if requires_grad is True
    dependencies: List[Leaf] = []

    if self.requires_grad:
        def _bkwd(grad: np.ndarray) -> np.ndarray:
            r"""
            Backward function for the clip operation.

            The gradient is passed through for values within the range [min_value, max_value].
            For values outside this range, the gradient is zero because the operation is not differentiable
            at the boundaries.

            Args:
                grad (np.ndarray): The gradient passed from the downstream operation.

            Returns:
                np.ndarray: The gradient for the input tensor.
            """

            # Create a mask for values within the clipping range
            mask = np.ones_like(self.data)

            # Apply the mask
            if min_value is not None:
                mask[self.data <= min_value] = 0
            if max_value is not None:
                mask[self.data >= max_value] = 0

            # Multiply the gradient by the mask
            return grad * mask

        dependencies.append(Leaf(value=self, grad_fn=_bkwd))

    # Return a new tensor with the clipped values and dependencies
    return Tensor(output, self.requires_grad, dependencies)
```


**Example:**

```python
# Example 1: Clipping tensor values to a range [0, 1]
a = Tensor([[0.5, -0.3], [1.2, 2.0]], requires_grad=True)
clipped_a = a.clip(min_value=0.0, max_value=1.0)

print("Original Tensor:\n", a.data)
print("Clipped Tensor:\n", clipped_a.data)

# Perform backward pass to test gradients
grad_output = np.ones_like(a.data)
clipped_a.backward(grad_output)
print("Gradients:\n", a.grad)

```

**Output:**

```
Original Tensor:
 [[ 0.5 -0.3]
 [ 1.2  2. ]]
Clipped Tensor:
 [[0.5 0. ]
 [1.  1. ]]
Gradients:
 [[1. 0.]
 [1. 0.]]

```


## `where`

<iframe width="1707" height="765" src="https://www.youtube.com/embed/4_kJy6hmv2E?list=PLWUV973D6J8imrTO4yJk3aI0NKJZgzFeG" title="Mastering Tensor Operations: Comparison, Where, and More in MicroTorch" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

The `where` method is a powerful element-wise selection operation for tensors, similar to NumPy's `np.where`. It allows conditional selection of elements from two tensors (`a` and `b`) based on a boolean condition. This is particularly useful in deep learning for masking values, implementing conditional operations, and handling gradients properly in automatic differentiation.

This function ensures that during backpropagation, gradients are propagated only to the selected elements of `a` and `b`, preventing unnecessary computations.

The comparison operators allow us to create boolean tensors by comparing tensor values element-wise. These operators are essential for conditional operations like `where`.

```python
# Comparison Operators
def __lt__(self, other: Data) -> "Tensor":
    """
    Less than operator (<).
    
    Creates a boolean tensor where each element is True if the corresponding
    element in self is less than the corresponding element in other.
    
    Args:
        other (Data): Value to compare against. Can be a Tensor or a compatible data type.
        
    Returns:
        Tensor: A boolean tensor with the comparison results.
    """

    other = Tensor.data_gate(other)
    return Tensor(self.data < other.data)

def __gt__(self, other: Data) -> "Tensor":
    """
    Greater than operator (>).
    
    Creates a boolean tensor where each element is True if the corresponding
    element in self is greater than the corresponding element in other.
    
    Args:
        other (Data): Value to compare against. Can be a Tensor or a compatible data type.
        
    Returns:
        Tensor: A boolean tensor with the comparison results.
    """

    other = Tensor.data_gate(other)
    return Tensor(self.data > other.data)

def __eq__(self, other: Data) -> "Tensor":
    """
    Equal to operator (==).
    
    Creates a boolean tensor where each element is True if the corresponding
    element in self is equal to the corresponding element in other.
    
    Args:
        other (Data): Value to compare against. Can be a Tensor or a compatible data type.
        
    Returns:
        Tensor: A boolean tensor with the comparison results.
    """

    other = Tensor.data_gate(other)
    return Tensor(self.data == other.data)

def __le__(self, other: Data) -> "Tensor":
    """
    Less than or equal to operator (<=).
    
    Creates a boolean tensor where each element is True if the corresponding
    element in self is less than or equal to the corresponding element in other.
    
    Args:
        other (Data): Value to compare against. Can be a Tensor or a compatible data type.
        
    Returns:
        Tensor: A boolean tensor with the comparison results.
    """

    other = Tensor.data_gate(other)
    return Tensor(self.data <= other.data)

def __ge__(self, other: Data) -> "Tensor":
    """
    Greater than or equal to operator (>=).
    
    Creates a boolean tensor where each element is True if the corresponding
    element in self is greater than or equal to the corresponding element in other.
    
    Args:
        other (Data): Value to compare against. Can be a Tensor or a compatible data type.
        
    Returns:
        Tensor: A boolean tensor with the comparison results.
    """

    other = Tensor.data_gate(other)
    return Tensor(self.data >= other.data)

def __ne__(self, other: Data) -> "Tensor":
    """
    Not equal to operator (!=).
    
    Creates a boolean tensor where each element is True if the corresponding
    element in self is not equal to the corresponding element in other.
    
    Args:
        other (Data): Value to compare against. Can be a Tensor or a compatible data type.
        
    Returns:
        Tensor: A boolean tensor with the comparison results.
    """

    other = Tensor.data_gate(other)
    return Tensor(self.data != other.data)

```

**Example:**

```python
# Create two tensors
a = Tensor(np.array([1, 2, 3, 4]))
b = Tensor(np.array([2, 2, 0, 5]))

# Compare the tensors
result_lt = a < b
print(result_lt.data)
# Output: [True False False True]

result_eq = a == b
print(result_eq.data)
# Output: [False True False False]

# Use with where operation
c = Tensor.where(a > b, a, b)
print(c.data)
# Output: [2 2 3 5]
```


The `where` method executes the main logic of selecting between two tensors `a` and `b` based on a condition. **Gradient for `a`:** If a value from tensor `a` was chosen (when `condition == True`), the gradient should flow to `a`. Otherwise, the gradient for `a` is zero.

```python
def _bkwd_a(grad: np.ndarray) -> np.ndarray:
    return np.where(condition.data, grad, 0.0)
```

**`np.where(condition.data, grad, 0.0)`** means: pass the gradient to `a` where the condition is `True` and zero out the gradient where the condition is `False`.

**Gradient for `b`:** If a value from tensor `b` was chosen (when `condition == False`), the gradient should flow to `b`. Otherwise, the gradient for `b` is zero.

```python
def _bkwd_b(grad: np.ndarray) -> np.ndarray:
    return np.where(condition.data, 0.0, grad)
```


**Implementation:**

```python
@staticmethod
def where(condition: "Tensor", a: "Tensor", b: "Tensor") -> "Tensor":
    r"""
    Performs element-wise selection based on a condition.

    This function returns a tensor where each element is taken from `a` if the corresponding 
    element in `condition` is True, otherwise from `b`. It supports automatic differentiation.

    Args:
        condition (Tensor): A boolean tensor where True selects from `a`, and False selects from `b`.
        a (Tensor): The tensor providing values where `condition` is True.
        b (Tensor): The tensor providing values where `condition` is False.

    Returns:
        Tensor: A tensor with elements from `a` where `condition` is True, and from `b` otherwise.

    Example:
        >>> cond = Tensor(np.array([[True, False], [False, True]]))
        >>> x = Tensor(np.array([[1, 2], [3, 4]]), requires_grad=True)
        >>> y = Tensor(np.array([[10, 20], [30, 40]]), requires_grad=True)
        >>> result = Tensor.where(cond, x, y)
        >>> print(result)
        Tensor([[ 1, 20],
                [30,  4]], requires_grad=True)

    Notes:
        - The returned tensor has `requires_grad=True` if either `a` or `b` requires gradients.
        - The backward function ensures gradients are passed only to the selected elements.
    """

    output = np.where(condition.data, a.data, b.data)  # Element-wise selection
    requires_grad = a.requires_grad or b.requires_grad
    dependencies: List[Leaf] = []

    if a.requires_grad:
        def _bkwd_a(grad: np.ndarray) -> np.ndarray:
            r"""
            Backward function for `a`.

            This ensures gradients flow only to the elements selected from `a`.

            Args:
                grad (np.ndarray): Gradient of the output tensor.

            Returns:
                np.ndarray: Gradient for `a`, masked where `condition` is False.
            """
            return np.where(condition.data, grad, 0.0)

        dependencies.append(Leaf(a, _bkwd_a))

    if b.requires_grad:
        def _bkwd_b(grad: np.ndarray) -> np.ndarray:
            r"""
            Backward function for `b`.

            This ensures gradients flow only to the elements selected from `b`.

            Args:
                grad (np.ndarray): Gradient of the output tensor.

            Returns:
                np.ndarray: Gradient for `b`, masked where `condition` is True.
            """
            return np.where(condition.data, 0.0, grad)

        dependencies.append(Leaf(b, _bkwd_b))

    return Tensor(output, requires_grad, dependencies)
```

**Example**

```python
# Define a condition tensor
condition = Tensor(np.array([[True, False], [False, True]]))

# Define input tensors
x = Tensor(np.array([[1, 2], [3, 4]]), requires_grad=True)
y = Tensor(np.array([[10, 20], [30, 40]]), requires_grad=True)

# Apply the where function
result = Tensor.where(condition, x, y)

# Print result
print(result)
# Output:
# Tensor([[ 1, 20],
#         [30,  4]], requires_grad=True)

# Backward pass
grad_output = Tensor(np.array([[1.0, 1.0], [1.0, 1.0]]))
result.backward(grad_output)

# Check gradients
print(x.grad)  # Should have gradients only where condition is True
# [[1. 0.]
#  [0. 1.]]

print(y.grad)  # Should have gradients only where condition is False
# [[0. 1.]
#  [1. 0.]]

# You can use the conditional operation
# Create two tensors
a = Tensor(np.array([1, 2, 3, 4]))
b = Tensor(np.array([2, 2, 0, 5]))

# Compare the tensors using a conditional operation
result_lt = a < b
print(result_lt.data)
# Output: [True False False True]

# Use the `where` operation to select values based on the comparison result
result = Tensor.where(result_lt, a, b)
print(result.data)
# Output: [1 2 3 5]
```


### Expanding `where` - More Useful Tensor Operations

With the static method `where`, we can implement several useful tensor operations like `maximum` and `minimum`.

```python
@staticmethod
def maximum(a: Data, b: Data) -> "Tensor":
    r"""
    Apply element-wise max operation: max(a: "Tensor", b: "Tensor") -> "Tensor"
    Returns a Tensor with the result of element-wise maximum.
    """

    a, b = Tensor.data_gate(a), Tensor.data_gate(b)

    return Tensor.where(a > b, a, b)

@staticmethod
def minimum(a: Data, b: Data) -> "Tensor":
    r"""
    Apply element-wise min operation: min(a: "Tensor", b: "Tensor") -> "Tensor"
    Returns a Tensor with the result of element-wise minimum.
    """

    a, b = Tensor.data_gate(a), Tensor.data_gate(b)

    return Tensor.where(a < b, a, b)

```

These implementations are clean and use where for efficient, element-wise comparisons. We can push this further and implement advanced operations in a concise way. For example - the `threshold` method sets values above a given threshold while leaving others unchanged.

```python
def threshold(self, threshold: float, value: float) -> "Tensor":
    return Tensor.where(self > threshold, self, Tensor(value))
```

The `masked_fill` method replaces values based on a boolean mask.

```python
def masked_fill(self, mask: "Tensor", value: float) -> "Tensor":
    return Tensor.where(mask, Tensor(value), self)
```

The `sign` method returns the sign of each element in the tensor

```python
def sign(self) -> "Tensor":
    return Tensor.where(
        self > 0, Tensor(1),
        Tensor.where(self < 0, Tensor(-1), Tensor(0))
    )
```

We can also rewrite the `clip` method using where. This simplifies the code by handling both the forward and backward passes through the `where` implementation.

```python
def clip(self, min_value: Optional[float] = None, max_value: Optional[float] = None) -> "Tensor":
    return Tensor.where(
        self < min_value, Tensor(min_value),
        Tensor.where(self > max_value, Tensor(max_value), self)
    )
```

By using `where`, we reduce code duplication and ensure consistency across these tensor operations.


## Indexing

Indexing in the `Tensor` class allows selecting specific elements from the tensor using another tensor or a NumPy array as an index. This operation is useful for extracting sub-tensors or performing advanced slicing operations.

<iframe width="1707" height="934" src="https://www.youtube.com/embed/mk2FkpeZOAM?list=PLWUV973D6J8imrTO4yJk3aI0NKJZgzFeG" title="" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

When `requires_grad=True`, the backward pass ensures that gradients are propagated correctly by constructing a zero tensor and filling only the indexed positions with the incoming gradient.

Valid Python index types:

* `int` (for scalar indexing)
* `slice` (for range selection like x[1:3])
* `Tuple` (for multi-dimensional indexing)
* `List` (for list-based indexing)

Also we can include the `np.ndarray` and `Tensor` for flexibility! 


```python
IndexType = Union[int, slice, Tuple, List[int], np.ndarray, "Tensor"]

def __getitem__(self, index: IndexType) -> "Tensor":
    r"""
    Perform indexing on the tensor.

    This method allows for selecting specific elements from the tensor using 
    another tensor or a NumPy array as an index.

    Args:
        index (IndexType): The indices to select from the tensor.

    Returns:
        Tensor: A new tensor containing the selected elements.

    Example:
        >>> x = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
        >>> idx = Tensor([0, 2])
        >>> y = x[0, idx]  # Selects elements [1, 3]
        >>> print(y)
        Tensor([1, 3], requires_grad=True, shape=(2,))

    Notes:
        - If `requires_grad=True`, the backward function ensures that the gradient 
          is only applied to the indexed positions.
    """

    # Normalize tensor-based indexing to numpy arrays
    if isinstance(index, (Tensor, np.ndarray)):
        index = Tensor.data_gate(index).data

    output = self.data[index]  # Perform indexing operation
    dependencies = []

    if self.requires_grad:
        def _bkwd(grad: np.ndarray) -> np.ndarray:
            r"""
            Backward pass for tensor indexing.

            This function constructs a zero tensor of the same shape as the original 
            tensor and assigns the incoming gradient only to the indexed positions.

            Args:
                grad (np.ndarray): Gradient from the next layer.

            Returns:
                np.ndarray: Gradient propagated back to the indexed positions.
            """

            full_grad = np.zeros_like(self.data)
            # Handle multiple uses of the same index correctly
            np.add.at(full_grad, index, grad)
            return full_grad

        dependencies.append(Leaf(value=self, grad_fn=_bkwd))
        
    return Tensor(output, self.requires_grad, dependencies)
```

**Example Usage:**

```python
# Create a tensor
x = Tensor([[10, 20, 30], [40, 50, 60]], requires_grad=True)

# Indexing operation
idx = Tensor([0, 2])  # Selecting elements at index 0 and 2
y = x[0, idx]  # Retrieves [10, 30]

# Print result
print(y)  # Tensor([10, 30], requires_grad=True, shape=(2,))

# Backward pass
y.backward(Tensor([1.0, 1.0]))  # Set gradient to ones

# Check gradients
print(x.grad)  
# Output should propagate gradients only to the selected positions:
# [[1.  0.  1.]
#  [0.  0.  0.]]
```


## More Tensor ops

### `abs`

The **absolute value** operation computes the magnitude of each element in a tensor, disregarding the sign. 

$$\text{abs}(x) = |x|$$

The derivative of $\text{abs}(x)$ is:

$$\frac{d}{dx} |x| = \text{sgn}(x)$$

where the sign function $\text{sgn}(x)$ is defined as:

$$\text{sgn}(x) =
\begin{cases}
  1, & \text{if } x > 0 \\
  -1, & \text{if } x < 0 \\
  0, & \text{if } x = 0
\end{cases}$$

Python Code for `abs` Operation with the Derivative


```python
def abs(self) -> "Tensor":
    r"""
    Computes the absolute value of the tensor's elements.

    The absolute value of each element is computed element-wise, and the result is returned as a new tensor.

    Returns:
        Tensor: A new tensor containing the absolute values of the input tensor's elements.
    
    The derivative of the absolute value function is handled in the backward pass using the sign of the input tensor.
    The gradient for positive values is 1, for negative values is -1, and the gradient is undefined at zero.
    """

    # Perform absolute value operation on the data
    output = np.abs(self.data)

    # Initialize the list of dependencies for gradient calculation
    dependencies: List[Leaf] = []

    # Backward function to compute the gradient for the absolute value operation
    def _bkwd(grad: np.ndarray) -> np.ndarray:
        r"""
        Compute the gradient of the absolute value operation.
        
        Args:
            grad (np.ndarray): The gradient passed from the downstream operation.
        
        Returns:
            np.ndarray: The gradient for the input tensor.
        
        The gradient of abs(x) is the sign of x:
        - If x > 0, the gradient is 1.
        - If x < 0, the gradient is -1.
        - The gradient is undefined at x = 0.
        """

        # The derivative of abs(x) is the sign of x: 1 for positive x, -1 for negative x
        return grad * np.sign(self.data)

    # If the tensor requires gradients, add the backward function to the dependencies list
    if self.requires_grad:
        dependencies.append(
            Leaf(
                value=self,  # The input tensor
                grad_fn=_bkwd  # The backward function to compute the gradients
            )
        )

    # Return a new tensor containing the absolute values, with the gradient dependencies if needed
    return Tensor(output, self.requires_grad, dependencies)

```

We can also leverage `where` to implement the `abs` method. This approach removes the need for an explicit backward implementation since `where` already handles it internally. Here's a clean, one-liner implementation:

```python
def abs(self) -> "Tensor":
    return Tensor.where(self >= 0, self, -self)
```


### `max`

<iframe width="1233" height="694" src="https://www.youtube.com/embed/olINYduKFJY?list=PLWUV973D6J8imrTO4yJk3aI0NKJZgzFeG" title="" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

The **max operation** returns the maximum value of a tensor along a specified axis. If no axis is specified, it returns the maximum value from the entire tensor.

For differentiation, the gradient of the maximum function is defined as:

$$\frac{d}{dx} \max(X) =
\begin{cases}
  1, & \text{if } x \text{ is the maximum value} \\
  0, & \text{otherwise}
\end{cases}$$


During backpropagation, only the maximum value(s) receive a gradient, while all other elements receive 0. But things are more complicated and to deconvolve here are the general steps for backward calculation:

* Identify which values are the maximum across the specified axis (or globally).

* Create a mask that indicates which elements of the original tensor were involved in the maximum value (i.e., the `1`'s).

* Distribute gradients only to those maximum values, propagating the gradient from downstream only to the maximum values.

* Handle multiple maxima (if two values are the same and are both maximum, gradients are split equally between them).


For the forward pass it's pretty obvious:

```python
def max(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> "Tensor":
    # Calculate the maximum values in forward mode
    output = np.max(self.data, axis=axis, keepdims=keepdims)

    dependencies = []

    # backward is here...

    # Return a new tensor containing the absolute values, with the gradient dependencies if needed
    return Tensor(output, self.requires_grad, dependencies)

```


And for the backward let's take an approach where we work through each test case, identify what it requires from the `max` method, and implement the corresponding logic step by step.


**Case 1: Simple case - max with no axis (scalar output)**

```python
a1 = Tensor([1, 2, 3, 4], requires_grad=True)
b1 = a1.max()  # Should be 4

b1.backward()
print(a1.grad)  # Expected: [0, 0, 0, 1]
```

During backpropagation, assign a gradient of 1 to the position of the maximum values by identifying the position of the max value using a mask and distributing the incoming gradient across the max values.

```python
if self.requires_grad:
    def _bkwd(grad: np.ndarray) -> np.ndarray:
        # Create a mask where max values are True
        mask = (self.data == output)
        return mask * grad

    dependencies.append(Leaf(value=self, grad_fn=_bkwd))

```

The gradient should match the expected value: `[0, 0, 0, 1]`. 

**Case 2: Max with Duplicate Maximums**

```python
a2 = Tensor([1, 4, 4, 2], requires_grad=True)
b2 = a2.max()
b2.backward()
print(a2.grad)  # Expected: [0, 0.5, 0.5, 0]
```

When there are **multiple maximum values**, we need to **split the gradient equally** among them. We handle the backward pass by identifying the position of the max value using a mask and distributing the incoming gradient across the max values.

```python
# Normalize by the number of max occurrences
count = np.sum(mask)
# Split the gradient equally
return mask * (grad / count)

```

**Case 3: Max Along a Specific Axis like `axis=0`**

```python
a3 = Tensor([[1, 5],
             [5, 4]], requires_grad=True)
b3 = a3.max(axis=0)  # Should be [5, 5]
b3.backward(np.array([1, 1]))
print(a3.grad)  # Expected: [[0, 1], [1, 0]]

```

Compute the maximum along a specified axis and expand the gradient correctly along that axis during backpropagation.

We update the `_bkwd` function:

```python
if axis is None:
    # For flattened tensor max, just count total max elements
    count = np.sum(mask)
    return mask * (grad / count)

# Count max occurrences
count = np.sum(mask, axis=axis, keepdims=True)
grad_expanded = np.expand_dims(grad, axis=axis)

# Normalize and Apply gradient
return mask * (grad_expanded / count)
```

**Case 4: Handling `keepdims=True`**

We use `keepdims` to ensure the gradient shape aligns with the input tensor during backpropagation. 

```python
a4 = Tensor([[1, 5],
             [5, 4]], requires_grad=True)
b4 = a4.max(axis=0, keepdims=True)
b4.backward(np.ones_like(b4.data))

print(a4.grad)  # Expected: [[0, 1], [1, 0]]
```

If `keepdims=True` the output already has the correct shape, so no expansion is needed. This preserves the dimensions and ensures the gradient is applied correctly. If `keepdims=False` the output is reduced along the specified axis, so we must expand the gradient to match the original shape for proper broadcasting during multiplication.

```python
grad_expanded = grad if keepdims else np.expand_dims(grad, axis=axis)
```

Or we can create a short version:

```python
count = np.sum(mask) if axis is None \
    else np.sum(mask, axis=axis, keepdims=True)

grad_expanded = grad if keepdims or axis is None \
    else np.expand_dims(grad, axis=axis)

return mask * (grad_expanded / count)
```

**Full implementation:**

```python
def max(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> "Tensor":
    r"""
    Computes the maximum value along the specified axis.

    This function returns the maximum value(s) of the tensor, either element-wise (if no axis is specified) 
    or along a given axis. The backward pass ensures that only the maximum elements receive gradients.

    Args:
        axis (Optional[Union[int, Tuple[int]]]): The axis along which to compute the maximum.
            If None, the maximum of the entire tensor is returned.
        keepdims (bool): If True, retains reduced dimensions with size 1.
    
    Returns:
        Tensor: A new tensor containing the maximum values along the given axis.

    The gradient is computed during backpropagation by assigning a gradient of 1 
    to the maximum element(s) and 0 elsewhere.
    """

    # Calculate the maximum values
    output = np.max(self.data, axis=axis, keepdims=keepdims)

    dependencies = []

    if self.requires_grad:
        def _bkwd(grad: np.ndarray) -> np.ndarray:
            # Handle multi-dimensional case
            output_expanded = output if keepdims or axis is None else np.expand_dims(output, axis=axis)
            # Create a mask where only the max values are True
            mask = (self.data == output_expanded)

            if axis is None:
                # For flattened tensor max, just count total max elements
                count = np.sum(mask)
                return mask * (grad / count)

            # Count max occurrences
            count = np.sum(mask, axis=axis, keepdims=True)
            grad_expanded = grad if keepdims else np.expand_dims(grad, axis=axis)

            # Normalize and Apply gradient
            return mask * (grad_expanded / count)

        dependencies.append(Leaf(value=self, grad_fn=_bkwd))

    return Tensor(output, self.requires_grad, dependencies)

```


**Example:**

```python
print("============ Testing max backward ============")

# Test 1: Simple case - max with no axis (scalar result)
print("\nTest 1: max with no axis (scalar output)")
a1 = Tensor([1, 2, 3, 4], requires_grad=True)
b1 = a1.max()  # Should be 4
b1.backward()
print(f"Input tensor: {a1.data}")
print(f"Max value: {b1.data}")
print(f"Gradient: {a1.grad}")
# Expected: Only the position with max value (4) should have gradient 1, others 0
expected_grad1 = np.array([0, 0, 0, 1])
print(f"Test passed: {np.allclose(a1.grad, expected_grad1)}")

# Test 2: Max with duplicate maximum values
print("\nTest 2: max with duplicate maximums")
a2 = Tensor([1, 4, 4, 2], requires_grad=True)
b2 = a2.max()
b2.backward()
print(f"Input tensor: {a2.data}")
print(f"Max value: {b2.data}")
print(f"Gradient: {a2.grad}")
# Expected: Positions with max value (4) should have gradient 0.5 each (1/count)
expected_grad2 = np.array([0, 0.5, 0.5, 0])
print(f"Test passed: {np.allclose(a2.grad, expected_grad2)}")

# Test 3: Max along a specific axis
print("\nTest 3: max along axis=0")
a3 = Tensor([[1, 5],
                [5, 4]], requires_grad=True)
b3 = a3.max(axis=0)  # Should be [5, 5]
b3.backward(np.array([1, 1]))
print(f"Input tensor:\n{a3.data}")
print(f"Max value: {b3.data}")
print(f"Gradient:\n{a3.grad}")
# Expected: First column: [0, 1], Second column: [1, 0]
expected_grad3 = np.array([[0, 1], [1, 0]])
print(f"Test passed: {np.allclose(a3.grad, expected_grad3)}")

# Test 4: Max along axis with keepdims=True
print("\nTest 4: max along axis=0, keepdims=True")
a4 = Tensor([[1, 5],
                [5, 4]], requires_grad=True)
b4 = a4.max(axis=0, keepdims=True)
b4.backward(np.ones_like(b4.data))
print(f"Input tensor:\n{a4.data}")
print(f"Max value:\n{b4.data}")
print(f"Gradient:\n{a4.grad}")
expected_grad4 = np.array([[0, 1], [1, 0]])
print(f"Test passed: {np.allclose(a4.grad, expected_grad4)}")

# Test 5: Max along axis with duplicate maximums
print("\nTest 5: max along axis with duplicate maximums")
a5 = Tensor([[5, 3],
                [5, 4]], requires_grad=True)
b5 = a5.max(axis=0)  # Should be [5, 4]
b5.backward(np.array([1, 1]))
print(f"Input tensor:\n{a5.data}")
print(f"Max value: {b5.data}")
print(f"Gradient:\n{a5.grad}")
# Expected: First column has two 5s, so grad = [0.5, 0.5], Second column: [0, 1]
expected_grad5 = np.array([[0.5, 0], [0.5, 1]])
print(f"Test passed: {np.allclose(a5.grad, expected_grad5)}")

# Test 6: Max over multiple dimensions
print("\nTest 6: max over multiple dimensions")
a6 = Tensor([[[1, 2], [3, 4]],
                [[5, 6], [7, 8]]], requires_grad=True)
b6 = a6.max()  # Should be 8
b6.backward()
print(f"Input tensor shape: {a6.shape}")
print(f"Max value: {b6.data}")
print(f"Gradient at max position: {a6.grad[1,1,1]}")
# Only position of max value (8) should have gradient = 1
expected_position = (1, 1, 1)  # In 0-indexed, (1,1,1) is the last element
expected_grad6 = np.zeros((2, 2, 2))
expected_grad6[expected_position] = 1
print(f"Test passed: {np.allclose(a6.grad, expected_grad6)}")

# Test 7: Another axis test with 3D tensor
print("\nTest 7: max on axis in 3D tensor")
a7 = Tensor([[[1, 2], [3, 4]],
                [[5, 6], [7, 8]]], requires_grad=True)
b7 = a7.max(axis=1)  # Max along middle dimension
print(f"Input tensor shape: {a7.shape}")
print(f"Output shape: {b7.shape}")
print(f"Output data: \n{b7.data}")
b7.backward(np.ones_like(b7.data))
print(f"Gradient: \n{a7.grad}")
expected_grad7 = np.array([[[0, 0], [1, 1]],
                            [[0, 0], [1, 1]]])
print(f"Test passed: {np.allclose(a7.grad, expected_grad7)}")

# Return overall test result
print("\n============ Summary ============")
print("All tests should pass if each element of the gradient is correct.")

```


### `min`

<iframe width="1233" height="694" src="https://www.youtube.com/embed/kuElg7VtSII?list=PLWUV973D6J8imrTO4yJk3aI0NKJZgzFeG" title="" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

The **min operation** returns the minimum value of a tensor along a specified axis. If no axis is specified, it returns the minimum value from the entire tensor.

For differentiation, the gradient of the minimum function is defined as:

$$\frac{d}{dx} \min(X) = \begin{cases} 1, & \text{if } x \text{ is the minimum value} \\ 0, & \text{otherwise} \end{cases}$$

The gradient is the same as for the `max` operation, we can create a `bkwd` method, that can serve for both methods:

```python
def bkwd_minmax(
    self,
    output: np.ndarray,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdims: bool = False
) -> np.ndarray:
    def _bkwd(grad: np.ndarray) -> np.ndarray:
        count = np.sum(mask) if axis is None \
            else np.sum(mask, axis=axis, keepdims=True)

        grad_expanded = grad if keepdims or axis is None \
            else np.expand_dims(grad, axis=axis)

        return mask * (grad_expanded / count)

    return _bkwd

```

Then we can make refactoring for the `max`:

```python
def max(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> "Tensor":
    r"""
    Computes the maximum value along the specified axis.

    This function returns the maximum value(s) of the tensor, either element-wise (if no axis is specified) 
    or along a given axis. The backward pass ensures that only the maximum elements receive gradients.

    Args:
        axis (Optional[Union[int, Tuple[int]]]): The axis along which to compute the maximum.
            If None, the maximum of the entire tensor is returned.
        keepdims (bool): If True, retains reduced dimensions with size 1.
    
    Returns:
        Tensor: A new tensor containing the maximum values along the given axis.

    The gradient is computed during backpropagation by assigning a gradient of 1 
    to the maximum element(s) and 0 elsewhere.
    """

    # Calculate the maximum values
    output = np.max(self.data, axis=axis, keepdims=keepdims)

    dependencies = []

    if self.requires_grad:
        dependencies.append(
            Leaf(value=self, grad_fn=self.bkwd_minmax(output, axis, keepdims))
        )

    return Tensor(output, self.requires_grad, dependencies)

```

And now we can implement `min`, the only difference lay in the forward step, calculate the minimum values for the output:

```python
def min(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> "Tensor":
    r"""
    Computes the minimum value along the specified axis.

    This function returns the minimum value(s) of the tensor, either element-wise (if no axis is specified) 
    or along a given axis. The backward pass ensures that only the minimum elements receive gradients.

    Args:
        axis (Optional[int]): The axis along which to compute the minimum.
            If None, the minimum of the entire tensor is returned.
        keepdims (bool): If True, retains reduced dimensions with size 1.
    
    Returns:
        Tensor: A new tensor containing the minimum values along the given axis.

    The gradient is computed during backpropagation by assigning a gradient of 1 
    to the minimum element(s) and 0 elsewhere.
    """

    # Calculate the minimum values
    output = np.min(self.data, axis=axis, keepdims=keepdims)

    dependencies = []

    if self.requires_grad:
        dependencies.append(
            Leaf(value=self, grad_fn=self.bkwd_minmax(output, axis, keepdims))
        )

    return Tensor(output, self.requires_grad, dependencies)

```


**Example:**

```python
# Create a tensor
x = Tensor(np.array([[1, 3, 5], [2, 8, 4]]), requires_grad=True)

# Compute the minimum along axis 1
result = x.min(axis=1, keepdims=True)

# Print the result
print(result)
# Output:
# Tensor([[1],
#         [2]], requires_grad=True)

# Backward pass
grad_output = Tensor(np.array([[1.0], [1.0]]))
result.backward(grad_output)

# Check gradients
print(x.grad)
# Expected Output:
# [[1. 0. 0.]
#  [1. 0. 0.]]
```


### `sum`

The **sum** operation computes the sum of all elements or along a specified axis. The formula for the sum is straightforward:

$$\text{sum}(x) = \sum_{i} x_i$$

The derivative of the sum operation with respect to each element is simply 1:

$$\frac{d}{dx} \sum x_i = 1$$

In the forward pass, we use `np.sum()` to compute the sum of tensor elements along the given axis (or across all elements if no axis is specified).


```python
def sum(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> "Tensor":
    # Compute the sum of the tensor along the specified axis (or over all elements)
    output = np.sum(self.data, axis=axis, keepdims=keepdims)

    dependencies = []

    if self.requires_grad:
        def _bkwd(grad: np.ndarray) -> np.ndarray:
            # ...
    # ...

    return Tensor(output, self.requires_grad, dependencies)
```


In the backward pass we need to compute the gradient with respect to the sum of tensor elements. First, we initialize `full_grad` as an array of ones with the same shape as the input tensor. This represents the gradient that will be propagated back through the tensor.

```python
def _bkwd(grad: np.ndarray) -> np.ndarray:
    # Initialize a gradient array of the same shape as the input
    full_grad = np.ones_like(self.data)
```


When `axis is None`, we've summed over all elements

```python
if axis is None:
    return full_grad * grad

```


If `axis` is specified and `keepdims=True`, we maintain the shape of the input tensor during the backward pass. If there are multiple elements summed over an axis, each element gets a share of the gradient proportional to its contribution.

```python
grad_expanded = grad if keepdims else np.expand_dims(grad, axis=axis)

return full_grad * grad_expanded

```

**Full implementation:**

```python
def sum(self, axis: int = None, keepdims: bool = False) -> "Tensor":
    r"""
    Computes the sum of all elements in the tensor along a specified axis.

    Args:
        axis (int or tuple of ints, optional): Axis or axes along which a sum is performed. 
            The default is to sum over all dimensions.
        keepdims (bool, optional): If True, the reduced axes are retained with length 1, 
            otherwise the result is reshaped to eliminate the reduced axes. Default is False.

    Returns:
        Tensor: A new tensor with the summed values and the gradient dependencies.
    
    The sum operation accumulates all elements along a given axis (or all elements if axis is None).
    The gradient for this operation is computed by broadcasting the incoming gradient across 
    the axis and summing it back up for each element.
    """

    # Perform summation over specified axis
    output = np.sum(self.data, axis=axis, keepdims=keepdims)

    # Initialize the list of dependencies for gradient calculation
    dependencies: List[Leaf] = []

    if self.requires_grad:
        # Backward function to calculate the gradients for the sum operation
        def _bkwd(grad: np.ndarray) -> np.ndarray:
            r"""
            Compute the gradient of the sum operation. The gradient is summed along the specified axis.
            
            Args:
                grad (np.ndarray): The gradient passed from the downstream operation.

            Returns:
                np.ndarray: The gradient for the input tensor.
            
            If `keepdims` is True, the gradient is broadcasted to match the original tensor's dimensions.
            """

            # Initialize a gradient array of the same shape as the input
            full_grad = np.ones_like(self.data)

            # When axis is None, we've summed over all elements
            if axis is None:
                return full_grad * grad

            grad_expanded = grad if keepdims else np.expand_dims(grad, axis=axis)

            return full_grad * grad_expanded

        # Add the backward function to the dependencies list
        dependencies.append(
            Leaf(
                value=self,  # The input tensor
                grad_fn=_bkwd  # The backward function to compute the gradients
            )
        )

    # Return a new tensor containing the sum, with the gradient dependencies if needed
    return Tensor(output, self.requires_grad, dependencies)

```


### `mean`

The **mean** operation computes the average of all elements or along a specified axis. The formula for the mean is:

$$\text{mean}(x) = \frac{1}{n} \sum_{i} x_i$$

where $n$ is the number of elements.

The derivative of the mean operation with respect to each element is:

$$\frac{d}{dx} \frac{1}{n} \sum x_i = \frac{1}{n}$$


```python
def mean(self, axis: Optional[int] = None, keepdims: bool = False) -> "Tensor":
    r"""
    Computes the mean of elements along a specified axis.

    The mean is calculated as the sum of the elements divided by the number of elements along the given axis.

    Args:
        axis (Optional[int]): Axis along which to compute the mean. If None, computes the mean of all elements.
        keepdims (bool): Whether to keep the reduced dimensions in the output.

    Returns:
        Tensor: A new tensor containing the mean of the elements along the specified axis.

    The mean is calculated using the formula:
        mean = sum(elements) / number_of_elements
    """

    count = self.data.shape[axis] if axis is not None else self.size
    return self.sum(axis=axis, keepdims=keepdims) / count

```


## More Activation Functions!

With our enhanced `Tensor` class, we can now build **complex functions** with ease. Let's explore essential activation functions — **ReLU**, **LeakyReLU** and **Softmax** — which are widely used in modern **neural network architectures**.


```python
class ReLU(Module):
    def forward(self, input: Tensor) -> Tensor:
        # Apply ReLU: max(0, x)
        return Tensor.maximum(0, input)
    
class LeakyReLU(Module):
    def __init__(self, alpha: float = 0.01):
        super().__init__()
        self.alpha = alpha

    def forward(self, input: Tensor) -> Tensor:
        # Apply LeakyReLU: max(0, x) + alpha * min(0, x)
        return Tensor.maximum(0, input) + self.alpha * Tensor.minimum(0, input)

class Softmax(Module):
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, input: Tensor) -> Tensor:
        # For numerical stability, subtract the maximum value along the specified dimension.
        exp_input = (input - input.max(axis=self.dim, keepdims=True)).exp()
        return exp_input / exp_input.sum(axis=self.dim, keepdims=True)

```

These implementations **highlight the power of the `Tensor` class** - with minimal code, we can define functions that form the **building blocks** of deep learning models. Thanks to the design of our framework, we only need to implement the **forward** method - **backpropagation** is handled automatically!


## Summary

We now have a solid foundation for building the remaining tools needed to construct and train our Deep Neural Network. These fundamental tensor operations will enable automatic differentiation, making it easier to implement anything we need.
