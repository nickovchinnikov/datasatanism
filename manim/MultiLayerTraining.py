from dataclasses import dataclass
from typing import List, Literal, Tuple

import numpy as np

from manim import *


@dataclass
class Parameter:
    name: str
    data: np.ndarray
    grad: np.ndarray


class Module:
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
        
    def forward(self, *args, **kwargs):
        raise NotImplementedError
    
    def backward(self, *args, **kwargs):
        raise NotImplementedError
    
    def parameters(self) -> List[Parameter]:
        return []
    
    def zero_grad(self):
        for param in self.parameters():
            param.grad.fill(0)


# Define a custom type alias for initialization methods
InitMethod = Literal["xavier", "he", "he_leaky", "normal", "uniform"]

def parameter(
    input_size: int,
    output_size: int,
    init_method: InitMethod = "xavier",
    gain: float = 1,
    alpha: float = 0.01
) -> np.ndarray:
    weights = np.random.randn(input_size, output_size)

    if init_method == "xavier":
        std = gain * np.sqrt(1.0 / input_size)
        return std * weights
    if init_method == "he":
        std = gain * np.sqrt(2.0 / input_size)
        return std * weights
    if init_method == "he_leaky":
        std = gain * np.sqrt(2.0 / (1 + alpha**2) * (1 / input_size))
        return std * weights
    if init_method == "normal":
        return gain * weights
    if init_method == "uniform":
        return gain * np.random.uniform(-1, 1, size=(input_size, output_size))

    raise ValueError(f"Unknown initialization method: {init_method}")


class Linear(Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        init_method: InitMethod = "xavier"
    ):
        self.input: np.ndarray = None

        self.weights: np.ndarray = parameter(input_size, output_size, init_method)
        self.d_weights: np.ndarray = np.zeros_like(self.weights)

        self.biases: np.ndarray = np.zeros((1, output_size))
        self.d_biases: np.ndarray = np.zeros_like(self.biases)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        x1 = x @ self.weights + self.biases
        return x1
    
    def backward(self, d_out: np.ndarray) -> np.ndarray:
        self.d_weights = self.input.T @ d_out
        self.d_biases = np.sum(d_out, axis=0, keepdims=True)

        return d_out @ self.weights.T

    def parameters(self):
        return [
            Parameter(
                name="weights",
                data=self.weights,
                grad=self.d_weights
            ),
            Parameter(
                name="biases",
                data=self.biases,
                grad=self.d_biases
            ),
        ]


class BCELoss(Module):
    def forward(
        self, pred: np.ndarray, target: np.ndarray, epsilon: float = 1e-7
    ) -> np.ndarray:        
        loss = -(
            target * np.log(pred + epsilon) + 
            (1 - target) * np.log(1 - pred + epsilon)
        )

        return np.mean(loss)

    def backward(
        self, pred: np.ndarray, target: np.ndarray, epsilon: float = 1e-7
    ) -> np.ndarray:
        grad = (pred - target) / (pred * (1 - pred) + epsilon)
        return grad


class Sigmoid(Module):
    def forward(self, x: np.ndarray):
        # Apply the Sigmoid function element-wise
        self.output = 1 / (1 + np.exp(-x))
        return self.output

    def backward(self, d_out: np.ndarray):
        # Derivative of the Sigmoid function: sigmoid * (1 - sigmoid)
        ds = self.output * (1 - self.output)
        return d_out * ds


class SGD:
    def __init__(
        self,
        lr: float = 0.01,
        momentum: float = 0.0
    ):
        self.lr = lr
        self.momentum = momentum
        self.velocity = {}

    def step(self, module: Module):
        for param in module.parameters():
            param_id = param.name

            # Init velocity if not exists
            if param_id not in self.velocity:
                self.velocity[param_id] = np.zeros_like(param.data)

            grad = param.grad.copy()

            # Update momentum
            self.velocity[param_id] = self.momentum * self.velocity[param_id] - self.lr * grad

            # Update parameters
            param.data += self.velocity[param_id]


class LeakyReLU(Module):
    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha

    def forward(self, x: np.ndarray):
        self.input = x
        return np.where(x > 0, x, self.alpha * x)

    def backward(self, d_out: np.ndarray):
        dx = np.ones_like(self.input)
        dx[self.input < 0] = self.alpha
        return d_out * dx


class Sequential(Module):
    def __init__(self, layers: List[Module]):
        self.layers = layers

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, d_out: np.ndarray, lr: float = 0.001) -> np.ndarray:
        for layer in reversed(self.layers):
            d_out = layer.backward(d_out)
        return d_out
    
    def parameters(self) -> List[Parameter]:
        params = []
        for i, layer in enumerate(self.layers):
            for param in layer.parameters():
                # Add unique prefix name for optimization step
                param.name = f"layer_{i}_{param.name}"
                params.append(param)
        return params


def make_spiral_dataset(
    n_samples: int = 100,
    noise: float = 0.2,
    seed: int = None,
    x_range: Tuple[int, int] = (-1, 1),
    y_range: Tuple[int, int] = (-1, 1)
):
    # Install the random seed
    if seed:
        np.random.seed(seed)

    n = n_samples // 2  # Split samples between two spirals

    # Generate first spiral
    theta1 = np.sqrt(np.random.rand(n)) * 4 * np.pi
    r1 = 2 * theta1 + np.pi
    x1 = np.stack([r1 * np.cos(theta1), r1 * np.sin(theta1)], axis=1)

    # Generate second spiral
    theta2 = np.sqrt(np.random.rand(n)) * 4 * np.pi
    r2 = -2 * theta2 - np.pi
    x2 = np.stack([r2 * np.cos(theta2), r2 * np.sin(theta2)], axis=1)

    # Combine spirals and add noise
    X = np.vstack([x1, x2])
    X += np.random.randn(n_samples, 2) * noise

    # Scale X to fit within the specified x and y ranges
    X[:, 0] = np.interp(X[:, 0], (X[:, 0].min(), X[:, 0].max()), x_range)
    X[:, 1] = np.interp(X[:, 1], (X[:, 1].min(), X[:, 1].max()), y_range)

    # Create labels
    y_range = np.zeros(n_samples)
    y_range[:n] = 0  # First spiral
    y_range[n:] = 1  # Second spiral

    return X, y_range


class DecisionBoundaryAnimation(Scene):
    def construct(self):
        n_epoch = 500
        # Generate synthetic classification data
        n_samples = 500

        x, y_target = make_spiral_dataset(n_samples=n_samples, noise=1.5, seed=1)
        y_target = y_target.reshape(-1, 1)

        # Model architecture
        model = Sequential([
            Linear(x.shape[1], 128, init_method="he_leaky"),
            LeakyReLU(alpha=0.01),
            Linear(128, 64, init_method="he_leaky"),
            LeakyReLU(alpha=0.01),
            Linear(64, 1, init_method="xavier"),
            Sigmoid()
        ])

        bce = BCELoss()
        optimizer = SGD(lr=0.001, momentum=0.9)

        # Configuration
        self.axes_config = {
            "x_range": [-1, 1, 0.2],
            "y_range": [-1, 1, 0.2],
            "x_length": 7,
            "y_length": 7,
            "axis_config": {"include_tip": False},
        }

        # Create coordinate system
        axes = Axes(**self.axes_config)
        self.play(Create(axes))

        # Helper function to convert numpy coordinates to manim points
        def to_manim_point(point):
            return axes.c2p(point[0], point[1])

        # Create scatter plot dots for the dataset
        dots = VGroup()
        colors = []
        for i in range(len(x)):
            point = x[i]
            color = BLUE if y_target[i] == 0 else RED
            dot = Dot(to_manim_point(point), color=color, radius=0.05)
            dots.add(dot)
            colors.append(color)

        self.play(Create(dots))

        # Create grid for decision boundary
        x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
        y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
        grid_size = 100
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_size),
                            np.linspace(y_min, y_max, grid_size))
        grid = np.c_[xx.ravel(), yy.ravel()]

        def create_boundary_mesh():
            y_pred = model.forward(grid)
            Z = (y_pred > 0.5).astype(int)
            Z = Z.reshape(xx.shape)

            # Create mesh of rectangles for visualization
            mesh = VGroup()
            dx = (x_max - x_min) / (grid_size - 1)
            dy = (y_max - y_min) / (grid_size - 1)

            for i in range(grid_size - 1):
                for j in range(grid_size - 1):
                    if Z[i, j] == 1:
                        rect = Rectangle(
                            width=dx * 3,
                            height=dy * 3,
                            fill_color=RED,
                            fill_opacity=0.2,
                            stroke_width=0
                        )
                        rect.move_to(axes.c2p(
                            x_min + j * dx + dx/2,
                            y_min + i * dy + dy/2
                        ))
                        mesh.add(rect)
            return mesh


        # Animation loop
        n_frames = 50
        epochs_per_frame = n_epoch // n_frames

        # Initial boundary
        old_mesh = create_boundary_mesh()
        self.play(FadeIn(old_mesh))

        # Title with epoch counter
        title = Text("Epoch: 0").to_corner(DR)
        self.play(Write(title))

        for frame in range(n_frames):
            # Train for several epochs
            for _ in range(epochs_per_frame):
                y_pred = model(x)
                _ = bce(y_pred, y_target)
                model.zero_grad()
                grad = bce.backward(y_pred, y_target)
                model.backward(grad)
                optimizer.step(model)

            new_mesh = create_boundary_mesh()
            new_title = Text(f"Epoch: {(frame + 1) * epochs_per_frame}").to_corner(DR)

            self.play(
                FadeTransform(old_mesh, new_mesh),
                Transform(title, new_title),
                run_time=0.5
            )

            old_mesh = new_mesh

        # Final pause
        self.wait(2)

