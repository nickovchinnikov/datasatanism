import numpy as np

from manim import *


class ActivationFunctions(Scene):
    def create_coordinate_system(self):
        axes = Axes(
            x_range=[-5, 5, 1],
            y_range=[-2, 2, 1],
            tips=False,
            axis_config={"include_numbers": True}
        ).scale(0.7)
        return axes

    def show_transformation_scene(self):
        # Create input points and grid
        dots = VGroup(*[
            Dot(point=[x/2, x/2, 0], radius=0.05, color=BLUE) 
            for x in range(-8, 9)
        ])
        
        # Create axes
        axes = self.create_coordinate_system()
        
        # Add title
        title = Text("Non-linear Transformation", font_size=36)
        title.to_edge(UP)
        
        # Show initial points
        self.play(
            Write(title),
            Create(axes),
            Create(dots)
        )
        self.wait()
        
        # Apply tanh transformation
        def tanh(x): return np.tanh(x)
        
        transformed_dots = VGroup(*[
            Dot(point=[d.get_center()[0], tanh(d.get_center()[0]), 0], radius=0.05, color=RED)
            for d in dots
        ])
        
        # Draw sigmoid function
        sigmoid_curve = axes.plot(
            tanh,
            color=YELLOW
        )
        
        # Show transformation
        self.play(
            Transform(dots, transformed_dots),
            Create(sigmoid_curve)
        )
        self.wait(2)
        
        self.play(
            FadeOut(dots),
            FadeOut(sigmoid_curve),
            FadeOut(axes),
            FadeOut(title)
        )

    def show_activation_functions(self):
        # Create axes
        axes = self.create_coordinate_system()
        
        # Define activation functions
        def sigmoid(x): return 1 / (1 + np.exp(-x))
        def tanh(x): return np.tanh(x)
        def relu(x): return max(0, x)
        def leaky_relu(x): return x if x > 0 else 0.1 * x
        
        # Create function graphs
        sigmoid_graph = axes.plot(sigmoid, color=RED)
        tanh_graph = axes.plot(tanh, color=BLUE)
        relu_graph = axes.plot(
            relu,
            x_range=[-5, 2],
            color=GREEN
        )
        leaky_relu_graph = axes.plot(
            leaky_relu,
            x_range=[-5, 2],
            color=YELLOW
        )
        
        # Create labels
        labels = VGroup(
            Text("Sigmoid: Outputs between 0 and 1", color=RED),
            Text("Tanh: Outputs between -1 and 1", color=BLUE),
            Text("ReLU: No upper limit, removes negatives", color=GREEN),
            Text("Leaky ReLU: Allows small negative values", color=YELLOW)
        ).arrange(DOWN, aligned_edge=LEFT).scale(0.4)
        labels.to_corner(DOWN + LEFT)
        
        # Show functions one by one
        title = Text("Activation Functions", font_size=36)
        title.to_edge(UP)
        
        self.play(Write(title))
        self.play(Create(axes))
        
        for graph, label in zip(
            [sigmoid_graph, tanh_graph, relu_graph, leaky_relu_graph],
            labels
        ):
            self.play(
                Create(graph),
                Write(label)
            )
            self.wait(2)

    def construct(self):
        # Scene 1: Non-linear transformation
        self.show_transformation_scene()
        
        # Scene 2: Different activation functions
        self.show_activation_functions()
        self.wait(2)


class NormalDistribution(Scene):
    def create_histogram(self):
        # Generate normal distribution data
        np.random.seed(42)
        data = np.random.normal(0, 1, 1000)
        
        # Create histogram
        hist, bins = np.histogram(data, bins=30, range=[-3, 3])
        
        # Create bar chart
        chart = BarChart(
            values=hist,
            bar_names=[
                f"{bins[i]:.1f}" if any(np.isclose(float(bins[i]), [-3, -2, -1, 0, 1, 2, 3], atol=0.05)) else ""
                for i in range(len(bins)-1)
            ],
            y_range=[0, max(hist), int(max(hist)/4)],
            x_length=6,
            y_length=4,
            bar_width=0.15,
            bar_fill_opacity=0.7,
            bar_colors=[BLUE],
        ).scale(0.8)

        # Add title
        title = Text("Gaussian (Normal) Distribution", font_size=36)
        title.to_edge(UP)
        
        # Add normal curve
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[0, max(hist), int(max(hist)/4)],
            tips=False
        ).scale(0.8)
        
        curve = axes.plot(
            lambda x: max(hist) * np.exp(-x**2/2) / np.sqrt(2*np.pi),
            x_range=[-2, 2, 0.1],
            color=RED
        )
        
        VGroup(chart, curve).move_to(ORIGIN)
        
        return VGroup(title, chart, curve)

    def second_scene(self):
        title = Text("Weight Initialization Formula", font_size=36)
        title.to_edge(UP)
        
        formula = MathTex(
            "W", r"\sim", r"\mathcal{N}(0, \sigma^2)"
        )
        
        explanation = Text(
            "Weights drawn from normal distribution\nwith mean 0 and variance σ²",
            font_size=30
        ).next_to(formula, DOWN, buff=1.0)
        
        return VGroup(title, formula, explanation)

    def third_scene(self):
        title = Text("Probability Density Function", font_size=36)
        title.to_edge(UP)
        
        # General form
        general_header = Text("General Normal Distribution:", font_size=24)
        general_formula = MathTex(
            r"\mathcal{N}(x \mid \mu, \sigma^2) = \frac{1}{\sqrt{2 \pi \sigma^2}} e^{-\frac{(x - \mu)^2}{2 \sigma^2}}"
        )
        
        # Standard form
        standard_header = Text("Standard Normal Distribution (μ=0, σ²=1):", font_size=24)
        standard_formula = MathTex(
            r"\mathcal{N}(x \mid 0, 1) = \frac{1}{\sqrt{2 \pi}} e^{-\frac{x^2}{2}}"
        )
        
        # Arrange elements
        VGroup(general_header, general_formula).arrange(DOWN, buff=0.3)
        VGroup(standard_header, standard_formula).arrange(DOWN, buff=0.3)
        
        formulas = VGroup(
            VGroup(general_header, general_formula),
            VGroup(standard_header, standard_formula)
        ).arrange(DOWN, buff=1)
        
        return VGroup(title, formulas)

    def construct(self):
        # Scene 1: Normal Distribution Histogram
        scene1 = self.create_histogram()
        self.play(Write(scene1[0]))  # Title
        self.play(Create(scene1[1]))  # Histogram
        self.play(Create(scene1[2]))  # Curve
        self.wait(5)
        self.play(FadeOut(scene1))
        
        # Scene 2: Weight Initialization Formula
        scene2 = self.second_scene()
        self.play(Write(scene2[0]))  # Title
        self.play(Write(scene2[1]))  # Formula
        self.play(Write(scene2[2]))  # Explanation
        self.wait(5)
        self.play(FadeOut(scene2))
        
        # Scene 3: PDF Formulas
        scene3 = self.third_scene()
        self.play(Write(scene3[0]))  # Title
        
        # Reveal formulas one by one
        formulas = scene3[1]
        for formula_group in formulas:
            self.play(
                Write(formula_group[0]),  # Header
                Write(formula_group[1]),  # Formula
                run_time=2
            )
            self.wait(1)
        
        self.wait(5)


class VanishingGradient(Scene):
    def construct(self):
        # Add title and explanation
        title = Text("Vanishing Gradient Problem", font_size=36)
        title.to_edge(UP)
        explanation = Text(
            "Gradients diminish as they propagate backwards",
            font_size=24
        )
        explanation.next_to(title, DOWN)
        
        self.play(
            Write(title),
            Write(explanation)
        )
        self.wait(1)

        # Create a neural network structure
        layers = VGroup()
        num_layers = 6
        neurons_per_layer = 4
        layer_spacing = 1.5
        
        # Create layers of neurons
        for i in range(num_layers):
            layer = VGroup()
            for j in range(neurons_per_layer):
                neuron = Circle(radius=0.2, color=BLUE)
                neuron.move_to([i * layer_spacing - 3, (j - neurons_per_layer/2) * 0.8, 0])
                layer.add(neuron)
            layers.add(layer)
        
        # Add connection lines between layers
        connections = VGroup()
        for i in range(num_layers - 1):
            for neuron1 in layers[i]:
                for neuron2 in layers[i + 1]:
                    line = Line(
                        neuron1.get_center(), 
                        neuron2.get_center(),
                        stroke_opacity=0.3
                    )
                    connections.add(line)
        
        network = VGroup(layers, connections).move_to(ORIGIN)
        
        # Animation sequence
        self.play(Create(network))
        # self.play(Create(layers), Create(connections))
        self.wait(2)
        
        # Gradient flow animation
        gradient_values = [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125]
        
        # Create labels for gradient values
        labels = VGroup()
        for i, value in enumerate(gradient_values):
            label = Text(f"{value:.4f}", font_size=24)
            label.next_to(layers[i], UP)
            labels.add(label)
        
        self.play(Create(labels))
        
        # Animate gradient flow with color intensity
        for i in range(num_layers):
            layer_copy = layers[i].copy()
            layer_copy.set_color(RED)
            layer_copy.set_opacity(gradient_values[i])
            self.play(
                Transform(layers[i], layer_copy),
                run_time=0.5
            )
        
        self.wait(5)


class ExplodingGradient(Scene):
    def construct(self):
        # Add title and explanation
        title = Text("Exploding Gradient Problem", font_size=36)
        title.to_edge(UP)
        explanation = Text(
            "Gradients grow exponentially as they propagate backwards",
            font_size=24
        )
        explanation.next_to(title, DOWN)
       
        self.play(
            Write(title),
            Write(explanation)
        )
        self.wait(1)
        
        # Create a neural network structure
        layers = VGroup()
        num_layers = 6
        neurons_per_layer = 4
        layer_spacing = 1.5
       
        # Create layers of neurons
        for i in range(num_layers):
            layer = VGroup()
            for j in range(neurons_per_layer):
                neuron = Circle(radius=0.2, color=BLUE)
                neuron.move_to([i * layer_spacing - 3, (j - neurons_per_layer/2) * 0.8, 0])
                layer.add(neuron)
            layers.add(layer)
       
        # Add connection lines between layers
        connections = VGroup()
        for i in range(num_layers - 1):
            for neuron1 in layers[i]:
                for neuron2 in layers[i + 1]:
                    line = Line(
                        neuron1.get_center(),
                        neuron2.get_center(),
                        stroke_opacity=0.3
                    )
                    connections.add(line)
       
        network = VGroup(layers, connections).move_to(ORIGIN)
       
        # Animation sequence
        self.play(Create(network))
        self.wait(2)
       
        # Exploding gradient values (exponential growth)
        gradient_values = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]  # Powers of 2
       
        # Create labels for gradient values
        labels = VGroup()
        for i, value in enumerate(gradient_values):
            label = Text(f"{value:.1f}", font_size=24)
            label.next_to(layers[i], UP)
            labels.add(label)
       
        self.play(Create(labels))
       
        # Animate gradient flow with color intensity and size increase
        for i in range(num_layers):
            layer_copy = layers[i].copy()
            layer_copy.set_color(RED)
            
            # Make neurons grow in size based on gradient value
            scale_factor = min(1 + gradient_values[i]/64, 3)  # Cap the maximum size
            layer_copy.scale(scale_factor)
            
            # Increase opacity with gradient value
            opacity = min(gradient_values[i]/32 + 0.3, 1.0)  # Ensure opacity doesn't exceed 1
            layer_copy.set_opacity(opacity)
            
            # Make connections thicker
            if i < num_layers - 1:
                for conn in connections[i * neurons_per_layer * neurons_per_layer : 
                                     (i + 1) * neurons_per_layer * neurons_per_layer]:
                    conn.set_stroke(width=gradient_values[i]/8)
            
            self.play(
                Transform(layers[i], layer_copy),
                run_time=0.5
            )
            
            # Add warning symbols when gradients get very large
            if gradient_values[i] > 8:
                warning = Text("!", color=YELLOW).next_to(layers[i], DOWN)
                self.play(FadeIn(warning))
       
        # Add final warning about instability
        warning_text = Text(
            "Large gradients can cause training instability!",
            font_size=20,
            color=RED
        ).to_edge(DOWN)
        self.play(Write(warning_text))
       
        self.wait(5)


class XavierInitialization(Scene):
    def construct(self):
        # Scene 1: Formulas and Description
        
        # Title
        title = Text("Xavier (Glorot) Initialization", font_size=36)
        title.to_edge(UP)
        
        # Description
        description = Text(
            "This is achieved by initializing weights from a uniform distribution between:",
            font_size=24
        )
        description.next_to(title, DOWN, buff=1.0)
        
        # Uniform distribution formula
        uniform_formula = MathTex(
            r"W \in (-\sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}}, \sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}})"
        )
        uniform_formula.next_to(description, DOWN, buff=1.0)
        
        # Variables explanation in MathTex
        vars_explanation = MathTex(
            r"\text{where } n_{\text{in}} \text{ and } n_{\text{out}} \text{ are the number of input and output neurons, respectively}"
        ).scale(0.8)
        vars_explanation.next_to(uniform_formula, DOWN, buff=1.0)

        # Animate first scene
        self.play(Write(title))
        self.play(Write(description))
        self.play(Write(uniform_formula))
        self.play(Write(vars_explanation))

        self.wait(5)

        self.play(FadeOut(description, uniform_formula, vars_explanation))
        
        # Normal distribution formula
        normal_title = Text("Normal Distribution Form:", font_size=24)
        normal_title.next_to(title, DOWN, buff=1.0)

        normal_formula = MathTex(
            r"W \sim \mathcal{N}(0, \frac{1}{n_{\text{in}}})"
        )
        normal_formula.next_to(normal_title, DOWN, buff=1.0)
        
        # Normal distribution explanation in MathTex
        normal_explanation = MathTex(
            r"\text{where } \mathcal{N}(0, \frac{1}{n_{\text{in}}}) \text{ is the standard normal distribution}",
            r"\text{with mean } 0 \text{ and variance } \frac{1}{n_{\text{in}}}"
        ).scale(0.8)
        normal_explanation.arrange(DOWN, buff=0.3)
        normal_explanation.next_to(normal_formula, DOWN, buff=1.0)
        
        self.play(Write(normal_title))
        self.play(Write(normal_formula))
        self.play(Write(normal_explanation))

        self.wait(5)
        
        self.play(FadeOut(normal_title, normal_formula, normal_explanation))
        
        # Scene 2: Distribution and Activation Functions
        
        # Generate Xavier distributed data
        n_in = 2  # Example input size
        data = np.random.normal(0, 1, 1000) * np.sqrt(1. / n_in)
        
        # Create histogram
        hist, bins = np.histogram(data, bins=30, range=[-3, 3])
        
        # Create bar chart
        chart = BarChart(
            values=hist,
            bar_names=[
                f"{bins[i]:.1f}" if any(np.isclose(float(bins[i]), [-3, -2, -1, 0, 1, 2, 3], atol=0.05)) else ""
                for i in range(len(bins)-1)
            ],
            y_range=[0, max(hist), int(max(hist)/4)],
            x_length=6,
            y_length=4,
            bar_width=0.15,
            bar_fill_opacity=0.7,
            bar_colors=[BLUE],
        ).scale(0.8)
        
        chart.shift(LEFT * 3.5)
        
        # Create grid for activation functions
        right_grid = NumberPlane(
            x_range=[-4, 4, 1],
            y_range=[-1.5, 1.5, 0.5],
            x_length=6,
            y_length=3,
        ).shift(RIGHT * 3.5)
        
        # Create activation functions
        sigmoid = VMobject()
        tanh = VMobject()
        
        x_vals_act = np.linspace(-4, 4, 1000)
        sigmoid_vals = 1 / (1 + np.exp(-x_vals_act))
        tanh_vals = np.tanh(x_vals_act)
        
        sigmoid_points = [right_grid.c2p(x, y) for x, y in zip(x_vals_act, sigmoid_vals)]
        tanh_points = [right_grid.c2p(x, y) for x, y in zip(x_vals_act, tanh_vals)]
        
        sigmoid.set_points_smoothly(sigmoid_points)
        tanh.set_points_smoothly(tanh_points)
        
        sigmoid.set_color(RED)
        tanh.set_color(GREEN)
        
        # Labels
        dist_title = Text("Xavier Distribution", font_size=24).next_to(chart, UP)
        act_title = Text("Activation Functions", font_size=24).next_to(right_grid, UP)
        
        sigmoid_label = Text("Sigmoid", font_size=20, color=RED) # .next_to(right_grid, DOWN)
        tanh_label = Text("Tanh", font_size=20, color=GREEN) # .next_to(sigmoid_label, RIGHT)

        labels = VGroup(
            sigmoid_label,
            tanh_label
        ).arrange(RIGHT, buff=0.5).next_to(right_grid, DOWN)
        
        # Animation sequence
        self.play(
            Create(chart),
            Create(right_grid)
        )
        self.play(
            Write(dist_title),
            Write(act_title)
        )
        self.play(
            Create(sigmoid),
            Create(tanh)
        )

        self.play(Write(labels))

        self.wait(5)


class ReLUEffect(Scene):
    def construct(self):
        # Title
        title = Text("ReLU Effect on Distribution", font_size=36)
        title.to_edge(UP)

        # ReLU formula
        relu_formula = MathTex(
            r"\text{ReLU}(x) = \max(0, x)"
        ).scale(0.8)
        relu_formula.next_to(title, DOWN, buff=0.5)
        
        # Create coordinate system
        axes = Axes(
            x_range=[-4, 4, 1],
            y_range=[-2, 2, 1],
            x_length=8,
            y_length=4,
            axis_config={"include_tip": True},
        ).next_to(relu_formula, DOWN, buff=1.0).scale(0.8)
        
        # Add labels
        x_label = axes.get_x_axis_label("x")
        y_label = axes.get_y_axis_label("y")

        def relu(x): return max(0, x)
        relu_graph = axes.plot(
            relu,
            x_range=[-5, 2],
            color=GREEN
        )
        
        # Create dots to demonstrate transformation
        n_points = 15
        random_x = np.linspace(-4, 2, n_points)
        original_dots = VGroup(*[
            Dot(axes.c2p(x, x), color=BLUE)
            for x in random_x
        ])
        
        transformed_dots = VGroup(*[
            Dot(axes.c2p(x, relu(x)), color=GREEN)
            for x in random_x
        ])
        
        # Create dashed lines to show transformation
        dashed_lines = VGroup(*[
            DashedLine(
                start=axes.c2p(x, x),
                end=axes.c2p(x, max(0, x)),
                color=YELLOW,
                dash_length=0.1
            )
            for x in random_x
        ])
        
        # Create y=x line
        identity_line = axes.plot(
            lambda x: x,
            x_range=[-4, 4],
            color=BLUE_E,
            stroke_width=1
        )
        
        # Create negative region indicator
        negative_region = Rectangle(
            width=axes.x_length/2,
            height=axes.y_length,
            color=RED_E,
            fill_opacity=0.2
        )
        negative_region.move_to(axes.c2p(-2, 0))
        
        negative_text = Text("Negative values set to zero", font_size=16, color=RED)
        negative_text.next_to(negative_region, UP)
        
        # Animation sequence
        self.play(
            Write(title),
            Create(axes),
            Write(x_label),
            Write(y_label)
        )
        
        self.play(Write(relu_formula))
        
        self.play(Create(identity_line))
        self.add(negative_region, negative_text)
        
        # Show original points
        self.play(Create(original_dots))
        
        # Create ReLU function
        self.play(Create(relu_graph))
        
        # Transform points
        self.play(
            Create(dashed_lines),
            Transform(original_dots, transformed_dots)
        )
        
        self.wait(5)


class LeakyReLUEffect(Scene):
    def construct(self):
        # Title
        title = Text("LeakyReLU - No Zero Out! α = 0.3 for demonstration.", font_size=36)
        title.to_edge(UP)
        
        # LeakyReLU formula
        relu_formula = MathTex(
            r"\text{LeakyReLU}(x) = \begin{cases} x & \text{if } x > 0 \\ 0.3x & \text{if } x \leq 0 \end{cases}"
        ).scale(0.8)
        relu_formula.next_to(title, DOWN, buff=0.5)
       
        # Create coordinate system
        axes = Axes(
            x_range=[-4, 4, 1],
            y_range=[-2, 2, 1],
            x_length=8,
            y_length=4,
            axis_config={"include_tip": True},
        ).next_to(relu_formula, DOWN, buff=1.0).scale(0.8)
       
        # Add labels
        x_label = axes.get_x_axis_label("x")
        y_label = axes.get_y_axis_label("y")
        
        # Define LeakyReLU function
        def leaky_relu(x): 
            return x if x > 0 else 0.3 * x
        
        leaky_relu_graph = axes.plot(
            leaky_relu,
            x_range=[-4, 2],
            color=GREEN
        )
       
        # Create dots to demonstrate transformation
        n_points = 15
        random_x = np.linspace(-4, 2, n_points)
        original_dots = VGroup(*[
            Dot(axes.c2p(x, x), color=BLUE)
            for x in random_x
        ])
       
        transformed_dots = VGroup(*[
            Dot(axes.c2p(x, leaky_relu(x)), color=GREEN)
            for x in random_x
        ])
       
        # Create dashed lines to show transformation
        dashed_lines = VGroup(*[
            DashedLine(
                start=axes.c2p(x, x),
                end=axes.c2p(x, leaky_relu(x)),
                color=YELLOW,
                dash_length=0.1
            )
            for x in random_x
        ])
       
        # Create y=x line
        identity_line = axes.plot(
            lambda x: x,
            x_range=[-4, 2],
            color=BLUE_E,
            stroke_width=1
        )
       
        # Create negative region indicator with different text
        negative_region = Rectangle(
            width=axes.x_length/2,
            height=axes.y_length,
            color=YELLOW_E,
            fill_opacity=0.2
        )
        negative_region.move_to(axes.c2p(-2, 0))
       
        negative_text = Text("Negative values scaled by 0.3\nMaintains gradient!", 
                           font_size=16, color=YELLOW_E)
        negative_text.next_to(negative_region, UP)
       
        # Animation sequence
        self.play(
            Write(title),
            Create(axes),
            Write(x_label),
            Write(y_label)
        )
       
        self.play(Write(relu_formula))
       
        self.play(Create(identity_line))
        self.add(negative_region, negative_text)
       
        # Show original points
        self.play(Create(original_dots))
       
        # Create LeakyReLU function
        self.play(Create(leaky_relu_graph))
       
        # Transform points
        self.play(
            Create(dashed_lines),
            Transform(original_dots, transformed_dots)
        )
       
        self.wait(5)


class WeightInitialization(Scene):
    def create_histogram(self, weights, color, title):
        # Create histogram using numpy
        hist, bins = np.histogram(weights.flatten(), bins=50, range=[-3, 3.1])
        
        # Create bar chart
        chart = BarChart(
            values=hist,
            bar_names=[
                f"{bins[i]:.1f}" if any(np.isclose(float(bins[i]), [-3, -2, -1, 0, 1, 2, 3], atol=0.05)) else ""
                for i in range(len(bins)-1)
            ],
            y_range=[0, max(hist), int(max(hist)/4)],
            x_length=6,
            y_length=4,
            bar_width=0.1,
            bar_fill_opacity=0.7,
            bar_colors=[color],
        ).scale(0.8)
        
        # Add title
        title = Text(title, font_size=24).next_to(chart, UP)
        
        return VGroup(chart, title)


    def create_activation_plot(self, activation_func, color, title):
        # Create axes
        axes = Axes(
            x_range=[-1, 1, 0.25],
            y_range=[-1, 1, 0.25],
            tips=False,
            axis_config={"include_numbers": True}
        ).scale(0.5)
        
        # Plot activation function
        graph = axes.plot(activation_func, color=color)
        
        # Add title
        title = Text(title, font_size=24).next_to(axes, UP)
        return VGroup(axes, graph, title)


    def construct(self):
        # Setup parameters
        input_size, output_size = 2, 2000
        alpha = 0.3

        # Random Normal Initialization
        weights_random = np.random.randn(input_size, output_size)
        def relu(x): return max(0, x)
        
        # Create and animate random normal initialization
        title = Text("Random Normal Initialization", font_size=36)
        self.play(Write(title))
        self.wait()
        self.play(title.animate.to_edge(UP))
        
        hist = self.create_histogram(weights_random, RED, "Weight Distribution")
        hist.shift(LEFT * 3)
        activation = self.create_activation_plot(
            lambda x: np.maximum(0, x),
            RED,
            "ReLU Activation"
        )
        activation.shift(RIGHT * 3)
        
        self.play(Create(hist))
        self.play(Create(activation))
        self.wait(5)
        self.play(FadeOut(VGroup(hist, activation, title)))

        # Xavier Initialization
        weights_xavier = np.random.randn(input_size, output_size) * np.sqrt(1. / input_size)
        def tanh(x): return np.tanh(x)
        
        title = Text("Xavier Initialization", font_size=36)
        self.play(Write(title))
        self.play(title.animate.to_edge(UP))
        
        hist = self.create_histogram(weights_xavier, GREEN, "Weight Distribution")
        hist.shift(LEFT * 3)
        activation = self.create_activation_plot(
            tanh,
            GREEN,
            "Tanh Activation"
        )
        activation.shift(RIGHT * 3)
        
        self.play(Create(hist))
        self.play(Create(activation))
        self.wait(5)
        self.play(FadeOut(VGroup(hist, activation, title)))

        # He Initialization
        weights_he = np.random.randn(input_size, output_size) * np.sqrt(2. / input_size)
        
        title = Text("He Initialization", font_size=36)
        self.play(Write(title))
        self.play(title.animate.to_edge(UP))
        
        hist = self.create_histogram(weights_he, BLUE, "Weight Distribution")
        hist.shift(LEFT * 3)
        activation = self.create_activation_plot(
            lambda x: np.maximum(0, x),
            BLUE,
            "ReLU Activation"
        )
        activation.shift(RIGHT * 3)
        
        self.play(Create(hist))
        self.play(Create(activation))
        self.wait(5)
        self.play(FadeOut(VGroup(hist, activation, title)))

        # Leaky He Initialization
        weights_leaky_he = np.random.randn(input_size, output_size) * np.sqrt(2. / ((1 + alpha**2) * input_size))
        
        title = Text("Leaky He Initialization", font_size=36)
        self.play(Write(title))
        self.play(title.animate.to_edge(UP))
        
        hist = self.create_histogram(weights_leaky_he, GRAY, "Weight Distribution")
        hist.shift(LEFT * 3)
        activation = self.create_activation_plot(
            lambda x: np.maximum(alpha * x, x),
            GRAY,
            "Leaky ReLU Activation"
        )
        activation.shift(RIGHT * 3)
        
        self.play(Create(hist))
        self.play(Create(activation))
        self.wait(5)
        self.play(FadeOut(VGroup(hist, activation, title)))
        self.wait(2)
