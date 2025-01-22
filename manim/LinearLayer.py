from manim import *


class ForwardModeAnimation(Scene):
    def construct(self):
        headers_size = 50
        math_size = 50
        math_legend = 35

        # Scene 1: Linear Layer Definition
        title1 = Text("Forward Mode for Linear Layer", font_size=headers_size)
        title1.to_edge(UP)
        
        eq1 = MathTex(
            r"A_i(\mathbf{x}) = \mathbf{x}\mathbf{w}_i + b_i",
            font_size=math_size
        ).next_to(title1, DOWN, buff=1.5)
        
        legend1 = VGroup(
            Tex(r"$\mathbf{x}$ : input to layer $i$", font_size=math_legend),
            Tex(r"$\mathbf{w}_i$ : weights", font_size=math_legend),
            Tex(r"$b_i$ : biases", font_size=math_legend)
        ).arrange(DOWN, aligned_edge=LEFT)
        legend1.next_to(eq1, DOWN, buff=1)
        
        self.play(Write(title1))
        self.play(Write(eq1))
        self.play(Write(legend1))

        self.wait(5)

        self.play(
            *[FadeOut(mob) for mob in [title1, eq1, legend1]]
        )
        
        # Scene 2: Single Network Step
        title2 = Text("Single Network Step", font_size=headers_size)
        title2.to_edge(UP)
        
        eq2 = MathTex(
            r"f_i(\mathbf{x}) = \sigma(A_i(\mathbf{x}))",
            font_size=math_size
        ).next_to(title2, DOWN, buff=1.5)
        
        legend2 = VGroup(
            Tex(r"$f_i$ : output of layer $i$", font_size=math_legend),
            Tex(r"$\sigma$ : activation function", font_size=math_legend),
            Tex(r"$A_i$ : linear transformation", font_size=math_legend)
        ).arrange(DOWN, aligned_edge=LEFT)
        legend2.next_to(eq2, DOWN, buff=1)
        
        self.play(Write(title2))
        self.play(Write(eq2))
        self.play(Write(legend2))

        self.wait(5)

        self.play(
            *[FadeOut(mob) for mob in [title2, eq2, legend2]]
        )
        
        # Scene 3: Full Network Transformation
        title3 = Text("Full Network Transformation", font_size=headers_size)
        title3.to_edge(UP)
        
        eq3 = MathTex(
            r"f(\mathbf{x}) = \sigma(A_L(\sigma(A_{L-1}( \dots \sigma(A_1(\mathbf{x})) \dots ))))",
            font_size=math_size
        ).next_to(title3, DOWN, buff=1.5)
        
        legend3 = VGroup(
            Tex(r"$f$ : complete network function", font_size=math_legend),
            Tex(r"$L$ : number of layers", font_size=math_legend)
        ).arrange(DOWN, aligned_edge=LEFT)
        legend3.next_to(eq3, DOWN, buff=1)
        
        self.play(Write(title3))
        self.play(Write(eq3))
        self.play(Write(legend3))

        self.wait(5)

        self.play(
            *[FadeOut(mob) for mob in [title3, eq3, legend3]]
        )
        
        # Scene 4: Functional Composition
        title4 = Text("Functional Composition View", font_size=headers_size)
        title4.to_edge(UP)
        
        eq4 = MathTex(
            r"f(\mathbf{x}) = A_L \circ \sigma \circ A_{L-1} \circ \dots \circ \sigma \circ A_1 (\mathbf{x})",
            font_size=math_size
        ).next_to(title4, DOWN, buff=1.5)
        
        legend4 = VGroup(
            Tex(r"$\circ$ : function composition", font_size=math_legend),
            Tex(r"Forward pass: left to right computation", font_size=math_legend),
            Tex(r"Intermediate values stored for backward pass", font_size=math_legend)
        ).arrange(DOWN, aligned_edge=LEFT)
        legend4.next_to(eq4, DOWN, buff=1)
        
        self.play(Write(title4))
        self.play(Write(eq4))
        self.play(Write(legend4))

        self.wait(5)


class BackpropagationExplanation(Scene):
    def construct(self):
        # Title
        title = Text("Backpropagation & The Chain Rule", font_size=40)
        self.play(Write(title))
        self.wait()
        self.play(title.animate.scale(0.6).to_edge(UP))

        # Goal explanation
        goal_text = Text("Goal: Compute gradient of loss function", font_size=32)
        goal_math = MathTex(r"\nabla \mathcal{L}")
        goal_group = VGroup(goal_text, goal_math).arrange(RIGHT, buff=0.5)
        
        self.play(Write(goal_group))
        self.wait()
        self.play(goal_group.animate.scale(0.8).next_to(title, DOWN))

        # Chain Rule Basic Example
        chain_rule_title = Text("Chain Rule", font_size=36)
        chain_rule_basic = MathTex(
            r"\frac{d}{dx} f(g(x)) = f'(g(x)) \cdot g'(x)"
        )
        chain_group = VGroup(chain_rule_title, chain_rule_basic).arrange(DOWN)
        
        self.play(Write(chain_group))
        self.wait(5)

        # Neural Network Structure
        nn_layers = VGroup()
        for i in range(4):
            layer = VGroup()
            num_neurons = 3 if i in [1, 2] else 2
            for j in range(num_neurons):
                neuron = Circle(radius=0.2, color=BLUE)
                neuron.move_to([2*i - 3, j - (num_neurons-1)/2, 0])
                layer.add(neuron)
            nn_layers.add(layer)

        # Add connections between layers
        connections = VGroup()
        for i in range(len(nn_layers)-1):
            layer = nn_layers[i]
            next_layer = nn_layers[i+1]
            for n1 in layer:
                for n2 in next_layer:
                    line = Line(n1.get_center(), n2.get_center(), stroke_opacity=0.5)
                    connections.add(line)

        # Neural Network Labels
        input_label = Text("x", font_size=24).next_to(nn_layers[0], DOWN)
        hidden_labels = VGroup(
            Text("A₁", font_size=24).next_to(nn_layers[1], DOWN),
            Text("A₂", font_size=24).next_to(nn_layers[2], DOWN)
        )
        loss_label = MathTex(r"\mathcal{L}").next_to(nn_layers[3], RIGHT)

        # Show network structure
        self.play(
            FadeOut(chain_group),
            Create(nn_layers),
            Create(connections),
            Write(input_label),
            Write(hidden_labels),
            Write(loss_label)
        )
        self.wait(5)

        # Backward propagation visualization
        gradient_color = RED
        backward_arrows = VGroup()
        for i in range(len(nn_layers)-1, 0, -1):
            layer = nn_layers[i]
            prev_layer = nn_layers[i-1]
            arrow = Arrow(
                layer.get_center(),
                prev_layer.get_center(),
                color=gradient_color,
                buff=0.3
            )
            backward_arrows.add(arrow)
            self.play(GrowArrow(arrow))
            self.wait(0.5)

        # Final formula
        final_formula = MathTex(
            r"\mathcal{L} = f(A_L(\sigma(A_{L-1}(\dots \sigma(A_1(x)) \dots))))"
        ).scale(0.8).to_edge(DOWN)
        
        self.play(Write(final_formula))
        self.wait(5)

        # Cleanup
        self.play(
            *[FadeOut(mob) for mob in self.mobjects]
        )

    def create_layered_text(self, text_list, direction=DOWN):
        result = VGroup()
        for text in text_list:
            result.add(Text(text, font_size=24))
        result.arrange(direction)
        return result


class BackwardEquations(Scene):
    def construct(self):
        # Scene 1: Forward Pass
        title1 = Text("Forward Pass", font_size=36)
        title1.to_edge(UP)
        
        forward_eq = MathTex(
            r"f(\mathbf{x}) = A_L \circ \sigma \circ A_{L-1} \circ \dots \circ \sigma \circ A_1 (\mathbf{x})",
            font_size=36
        )
        
        explanation1 = Text(
            "The forward pass computes these transformations sequentially,\n\n"
            "storing intermediate values for use during the backward pass.",
            font_size=24,
            t2c={"forward pass": BLUE, "backward pass": RED}
        )
        explanation1.next_to(forward_eq, DOWN, buff=1)

        self.play(Write(title1))
        self.play(Write(forward_eq))
        self.play(Write(explanation1))
        self.wait(5)
        self.play(
            *[FadeOut(mob) for mob in self.mobjects]
        )

        # Scene 2: Backward Flow Initial
        title2 = Text("Backward Chain", font_size=36)
        title2.to_edge(UP)
        
        backward_eq = MathTex(
            r"\nabla f = \nabla A_1 \circ \sigma' \circ \nabla A_2 \circ \dots \circ \sigma' \circ \nabla A_L (\mathcal{L})",
            font_size=36
        )
        
        self.play(Write(title2))
        self.play(Write(backward_eq))
        self.wait(5)
        self.play(
            *[FadeOut(mob) for mob in self.mobjects]
        )

        # Scene 3: Backward Flow Details
        title3 = Text("Gradient Flow Details", font_size=36)
        title3.to_edge(UP)
        
        chain_rule = MathTex(
            r"\frac{\partial \mathcal{L}}{\partial x_{\text{out}}} = "
            r"\frac{\partial \mathcal{L}}{\partial A_L} \cdot \sigma' \cdot "
            r"\frac{\partial A_{L-1}}{\partial x_{\text{out}}}",
            font_size=36
        )
        
        explanation3 = Text(
            "The chain rule propagates gradients backward,\n\n"
            "starting with the gradient of the loss",
            font_size=24,
            t2c={"chain rule": YELLOW, "gradients": RED}
        )
        explanation3.next_to(chain_rule, DOWN, buff=1)

        self.play(Write(title3))
        self.play(Write(chain_rule))
        self.play(Write(explanation3))
        self.wait(5)
        self.play(
            *[FadeOut(mob) for mob in self.mobjects]
        )

        # Scene 4: Final Gradient Vector
        title4 = Text("Gradients of the Layer", font_size=36)
        title4.to_edge(UP)
        
        gradient_vector = MathTex(
            r"\nabla A_i(\mathbf{x}) = \begin{bmatrix} "
            r"\frac{\partial A_i(\mathbf{x})}{\partial \mathbf{w_i}} \\ "
            r"\frac{\partial A_i(\mathbf{x})}{\partial \mathbf{b_i}} "
            r"\end{bmatrix} = "
            r"\begin{bmatrix} "
            r"\mathbf{x}^T \\ "
            r"1 "
            r"\end{bmatrix}",
            font_size=36
        )
        
        explanation4 = Text(
            "Gradient vector with respect to weights and biases",
            font_size=24,
            t2c={"weights": BLUE, "biases": GREEN}
        )
        explanation4.next_to(gradient_vector, DOWN, buff=1)

        self.play(Write(title4))
        self.play(Write(gradient_vector))
        self.play(Write(explanation4))
        self.wait(5)
        self.play(
            *[FadeOut(mob) for mob in self.mobjects]
        )


class BiasGradientAnimation(Scene):
    def construct(self):
        # Initial equation setup
        total_derivative = MathTex(
            r"\frac{\partial L}{\partial b_i} = \frac{\partial L}{\partial A_i} \cdot \frac{\partial A_i}{\partial b_i}"
        ).scale(1.2)
        
        # Second equation
        substituted = MathTex(
            r"\frac{\partial L}{\partial b_i} = \frac{\partial L}{\partial A_i} \cdot 1 = \frac{\partial L}{\partial A_i}"
        ).scale(1.2)
        
        # Position first equation
        total_derivative.shift(UP * 2)
        
        # Create batch visualization elements
        batch_rect = Rectangle(height=3, width=4)
        batch_grid = VGroup()
        for i in range(4):
            for j in range(3):
                cell = Rectangle(height=1, width=1)
                cell.move_to(batch_rect.get_center() + RIGHT * (j-1) + UP * (1-i))
                batch_grid.add(cell)
        
        # Labels
        samples_label = Text("Samples", font_size=24).next_to(batch_rect, LEFT)
        features_label = Text("Features", font_size=24).next_to(batch_rect, UP)
        
        # Code implementation
        code = Code(
            code="""db = np.sum(d_out, axis=0, keepdims=True)""",
            language="python",
            font="Monospace",
            font_size=24
        ).shift(DOWN * 2)
        
        # Animation sequence
        self.play(Write(total_derivative))
        self.wait()
        
        # Show substitution
        substituted.next_to(total_derivative, DOWN)
        self.play(Write(substituted))
        self.wait()
        
        # Fade out equations and show batch visualization
        self.play(
            FadeOut(total_derivative),
            FadeOut(substituted),
            FadeIn(batch_rect),
            FadeIn(batch_grid),
            FadeIn(samples_label),
            FadeIn(features_label)
        )
        self.wait(5)
        
        # Highlight summation along axis=0
        arrows = VGroup()
        for j in range(3):  # For each column
            arrow = Arrow(
                start=batch_grid[j*4].get_top(),
                end=batch_grid[j*4 + 3].get_bottom(),
                color=YELLOW
            )
            arrow.next_to(batch_grid[j*4:j*4+4], RIGHT, buff=0.2)
            arrows.add(arrow)
        
        self.play(Create(arrows))
        self.wait()
        
        # Show code implementation
        self.play(Write(code))
        self.wait()
        
        # Final result visualization
        result_rect = Rectangle(height=1, width=4, color=GREEN)
        result_rect.next_to(batch_rect, DOWN, buff=1)
        result_label = Text("Resulting Bias Gradient", font_size=24).next_to(result_rect, DOWN)
        
        self.play(
            Create(result_rect),
            Write(result_label)
        )
        self.wait(5)


