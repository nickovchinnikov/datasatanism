from manim import *


class LinearLayer(Scene):
    def construct(self):
        # Scene 1: Main equation
        equation = MathTex(
            r"\hat{x}_{i+1} = \sigma(\mathbf{w}_i^T \mathbf{x} + b_i)"
        ).scale(1.2)
        
        # Description text
        description = VGroup(
            Text("where:", font_size=24),
            MathTex(r"\mathbf{w}_i^T", r"\text{ are the layer's weights}", font_size=35),
            MathTex(r"b_i", r"\text{ represents the biases}", font_size=35),
            Text("The term", font_size=24),
            MathTex(r"\mathbf{w}_i^T \mathbf{x} + b_i", r"\text{ applies matrix multiplication and adds biases}", font_size=35)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        
        # Position equation and description
        equation.to_edge(UP, buff=1)
        description.next_to(equation, DOWN, buff=1)
        
        # Animate equation and description
        self.play(Write(equation), run_time=2)
        self.wait()
        
        for part in description:
            self.play(Write(part), run_time=1)
        self.wait(5)
        
        # Fade out description for matrix multiplication example
        self.play(
            FadeOut(description),
            # equation.animate.scale(0.8).to_corner(UL)
        )
        
        # Matrix multiplication example
        # Create matrices with brackets
        matrix_W = Matrix([
            ["w_{11}", "w_{12}", "w_{13}"],
            ["w_{21}", "w_{22}", "w_{23}"],
            ["w_{31}", "w_{32}", "w_{33}"]
        ], v_buff=0.8).scale(0.8)
        
        vector_x = Matrix([
            ["x_1"],
            ["x_2"],
            ["x_3"]
        ], v_buff=0.8).scale(0.8)
        
        bias_vector = Matrix([
            ["b_1"],
            ["b_2"],
            ["b_3"]
        ], v_buff=0.8).scale(0.8)
        
        # Color coding
        matrix_W.set_color(BLUE)
        vector_x.set_color(GREEN)
        bias_vector.set_color(RED)

        # Add multiplication and addition symbols
        times = MathTex(r"\cdot") #.next_to(matrix_W, RIGHT, buff=0.2)
        plus = MathTex(r"+") #.next_to(vector_x, RIGHT, buff=0.2)
        
        # Result arrow and matrix
        arrow = MathTex(r"\Rightarrow") # .next_to(bias_vector, RIGHT, buff=0.5)
        
        result_matrix = Matrix([
            ["w_{11}x_1 + w_{12}x_2 + w_{13}x_3 + b_1"],
            ["w_{21}x_1 + w_{22}x_2 + w_{23}x_3 + b_2"],
            ["w_{31}x_1 + w_{32}x_2 + w_{33}x_3 + b_3"]
        ], v_buff=0.8).scale(0.8)
        
        # result_matrix.next_to(arrow, RIGHT, buff=0.5)
        
        # Show result
        # self.play(Write(arrow), Write(result_matrix))

        # Position matrices
        matrix_group = VGroup(
            matrix_W, 
            times, 
            vector_x, 
            plus, 
            bias_vector, 
            arrow, 
            result_matrix
        ).arrange(RIGHT) # , buff=0.5)
        matrix_group.next_to(equation, DOWN, buff=2)
        
        # Show matrix multiplication
        self.play(
            Write(matrix_W),
            Write(times),
            Write(vector_x),
            Write(plus),
            Write(bias_vector),
            Write(arrow), 
            Write(result_matrix),
            run_time=2
        )
        
        self.wait(5)
        
        # Add final note about activation
        activation_note = Text("Finally, apply activation function σ to each element", 
                             font_size=24).next_to(matrix_group, DOWN, buff=1)
        self.play(Write(activation_note))
        self.wait(5)
