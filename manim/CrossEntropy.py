from manim import *


class BCEPresentation(Scene):
    def construct(self):
        # Slide 1: What is Binary?
        self.binary_definition_slide()
        self.wait(5)
        self.clear()
        
        # Slide 2: Vector Distribution
        self.vector_distribution_slide()
        self.wait(5)
        self.clear()
        
        # Slide 3: BCE Formula
        self.bce_formula_slide()
        self.wait(5)

    def binary_definition_slide(self):
        title = Text("What is Binary Classification?", font_size=40)
        title.to_edge(UP)
        
        binary_def = Text(
            "Binary: Having two possible values (0 or 1)",
            font_size=30
        ).next_to(title, DOWN, buff=1)
        
        # Example classes
        class_formula = MathTex(
            r"f(x) = \begin{cases} 1 & \text{if positive class} \\ 0 & \text{if negative class} \end{cases}"
        ).next_to(binary_def, DOWN, buff=1)
        
        # Examples with pictures
        example = Text("Examples:", font_size=25).next_to(class_formula, DOWN, buff=0.5)
        examples = VGroup(
            Text("• Spam (1) vs Not Spam (0)", font_size=25),
            Text("• Fraud (1) vs Normal (0)", font_size=25)
        ).arrange(DOWN, aligned_edge=LEFT).next_to(example, DOWN, buff=0.5)
        
        # Animations
        self.play(Write(title))
        self.play(FadeIn(binary_def))
        self.play(Write(class_formula))
        self.play(
            Write(example),
            Write(examples[0]),
            Write(examples[1])
        )

    def vector_distribution_slide(self):
        title = Text("Vector Distribution Comparison", font_size=40)
        title.to_edge(UP)
        
        # True labels
        true_label = MathTex(r"y = [0, 1, 1, 1, 0, 1, 1, 1]")
        true_label.next_to(title, DOWN, buff=1)
        
        # Predicted labels
        pred_label = MathTex(r"\hat{y} = [1, 1, 1, 0, 0, 1, 1, 0]")
        pred_label.next_to(true_label, DOWN, buff=1)
        
        # Creating numbers for visualization
        true_values = [0, 1, 1, 1, 0, 1, 1, 1]
        pred_values = [1, 1, 1, 0, 0, 1, 1, 0]
        
        true_nums = VGroup(*[
            Text(str(val), font_size=30) for val in true_values
        ])
        pred_nums = VGroup(*[
            Text(str(val), font_size=30) for val in pred_values
        ])
        
        # Arrange numbers horizontally
        true_nums.arrange(RIGHT, buff=0.7)
        pred_nums.arrange(RIGHT, buff=0.7)
        
        # Position number groups
        true_nums.next_to(pred_label, DOWN, buff=1)
        pred_nums.next_to(true_nums, DOWN, buff=1)
        
        # Color numbers based on values
        for i, (true_val, pred_val) in enumerate(zip(true_values, pred_values)):
            true_nums[i].set_color(BLUE if true_val else RED)
            pred_nums[i].set_color(BLUE if pred_val else RED)
        
        # Add labels for true and predicted values
        true_text = Text("True:", font_size=25).next_to(true_nums, LEFT, buff=0.5)
        pred_text = Text("Pred:", font_size=25).next_to(pred_nums, LEFT, buff=0.5)
        
        # Animations
        self.play(Write(title))
        self.play(Write(true_label))
        self.play(Write(pred_label))
        self.wait(5)
        self.play(
            Write(true_text),
            Write(pred_text),
            *[Write(num) for num in true_nums],
            *[Write(num) for num in pred_nums]
        )
        
        # Add difference highlights
        for i in range(8):
            if true_values[i] != pred_values[i]:
                surround = SurroundingRectangle(
                    VGroup(true_nums[i], pred_nums[i]),
                    color=YELLOW
                )
                self.play(Create(surround))
                self.wait(1)

    def bce_formula_slide(self):
        title = Text("Binary Cross-Entropy Formula", font_size=40)
        title.to_edge(UP)
        
        # Full BCE formula
        bce_formula = MathTex(
            r"\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} ",
            r"y_i \log(p_i)",
            r" + ",
            r"(1-y_i) \log(1-p_i)"
        )
        
        # Position the formula
        bce_formula.next_to(title, DOWN, buff=1)
        
        # Create explanations for each part
        explanations = VGroup(
            Text("Where:", font_size=25),
            MathTex(r"y_i \in \{0, 1\}", r"\text{ is the true label (ground truth)}"),
            MathTex(r"p_i \in [0, 1]", r"\text{ is the predicted probability}"),
            MathTex(r"N", r"\text{ is the total number of samples}")
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        explanations.next_to(bce_formula, DOWN, buff=1)
        
        # Animations
        self.play(Write(title))
        
        # Reveal formula parts sequentially
        self.play(Write(bce_formula[0]))
        self.play(Write(bce_formula[1]))
        self.play(Write(bce_formula[2]))
        self.play(Write(bce_formula[3]))
        
        self.wait(5)

        # Highlight different parts with explanations
        self.play(Write(explanations))
        
        # Highlight important parts of the formula
        highlight_boxes = [
            SurroundingRectangle(bce_formula[1], color=BLUE),
            SurroundingRectangle(bce_formula[3], color=RED)
        ]
        
        for box in highlight_boxes:
            self.play(Create(box))
            self.wait(1)
            self.play(FadeOut(box))
    