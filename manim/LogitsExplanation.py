import numpy as np

from manim import *


class LogitsExplanation(Scene):
    def construct(self):
        # Title
        title = Text("Understanding Logits", font_size=40)
        self.play(Write(title))
        self.play(title.animate.to_edge(UP))
        self.wait(2)

        # Add x-axis first since we'll reference it
        x_axis = Line(LEFT*4, RIGHT*4)
        x_axis.next_to(ORIGIN, DOWN, buff=1.5)
        zero_label = Text("0", font_size=24).next_to(x_axis, LEFT)

        # Create initial logits visualization
        logits_values = [1.35, -0.35, 1.0, 0.36, -0.64]
        logits_bars = VGroup()
        logits_labels = VGroup()
        
        # Create bars for logits
        for i, val in enumerate(logits_values):
            bar = Rectangle(
                height=abs(val)/2,
                width=1,
                fill_opacity=0.8,
                color=BLUE if val >= 0 else RED
            )
            # Position horizontally relative to x-axis
            bar.move_to(x_axis.get_center() + RIGHT*(1.5*i - 3))
            
            if val < 0:
                # For negative values, align top with x-axis
                bar.align_to(x_axis, UP)
            else:
                # For positive values, align bottom with x-axis
                bar.align_to(x_axis, DOWN)
            
            label = Text(f"{val:.1f}", font_size=24)
            label.next_to(bar, DOWN if val >= 0 else UP)
            
            class_label = Text(f"Class {i+1}", font_size=24)
            class_label.next_to(bar, UP, buff=1)
            
            logits_bars.add(bar)
            logits_labels.add(label, class_label)

        # Create logits explanation
        logits_exp = Text(
            "Logits: Raw unbounded model outputs",
            font_size=28
        ).to_edge(UP, buff=1.5)

        # Show initial state
        self.play(
            Create(x_axis),
            Write(zero_label),
            *[GrowFromCenter(bar) for bar in logits_bars],
            *[Write(label) for label in logits_labels]
        )
        self.play(Write(logits_exp))
        self.wait(5)

        # Transform to probabilities using softmax
        softmax_values = np.exp(logits_values) / np.sum(np.exp(logits_values))
        softmax_bars = VGroup()
        softmax_labels = VGroup()
        softmax_class_labels = VGroup()

        # Create probability bars
        for i, val in enumerate(softmax_values):
            bar = Rectangle(
                height=val*3,  # Scale for visibility
                width=1,
                fill_opacity=0.8,
                color=GREEN
            )
            # Position bars relative to x-axis
            bar.move_to(x_axis.get_center() + RIGHT*(1.5*i - 3))
            bar.align_to(x_axis, DOWN)
            
            label = Text(f"{val:.2f}", font_size=24)
            label.next_to(bar, UP)
            
            # Create new class labels for probability state
            class_label = Text(f"Class {i+1}", font_size=24)
            class_label.next_to(bar, UP, buff=1)
            
            softmax_bars.add(bar)
            softmax_labels.add(label)
            softmax_class_labels.add(class_label)

        # Add softmax explanation
        softmax_exp = Text(
            "Softmax: Normalized probabilities (0 to 1)",
            font_size=28
        ).to_edge(UP, buff=1.5)

        # Transform to probabilities
        self.play(
            Transform(logits_exp, softmax_exp),
            *[Transform(old_bar, new_bar) 
              for old_bar, new_bar in zip(logits_bars, softmax_bars)],
            *[Transform(old_label, new_label) 
              for old_label, new_label in zip(
                  [label for i, label in enumerate(logits_labels) if i % 2 == 0],
                  softmax_labels
              )],
            *[Transform(old_class_label, new_class_label)
              for old_class_label, new_class_label in zip(
                  [label for i, label in enumerate(logits_labels) if i % 2 == 1],
                  softmax_class_labels
              )]
        )
        self.wait(5)

        # Final scene cleanup
        self.play(
            *[FadeOut(mob) 
              for mob in self.mobjects]
        )
        self.wait()


class LogitsConcept(Scene):
    def construct(self):
        # Scene 1: Unbounded Values
        # Create number line
        number_line = NumberLine(
            x_range=[-5, 5, 1],
            length=10,
            include_numbers=True,
            include_tip=True,
        )
        number_line_label = Text("Logits Range", font_size=36)
        number_line_label.next_to(number_line, UP, buff=0.5)

        # Create example points on the number line
        points = [(-3.5, RED), (0, YELLOW), (2.8, GREEN)]
        dots = VGroup()
        labels = VGroup()
        
        for val, color in points:
            dot = Dot(number_line.number_to_point(val), color=color)
            label = Text(f"{val}", font_size=24, color=color)
            label.next_to(dot, UP, buff=0.3)
            dots.add(dot)
            labels.add(label)

        # Create explanation text
        unbounded_text = Text(
            "Logits are unbounded: can be negative, zero, or positive",
            font_size=28
        ).to_edge(UP, buff=1)

        # Animate first scene
        self.play(Create(number_line), Write(number_line_label))
        self.play(Write(unbounded_text))
        self.play(
            *[GrowFromCenter(dot) for dot in dots],
            *[Write(label) for label in labels]
        )
        self.wait(10)

        # Transition to Scene 2
        self.play(
            *[FadeOut(mob) for mob in [number_line, number_line_label, dots, labels, unbounded_text]]
        )

        # Scene 2: Class Assignment
        # Create input example
        input_text = Text("Input Image:", font_size=32).shift(LEFT * 4 + UP * 2)
        # input_box = Square(side_length=2, color=BLUE)
        # input_box.next_to(input_text, DOWN)
        input_box = ImageMobject("./assets/cat.jpg").set_width(2)
        input_box.next_to(input_text, DOWN)
        
        # Create class boxes
        classes = ["Cat", "Dog", "Bird"]
        logit_values = [2.5, 1.2, -0.8]
        class_boxes = VGroup()
        class_labels = VGroup()
        logit_labels = VGroup()
        arrows = VGroup()

        for i, (cls, val) in enumerate(zip(classes, logit_values)):
            # Create class box
            box = Rectangle(height=1, width=2)
            box.shift(RIGHT * 2 + UP * (1.5 - i * 1.5))
            
            # Create labels
            class_text = Text(cls, font_size=24)
            class_text.move_to(box)
            
            logit_text = Text(f"Logit: {val}", font_size=20, color=YELLOW)
            logit_text.next_to(box, RIGHT)
            
            # Create arrow
            arrow = Arrow(input_box.get_right(), box.get_left(), color=BLUE)
            
            class_boxes.add(box)
            class_labels.add(class_text)
            logit_labels.add(logit_text)
            arrows.add(arrow)

        # Create explanation text
        assignment_text = Text(
            "Logits represent unnormalized confidence scores",
            font_size=28
        ).to_edge(UP)

        # Animate second scene
        self.play(
            Write(input_text),
            FadeIn(input_box)
        )
        self.play(
            *[Create(box) for box in class_boxes],
            *[Write(label) for label in class_labels]
        )
        self.play(
            *[GrowArrow(arrow) for arrow in arrows],
            run_time=1.5
        )
        self.play(
            Write(assignment_text),
            *[Write(label) for label in logit_labels]
        )
        self.wait(5)

        # Highlight highest logit
        highlight_box = class_boxes[0].copy()
        highlight_box.set_color(GREEN)
        self.play(
            Create(highlight_box),
            logit_labels[0].animate.set_color(GREEN)
        )
        self.wait(5)

        # Final cleanup
        self.play(
            *[FadeOut(mob) for mob in self.mobjects]
        )