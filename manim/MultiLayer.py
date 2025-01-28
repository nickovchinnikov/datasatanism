import itertools

import numpy as np

from manim import *


def create_textbox(string, color=BLUE, text_color=WHITE, font_size=23, padding=0.4, fill_opacity=0.5):
    text = Tex(string, color=text_color, font_size=font_size)
    box = Rectangle(
        height=text.height + padding,
        width=text.width + padding,
        fill_color=color,
        fill_opacity=fill_opacity, stroke_color=color
    )
    text.move_to(box.get_center())
    return VGroup(box, text)


class Perceptron(ZoomedScene):
    def __init__(self, **kwargs):
        ZoomedScene.__init__(
            self,
            zoom_factor=0.12,
            zoomed_display_height=2,
            zoomed_display_width=2,
            image_frame_stroke_width=1,
            zoomed_camera_config={
                "default_frame_stroke_width": 6,
            },
            **kwargs
        )

    def construct(self):
        # #####
        # Image scene
        # #####
        self.camera.frame.save_state()

        digit_image = ImageMobject('./assets/five.png').scale(3).shift(2 * LEFT)

        self.add(digit_image)

        zoomed_camera = self.zoomed_camera
        zoomed_display = self.zoomed_display
        frame = zoomed_camera.frame
        zoomed_display_frame = zoomed_display.display_frame

        frame.move_to(digit_image.get_center() + 1.55 * UP + 0.1 * RIGHT)
        frame.set_color(RED_B)
        zoomed_display_frame.set_color(RED)
        zoomed_display.shift(2 * LEFT)

        camera_text_params = {"color": BLUE, "font_size": 36}

        zoomed_camera_texts = [
            Tex(
                "Pixel input: $x_{i" + (f'+{i}' if i > 0 else '') + "}$",
                **camera_text_params
            ).next_to(zoomed_display_frame, DOWN)
            for i in range(4)
        ]

        self.play(Create(frame))
        self.activate_zooming()

        shift = 0.242
        duration = 1

        self.play(self.get_zoomed_display_pop_out_animation(), Create(zoomed_camera_texts[0]))
        
        self.wait(duration=duration)
        self.play(Uncreate(zoomed_camera_texts[0]), frame.animate.shift(shift * RIGHT), Create(zoomed_camera_texts[1]))

        self.wait(duration=duration)
        self.play(Uncreate(zoomed_camera_texts[1]), frame.animate.shift(shift * RIGHT), Create(zoomed_camera_texts[2]))

        self.wait(duration=duration)
        self.play(Uncreate(zoomed_camera_texts[2]), frame.animate.shift(shift * RIGHT), Create(zoomed_camera_texts[3]))

        self.wait(duration=3)

        self.play(Uncreate(zoomed_display_frame), Uncreate(zoomed_camera_texts[3]), FadeOut(frame))
        self.remove(digit_image)

        self.camera.frame.restore()

        # Clear before run the next step
        self.clear()

        # #####
        # Perceptron scene
        # #####

        # r"$\sum_{i=0}^{n}{w_i x_i} = w^Tx$"
        margin_left = 6
        margin_right = 2
        font_size_inside_layers = 50
        font_size_labels = 20

        # Sum
        sum_node = LabeledDot(
            Tex(r"$\sum_{i=0}^{n}{w_i x_i}$", color=WHITE, font_size=45), color=BLUE).shift(LEFT)
        sum_label = Text(
            "Weighted sum", font_size=font_size_labels).next_to(sum_node, UP)
        sum_layer = VGroup(sum_node, sum_label)

        # Activation function
        axes = Axes(
            x_range=[-2, 2, 2],
            y_range=[-0.1, 0.9, 2],
            x_length=4,
            y_length=4,
            axis_config={'include_numbers': False},
            tips=False,
            x_axis_config={'color': ORANGE},
            y_axis_config={'color': ORANGE},
        )
        # Step function
        graph = axes.plot(lambda x: 0 if x <= 0 else 0.8,
                          use_smoothing=False, x_range=[-2, 2, 0.001], color=YELLOW)
        sigmoid_plot = VGroup(axes, graph).scale(0.3)

        sigmoid_frame = Dot(color=BLUE, radius=0.9)
        sigmoid_plot.move_to(sigmoid_frame.get_center())
        sigmoid_plot_and_frame = VGroup(sigmoid_frame, sigmoid_plot).next_to(
            sum_node, margin_right * RIGHT)
        sigmoid_label = Text(
            "Activation function", font_size=font_size_labels).next_to(sigmoid_frame, UP)

        sigmoid_layer = VGroup(sigmoid_plot_and_frame, sigmoid_label)

        # Activation function code
        code = '''xw = np.dot(x, w)
if xw >= 0:
    return 1
else:
    return -1
        '''
        activation_function_code = Code(
            code=code, tab_width=4, background="window",
            language="Python", font="Monospace"
        ).next_to(sum_node, margin_right * RIGHT)

        # Activation function math
        activation_function_math = Tex(r"""$
            \begin{cases}
                1,  & \mbox{if } w^Tx \geq 0 \\
                -1, & \mbox{otherwise}
            \end{cases}
        $""", font_size=40).next_to(sigmoid_plot_and_frame, margin_right * RIGHT)

        # Weights
        weights_params = {"color": BLUE,
                          "font_size": font_size_inside_layers}
        w_2 = create_textbox(
            r"$w_2$", **weights_params).next_to(sum_node, margin_left * LEFT)
        w_3 = create_textbox(
            r"$w_3$", **weights_params).next_to(w_2, DOWN)
        w_n = create_textbox(
            r"$w_n$", **weights_params).next_to(w_3, 3 * DOWN)
        w_1 = create_textbox(
            r"$w_1$", **weights_params).next_to(w_2, UP)
        w_0 = create_textbox(
            r"$w_0$", **weights_params).next_to(w_1, UP)

        weights = [w_0, w_1, w_2, w_3, w_n]
       
        w_num_2 = create_textbox(
            r"0.4", **weights_params).move_to(w_2)
        w_num_3 = create_textbox(
            r"0.5", **weights_params).next_to(w_2, DOWN)
        w_num_n = create_textbox(
            r"0.1", **weights_params).next_to(w_3, 3 * DOWN)
        w_num_1 = create_textbox(
            r"0.2", **weights_params).next_to(w_2, UP)
        w_num_0 = create_textbox(
            r"0.1", **weights_params).next_to(w_1, UP)
        
        weights_numbers = [w_num_0, w_num_1, w_num_2, w_num_3, w_num_n]

        weights_vdots = DashedLine(w_3.get_bottom(), w_n.get_top(),
                                   color=BLUE, dash_length=0.1)
        weights_text = Text(
            "Weights", font_size=font_size_labels).next_to(w_0, UP)

        weights_layer = VGroup(weights_text, weights_vdots, *weights)

        weights_framebox = SurroundingRectangle(weights_layer, buff = .1)

        # Inputs
        input_params = {"font_size": font_size_inside_layers, "fill_opacity": 1}
        x_2 = create_textbox(
            r"$x_2$", color=LIGHT_GRAY, text_color=WHITE, **input_params).next_to(w_2, margin_left * LEFT)
        x_1 = create_textbox(
            r"$x_1$", color=WHITE, text_color=BLACK, **input_params).next_to(x_2, UP)
        x_0 = create_textbox(
            r"$x_0$", color=BLUE, text_color=WHITE, **input_params).next_to(x_1, UP)
        x_3 = create_textbox(
            r"$x_3$", color=DARK_GREY, text_color=WHITE, **input_params).next_to(x_2, DOWN)
        x_n = create_textbox(
            r"$x_n$", color=WHITE, text_color=BLACK, **input_params).next_to(x_3, 3 * DOWN)

        inputs = [x_0, x_1, x_2, x_3, x_n]

        weights_vdots = DashedLine(x_3.get_bottom(), x_n.get_top(),
                                   color=BLUE, dash_length=0.1)

        inputs_text = Text(
            "Inputs", font_size=font_size_labels).next_to(x_0, UP)

        inputs_group = VGroup(inputs_text, weights_vdots, *inputs)

        inputs_text_bias = Tex(
            r"Bias \\ $x_0=1$", font_size=font_size_labels + 10
        ).next_to(x_0, UP)


        # Equation explanation
        sum_equations = [
            MathTex(f"=x_{i}w_{i}").next_to(weights[i], 2 * RIGHT)
            for i in range(4)
        ]
        sum_equations.append(MathTex("=x_nw_n").next_to(weights[4], 2 * RIGHT))

        sum_result_equation = MathTex(
            "x_0w_0 +", "x_1w_1 +", "x_2w_2 +", "x_3w_3 + ", "\cdots", "+ x_nw_n"
        ).next_to(w_0, 2 * RIGHT)

        sum_result_equation_sigma = MathTex(
            "=\sum_{i=0}^{n}{w_i x_i}"
        ).next_to(sum_result_equation, DOWN)

        sum_result_vector_form = MathTex(
            r"""
            =\begin{bmatrix}
                x_0 & x_1 & x_2 & \cdots & x_n
            \end{bmatrix}
            \begin{bmatrix}
                w_0     \\
                w_1     \\
                w_2     \\
                \vdots  \\
                w_n
            \end{bmatrix}
            """
        ).next_to(sum_result_equation, DOWN)

        sum_result_equation_dot_product = MathTex(
            "=x^Tw"
        ).next_to(sum_result_vector_form, DOWN)


        # Arrows params
        arrows_params = {"color": RED, "stroke_width": 4}

        # Arrows between inputs and weights
        arrows_input_weights = VGroup(*[Arrow(start=inputs[i].get_right(
        ), end=weights[i].get_left(), **arrows_params) for i in range(len(weights))])

        # Arrows between weights and sum
        arrows_weights_sum = VGroup(*[Arrow(start=weights[i].get_right(
        ), end=sum_node.get_left() + ((2 - i) * 0.2 * UP), **arrows_params) for i in range(len(weights))])

        arrow_weighted_sum_activation = Arrow(
            start=sum_node.get_right(), end=sigmoid_plot_and_frame.get_left(), **arrows_params)
        arrow_activation_output = Arrow(
            start=sigmoid_plot_and_frame.get_right(), end=activation_function_math.get_left(), **arrows_params)


        # Render items


        # Inputs and weights
        transfer_time_short = 4
        transfer_time_long = 7

        self.play(Create(inputs_group))
        self.wait(transfer_time_short)

        self.play(Create(weights_layer), Create(arrows_input_weights))
        self.wait(transfer_time_short)

        # Equations
        self.play(*map(Write, sum_equations))
        self.wait(transfer_time_long)
        self.play(*map(Uncreate, sum_equations))

        self.play(Create(sum_result_equation))
        self.wait(transfer_time_short)

        self.play(Create(sum_result_equation_sigma))
        self.wait(transfer_time_short)

        self.play(Uncreate(sum_result_equation_sigma))
        self.play(Create(sum_result_vector_form))
        self.wait(transfer_time_short)

        self.play(Create(sum_result_equation_dot_product))
        self.wait(transfer_time_long)

        self.play(
            Uncreate(sum_result_equation),
            Uncreate(sum_result_vector_form),
            Uncreate(sum_result_equation_dot_product)
        )

        # Sum layer
        self.play(Create(sum_layer), Create(arrows_weights_sum))
        self.wait(transfer_time_short)

        # Code example
        self.play(Create(activation_function_code))
        self.wait(transfer_time_long)
        self.play(Uncreate(activation_function_code))

        # Activation layer
        self.play(Create(arrow_weighted_sum_activation), Create(sigmoid_layer))
        self.wait(transfer_time_short)

        self.play(Create(activation_function_math), Create(arrow_activation_output))
        self.wait(transfer_time_short)
        
        # Bias term
        self.play(Uncreate(inputs_text))
        self.play(Create(inputs_text_bias))
        self.wait(transfer_time_short)

        # Weights framebox
        self.play(Create(weights_framebox))
        self.wait(transfer_time_short)

        self.play(*map(Uncreate, weights))
        self.play(*map(Create, weights_numbers))
        self.wait(transfer_time_short)

        self.wait(duration=10)


class LinearOR(Scene):
    def construct(self):
        ###
        # Rendering const
        ###
        # transfer_time_short = 4
        # transfer_time_long = 7

        ###
        # Equations
        ###
        tex_to_color_map = {
            'x': BLUE,
            'x_i': BLUE,
            'x_0': BLUE,
            'x_1': BLUE,
            'x_2': BLUE,
            'x_n': BLUE
        }
        perceptron_equation = MathTex(
            r"\sum_{i=0}^{n}w_ix_i=w^Tx",
            tex_to_color_map=tex_to_color_map
        )
        perceptron_equation_2 = MathTex(
            r"w^Tx=w_0x_0+w_1x_1+w_2x_2+\dots+w_nx_n",
            tex_to_color_map=tex_to_color_map
        )
        bias_term = MathTex(
            r"\text{Bias term: } x_0 = 1",
        )
        perceptron_equation_explained = MathTex(
            r"w^Tx=w_0+w_1x_1+w_2x_2",
            tex_to_color_map=tex_to_color_map
        )
        activation_function = MathTex(r"""
            \begin{cases}
                1,  & \mbox{if } w^Tx \geq 0 \\
                -1, & \mbox{otherwise}
            \end{cases}
        """)

        perceptron_equations_group = VGroup(
            perceptron_equation,
            perceptron_equation_2,
            bias_term,
            perceptron_equation_explained,
            activation_function
        ).arrange(DOWN)

        perceptron_equation_explained_sr = SurroundingRectangle(
            perceptron_equation_explained
        )

        ###
        # Construct the OR table
        ###

        # Make colors for 0 / 1 as RED / GREEN
        def table_colorizer(table, table_content):
            for (r_idx, row) in enumerate(table_content):
                for (c_idx, cell) in enumerate(row):
                    ent = table.get_entries_without_labels((r_idx + 1, c_idx + 1))
                    if cell == "0":
                        ent.set_color(RED)
                    if cell == "1":
                        ent.set_color(GREEN)

        or_table_full_content = np.array([
            [0, 0, 0],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1]
        ]).astype(str)

        or_table_labels = [
            MathTex("x_1"),
            MathTex("x_2"),
            MathTex("OR(x_1,x_2)")
        ]

        or_table_full = MathTable(
            or_table_full_content.tolist(),
            col_labels=or_table_labels,
            include_outer_lines=True
        ).move_to(LEFT * 3).scale(0.7)

        table_colorizer(or_table_full, or_table_full_content)

        lab = or_table_full.get_labels()
        colors = itertools.repeat(BLUE, 3)

        for (k, color) in enumerate(colors):
            lab[k].set_color(color)

        or_table_annotation = MathTex(
            r"1 = \text{True}, 0 = \text{False}",
            tex_to_color_map={"1": GREEN, "True": GREEN, "0": RED, "False": RED}
        ).next_to(or_table_full, UP)

        or_table_full_rows_sr = [
            SurroundingRectangle(or_table_full.get_rows()[i], buff=MED_SMALL_BUFF)
            for i in range(1, 4)
        ]

        or_table_perceptron_equation_explained = perceptron_equation_explained.copy()

        or_table_perceptron_activation = activation_function.copy()
        
        or_table_perceptron_first_equation1 = MathTex(
            r"w_0 + w_1 \cdot 0 + w_2 \cdot 0 < 0"
        )

        or_table_perceptron_first_equation2 = MathTex(
            r"\Rightarrow w_0 < 0"
        )

        or_table_equations_group = VGroup(
            or_table_perceptron_activation,
            or_table_perceptron_equation_explained,
            or_table_perceptron_first_equation1,
            or_table_perceptron_first_equation2
        ).arrange(DOWN, center=False).next_to(or_table_full, RIGHT)

        ###
        # Second table
        ###
        or_table_perceptron_content = np.concatenate(
            (
                or_table_full_content,
                np.array([
                    ["w_0 + w_1 \cdot 0 + w_2 \cdot 0 < 0"],
                    ["w_0 + w_1 \cdot 1 + w_2 \cdot 0 \geq 0"],
                    ["w_0 + w_1 \cdot 0 + w_2 \cdot 1 \geq 0"],
                    ["w_0 + w_1 \cdot 1 + w_2 \cdot 1 \geq 0"]
                ])
            ),
            axis=1
        )

        or_table_perceptron_labels = [
            MathTex("x_1"),
            MathTex("x_2"),
            MathTex("OR(x_1,x_2)"),
            MathTex("w_0+w_1x_1+w_2x_2")
        ]

        or_table_perceptron = MathTable(
            or_table_perceptron_content.tolist(),
            col_labels=or_table_perceptron_labels,
            include_outer_lines=True
        )

        table_colorizer(or_table_perceptron, or_table_perceptron_content)

        lab = or_table_perceptron.get_labels()
        colors = itertools.repeat(BLUE, 4)

        for (k, color) in enumerate(colors):
            lab[k].set_color(color)

        or_table_perceptron_rows_sr = [
            SurroundingRectangle(or_table_perceptron.get_rows()[i], buff=MED_SMALL_BUFF)
            for i in range(1, 4)
        ]

        ###
        # Final system of equations
        ###

        final_system_of_equations1 = MathTex(
            r"""
                    \begin{cases}
                        w_0 + w_1 \cdot 0 + w_2 \cdot 0 < 0 \\
                        w_0 + w_1 \cdot 1 + w_2 \cdot 0 \geq 0 \\
                        w_0 + w_1 \cdot 0 + w_2 \cdot 1 \geq 0 \\
                        w_0 + w_1 \cdot 1 + w_2 \cdot 1 \geq 0
                    \end{cases}
            """
        )

        final_system_of_equations2 = MathTex(
            r"""
                \Rightarrow
                \begin{cases}
                    w_0 < 0 \\
                    w_0 + w_1 \geq 0 \\
                    w_0 + w_2 \geq 0 \\
                    w_0 + w_1 + w_2 \geq 0
                \end{cases}
            """
        )

        final_system_of_equations3 = MathTex(
            r"""
                \Rightarrow
                \begin{cases}
                    w_0 < 0 \\
                    w_1 \geq -w_0 \\
                    w_2 \geq -w_0 \\
                    w_1 + w_2 \geq -w_0
                \end{cases}
            """
        )

        final_system_of_equations_gr1 = VGroup(
            final_system_of_equations1,
            final_system_of_equations2
        ).arrange(RIGHT)

        final_system_of_equations_gr2 = VGroup(
            final_system_of_equations2.copy(),
            final_system_of_equations3
        ).arrange(RIGHT)

        ###
        # Visualization OR
        ###
        ax = Axes(
            x_range=[-0.3, 1.3],
            y_range=[-0.5, 1.5],
            tips=False,
            axis_config={"include_tip": False},
        )

        labels = ax.get_axis_labels(
            x_label=MathTex("x_1", color=BLUE),
            y_label=MathTex("x_2", color=BLUE)
        )

        dots = [
            Dot(
                point=[ax.coords_to_point(*row[0:2])],
                color=RED if row[0:2] == [0, 0] else GREEN
            )
            for row in or_table_full_content.astype(int).tolist()
        ]

        labels_dots = [
            MathTex(
                f"({row[0]}, {row[1]})",
                color=RED if row[0:2] == [0, 0] else GREEN
            ).next_to(dots[idx], UP + RIGHT)
            for (idx, row) in enumerate(or_table_full_content.astype(int).tolist())
        ]

        or_plot_group = VGroup(
            ax,
            labels,
            *dots,
            *labels_dots
        ).scale(0.5).next_to(or_table_full, RIGHT)

        graph_line = ax.plot(lambda x: (1 - 1.1 * x) / 1.1, x_range=[-0.3, 1.3, 0.001], color=MAROON)

        ###
        # Rendering
        ###
        transfer_time_short = 4
        transfer_time_long = 7

        # Start from the perceptrone equation
        self.play(Create(perceptron_equations_group[0]))
        self.wait(transfer_time_short)

        self.play(Create(perceptron_equations_group[1]))
        self.wait(transfer_time_short)

        self.play(Create(perceptron_equations_group[2]))
        self.wait(transfer_time_short)

        self.play(Create(perceptron_equations_group[3]))
        self.wait(transfer_time_short)

        self.play(Create(perceptron_equations_group[4]))
        self.wait(transfer_time_short)

        self.play(Create(perceptron_equation_explained_sr))
        self.wait(transfer_time_short)

        self.play(
            Uncreate(perceptron_equations_group),
            Uncreate(perceptron_equation_explained_sr)
        )

        self.play(or_table_full.create(), Create(or_table_annotation))

        ###
        # Render OR graph
        ###
        self.play(Create(or_plot_group))
        self.wait(transfer_time_short)
        self.play(Create(graph_line))

        self.wait(transfer_time_long)

        self.play(Uncreate(or_plot_group), Uncreate(graph_line))

        self.play(Create(or_table_equations_group[0]))
        self.wait(transfer_time_short)

        self.play(Create(or_table_equations_group[1]))
        self.wait(transfer_time_short)

        self.play(Create(or_table_full_rows_sr[0]))
        self.play(Create(or_table_equations_group[2]))
        self.wait(transfer_time_short)

        self.play(Create(or_table_equations_group[3]))
        self.wait(transfer_time_short)

        self.play(
            Uncreate(or_table_full_rows_sr[0]),
            Uncreate(or_table_equations_group),
            Uncreate(perceptron_equation_explained_sr),
            Uncreate(or_table_full),
            Uncreate(or_table_annotation)
        )

        ###
        # Render table and final system of inequalities
        ###

        self.play(Create(or_table_perceptron))
        self.wait(transfer_time_long)

        self.play(Uncreate(or_table_perceptron))

        self.play(Create(final_system_of_equations_gr1[0]))
        self.play(Create(final_system_of_equations_gr1[1]))
        self.wait(transfer_time_long)

        self.play(Uncreate(final_system_of_equations_gr1))

        self.play(Create(final_system_of_equations_gr2[0]))
        self.play(Create(final_system_of_equations_gr2[1]))
        self.wait(transfer_time_long)

        self.play(Uncreate(final_system_of_equations_gr2))


class NonLinearXOR(Scene):
    def construct(self):
        # Make colors for 0 / 1 as RED / GREEN
        def table_colorizer(table, table_content):
            for (r_idx, row) in enumerate(table_content):
                for (c_idx, cell) in enumerate(row):
                    ent = table.get_entries_without_labels((r_idx + 1, c_idx + 1))
                    if cell == "0":
                        ent.set_color(RED)
                    if cell == "1":
                        ent.set_color(GREEN)

        xor_table_full_content = np.array([
            [0, 0, 0],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 0]
        ]).astype(str)

        xor_table_labels = [
            MathTex("x_1"),
            MathTex("x_2"),
            MathTex("XOR(x_1,x_2)")
        ]

        xor_table_full = MathTable(
            xor_table_full_content.tolist(),
            col_labels=xor_table_labels,
            include_outer_lines=True
        ).move_to(LEFT * 3).scale(0.7)

        table_colorizer(xor_table_full, xor_table_full_content)

        xor_table_annotation = MathTex(
            r"1 = \text{True}, 0 = \text{False}",
            tex_to_color_map={"1": GREEN, "True": GREEN, "0": RED, "False": RED}
        ).next_to(xor_table_full, UP)

        ###
        # Visualization XOR
        ###
        ax = Axes(
            x_range=[-0.3, 1.3],
            y_range=[-0.5, 1.5],
            tips=False,
            axis_config={"include_tip": False},
        )

        labels = ax.get_axis_labels(
            x_label=MathTex("x_1", color=BLUE),
            y_label=MathTex("x_2", color=BLUE)
        )

        dots = [
            Dot(
                point=[ax.coords_to_point(*row[0:2])],
                color=RED if row[0:2] in [[0, 0], [1, 1]] else GREEN
            )
            for row in xor_table_full_content.astype(int).tolist()
        ]

        labels_dots = [
            MathTex(
                f"({row[0]}, {row[1]})",
                color=RED if row[0:2] in [[0, 0], [1, 1]] else GREEN
            ).next_to(dots[idx], UP + RIGHT)
            for (idx, row) in enumerate(xor_table_full_content.astype(int).tolist())
        ]

        xor_plot_group = VGroup(
            ax,
            labels,
            *dots,
            *labels_dots
        ).scale(0.5).next_to(xor_table_full, RIGHT)

        graph_line1 = ax.plot(lambda x: -x + 0.5, x_range=[-0.3, 1.3, 0.001], color=BLUE)
        graph_line2 = ax.plot(lambda x: -x + 1.5, x_range=[-0.3, 1.3, 0.001], color=BLUE)

        graph_lines_group = VGroup(graph_line1, graph_line2)

        ###
        # Final system of equations
        ###

        final_system_of_equations1 = MathTex(
            r"""
                    \begin{cases}
                        w_0 + w_1 \cdot 0 + w_2 \cdot 0 < 0 \\
                        w_0 + w_1 \cdot 1 + w_2 \cdot 0 \geq 0 \\
                        w_0 + w_1 \cdot 0 + w_2 \cdot 1 \geq 0 \\
                        w_0 + w_1 \cdot 1 + w_2 \cdot 1 < 0
                    \end{cases}
            """
        )

        final_system_of_equations2 = MathTex(
            r"""
                \Rightarrow
                \begin{cases}
                    w_0 < 0 \\
                    w_0 + w_1 \geq 0 \\
                    w_0 + w_2 \geq 0 \\
                    w_0 + w_1 + w_2 < 0
                \end{cases}
            """
        )

        final_system_of_equations3 = MathTex(
            r"""
                \Rightarrow
                \begin{cases}
                    w_0 < 0 \\
                    w_1 \geq -w_0 \\
                    w_2 \geq -w_0 \\
                    w_1 + w_2 < -w_0
                \end{cases}
            """
        )

        final_system_of_equations_gr1 = VGroup(
            final_system_of_equations1,
            final_system_of_equations2
        ).arrange(RIGHT)

        final_system_of_equations_gr2 = VGroup(
            final_system_of_equations2.copy(),
            final_system_of_equations3
        ).arrange(RIGHT)

        conclusion = Text("Contradiction", color=RED)\
            .next_to(final_system_of_equations3, UP)

        ###
        # Rendering
        ###
        transfer_time_short = 4
        transfer_time_long = 7

        self.play(xor_table_full.create(), Create(xor_table_annotation))

        self.play(Create(xor_plot_group))
        self.wait(transfer_time_long)

        self.play(Create(graph_lines_group))
        self.wait(transfer_time_long)

        self.play(
            Uncreate(xor_table_annotation),
            Uncreate(xor_table_full),
            Uncreate(xor_plot_group),
            Uncreate(graph_lines_group)
        )

        self.play(Create(final_system_of_equations_gr1))
        self.wait(transfer_time_long)
        self.play(Uncreate(final_system_of_equations_gr1))

        self.play(Create(final_system_of_equations_gr2), Create(conclusion))
        self.wait(transfer_time_long)
