import numpy as np

from manim import *


class QFSPFunction(ThreeDScene):
    def construct(self):
        # Define the quadratic function with sinusoidal perturbations
        def qfsp(x, y):
            return x**2 + y**2 + 10 * np.sin(x) * np.sin(y)
        
        # Add the title with the formula
        title = Tex(
            r"$f(x, y) = x^2 + y^2 + 10 \sin(x) \sin(y)$", font_size=48
        ).to_corner(DL).shift(UP * 0.3)
        self.add(title)
        # Use `add_fixed_in_frame_mobjects` to ensure it stays fixed
        self.add_fixed_in_frame_mobjects(title)

        # Set up the axes
        axes = ThreeDAxes(
            x_range=(-5, 5, 1),
            y_range=(-5, 5, 1),
            z_range=(0, 21, 5),
            axis_config={"color": WHITE}  # Changed axis color to white for visibility against black background
        )
        self.add(axes)

        # Create the surface plot
        resolution = 20
        surface = Surface(
            lambda u, v: axes.c2p(u, v, qfsp(u, v)),
            u_range=[-5, 5],
            v_range=[-5, 5],
            resolution=(resolution, resolution),
            fill_opacity=0.7,
            checkerboard_colors=[BLUE_D, BLUE_E]
        )

        # Add the surface to the scene
        self.add(surface)

        # Set up the camera for a specific location
        # Adjust these values to find the best view for your plot
        self.set_camera_orientation(phi=60 * DEGREES, theta=30 * DEGREES, distance=15)

        # Animate the rotation of the scene (commented out for a fixed camera position)
        self.begin_ambient_camera_rotation(rate=0.2)
        self.wait(15)
        self.begin_ambient_camera_rotation(rate=0.1, about="phi")
        self.wait(10)

