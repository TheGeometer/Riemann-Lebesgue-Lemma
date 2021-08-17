from abc import ABC

import colour
from manim import *
from numpy import linalg
import math
import numpy as np
import random
from typing import Awaitable

step_size = 0.05


def construct_partition(f, a, b, epsilon):
    partition = [min(a, b)]
    current_x = partition[0]
    f_max = f(partition[0])
    f_min = f(partition[0])
    max_diff = f_max - f_min

    while current_x <= max(a, b):
        if f(current_x) < f_min:
            f_min = f(current_x)
        if f(current_x) > f_max:
            f_max = f(current_x)

        max_diff = f_max - f_min

        if max_diff > epsilon:
            partition.append(current_x)
            f_min = f(partition[-1])
            f_max = f(partition[-1])

        current_x += step_size

    partition.append(max(a, b))

    return partition


def min_max_coords(f, a, b):
    step_size = 0.025

    minab = min(a, b)
    maxab = max(a, b)

    current_x = minab
    f_max = f(minab)
    f_min = f(minab)
    min_x = minab
    max_x = minab

    while current_x < maxab + step_size:
        if f(current_x) < f_min:
            f_min = f(current_x)
            min_x = current_x
        if f(current_x) >= f_max:
            f_max = f(current_x)
            max_x = current_x
        current_x += step_size

    return [min_x, max_x]


def integrate(f, a, b):
    delta_x = ((b - a) / 10000)
    iterations = int(abs((b - a) / delta_x))
    area = 0.0
    x = a
    for _ in range(iterations):
        delta_area = f(x) * delta_x
        x = x + delta_x
        area = area + delta_area

    return area


def integral(f, lambd):
    return integrate(lambda x: f(x) * np.sin(lambd * x), 0, 1)


def point_on_axes(axes, x, y):
    return [axes.x_axis.number_to_point(x)[0], axes.y_axis.number_to_point(y)[1], 0]


def point_on_graph(axes, func, x):
    return point_on_axes(axes, x, func(x))


def sq(x):
    return x ** 2


def sine(x):
    return 40*np.sin(x)+50


def draw_min_max(axes, func, a, b):
    min_x = min_max_coords(func, a, b)[0]
    max_x = min_max_coords(func, a, b)[1]

    min_point = point_on_graph(axes, func, min_x)
    max_point = point_on_graph(axes, func, max_x)

    min_dot = Dot(min_point, fill_color=RED)
    max_dot = Dot(max_point, fill_color=RED)

    stroke_width = 3

    boundary_min = DashedLine(point_on_axes(axes, a, 0), point_on_axes(axes, a, axes.y_range[1] - 1), color=GREEN_E,
                              stroke_width=stroke_width)
    boundary_max = DashedLine(point_on_axes(axes, b, 0), point_on_axes(axes, b, axes.y_range[1] - 1), color=GREEN_E,
                              stroke_width=stroke_width)

    anims = [AnimationGroup(Write(boundary_min), Write(boundary_max)), AnimationGroup(Create(min_dot), Create(max_dot))]

    x_line_to_min = DashedLine(point_on_axes(axes, min_x, 0), point_on_graph(axes, func, min_x),
                               stroke_width=stroke_width, color=RED)
    x_line_to_max = DashedLine(point_on_axes(axes, max_x, 0), point_on_graph(axes, func, max_x),
                               stroke_width=stroke_width, color=RED)

    y_line_to_min = DashedLine(point_on_graph(axes, func, min_x), point_on_axes(axes, 0, func(min_x)),
                               stroke_width=stroke_width, color=RED)
    y_line_to_max = DashedLine(point_on_graph(axes, func, max_x), point_on_axes(axes, 0, func(max_x)),
                               stroke_width=stroke_width, color=RED)

    anims.append(AnimationGroup(Write(x_line_to_min), Write(x_line_to_max)))
    anims.append(AnimationGroup(Write(y_line_to_min), Write(y_line_to_max)))

    run_time = 1
    destroy_anims_y = AnimationGroup(Unwrite(y_line_to_min, run_time=run_time),
                                     Unwrite(y_line_to_max, run_time=run_time),
                                     lag_ratio=0)
    destroy_anims_x = AnimationGroup(Unwrite(x_line_to_min, run_time=run_time),
                                     Unwrite(x_line_to_max, run_time=run_time),
                                     lag_ratio=0)

    destroy_anims = AnimationGroup(*[destroy_anims_y, destroy_anims_x], lag_ratio=1)

    destroy_all = AnimationGroup(destroy_anims,
                                 Uncreate(min_dot, run_time=2 * run_time),
                                 Uncreate(max_dot, run_time=2 * run_time))

    anims.append(destroy_all)
    anims.append(Unwrite(boundary_min))
    anims.append(Unwrite(boundary_max))

    return anims


def draw_min_max_epsilon(axes, f, a, b, epsilon):
    epsilon_stick = Line(point_on_axes(axes, 0, 0),
                         point_on_axes(axes, 0, epsilon),
                         color=BLUE_D,
                         stroke_width=3).shift(LEFT / 2)
    epsilon_text = MathTex('\\epsilon').scale(0.75)
    brace = Brace(epsilon_stick, sharpness=0.5, direction=[-1, 0, 0], color=BLUE_D)
    brace.next_to(epsilon_stick, LEFT, buff=SMALL_BUFF)
    epsilon_text.next_to(brace, LEFT, buff=SMALL_BUFF).set_color_by_tex('eps', BLUE_D)

    xicks = construct_partition(f, a, b, epsilon)
    anims = [AnimationGroup(Write(epsilon_text),
                            Write(epsilon_stick),
                            Write(brace)),
             AnimationGroup(Unwrite(epsilon_text), Unwrite(brace))]

    for anim in [Create(axes.x_axis.get_tick(num).set_color(GREEN_E)) for num in xicks]:
        anims.append(anim)

    min_x_vals = [min_max_coords(f, xicks[ind], xicks[ind + 1])[0] for ind in range(len(xicks) - 1)]
    max_x_vals = [min_max_coords(f, xicks[ind], xicks[ind + 1])[1] for ind in range(len(xicks) - 1)]

    interval_anims = [draw_min_max(axes, f, xicks[ind], xicks[ind + 1]) for ind in range(len(xicks)-1)]

    for index in range(len(xicks) - 1):

        f_min_max = [f(val) for val in min_max_coords(f, xicks[index], xicks[index + 1])]

        if index == 0:
            anims.append(interval_anims[index][0])
        else:
            anims.append(interval_anims[index][0].animations[1])

        anims.append(interval_anims[index][1])
        anims.append(interval_anims[index][2])
        anims.append(interval_anims[index][3])

        stick_destination = [epsilon_stick.get_x(),
                             point_on_axes(axes, 0, (f(min_x_vals[index]) + f(max_x_vals[index])) / 2)[1],
                             0]

        epsilon_stick.generate_target()
        epsilon_stick.target.move_to(stick_destination)

        anims.append(MoveToTarget(epsilon_stick))

        anims.append(interval_anims[index][4])

        if index == 0:
            anims.append(interval_anims[index][5])
        elif index == len(xicks)-2:
            anims.append(AnimationGroup(interval_anims[index-1][6], interval_anims[index][6]))
        else:
            anims.append(interval_anims[index-1][6])

    anims.append(Unwrite(epsilon_stick))

    return anims


class CreatePartitions(Scene):
    def construct(self):
        xicks = construct_partition(sine, 0, 10, 30)
        yicks = [num ** 2 for num in xicks]

        axes = Axes(
            x_range=[0, 11, step_size],
            y_range=[0, 100, step_size],
            x_axis_config={
                "longer_tick_multiple": 1,
                "include_numbers": False,
                "include_ticks": True,
                "show_only_custom": True,
                "custom_ticks": xicks,
                "include_tip": True
                # "numbers_to_exclude": nums_to_exclude,
            },
            y_axis_config={"include_tip": True,
                           "show_only_custom": True,
                           "custom_ticks": yicks},
            y_length=5,
            x_length=10,
            tips=False
        )

        labels = axes.get_axis_labels(
            x_label=MathTex("x"), y_label=MathTex("f(x)")
        )

        graph = axes.get_graph(
            sine,
            x_range=[0, 10],
            color=YELLOW
        )
        create_xicks = [Create(axes.x_axis.get_tick(num).set_color(GREEN_E)) for num in xicks]
        create_yicks = [FadeIn(tick, run_time=0.5) for tick in axes.y_axis.get_tick_marks()]

        xick_grp = AnimationGroup(*create_xicks, lag_ratio=1)

        draw_minmax = draw_min_max_epsilon(axes, sine, 0, 10, 40)

        self.play(Create(axes, run_time=2), Write(labels))
        self.play(Create(graph))
        self.wait()
        # self.play(xick_grp)
        self.wait()
        for anim in draw_minmax:
            self.play(anim)
        self.wait()
        # self.play(*create_yicks)


class IntroduceProblem(Scene):

    @staticmethod
    def integral(f, lambd):
        return integrate(lambda x: f(x) * np.cos(lambd * x), 0, 1)

    def construct(self):
        axes = Axes(
            x_range=[-0.1, 1.1, 0.1],
            y_range=[-10, 10, 0.1],
            x_axis_config={"numbers_to_include": np.array([0, 1]), "include_ticks": False},
            y_axis_config={"numbers_to_include": [-10, -5, 0, 5, 10], "include_ticks": False},
            x_length=10,
            y_length=5,
            tips=True
        ).shift(DOWN * 1)
        labels = axes.get_axis_labels(
            x_label=MathTex("x"), y_label=MathTex("y")
        )

        graph = axes.get_graph(
            lambda x: 120 * (x - 0.5) ** 3 - 9 * x ** 2 + 5,
            x_range=[0, 1],
            color=YELLOW
        )

        lambd = 25
        new_graph = axes.get_graph(
            lambda x: (120 * (x - 0.5) ** 3 - 9 * x ** 2 + 5) * np.cos(25 * x),
            x_range=[0, 1],
            color=YELLOW
        )

        equation = MathTex("f(x)").shift(UP * 3)

        self.play(Create(axes), Write(labels))
        self.play(Create(graph), Write(equation))

        new_equation = MathTex("f(x)", "\cos(\lambda x)").shift(UP * 3)
        lambda_onscreen = MathTex("\lambda = ", f"{lambd}").shift(UP * 3 + RIGHT * 5)

        self.play(Transform(graph, new_graph))
        self.play(
                  Transform(equation, new_equation),
                  FadeIn(lambda_onscreen),
                  Create(new_graph))
        self.wait()
        graph = new_graph
        area = axes.get_area(graph, [0, 1], color=BLUE, opacity=0.5, dx_scaling=0.5)
        self.play(FadeIn(area))
        self.wait()

        
maxn = 10


def weierstrass(x, b):
    y = 0
    for i in range(1, maxn):
        y += np.cos((b ** i) * PI * x) / (3 ** i)
    return 10 * y + 7


class Weierstrass(Scene):
    def construct(self):
        axes = Axes(
            x_range=[0, 1, 0.001],
            y_range=[-0.1, 15, 0.001],
            x_axis_config={
                "include_numbers": False,
                "include_ticks": False,
                "include_tip": True
            },
            y_length=5,
            x_length=10,
            y_axis_config={"include_ticks": False,
                           "include_numbers": False,
                           "include_tip": True},
            tips=False
        )

        f = axes.get_graph(
            lambda x: weierstrass(x, 0),
            x_range=[0, 1],
            color=YELLOW,
            stroke_width=1
        )

        def updater(mob, alpha):
            mob.become(axes.get_graph(lambda x: weierstrass(x, interpolate(0, 7, alpha)),
                                      x_range=[0, 1],
                                      color=YELLOW,
                                      stroke_width=1
                                      ))

        self.play(Create(axes))
        self.play(Create(f))
        self.play(UpdateFromAlphaFunc(f, updater, run_time=5))
        self.wait()
