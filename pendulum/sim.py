"""Cart-pole: same constants as Gym CartPole but with a continuous force, a PNG renderer for Clef,
and an LQR controller as ground truth. Pure numpy + PIL.

Any sim module used by pendulum.run must expose: DT, FORCES, TASK, SCENE, step, failed, lqr_force,
render, initial, numeric_state, angles_deg."""
import math
import random

import numpy as np
from PIL import Image, ImageDraw

from common.control import lqr_controller

G, MC, MP, L, DT, FMAX = 9.8, 1.0, 0.1, 0.5, 0.02, 10.0   # L = half pole length
TH_LIMIT, X_LIMIT = 12 * math.pi / 180, 2.4                # same failure box as Gym
FORCES = {"hard_left": -10, "left": -5, "nudge_left": -2, "none": 0,
          "nudge_right": 2, "right": 5, "hard_right": 10}
TASK = ("A cart runs on a horizontal track with a pole hinged on top. Keep the pole upright and the cart "
        "near the centre of the track. Physics: to catch a pole that leans to one side, push the cart "
        "toward that same side; the faster it falls, the harder you push.")
SCENE = "Red line = pole, black box = cart, grey ticks are fixed to the track."


def initial():
    return (0.0, 0.0, random.uniform(-0.05, 0.05), 0.0)


def numeric_state(s):
    return {"pole_angle_deg": round(math.degrees(s[2]), 2),
            "pole_angular_velocity_deg_per_s": round(math.degrees(s[3]), 1),
            "cart_position_m": round(s[0], 3), "cart_velocity_m_per_s": round(s[1], 3),
            "conventions": "positive angle = pole leaning right; positive position/velocity = right"}


def angles_deg(s):
    return [math.degrees(s[2])]


def step(s, f):
    """s = (x, x_dot, theta, theta_dot); theta > 0 = pole leaning right; f in newtons, + = right."""
    x, xd, th, thd = s
    f = max(-FMAX, min(FMAX, f))
    c, si = math.cos(th), math.sin(th)
    tmp = (f + MP * L * thd * thd * si) / (MC + MP)
    thdd = (G * si - c * tmp) / (L * (4 / 3 - MP * c * c / (MC + MP)))
    xdd = tmp - MP * L * thdd * c / (MC + MP)
    return (x + DT * xd, xd + DT * xdd, th + DT * thd, thd + DT * thdd)


def failed(s):
    return abs(s[0]) > X_LIMIT or abs(s[2]) > TH_LIMIT


def _linear():
    """Continuous A, B about the upright for state (x, x_dot, theta, theta_dot)."""
    d = L * (4 / 3 - MP / (MC + MP))
    a43, b4 = G / d, -1 / ((MC + MP) * d)
    a23, b2 = -MP * L * a43 / (MC + MP), 1 / (MC + MP) - MP * L * b4 / (MC + MP)
    Ac = np.array([[0, 1, 0, 0], [0, 0, a23, 0], [0, 0, 0, 1], [0, 0, a43, 0]], float)
    Bc = np.array([[0], [b2], [0], [b4]], float)
    return Ac, Bc


lqr_force = lqr_controller(*_linear(), np.diag([1.0, 1.0, 10.0, 1.0]), np.array([[0.1]]), FMAX)
"""lqr_force(s, period): LQR designed for a force held `period` seconds (the decision interval)."""


def render(s, size=(256, 160)):
    """Image right = +x. Track ticks give the model a fixed reference so cart motion is visible."""
    w, h = size
    img = Image.new("RGB", size, "white")
    d = ImageDraw.Draw(img)
    scale = w / (2 * X_LIMIT + 1)
    cx, cy = w / 2 + s[0] * scale, h * 0.75
    d.line([(0, cy + 12), (w, cy + 12)], fill="black", width=2)
    for m in range(-2, 3):
        tx = w / 2 + m * scale
        d.line([(tx, cy + 12), (tx, cy + 20)], fill="gray", width=2)
    d.rectangle([cx - 20, cy - 10, cx + 20, cy + 10], fill="black")
    plen = 2 * L * scale
    d.line([(cx, cy), (cx + plen * math.sin(s[2]), cy - plen * math.cos(s[2]))], fill="red", width=5)
    return img


if __name__ == "__main__":
    hold = 5                                  # decide at 10 Hz like the experiment
    s = (0.0, 0.0, 0.1, 0.0)
    for i in range(1000):
        if i % hold == 0:
            f = lqr_force(s, hold * DT)
        s = step(s, f)
        assert not failed(s), s
    assert abs(s[2]) < 0.01, s
    assert lqr_force((0, 0, 0.1, 0), DT) > 0, "leaning right must push right"
    render(s).save("/dev/null", format="PNG")
    print("ok  K(0.1s) =", np.round(lqr_force.gain(hold * DT), 2))
