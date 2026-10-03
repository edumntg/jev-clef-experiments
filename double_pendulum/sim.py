"""Double inverted pendulum on a cart (two links hinged in series, both upright). Same interface as
pendulum/sim.py so pendulum.run can drive it. Equations from the Lagrangian in D(q)q" + C(q,q')q' + G(q) = H u
form (Bogdanov 2004), integrated with RK4; LQR on the linearisation about the upright."""
import math
import random

import numpy as np
from PIL import Image, ImageDraw

from common.control import lqr_controller

G, M0, M1, M2, L1, L2 = 9.8, 1.0, 0.1, 0.1, 0.5, 0.5     # cart, link masses, full link lengths
DT, FMAX = 0.02, 20.0
TH_LIMIT, X_LIMIT = 20 * math.pi / 180, 2.4
FORCES = {"hard_left": -20, "left": -10, "nudge_left": -4, "none": 0,
          "nudge_right": 4, "right": 10, "hard_right": 20}
TASK = ("A cart runs on a horizontal track carrying a double pendulum: a lower pole hinged on the cart and "
        "an upper pole hinged on the tip of the lower one. Keep both poles upright and the cart near the "
        "centre. Physics: the upper pole matters most; move the cart under the direction the poles are "
        "falling, and the faster they fall the harder you push.")
SCENE = "Red line = lower pole, blue line = upper pole, black box = cart, grey ticks are fixed to the track."

# constant coefficients (uniform rods: COM at mid-length, I = m l^2 / 12)
d1 = M0 + M1 + M2
d2 = (M1 / 2 + M2) * L1
d3 = M2 * L2 / 2
d4 = (M1 / 3 + M2) * L1 ** 2
d5 = M2 * L1 * L2 / 2
d6 = M2 * L2 ** 2 / 3
f1, f2 = d2 * G, d3 * G


def _accel(q, dq, u):
    """q = (x, th1, th2) with th from upright, positive = leaning right."""
    _, t1, t2 = q
    c1, c2, c12, s1, s2, s12 = math.cos(t1), math.cos(t2), math.cos(t1 - t2), math.sin(t1), math.sin(t2), math.sin(t1 - t2)
    D = np.array([[d1, d2 * c1, d3 * c2],
                  [d2 * c1, d4, d5 * c12],
                  [d3 * c2, d5 * c12, d6]])
    C = np.array([[0, -d2 * s1 * dq[1], -d3 * s2 * dq[2]],
                  [0, 0, d5 * s12 * dq[2]],
                  [0, -d5 * s12 * dq[1], 0]])
    Gv = np.array([0, -f1 * s1, -f2 * s2])
    return np.linalg.solve(D, np.array([u, 0, 0]) - C @ dq - Gv)


def _deriv(z, u):
    q, dq = z[:3], z[3:]
    return np.concatenate([dq, _accel(q, dq, u)])


def step(s, f):
    """s = (x, x_dot, th1, th1_dot, th2, th2_dot); f in newtons, + = right. RK4."""
    f = max(-FMAX, min(FMAX, f))
    z = np.array([s[0], s[2], s[4], s[1], s[3], s[5]])
    k1 = _deriv(z, f)
    k2 = _deriv(z + DT / 2 * k1, f)
    k3 = _deriv(z + DT / 2 * k2, f)
    k4 = _deriv(z + DT * k3, f)
    z = z + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return (z[0], z[3], z[1], z[4], z[2], z[5])


def failed(s):
    return abs(s[0]) > X_LIMIT or abs(s[2]) > TH_LIMIT or abs(s[4]) > TH_LIMIT


def _linear():
    """Continuous A, B about the upright for state (x, x_dot, th1, th1_dot, th2, th2_dot)."""
    D0 = np.array([[d1, d2, d3], [d2, d4, d5], [d3, d5, d6]])
    Di = np.linalg.inv(D0)
    Ac = np.zeros((6, 6))                    # (q, dq) order first
    Ac[:3, 3:] = np.eye(3)
    Ac[3:, :3] = Di @ np.diag([0.0, f1, f2])
    Bc = np.zeros((6, 1))
    Bc[3:, 0] = Di @ np.array([1.0, 0, 0])
    perm = [0, 3, 1, 4, 2, 5]
    return Ac[np.ix_(perm, perm)], Bc[perm]


lqr_force = lqr_controller(*_linear(), np.diag([1.0, 1.0, 50.0, 5.0, 100.0, 5.0]), np.array([[0.05]]), FMAX)
"""lqr_force(s, period): LQR designed for a force held `period` seconds (the decision interval)."""


def initial():
    return (0.0, 0.0, random.uniform(-0.02, 0.02), 0.0, random.uniform(-0.02, 0.02), 0.0)


def numeric_state(s):
    return {"lower_pole_angle_deg": round(math.degrees(s[2]), 2),
            "lower_pole_angular_velocity_deg_per_s": round(math.degrees(s[3]), 1),
            "upper_pole_angle_deg": round(math.degrees(s[4]), 2),
            "upper_pole_angular_velocity_deg_per_s": round(math.degrees(s[5]), 1),
            "cart_position_m": round(s[0], 3), "cart_velocity_m_per_s": round(s[1], 3),
            "conventions": "angles measured from vertical; positive = leaning right; positive position/velocity = right"}


def angles_deg(s):
    return [math.degrees(s[2]), math.degrees(s[4])]


def render(s, size=(256, 192)):
    w, h = size
    img = Image.new("RGB", size, "white")
    d = ImageDraw.Draw(img)
    scale = w / (2 * X_LIMIT + 1)
    cx, cy = w / 2 + s[0] * scale, h * 0.8
    d.line([(0, cy + 12), (w, cy + 12)], fill="black", width=2)
    for m in range(-2, 3):
        tx = w / 2 + m * scale
        d.line([(tx, cy + 12), (tx, cy + 20)], fill="gray", width=2)
    d.rectangle([cx - 20, cy - 10, cx + 20, cy + 10], fill="black")
    x1, y1 = cx + L1 * scale * math.sin(s[2]), cy - L1 * scale * math.cos(s[2])
    x2, y2 = x1 + L2 * scale * math.sin(s[4]), y1 - L2 * scale * math.cos(s[4])
    d.line([(cx, cy), (x1, y1)], fill="red", width=5)
    d.line([(x1, y1), (x2, y2)], fill="blue", width=5)
    return img


if __name__ == "__main__":
    # upright is unstable (each pole falls further when leaning alone); pushing right straightens a right lean
    assert _accel(np.array([0, 0.1, 0.0]), np.zeros(3), 0.0)[1] > 0
    assert _accel(np.array([0, 0.0, 0.1]), np.zeros(3), 0.0)[2] > 0
    assert _accel(np.array([0, 0.1, 0.0]), np.zeros(3), 20.0)[1] < _accel(np.array([0, 0.1, 0.0]), np.zeros(3), 0.0)[1]
    hold = 5                                  # decide at 10 Hz like the experiment
    s = (0.0, 0.0, 0.05, 0.0, -0.03, 0.0)
    for i in range(1500):
        if i % hold == 0:
            f = lqr_force(s, hold * DT)
        s = step(s, f)
        assert not failed(s), (i, s)
    assert max(abs(s[2]), abs(s[4])) < 0.005, s
    print("ok  K(0.1s) =", np.round(lqr_force.gain(hold * DT), 1))
