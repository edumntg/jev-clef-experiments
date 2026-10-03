"""Discrete LQR for a continuous linear plant held with zero-order hold over `period` seconds.
numpy only: matrix exponential by scaling-and-squaring Taylor, Riccati by iteration."""
import functools

import numpy as np


def _expm(M):
    n = max(0, int(np.ceil(np.log2(max(np.linalg.norm(M, 1), 1e-12)))) + 1)
    A = M / 2 ** n
    E, term = np.eye(len(M)), np.eye(len(M))
    for k in range(1, 25):
        term = term @ A / k
        E = E + term
    for _ in range(n):
        E = E @ E
    return E


def dlqr(Ac, Bc, Q, R, period):
    """Gain K (1 x n as a vector) for u = -K x, where x is sampled every `period` and u is held."""
    n, m = Ac.shape[0], Bc.shape[1]
    M = np.zeros((n + m, n + m))
    M[:n, :n], M[:n, n:] = Ac * period, Bc * period
    E = _expm(M)
    A, B = E[:n, :n], E[:n, n:]
    P = Q.copy()
    for _ in range(100000):
        Pn = Q + A.T @ P @ A - A.T @ P @ B @ np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
        if np.abs(Pn - P).max() < 1e-9:
            break
        P = Pn
    return np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)[0]


def lqr_controller(Ac, Bc, Q, R, fmax):
    """Returns force(s, period) with the gain cached per period."""
    @functools.lru_cache(maxsize=None)
    def gain(period):
        return dlqr(Ac, Bc, Q, R, period)

    def force(s, period):
        return float(np.clip(-gain(round(period, 6)) @ np.array(s, float), -fmax, fmax))

    force.gain = gain
    return force


if __name__ == "__main__":
    # double integrator sanity: K must stabilise x'' = u at any period
    Ac, Bc = np.array([[0, 1], [0, 0.0]]), np.array([[0], [1.0]])
    f = lqr_controller(Ac, Bc, np.eye(2), np.array([[1.0]]), 10)
    x = np.array([1.0, 0.0])
    for _ in range(200):
        u = f(x, 0.1)
        A, B = np.array([[1, 0.1], [0, 1]]), np.array([0.005, 0.1])
        x = A @ x + B * u
    assert np.abs(x).max() < 1e-3, x
    print("ok")
