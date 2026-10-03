"""Balance a cart-pole with LQR (baseline), Jev (numeric state as JSON) or Clef (last N frames).

The sim is synchronous by default: it pauses while the model thinks, so you measure decision quality
only. --realtime instead keeps stepping with the stale force during the round trip, so you find the
latency at which the controller breaks.

The loop is generic over a sim module (see pendulum/sim.py for the interface); double_pendulum reuses it.

Examples:
  uv run pendulum --policy lqr
  uv run pendulum --policy jev
  uv run pendulum --policy clef --frames 4              # CLEF_MODEL in .env picks clef-flash (default) or clef
  uv run pendulum --policy clef --text-state --realtime
"""
import argparse
import collections
import csv
import json
import os
import random
import time

import numpy as np

from common.decision import ask, expected_value, model_name
from common.viewer import Viewer
from . import sim


def run(simmod, label, runs_dir):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", default="lqr", choices=["lqr", "jev", "clef"])
    ap.add_argument("--steps", type=int, default=1000, help=f"max physics steps (dt={simmod.DT} s)")
    ap.add_argument("--hold", type=int, default=5, help="physics steps per decision")
    ap.add_argument("--frames", type=int, default=4, help="frames sent to clef (1-4)")
    ap.add_argument("--frame-every", type=int, default=5, help="physics steps between captured frames")
    ap.add_argument("--text-state", action="store_true", help="also give clef the numbers (default: pixels only)")
    ap.add_argument("--argmax", action="store_true", help="apply the top bucket instead of the expected value")
    ap.add_argument("--realtime", action="store_true", help="keep simulating with the stale force while waiting")
    ap.add_argument("--no-gif", action="store_true", help="skip the GIF (always written to outputs/ otherwise)")
    ap.add_argument("--headless", action="store_true", help="no live window")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    forces = simmod.FORCES
    questions = {"force": {
        "type": "choice",
        "instructions": simmod.TASK + " Which force should be applied to the cart right now?",
        "criteria": {k: (f"Push the cart {'left' if v < 0 else 'right'} with {abs(v)} N" if v else "No force")
                     for k, v in forces.items()},
    }}

    random.seed(a.seed)
    s = simmod.initial()
    frames = collections.deque(maxlen=a.frames)
    movie, rows, lats = [], [], []
    t = 0
    view = None if a.headless else Viewer(f"{label} - {model_name(a.policy)}")
    caption = ""

    def advance(n, f):
        nonlocal s, t
        for _ in range(n):
            s = simmod.step(s, f)
            t += 1
            img = simmod.render(s) if (view or t % a.frame_every == 0) else None
            if t % a.frame_every == 0:
                frames.append(img)
                movie.append(img)
            if view:
                view.show(img, caption)
                time.sleep(simmod.DT)   # real-time pacing while nothing is pending
            if simmod.failed(s):
                return False
        return True

    frames.append(simmod.render(s))
    os.makedirs(runs_dir, exist_ok=True)
    name = f"{model_name(a.policy)}{'-rt' if a.realtime else ''}-{time.strftime('%Y%m%d-%H%M%S')}"
    f_prev, alive = 0.0, True
    while alive and t < a.steps:
        f_lqr = simmod.lqr_force(s, a.hold * simmod.DT)
        probs, lat, choice = None, 0.0, "lqr"
        if a.policy == "lqr":
            f = f_lqr
        else:
            if a.policy == "jev":
                state, imgs = simmod.numeric_state(s), None
            else:
                state = {"description": f"{len(frames)} consecutive frames of the system, oldest first, "
                                        f"{a.frame_every * simmod.DT:.2f} s apart. Image right = positive "
                                        f"direction. {simmod.SCENE}"}
                if a.text_state:
                    state["measurements"] = simmod.numeric_state(s)
                imgs = list(frames)
            answers, lat = ask(a.policy, state, questions, images=imgs)
            probs, choice = answers["force"]["probabilities"], answers["force"]["choice"]
            f = forces[choice] if a.argmax else expected_value(probs, forces)
            lats.append(lat)
        num = {k: v for k, v in simmod.numeric_state(s).items() if k != "conventions"}
        rows.append({"t": t, **num, "choice": choice, "f_model": f, "f_lqr": f_lqr,
                     "latency_s": lat, "probs": json.dumps(probs)})
        avg = sum(lats) / len(lats) * 1000 if lats else 0
        angles = "  ".join(f"{d:6.2f}deg" for d in simmod.angles_deg(s))
        caption = (f"t={t * simmod.DT:5.2f}s  angles {angles}  x={s[0]:6.2f}m\n"
                   f"last action: {choice:12s} force={f:6.2f} N  (lqr {f_lqr:6.2f})\n"
                   f"latency {lat * 1000:5.0f} ms   avg {avg:5.0f} ms   calls {len(lats)}")
        print(caption.replace("\n", "  "), flush=True)
        if a.realtime and lat:
            alive = advance(int(lat / simmod.DT), f_prev)   # stale action during the round trip
        if alive:
            alive = advance(a.hold, f)
        f_prev = f

    with open(f"{runs_dir}/{name}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    if not a.no_gif and movie:
        movie[0].save(f"{runs_dir}/{name}.gif", save_all=True, append_images=movie[1:],
                      duration=int(a.frame_every * simmod.DT * 1000), loop=0)

    fm, fl = np.array([r["f_model"] for r in rows]), np.array([r["f_lqr"] for r in rows])
    print(f"\n{model_name(a.policy)}: survived {t} steps = {t * simmod.DT:.1f} s"
          f"{' (fell)' if not alive else ' (max steps)'}  decisions={len(rows)}")
    print(f"mean |angle| = {np.mean([abs(v) for r in rows for k, v in r.items() if k.endswith('angle_deg')]):.2f} deg")
    if a.policy != "lqr" and len(rows) > 2:
        print(f"sign agreement with LQR = {np.mean(np.sign(fm) == np.sign(fl)):.0%}"
              f"   corr(f_model, f_lqr) = {np.corrcoef(fm, fl)[0, 1]:.2f}"
              f"   mean latency = {np.mean(lats) * 1000:.0f} ms")
    print(f"outputs: {runs_dir}/{name}.csv" + ("" if a.no_gif else f"  {runs_dir}/{name}.gif"))
    if view:
        view.close()


def main():
    run(sim, "pendulum", os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs"))


if __name__ == "__main__":
    main()
