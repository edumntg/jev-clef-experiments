"""Clef plays Super Mario Bros from pixels. Each decision: send the last 4 frames (plus a tiny JSON
state unless --no-state), ask, hold the resulting input for --hold emulator frames, glide to landing if
airborne, repeat.

Two ways to ask:
  default (perception)  Scores: "how far is the nearest enemy / pipe / hole?"  and code picks the button
                        (the model is the eyes, code is the policy; the System One way)
  --buttons             one Choice: "which button?"  (the model is the policy)

Views (--view): mask = full screen with everything behind Mario greyed out (default, cleared 1-1 with
CLEF_MODEL=clef), full = untouched screen, crop = window around Mario.

Examples:
  uv run mario --policy random                 # no API key needed, checks the emulator
  CLEF_MODEL=clef uv run mario                 # the config that reached the flag
  uv run mario --buttons --view full --scale 1 # the naive baseline
"""
import argparse
import collections
import csv
import json
import os
import random
import time

import gym_super_mario_bros  # noqa: F401  (registers the envs)
import gymnasium as gym
from nes_py.wrappers import JoypadSpace
from PIL import Image, ImageDraw

from common.decision import ask, model_name
from common.viewer import Viewer

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

ACTIONS = {  # name -> buttons; order defines the JoypadSpace index
    "right": ["right"],
    "right_run": ["right", "B"],
    "right_jump": ["right", "A"],
    "right_run_jump": ["right", "A", "B"],
    "jump": ["A"],
    "left": ["left"],
    "noop": ["NOOP"],
}
IDX = {k: i for i, k in enumerate(ACTIONS)}
CONTEXT = ("You are looking at consecutive frames of Super Mario Bros (NES), oldest first. Mario is the small "
           "red-and-brown figure; he moves right. Green pipes block the way and must be jumped. Goombas are "
           "brown mushroom-shaped enemies that walk left toward Mario: touching one from the side kills "
           "him, landing on top kills it. A jump takes about one second and covers about 4 tiles, so it "
           "must start when the obstacle or enemy is 1 to 3 tiles ahead, not later.")
TILES = "A tile is one brick in the floor, about Mario's width."
VIEW_NOTE = {
    "crop": (" Each frame is cropped around Mario: he stands near the left edge and everything to his right is "
             "what lies ahead; the view scrolls with him, so the floor and pipes move left as he advances."),
    "mask": " The grey area on the left of each frame is behind Mario and irrelevant; only look to the right of it.",
}
CROP_W, CROP_BACK, HUD = 160, 24, 32          # px: window width, px kept behind Mario, HUD rows dropped
BUTTON_QUESTIONS = {
    "action": {
        "type": "choice",
        "instructions": CONTEXT + " " + TILES + " Which input should be held for the next fraction of a second?",
        "criteria": {
            "right": "Walk right: nothing within 4 tiles ahead",
            "right_run": "Run right (B held): long clear stretch ahead",
            "right_jump": "Jump while moving right: a pipe, block step, small gap or goomba is 1 to 3 tiles ahead",
            "right_run_jump": "Long running jump right: a wide gap or a tall pipe is 1 to 3 tiles ahead",
            "jump": "Jump in place: hit a block above or dodge something without advancing",
            "left": "Step back left: let an enemy or hazard pass",
            "noop": "Wait: a moving platform or enemy needs to pass first",
        },
    },
    "danger": {
        "type": "noul",
        "instructions": "If Mario keeps walking right without jumping, will he die within the next second?",
    },
}
LEVELS = ["nothing of that kind ahead of Mario", "far: more than 4 tiles ahead",
          "medium: 2 to 4 tiles ahead", "close: less than 2 tiles ahead, about to reach it"]
PERCEPTION_QUESTIONS = {
    "enemy": {"type": "score", "criteria": LEVELS, "instructions": CONTEXT + " " + TILES +
              " How far ahead of Mario (to his right) is the nearest goomba or other enemy on the ground?"},
    "obstacle": {"type": "score", "criteria": LEVELS, "instructions": CONTEXT + " " + TILES +
                 " How far ahead of Mario is the nearest pipe, wall or step that blocks his path?"},
    "gap": {"type": "score", "criteria": LEVELS, "instructions": CONTEXT + " " + TILES +
            " How far ahead of Mario is the nearest hole in the brick floor, where the floor is missing "
            "and the blue sky reaches the bottom edge of the frame?"},
    "danger": {"type": "noul",
               "instructions": "If Mario keeps walking right without jumping, will he die within the next second?"},
}


class PerceptionPolicy:
    """Code is the policy; the model only answers what it sees. The model reports distance levels; code
    decides timing: walk by default, run to build speed when terrain is at medium range, and take ONE full
    running jump only when a threat is close. A jump takes ~3 decisions during which Mario cannot act,
    so early or false jumps are what get him killed."""

    # score 0..3 = nothing / far / medium / close. The model under-reports closeness by a tile or two,
    # so "medium-to-close" is already the moment to jump.
    JUMP = {"gap": 1.3, "obstacle": 1.8, "enemy": 1.4}
    RUN = 0.8

    def __call__(self, s):
        if s["gap"]["score"] > self.JUMP["gap"] or s["obstacle"]["score"] > self.JUMP["obstacle"] or s["danger"] > 0.5:
            return "right_run_jump"                      # terrain: long jump
        if s["enemy"]["score"] > self.JUMP["enemy"]:
            return "right_jump"                          # enemy: short hop, shorter blind window
        if max(s["obstacle"]["score"], s["gap"]["score"]) > self.RUN:   # terrain coming: build speed
            return "right_run"
        return "right"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", default="clef", choices=["random", "clef"])
    ap.add_argument("--level", default="SuperMarioBros-1-1-v0", help="gym id, e.g. SuperMarioBros-v0 for the full game")
    ap.add_argument("--hold", type=int, default=12, help="emulator frames per decision (60 fps)")
    ap.add_argument("--max-decisions", type=int, default=300)
    ap.add_argument("--no-state", action="store_true", help="pixels only, no JSON hints")
    ap.add_argument("--buttons", action="store_true", help="ask for the button directly instead of perception + code policy")
    ap.add_argument("--scale", type=int, default=2, help="upscale frames sent to the model")
    ap.add_argument("--view", default="mask", choices=["mask", "full", "crop"], help="what the model sees (see docstring)")
    ap.add_argument("--frame-gap", type=int, default=6, help="emulator frames between the 4 captured frames")
    ap.add_argument("--sample", action="store_true", help="sample the action from the probabilities instead of argmax")
    ap.add_argument("--no-gif", action="store_true")
    ap.add_argument("--headless", action="store_true", help="no live window")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    random.seed(a.seed)
    env = JoypadSpace(gym.make(a.level, render_mode="rgb_array"), list(ACTIONS.values()))
    obs, info = env.reset(seed=a.seed)
    ram = env.unwrapped.ram

    def crop_box():
        """Window around Mario (RAM 0x03AD = his screen x). None when the view is the untouched full frame."""
        if a.view == "full":
            return None
        left = min(max(int(ram[0x03AD]) - CROP_BACK, 0), obs.shape[1] - CROP_W)
        return (left, HUD, left + CROP_W, obs.shape[0])

    def model_view(img, box):
        if a.view == "mask":
            img = img.copy()
            ImageDraw.Draw(img).rectangle((0, HUD, box[0], img.height), fill=(90, 90, 90))
        elif a.view == "crop":
            img = img.crop(box)
        return img.resize((img.width * a.scale, img.height * a.scale), Image.NEAREST) if a.scale > 1 else img

    frames = collections.deque([(Image.fromarray(obs), crop_box())], maxlen=4)
    movie, rows = [frames[0][0]], []
    name = (f"{model_name(a.policy)}{'-buttons' if a.buttons else '-perception'}-{a.view}"
            f"-x{a.scale}-h{a.hold}-{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid() % 1000:03d}")
    os.makedirs(RUNS, exist_ok=True)
    every = max(1, a.frame_gap)          # the 4 frames span 4*frame_gap emulator frames (default 0.4 s)
    questions = BUTTON_QUESTIONS if a.buttons else PERCEPTION_QUESTIONS
    policy = PerceptionPolicy()
    if a.view != "full":
        questions = {k: {**q, "instructions": q["instructions"] + VIEW_NOTE[a.view]} for k, q in questions.items()}
    x_max, stuck, prev, done, lats = info["x_pos"], 0, "noop", False, []
    view = None if a.headless else Viewer("mario - " + model_name(a.policy), scale=3)
    caption, fc, last_cap = "", 0, 0

    def capture():
        nonlocal last_cap
        img = Image.fromarray(obs)
        frames.append((img, crop_box()))
        movie.append(img)
        last_cap = fc

    def play(action, nframes):
        """Step the emulator holding one input; capture a frame every `every` frames; update the window."""
        nonlocal obs, info, done, fc
        for _ in range(nframes):
            if done:
                return
            obs, _, term, trunc, info = env.step(IDX[action])
            done = term or trunc
            fc += 1
            if fc % every == 0:
                capture()
            if view:
                shown = Image.fromarray(obs)
                box = crop_box()
                if box:
                    ImageDraw.Draw(shown).rectangle(box, outline="yellow")
                view.show(shown, caption)
                time.sleep(1 / 60)

    for n in range(a.max_decisions):
        if last_cap != fc:
            capture()                        # the newest frame is always the current one
        state = None if a.no_state else {
            "mario_x": info["x_pos"], "time_left": info["time"], "size": info["status"],
            "decisions_without_progress": stuck,
            "frames": f"{len(frames)} frames, oldest first, about {every / 60:.2f} s apart"}
        if a.policy == "random":
            choice, probs, danger, lat = random.choice(list(ACTIONS)), None, None, 0.0
        else:
            answers, lat = ask(a.policy, state or "Super Mario Bros frames, oldest first.",
                               questions, images=[model_view(f, box) for f, box in frames])
            danger = answers["danger"]["noul"]
            if a.buttons:
                probs = answers["action"]["probabilities"]
                choice = (random.choices(list(probs), weights=list(probs.values()))[0]
                          if a.sample else answers["action"]["choice"])
            else:
                probs = {k: (round(v["score"], 2) if v["type"] == "score" else v["noul"]) for k, v in answers.items()}
                probs.update({k + "_close": v["probabilities"].get("3", 0.0) for k, v in answers.items() if v["type"] == "score"})
                choice = policy({k: v if v["type"] == "score" else v["noul"] for k, v in answers.items()})
            lats.append(lat)
        if stuck >= 3:                       # no progress for 3 decisions: fixed escape macro (both modes)
            choice = "escape"
            play("left", 16)
            play("right_run", 12)
            play("right_run_jump", 32)
            stuck = 0
        else:
            if "A" in ACTIONS[choice] and "A" in ACTIONS[prev]:
                play("right" if "right" in ACTIONS[choice] else "noop", 1)   # NES needs A released to re-jump
            play(choice, a.hold * (2 if choice == "right_run_jump" else 1))   # long jump = A held longer
        landed = 0
        while not done and ram[0x001D] != 0 and landed < 90:   # airborne: nothing to decide, glide to landing
            play("right", 1)
            landed += 1
        stuck = 0 if info["x_pos"] > x_max else stuck + 1
        x_max = max(x_max, info["x_pos"])
        rows.append({"decision": n, "x_pos": info["x_pos"], "choice": choice, "danger": danger,
                     "latency_s": lat, "probs": json.dumps(probs)})
        top = "" if not probs else "  " + " ".join(f"{k}:{v:.2f}" for k, v in
                                                     sorted(probs.items(), key=lambda kv: -kv[1])[:3])
        avg = sum(lats) / len(lats) * 1000 if lats else 0
        caption = (f"#{n:3d} x={info['x_pos']:4d}  last action: {choice:15s} danger={danger if danger is None else round(danger, 2)}\n"
                   f"latency {lat * 1000:5.0f} ms   avg {avg:5.0f} ms   calls {len(lats)}\n{top.strip()}")
        print(caption.replace("\n", "  "), flush=True)
        prev = choice if choice != "escape" else "right_run_jump"
        if done:
            break

    with open(f"{RUNS}/{name}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    if not a.no_gif:
        movie[0].save(f"{RUNS}/{name}.gif", save_all=True, append_images=movie[1:],
                      duration=int(every * 1000 / 60), loop=0)
    outcome = "FLAG" if info.get("flag_get") else ("died" if done else "stopped")
    print(f"\n{model_name(a.policy)}: {outcome} at x={x_max} after {len(rows)} decisions"
          f"  mean latency {sum(r['latency_s'] for r in rows) / len(rows) * 1000:.0f} ms"
          f"  outputs: {RUNS}/{name}.csv" + ("" if a.no_gif else f"  {RUNS}/{name}.gif"))
    env.close()
    if view:
        view.close()


if __name__ == "__main__":
    main()
