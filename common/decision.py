"""One client for both System One backends.
  clef -> Cloudflare Workers AI (text + up to 4 images). CLEF_MODEL picks "clef-flash" (9B, default)
          or "clef" (27B).
  jev  -> TypeSafe Jev via OpenRouter (text only). JEV_MODEL overrides the version.
Both take the same {state, questions} shape and return the same answers shape."""
import base64
import io
import json
import os
import time
import urllib.error
import urllib.request

from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_env():
    p = os.path.join(ROOT, ".env")
    if not os.path.exists(p):
        return
    for line in open(p):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_env()
JEV_MODEL = os.environ.get("JEV_MODEL", "typesafe/jev-1.13")
CLEF_MODEL = os.environ.get("CLEF_MODEL", "clef-flash")
if CLEF_MODEL not in ("clef", "clef-flash"):
    raise SystemExit(f"CLEF_MODEL must be 'clef' or 'clef-flash', got {CLEF_MODEL!r}")


def _need(*keys):
    for k in keys:
        if os.environ.get(k):
            return os.environ[k]
    raise SystemExit(f"{keys[0]} not set. Copy .env.example to .env and fill it in.")


def _post(url, body, headers, timeout=90):
    req = urllib.request.Request(url, json.dumps(body).encode(),
                                 {"Content-Type": "application/json", **headers})
    last = None
    for attempt in range(2):  # one retry: a single dropped read should not kill a run
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"HTTP {e.code} from {url}: {e.read()[:500].decode(errors='replace')}") from None
        except Exception as e:
            last = e
            time.sleep(1)
    raise RuntimeError(f"unreachable {url}: {last}")


def img_to_data_url(img):
    """PIL image or HxWx3 uint8 array -> data:image/png;base64,..."""
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def ask(backend, state, questions, images=None):
    """Returns (answers, latency_seconds). answers[qid] has .choice/.probabilities/.confidence,
    .noul, or .score depending on question type."""
    t0 = time.time()
    if backend == "clef":
        acct = _need("CLOUDFLARE_ACCOUNT_ID")
        tok = _need("CLOUDFLARE_API_TOKEN", "CLOUDFLARE_AUTH_TOKEN")
        body = {"model": CLEF_MODEL, "state": state, "questions": questions}
        if images:
            body["images"] = [img_to_data_url(i) for i in images[-4:]]
        r = _post(f"https://api.cloudflare.com/client/v4/accounts/{acct}/ai/run/@cf/cloudflare/{CLEF_MODEL}",
                  body, {"Authorization": f"Bearer {tok}"})
        if "success" in r and not r["success"]:
            raise RuntimeError(f"Workers AI error: {r.get('errors')}")
        r = r.get("result", r)
    elif backend == "jev":
        if images:
            raise ValueError("jev is text-only; describe the state as JSON instead")
        r = _post("https://openrouter.ai/api/alpha/decisions",
                  {"model": JEV_MODEL, "state": state, "questions": questions},
                  {"Authorization": f"Bearer {_need('OPENROUTER_API_KEY')}"})
    else:
        raise ValueError(f"unknown backend {backend!r} (use clef or jev)")
    return r["answers"], time.time() - t0


def model_name(backend):
    """Resolved model id, for run names and logs."""
    return {"clef": CLEF_MODEL, "jev": JEV_MODEL.split("/")[-1]}.get(backend, backend)


def expected_value(probabilities, values):
    """Probability-weighted value of a choice answer, e.g. a continuous force from force buckets."""
    return sum(p * values[k] for k, p in probabilities.items())


if __name__ == "__main__":
    assert abs(expected_value({"a": 0.25, "b": 0.75}, {"a": -10, "b": 10}) - 5.0) < 1e-9
    assert img_to_data_url(Image.new("RGB", (2, 2))).startswith("data:image/png;base64,")
    print("ok")
