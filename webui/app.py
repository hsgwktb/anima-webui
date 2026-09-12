"""Anima WebUI backend.

Serves an A1111-style front end for the Anima model and exposes an
AUTOMATIC1111-compatible API (/sdapi/v1/*) on top of the local ComfyUI.

    ComfyUI :8188  <--this app-->  browser (A1111-style UI / any A1111 client)

Anima is a Cosmos-Predict2-2B finetune, so the graph is:
    UNETLoader(anima-base-v1.0) + CLIPLoader(qwen_3_06b_base) + VAELoader(qwen_image_vae)
    -> CLIPTextEncode x2 -> EmptyLatentImage -> KSampler -> VAEDecode -> SaveImage
"""
import asyncio, base64, io, json, os, random, re, time, uuid
from urllib.parse import quote

import aiohttp
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from PIL import Image, ImageFilter, ImageOps

LANCZOS = getattr(Image, "Resampling", Image).LANCZOS

HERE = os.path.dirname(os.path.abspath(__file__))
COMFY = os.environ.get("COMFY_URL", "http://127.0.0.1:8188")
WS_URL = COMFY.replace("http://", "ws://").replace("https://", "wss://") + "/ws"
MODELS_DIR = "/content/ComfyUI/models"
CLIP_TYPE = os.environ.get("ANIMA_CLIP_TYPE", "stable_diffusion")
CLIENT_ID = str(uuid.uuid4())

app = FastAPI(title="Anima WebUI", version="1.0")

PROGRESS = {
    "progress": 0.0, "eta_relative": 0.0,
    "state": {"skipped": False, "interrupted": False, "job": "", "job_count": 1,
              "job_no": 0, "sampling_step": 0, "sampling_steps": 0},
    "current_image": None, "textinfo": "",
}
_ACTIVE = {"prompt_id": None, "steps": 0, "t0": 0.0, "step_times": []}

# ---------------------------------------------------------------- samplers
SAMPLER_MAP = {
    "Euler a": "euler_ancestral", "Euler": "euler", "LMS": "lms", "Heun": "heun",
    "DPM2": "dpm_2", "DPM2 a": "dpm_2_ancestral", "DPM++ 2S a": "dpmpp_2s_ancestral",
    "DPM++ 2M": "dpmpp_2m", "DPM++ SDE": "dpmpp_sde", "DPM++ 2M SDE": "dpmpp_2m_sde",
    "DPM++ 3M SDE": "dpmpp_3m_sde", "Restart": "restart", "DDIM": "ddim",
    "UniPC": "uni_pc", "LCM": "lcm",
}
SAMPLER_MAP.update({k + " Karras": v for k, v in list(SAMPLER_MAP.items())})

SCHEDULER_MAP = {
    "Automatic": "normal", "Karras": "karras", "Exponential": "exponential",
    "SGM Uniform": "sgm_uniform", "Normal": "normal", "Simple": "simple",
    "DDIM Unified": "ddim_uniform", "Beta": "beta", "Linear Quadratic": "linear_quadratic",
    "KL Optimal": "kl_optimal", "Align Your Steps": "ays",
}


def resolve_sampler(name, scheduler):
    name = (name or "Euler").strip()
    sampler = SAMPLER_MAP.get(name)
    sched = SCHEDULER_MAP.get((scheduler or "Automatic").strip(), "normal")
    if sampler is None:
        sampler = SAMPLER_MAP.get(name.replace(" Karras", ""), "euler")
        if name.endswith("Karras") and sched == "normal":
            sched = "karras"
    return sampler, sched


# ---------------------------------------------------------------- comfy IO
async def comfy_get(path, **params):
    async with aiohttp.ClientSession() as s:
        async with s.get(COMFY + path, params=params, timeout=aiohttp.ClientTimeout(total=60)) as r:
            return await r.json()


async def comfy_post(path, payload):
    async with aiohttp.ClientSession() as s:
        async with s.post(COMFY + path, json=payload, timeout=aiohttp.ClientTimeout(total=60)) as r:
            txt = await r.text()
            try:
                return r.status, json.loads(txt)
            except Exception:
                return r.status, {"raw": txt}


def list_models():
    out = []
    d = os.path.join(MODELS_DIR, "diffusion_models")
    if os.path.isdir(d):
        for f in sorted(os.listdir(d)):
            if f.endswith(".safetensors"):
                out.append({
                    "title": f,
                    "model_name": f,
                    "hash": None,
                    "sha256": None,
                    "filename": os.path.join(d, f),
                    "config": None,
                })
    return out


def pick_model(requested):
    """Resolve the requested checkpoint to a file present on disk."""
    names = [m["model_name"] for m in list_models()]
    if requested:
        base = os.path.basename(str(requested))
        if base in names:
            return base
        for n in names:
            if base and base.lower() in n.lower():
                return n
    for pref in ("anima-base-v1.0.safetensors", "anima-turbo-v1.1.safetensors",
                 "anima-turbo-v1.0.safetensors", "anima-aesthetic-v1.1.safetensors"):
        if pref in names:
            return pref
    return names[0] if names else "anima-base-v1.0.safetensors"


# ---------------------------------------------------------------- extra networks (loras)
LORA_DIR = os.path.join(MODELS_DIR, "loras")
PREVIEW_EXTS = ("png", "jpg", "jpeg", "webp", "gif")
LORA_EXTS = (".safetensors", ".ckpt", ".pt", ".sft")

# A1111 syntax, up to three trailing values: <lora:name:te:unet:dyn>
LORA_TAG_RE = re.compile(
    r"<(?:lora|lyco):([^:>]+?)(?::([^:>]+?))?(?::([^:>]+?))?(?::([^:>]+?))?>",
    re.IGNORECASE)


def find_preview(path):
    """A1111's lookup order: NAME.EXT, then NAME.preview.EXT, for each allowed ext."""
    stem = os.path.splitext(path)[0]
    for ext in PREVIEW_EXTS:
        for cand in ("%s.%s" % (stem, ext), "%s.preview.%s" % (stem, ext)):
            if os.path.isfile(cand):
                return cand
    return None


def scan_loras():
    out = []
    if not os.path.isdir(LORA_DIR):
        return out
    for root, _dirs, files in os.walk(LORA_DIR):
        for fn in files:
            if os.path.splitext(fn)[1].lower() not in LORA_EXTS:
                continue
            full = os.path.join(root, fn)
            rel = os.path.relpath(full, LORA_DIR).replace(os.sep, "/")
            st = os.stat(full)
            prev = find_preview(full)
            out.append({
                "name": os.path.splitext(rel)[0],
                "alias": os.path.splitext(fn)[0],
                "filename": full,
                "rel": rel,
                "subdir": os.path.dirname(rel),
                "size": st.st_size,
                "mtime": st.st_mtime,
                "ctime": st.st_ctime,
                "has_preview": bool(prev),
                "preview_ext": (os.path.splitext(prev)[1].lstrip(".") if prev else None),
            })
    out.sort(key=lambda x: x["name"].lower())
    return out


def resolve_lora(requested, loras):
    """Match a <lora:NAME> tag to a file: alias, relative name, basename, then substring."""
    q = str(requested or "").strip().replace("\\", "/").lower()
    if not q:
        return None
    for it in loras:
        if q in (it["alias"].lower(), it["name"].lower(), os.path.basename(it["rel"]).lower()):
            return it
    for it in loras:
        if q in it["name"].lower():
            return it
    return None


def parse_loras(text):
    """Strip <lora:...> tags out of a prompt; return (clean_text, [(name, te, unet)])."""
    found = []

    def _f(g):
        try:
            return float(g)
        except (TypeError, ValueError):
            return None

    def repl(m):
        te = _f(m.group(2))
        unet = _f(m.group(3))
        te = 1.0 if te is None else te
        found.append((m.group(1).strip(), te, te if unet is None else unet))
        return ""

    clean = LORA_TAG_RE.sub(repl, text or "")
    clean = re.sub(r"[ \t]{2,}", " ", clean)
    clean = re.sub(r",(\s*,)+", ",", clean)
    return clean.strip(), found


# ---------------------------------------------------------------- graph
def build_graph(p):
    req = (p.get("override_settings") or {}).get("sd_model_checkpoint")
    ckpt = pick_model(req)

    seed = p.get("seed", -1)
    try:
        seed = int(seed)
    except (TypeError, ValueError):
        seed = -1
    if seed < 0:
        seed = random.randint(0, 0xFFFFFFFF)

    sampler, sched = resolve_sampler(p.get("sampler_name"), p.get("scheduler"))
    w = max(64, int(p.get("width", 1024)))
    h = max(64, int(p.get("height", 1024)))
    bs = max(1, min(8, int(p.get("batch_size", 1))))
    steps = max(1, int(p.get("steps", 30)))
    cfg = float(p.get("cfg_scale", 4.0))

    pos, lora_tags = parse_loras(p.get("prompt"))
    neg, _ = parse_loras(p.get("negative_prompt"))

    g = {
        "1": {"class_type": "UNETLoader",
              "inputs": {"unet_name": ckpt, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": "qwen_3_06b_base.safetensors",
                         "type": CLIP_TYPE, "device": "default"}},
        "3": {"class_type": "VAELoader",
              "inputs": {"vae_name": "qwen_image_vae.safetensors"}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": pos, "clip": ["2", 0]}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": neg, "clip": ["2", 0]}},
        "6": {"class_type": "EmptyLatentImage",
              "inputs": {"width": w, "height": h, "batch_size": bs}},
        "7": {"class_type": "KSampler",
              "inputs": {"model": ["1", 0], "positive": ["4", 0], "negative": ["5", 0],
                         "latent_image": ["6", 0], "seed": seed, "steps": steps,
                         "cfg": cfg, "sampler_name": sampler, "scheduler": sched,
                         "denoise": 1.0}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}},
        "9": {"class_type": "SaveImage",
              "inputs": {"filename_prefix": "Anima", "images": ["8", 0]}},
    }

    # Anima loras are model-only (the official template uses LoraLoaderModelOnly),
    # so each tag becomes one node chained between the UNET and the sampler.
    known = scan_loras()
    model_ref = ["1", 0]
    loras_used, lora_missing = [], []
    for i, (name, te_w, unet_w) in enumerate(lora_tags):
        it = resolve_lora(name, known)
        if it is None:
            lora_missing.append(name)
            continue
        nid = "2%02d" % i
        g[nid] = {"class_type": "LoraLoaderModelOnly",
                  "inputs": {"lora_name": it["rel"], "strength_model": unet_w,
                             "model": model_ref}}
        model_ref = [nid, 0]
        loras_used.append({"name": it["alias"], "weight": unet_w, "file": it["rel"]})
    g["7"]["inputs"]["model"] = model_ref

    if p.get("enable_hr"):
        scale = float(p.get("hr_scale", 2.0))
        w2 = max(64, int(round(w * scale / 8) * 8))
        h2 = max(64, int(round(h * scale / 8) * 8))
        hr_steps = int(p.get("hr_second_pass_steps") or 0) or steps
        g["10"] = {"class_type": "LatentUpscale",
                   "inputs": {"upscale_method": "bilinear", "width": w2, "height": h2,
                              "crop": "disabled", "samples": ["7", 0]}}
        g["11"] = {"class_type": "KSampler",
                   "inputs": {"model": model_ref, "positive": ["4", 0], "negative": ["5", 0],
                              "latent_image": ["10", 0], "seed": seed, "steps": hr_steps,
                              "cfg": cfg, "sampler_name": sampler, "scheduler": sched,
                              "denoise": float(p.get("denoising_strength", 0.5))}}
        g["8"]["inputs"]["samples"] = ["11", 0]

    return (g, seed, ckpt, sampler, sched, steps, (w, h), bs, cfg,
            loras_used, lora_missing)



def infotext(p, seed, ckpt, sampler, sched, steps, size, cfg):
    a1111_sampler = (p.get("sampler_name") or "Euler")
    lines = [(p.get("prompt") or "").strip()]
    if (p.get("negative_prompt") or "").strip():
        lines.append("Negative prompt: " + p["negative_prompt"].strip())
    parts = [
        "Steps: %d" % steps,
        "Sampler: %s" % a1111_sampler,
        "Schedule type: %s" % (p.get("scheduler") or "Automatic"),
        "CFG scale: %s" % cfg,
        "Seed: %d" % seed,
        "Size: %dx%d" % size,
        "Model: %s" % ckpt,
    ]
    if p.get("enable_hr"):
        parts.append("Hires upscale: %s" % p.get("hr_scale", 2.0))
        parts.append("Hires upscaler: %s" % p.get("hr_upscaler", "Latent"))
        parts.append("Denoising strength: %s" % p.get("denoising_strength", 0.5))
    lines.append(", ".join(parts))
    return "\n".join(lines)


async def run_one(graph, timeout=900):
    """Queue a graph, wait for completion, return list of (bytes, filename)."""
    status, res = await comfy_post("/prompt", {"prompt": graph, "client_id": CLIENT_ID})
    if status != 200 or "prompt_id" not in res:
        raise RuntimeError("ComfyUI rejected the prompt: %s" % json.dumps(res)[:600])
    pid = res["prompt_id"]
    _ACTIVE["prompt_id"] = pid
    PROGRESS["state"]["interrupted"] = False

    t0 = time.time()
    while time.time() - t0 < timeout:
        async with aiohttp.ClientSession() as s:
            async with s.get(COMFY + "/history/" + pid,
                             timeout=aiohttp.ClientTimeout(total=30)) as r:
                hist = await r.json()
        if pid in hist:
            entry = hist[pid]
            st = entry.get("status", {})
            if st.get("status_str") == "error" or st.get("completed") is False and st.get("messages"):
                pass
            images = []
            for node in entry.get("outputs", {}).values():
                for im in node.get("images", []):
                    async with aiohttp.ClientSession() as s:
                        params = {"filename": im["filename"], "type": im.get("type", "output")}
                        if im.get("subfolder"):
                            params["subfolder"] = im["subfolder"]
                        async with s.get(COMFY + "/view", params=params,
                                         timeout=aiohttp.ClientTimeout(total=60)) as r:
                            images.append((await r.read(), im["filename"]))
            if not images:
                msgs = entry.get("status", {}).get("messages", [])
                raise RuntimeError("ComfyUI produced no image. status=%s" % json.dumps(msgs)[:600])
            return images
        await asyncio.sleep(0.6)
    raise RuntimeError("Timed out after %ss waiting for ComfyUI" % timeout)


# ---------------------------------------------------------------- ws progress
async def ws_progress_loop():
    """Track sampling progress from ComfyUI's websocket."""
    while True:
        try:
            async with aiohttp.ClientSession() as s:
                async with s.ws_connect(WS_URL + "?clientId=" + CLIENT_ID, heartbeat=20) as ws:
                    async for msg in ws:
                        if msg.type != aiohttp.WSMsgType.TEXT:
                            continue
                        try:
                            d = json.loads(msg.data)
                        except Exception:
                            continue
                        t = d.get("type")
                        data = d.get("data", {})
                        if t == "progress":
                            mx = max(1, int(data.get("max", 1)))
                            v = int(data.get("value", 0))
                            PROGRESS["state"]["sampling_step"] = v
                            PROGRESS["state"]["sampling_steps"] = mx
                            frac = v / mx
                            PROGRESS["progress"] = round(0.05 + 0.9 * frac, 3)
                            now = time.time()
                            if _ACTIVE["t0"]:
                                done = now - _ACTIVE["t0"]
                                est = done / max(frac, 0.02)
                                PROGRESS["eta_relative"] = round(max(0.0, est - done), 1)
                        elif t == "executing":
                            if data.get("node") is None:
                                PROGRESS["progress"] = 1.0
                                PROGRESS["eta_relative"] = 0.0
                            else:
                                _ACTIVE["t0"] = _ACTIVE["t0"] or time.time()
        except Exception:
            await asyncio.sleep(2)


@app.on_event("startup")
async def _startup():
    asyncio.create_task(ws_progress_loop())


# ---------------------------------------------------------------- pages / api
@app.get("/", response_class=HTMLResponse)
async def index():
    with open(os.path.join(HERE, "index.html"), encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.get("/internal/ping")
async def ping():
    try:
        async with aiohttp.ClientSession() as s:
            async with s.get(COMFY + "/system_stats",
                             timeout=aiohttp.ClientTimeout(total=5)) as r:
                st = await r.json()
        dev = (st.get("devices") or [{}])[0]
        return {"status": "ok", "comfy": dev.get("name"), "vram_free": dev.get("vram_free")}
    except Exception as e:
        return JSONResponse({"status": "comfy_unreachable", "error": repr(e)}, status_code=503)


@app.get("/internal/health")
async def health():
    models = [m["model_name"] for m in list_models()]
    te = os.path.exists(os.path.join(MODELS_DIR, "text_encoders", "qwen_3_06b_base.safetensors"))
    vae = os.path.exists(os.path.join(MODELS_DIR, "vae", "qwen_image_vae.safetensors"))
    comfy_ok, err = False, None
    try:
        await comfy_get("/system_stats")
        comfy_ok = True
    except Exception as e:
        err = repr(e)
    return {"comfy_ok": comfy_ok, "comfy_error": err, "clip_type": CLIP_TYPE,
            "diffusion_models": models, "text_encoder_ok": te, "vae_ok": vae,
            "loras": [it["alias"] for it in scan_loras()],
            "lora_dir": LORA_DIR}


@app.get("/sdapi/v1/sd-models")
async def sd_models():
    return list_models()


@app.post("/sdapi/v1/refresh-checkpoints")
async def refresh():
    return {}


# ---- extra networks (A1111's "Lora" / "Checkpoints" card tabs) ----
NO_PREVIEW = os.path.join(HERE, "card-no-preview.png")

EXTRA_TITLES = {
    "textual_inversion": "Textual Inversion",
    "hypernetworks": "Hypernetworks",
    "checkpoints": "Checkpoints",
    "lora": "Lora",
}


def _checkpoint_items():
    out = []
    for m in list_models():
        st = os.stat(m["filename"])
        prev = find_preview(m["filename"])
        out.append({
            "name": m["model_name"], "alias": m["model_name"],
            "filename": m["filename"], "rel": m["model_name"], "subdir": "",
            "size": st.st_size, "mtime": st.st_mtime, "ctime": st.st_ctime,
            "has_preview": bool(prev),
            "preview": ("/internal/preview?kind=checkpoints&name=%s" % quote(m["model_name"])) if prev else None,
        })
    out.sort(key=lambda x: x["name"].lower())
    return out


@app.get("/sdapi/v1/loras")
async def sd_loras():
    return [{"name": it["name"], "alias": it["alias"], "path": it["filename"],
             "metadata": {}} for it in scan_loras()]


@app.get("/internal/extra-networks")
async def extra_networks(kind: str = "lora"):
    """Card data for one extra-networks tab (mirrors A1111's page items)."""
    if kind == "lora":
        items = scan_loras()
        for it in items:
            it["preview"] = ("/internal/preview?kind=lora&name=%s" % quote(it["name"])) if it["has_preview"] else None
    elif kind == "checkpoints":
        items = _checkpoint_items()
    else:
        items = []
    dirs = sorted({it["subdir"] for it in items if it.get("subdir")})
    return {
        "kind": kind,
        "title": EXTRA_TITLES.get(kind, kind),
        "items": items,
        "dirs": dirs,
        # A1111: "Default multiplier for extra networks"
        "default_multiplier": 1.0,
    }


@app.get("/internal/preview")
async def preview(kind: str = "lora", name: str = ""):
    """Serve a card thumbnail, falling back to A1111's no-preview placeholder."""
    path = None
    if kind == "lora":
        it = resolve_lora(name, scan_loras())
        if it:
            path = find_preview(it["filename"])
    elif kind == "checkpoints":
        for m in list_models():
            if name in (m["model_name"], os.path.basename(m["filename"])):
                path = find_preview(m["filename"])
                break
    if path and os.path.isfile(path):
        low = path.lower()
        mime = "image/jpeg" if low.endswith((".jpg", ".jpeg")) else "image/" + low.rsplit(".", 1)[-1]
        with open(path, "rb") as f:
            return Response(f.read(), media_type=mime)
    if os.path.isfile(NO_PREVIEW):
        with open(NO_PREVIEW, "rb") as f:
            return Response(f.read(), media_type="image/png")
    return Response(status_code=404)


@app.get("/sdapi/v1/samplers")
async def samplers():
    return [{"name": n, "aliases": [], "options": {}} for n in SAMPLER_MAP if not n.endswith("Karras")]


@app.get("/sdapi/v1/schedulers")
async def schedulers():
    return [{"name": n, "label": n, "aliases": []} for n in SCHEDULER_MAP]


@app.get("/sdapi/v1/options")
async def options_get():
    return {"sd_model_checkpoint": pick_model(None), "sd_vae": "Automatic",
            "CLIP_stop_at_last_layers": 1, "samples_format": "png",
            "sampler_name": "Euler", "scheduler": "Simple", "steps": 30, "cfg_scale": 4.0}


@app.post("/sdapi/v1/options")
async def options_set(req: Request):
    return {}


@app.get("/sdapi/v1/progress")
async def progress(skip_current_image: bool = True):
    out = dict(PROGRESS)
    out["state"] = dict(PROGRESS["state"])
    return out


@app.post("/sdapi/v1/interrupt")
async def interrupt():
    try:
        await comfy_post("/interrupt", {})
    except Exception:
        pass
    try:
        await comfy_post("/queue", {"clear": True})
    except Exception:
        pass
    PROGRESS["state"]["interrupted"] = True
    PROGRESS["progress"] = 0.0
    return {}


@app.post("/sdapi/v1/skip")
async def skip():
    return await interrupt()


# ---------------------------------------------------------------- img2img
COMFY_INPUT = "/content/ComfyUI/input"


def _pil_from_b64(data):
    return Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB")


def _save_comfy_input(img, prefix):
    """ComfyUI's LoadImage reads from its own input/ dir, so park the file there."""
    os.makedirs(COMFY_INPUT, exist_ok=True)
    name = "%s_%s.png" % (prefix, uuid.uuid4().hex[:12])
    img.save(os.path.join(COMFY_INPUT, name))
    return name


def a1111_resize(mode, im, width, height):
    """A1111 modules/images.py:resize_image, modes 0/1/2 (LANCZOS, no upscaler).

    0 = Just resize (stretch), 1 = Crop and resize (cover + centre crop),
    2 = Resize and fill (contain + edge-replicated padding).
    """
    width = max(8, int(width))
    height = max(8, int(height))
    if mode == 0:
        return im.resize((width, height), LANCZOS)

    ratio = width / height
    src_ratio = im.width / im.height
    if mode == 1:
        src_w = width if ratio > src_ratio else im.width * height // im.height
        src_h = height if ratio <= src_ratio else im.height * width // im.width
        resized = im.resize((max(1, src_w), max(1, src_h)), LANCZOS)
        res = Image.new("RGB", (width, height))
        res.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))
        return res

    src_w = width if ratio < src_ratio else im.width * height // im.height
    src_h = height if ratio >= src_ratio else im.height * width // im.width
    resized = im.resize((max(1, src_w), max(1, src_h)), LANCZOS)
    res = Image.new("RGB", (width, height))
    res.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))
    if ratio < src_ratio:
        fill_height = height // 2 - src_h // 2
        if fill_height > 0:
            top = resized.crop((0, 0, resized.width, 1)).resize((width, fill_height), LANCZOS)
            bottom = resized.crop((0, resized.height - 1, resized.width, resized.height)).resize((width, fill_height), LANCZOS)
            res.paste(top, (0, 0))
            res.paste(bottom, (0, fill_height + src_h))
    elif ratio > src_ratio:
        fill_width = width // 2 - src_w // 2
        if fill_width > 0:
            left = resized.crop((0, 0, 1, resized.height)).resize((fill_width, height), LANCZOS)
            right = resized.crop((resized.width - 1, 0, resized.width, resized.height)).resize((fill_width, height), LANCZOS)
            res.paste(left, (0, 0))
            res.paste(right, (fill_width + src_w, 0))
    return res


def prepare_mask(p, init_img, mode, w, h):
    """A1111 mask handling: resize with the image, blur, then optional invert."""
    raw = p.get("mask")
    if not raw:
        return None
    m = Image.open(io.BytesIO(base64.b64decode(raw))).convert("L")
    if m.size != (w, h):
        m = a1111_resize(mode, m.convert("RGB"), w, h).convert("L")
    blur = float(p.get("mask_blur") or 0)
    if blur > 0:
        m = m.filter(ImageFilter.GaussianBlur(radius=blur))
    if int(p.get("inpainting_mask_invert") or 0) == 1:
        m = ImageOps.invert(m)
    return m


def build_img2img_graph(p, init_img, mask_img):
    req = (p.get("override_settings") or {}).get("sd_model_checkpoint")
    ckpt = pick_model(req)

    seed = p.get("seed", -1)
    try:
        seed = int(seed)
    except (TypeError, ValueError):
        seed = -1
    if seed < 0:
        seed = random.randint(0, 0xFFFFFFFF)

    sampler, sched = resolve_sampler(p.get("sampler_name"), p.get("scheduler"))
    w, h = init_img.size
    bs = max(1, min(8, int(p.get("batch_size", 1))))
    steps = max(1, int(p.get("steps", 30)))
    cfg = float(p.get("cfg_scale", 4.0))
    strength = min(1.0, max(0.0, float(p.get("denoising_strength", 0.75))))

    pos, lora_tags = parse_loras(p.get("prompt"))
    neg, _ = parse_loras(p.get("negative_prompt"))

    init_name = _save_comfy_input(init_img, "anima_i2i")

    g = {
        "1": {"class_type": "UNETLoader",
              "inputs": {"unet_name": ckpt, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": "qwen_3_06b_base.safetensors",
                         "type": CLIP_TYPE, "device": "default"}},
        "3": {"class_type": "VAELoader",
              "inputs": {"vae_name": "qwen_image_vae.safetensors"}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": pos, "clip": ["2", 0]}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": neg, "clip": ["2", 0]}},
        # LoadImage -> VAEEncode is A1111's "encode the init image" step
        "20": {"class_type": "LoadImage", "inputs": {"image": init_name, "upload": "image"}},
        "21": {"class_type": "VAEEncode", "inputs": {"pixels": ["20", 0], "vae": ["3", 0]}},
    }

    latent_ref = ["21", 0]
    if mask_img is not None:
        mask_name = _save_comfy_input(mask_img, "anima_mask")
        g["23"] = {"class_type": "LoadImage", "inputs": {"image": mask_name, "upload": "image"}}
        g["24"] = {"class_type": "ImageToMask", "inputs": {"image": ["23", 0], "channel": "red"}}
        g["25"] = {"class_type": "SetLatentNoiseMask",
                   "inputs": {"samples": latent_ref, "mask": ["24", 0]}}
        latent_ref = ["25", 0]

    if bs > 1:
        g["26"] = {"class_type": "RepeatLatentBatch",
                   "inputs": {"samples": latent_ref, "amount": bs}}
        latent_ref = ["26", 0]

    g["7"] = {"class_type": "KSampler",
              "inputs": {"model": ["1", 0], "positive": ["4", 0], "negative": ["5", 0],
                         "latent_image": latent_ref, "seed": seed, "steps": steps,
                         "cfg": cfg, "sampler_name": sampler, "scheduler": sched,
                         "denoise": strength}}
    g["8"] = {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}}
    g["9"] = {"class_type": "SaveImage",
              "inputs": {"filename_prefix": "Anima-i2i", "images": ["8", 0]}}

    known = scan_loras()
    model_ref = ["1", 0]
    loras_used, lora_missing = [], []
    for i, (name, te_w, unet_w) in enumerate(lora_tags):
        it = resolve_lora(name, known)
        if it is None:
            lora_missing.append(name)
            continue
        nid = "2%02d" % i
        g[nid] = {"class_type": "LoraLoaderModelOnly",
                  "inputs": {"lora_name": it["rel"], "strength_model": unet_w,
                             "model": model_ref}}
        model_ref = [nid, 0]
        loras_used.append({"name": it["alias"], "weight": unet_w, "file": it["rel"]})
    g["7"]["inputs"]["model"] = model_ref

    return (g, seed, ckpt, sampler, sched, steps, (w, h), bs, cfg,
            loras_used, lora_missing)


def infotext_img2img(p, seed, ckpt, sampler, sched, steps, size, cfg, loras, missing):
    txt = infotext(p, seed, ckpt, sampler, sched, steps, size, cfg)
    extra = ["Denoising strength: %s" % p.get("denoising_strength", 0.75)]
    if p.get("mask"):
        extra.append("Mask blur: %s" % p.get("mask_blur", 4))
    txt += "\n" + ", ".join(extra)
    if loras:
        txt += "\nLoras: " + ", ".join("%s: %s" % (x["name"], x["weight"]) for x in loras)
    if missing:
        txt += "\nNetworks with errors: " + ", ".join(missing)
    return txt


@app.post("/sdapi/v1/img2img")
async def img2img_api(req: Request):
    p = await req.json()
    init_list = p.get("init_images") or []
    if not init_list:
        return JSONResponse({"detail": "Init image not found"}, status_code=404)

    mode = int(p.get("resize_mode") or 0)
    if mode not in (0, 1, 2):
        mode = 0

    src = _pil_from_b64(init_list[0])
    w = int(p.get("width") or src.width)
    h = int(p.get("height") or src.height)
    init_img = a1111_resize(mode, src, w, h)
    w, h = init_img.size
    mask_img = prepare_mask(p, init_img, mode, w, h)

    (graph, seed, ckpt, sampler, sched, steps, size, bs, cfg,
     loras_used, lora_missing) = build_img2img_graph(p, init_img, mask_img)

    n_iter = max(1, min(16, int(p.get("n_iter", 1))))
    PROGRESS["state"]["job_count"] = n_iter

    out_images, infos, seeds = [], [], []
    for i in range(n_iter):
        PROGRESS["state"]["job_no"] = i
        if i > 0 and int(p.get("seed", -1) or -1) < 0:
            graph, seed, *_ = build_img2img_graph(p, init_img, mask_img)
        PROGRESS["progress"] = 0.02
        _ACTIVE["t0"] = time.time()
        for raw, _fn in await run_one(graph):
            out_images.append(base64.b64encode(raw).decode())
        infos.append(infotext_img2img(p, seed, ckpt, sampler, sched, steps, size, cfg,
                                      loras_used, lora_missing))
        seeds.append(seed)

    PROGRESS["progress"] = 1.0
    PROGRESS["eta_relative"] = 0.0
    info = json.dumps({
        "prompt": p.get("prompt", ""),
        "negative_prompt": p.get("negative_prompt", ""),
        "seed": seeds[0], "all_seeds": seeds,
        "width": size[0], "height": size[1],
        "sampler_name": p.get("sampler_name", "Euler"),
        "cfg_scale": cfg, "steps": steps,
        "denoising_strength": p.get("denoising_strength", 0.75),
        "batch_size": bs, "n_iter": n_iter,
        "infotexts": infos,
        "model": ckpt,
        "loras": loras_used,
        "lora_warnings": lora_missing,
    })
    return {"images": out_images, "parameters": p, "info": info,
            "loras": loras_used, "lora_warnings": lora_missing}


@app.post("/sdapi/v1/txt2img")
async def txt2img(req: Request):
    p = await req.json()
    (graph, seed, ckpt, sampler, sched, steps, size, bs, cfg,
     loras_used, lora_missing) = build_graph(p)
    n_iter = max(1, min(16, int(p.get("n_iter", 1))))

    PROGRESS["state"]["job_count"] = n_iter
    PROGRESS["progress"] = 0.02
    _ACTIVE["t0"] = time.time()

    out_images, infos, seeds = [], [], []
    for i in range(n_iter):
        PROGRESS["state"]["job_no"] = i
        if i > 0 and int(p.get("seed", -1) or -1) < 0:
            graph, seed, *_ = build_graph(p)
        PROGRESS["progress"] = 0.02
        _ACTIVE["t0"] = time.time()
        for raw, _fn in await run_one(graph):
            out_images.append(base64.b64encode(raw).decode())
        txt = infotext(p, seed, ckpt, sampler, sched, steps, size, cfg)
        if loras_used:
            txt += "\nLoras: " + ", ".join("%s: %s" % (x["name"], x["weight"]) for x in loras_used)
        if lora_missing:
            # A1111 reports these the same way rather than failing the whole job
            txt += "\nNetworks with errors: " + ", ".join(lora_missing)
        infos.append(txt)
        seeds.append(seed)

    PROGRESS["progress"] = 1.0
    PROGRESS["eta_relative"] = 0.0
    info = json.dumps({
        "prompt": p.get("prompt", ""),
        "negative_prompt": p.get("negative_prompt", ""),
        "seed": seeds[0], "all_seeds": seeds,
        "subseed": -1, "all_subseeds": [-1] * len(seeds),
        "width": size[0], "height": size[1],
        "sampler_name": p.get("sampler_name", "Euler"),
        "cfg_scale": cfg, "steps": steps,
        "batch_size": bs, "n_iter": n_iter,
        "infotexts": infos,
        "model": ckpt,
        "loras": loras_used,
        "lora_warnings": lora_missing,
    })
    return {"images": out_images, "parameters": p, "info": info,
            "loras": loras_used, "lora_warnings": lora_missing}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
