# Anima WebUI

An **AUTOMATIC1111-style web front end** for the
[Anima](https://huggingface.co/circlestone-labs/Anima) image model
(CircleStone Labs / Comfy Org — a 2B finetune of `nvidia/Cosmos-Predict2-2B-Text2Image`,
tuned for anime and non-photorealistic illustration), driving a local **ComfyUI**
through an **A1111-compatible `/sdapi/v1` API**.

## Why a front end instead of just using A1111?

Anima cannot be loaded by AUTOMATIC1111/Forge. It is a Cosmos DiT that ships as three separate
files and uses a **Qwen3-0.6B** text encoder:

| file | ComfyUI folder | size |
|---|---|---|
| `anima-base-v1.0.safetensors` | `models/diffusion_models/` | 4.18 GB |
| `qwen_3_06b_base.safetensors` | `models/text_encoders/` | 1.19 GB |
| `qwen_image_vae.safetensors` | `models/vae/` | 0.25 GB |

A1111's loader only understands SD1.5/SDXL/SD3/Flux-style single-file checkpoints, so the model
itself has to run in ComfyUI (which supports Anima natively). This project supplies the familiar
A1111 interface — and an A1111-compatible HTTP API — on top of it.

## What's here

```
webui/
├── index.html   # the whole UI: dark slate + orange Gradio-Default palette, A1111 layout
├── app.py       # FastAPI: serves the UI, exposes /sdapi/v1, talks to ComfyUI
└── card-no-preview.png   # A1111's placeholder art for extra-network cards
colab_setup.sh   # clone/pull + restart on a Colab (or any Linux) box
```

### Reproduced from A1111

- quicksettings row (Stable Diffusion checkpoint / SD VAE / CLIP stop at last layers), the top tab
  bar, prompt + negative prompt with A1111's placeholder text, the tall orange **Generate** button
  with the **Interrupt | Skip** overlay that covers it while sampling
- `Hires. fix` (working: latent upscale + second KSampler) and `Refiner` accordions, the
  `Sampling method` accordion (sampler / schedule type / steps / Restore faces / Tiling),
  Width/Height + swap, Batch count/size, CFG Scale, Seed, Script
- the output panel: gallery, action buttons, generation-parameters box, and a progress bar with
  percentage + ETA fed by ComfyUI's websocket
- **extra networks**: a second tab row (`Generation | Textual Inversion | Hypernetworks |
  Checkpoints | Lora`) with the card browser — `Search`, `Sort:` with A1111's four sort-key icons,
  sort direction, folder-list toggle, `Refresh`, folder-list buttons that drive the search box,
  16×24rem cards with the `⎘ / 🛈 / 🛠` hover buttons, preview lookup the A1111 way, and clicking a
  card inserts `<lora:NAME:1.0>` into the prompt
- **img2img**: the `mode_img2img` sub-tabs (`img2img / Sketch / Inpaint / Inpaint sketch /
  Inpaint upload / Batch`), drag-and-drop / click / Ctrl+V image input, the four-way `Resize mode`
  radio, `Resize to` / `Resize by` tabs, Denoising strength, the Inpaint accordion (mask blur,
  mask mode, masked content, inpaint area), plus `Send to inpaint` and a paintable mask canvas

## API

A1111-compatible subset, so A1111 clients can point at this host:

- `POST /sdapi/v1/txt2img` — A1111 payloads in (`prompt`, `negative_prompt`, `steps`, `cfg_scale`,
  `width`, `height`, `sampler_name`, `scheduler`, `seed`, `batch_size`, `n_iter`, `enable_hr`,
  `hr_scale`, `hr_second_pass_steps`, `denoising_strength`, `override_settings.sd_model_checkpoint`)
- `POST /sdapi/v1/img2img` — plus `init_images`, `resize_mode`, `mask`, `mask_blur`,
  `inpainting_mask_invert`, `inpainting_fill`, `inpaint_full_res`, `inpaint_full_res_padding`
- `GET /sdapi/v1/sd-models`, `/loras`, `/samplers`, `/schedulers`, `/options`, `/progress`
- `POST /sdapi/v1/interrupt`, `/skip`, `/options`, `/refresh-checkpoints`
- `GET /internal/extra-networks?kind=lora|checkpoints|…`, `/internal/preview`, `/internal/health`
- `GET /` — the UI, `GET /docs` — Swagger

Both `txt2img` and `img2img` return `{images: [b64], parameters, info}` with A1111-format
`infotexts`, and report extra-network problems the way A1111 does
(`Networks with errors: <name>`).

### Prompt syntax

`<lora:name>`, `<lora:name:0.8>`, `<lora:name:te:unet>` and `<lyco:...>`. Tags are stripped from the
text before encoding; each becomes a `LoraLoaderModelOnly` node chained between the UNET and the
sampler, which is what Anima's own official ComfyUI template does.

## Running it

Requirements: a ComfyUI (with Anima models present) reachable at `http://127.0.0.1:8188`, and the
Anima files placed as in the table above.

```bash
pip install fastapi uvicorn aiohttp pillow
cd webui
ANIMA_CLIP_TYPE=stable_diffusion python3 -m uvicorn app:app --host 127.0.0.1 --port 8001
```

On a Colab VM (or any Linux box) `colab_setup.sh` clones/updates this repo and (re)starts the
server:

```bash
ANIMA_REPO=https://github.com/hsgwktb/anima-webui.git bash colab_setup.sh
```

Then expose it (a quick tunnel, no auth — the URL is the only secret):

```bash
cloudflared tunnel --url http://127.0.0.1:8001
```

Measured on a Colab L4 with Anima resident (~10.9 / 23 GB VRAM):

| run | time |
|---|---|
| 1024x1024, 30 steps, txt2img | 43.4 s |
| 512x512, 8 steps, txt2img | 6.6 s |
| 512x512, 8 steps, with a LoRA | 8.5 s |

## Known gaps

- top-level tabs other than txt2img/img2img are UI-only; img2img's **Sketch**, **Inpaint sketch**
  and **Batch** sub-tabs are not connected to the backend
- `Resize mode` supports Just resize / Crop and resize / Resize and fill;
  "Just resize (latent upscale)" is not implemented
- inpainting regenerates the masked region through a ComfyUI latent noise mask, so A1111's
  **Masked content** modes have no direct equivalent and **Only masked** is not cropped/stitched
- the extra-networks panes are one shared DOM block moved between tabs, so their element ids stay
  `txt2img_*` even on the img2img tab
- `Automatic` schedule type maps to ComfyUI's `normal`; `Align Your Steps` passes `ays` straight through

## Licence / attribution

The interface deliberately reproduces AUTOMATIC1111's design. Markup structure, several CSS rules,
the extra-network sort / direction / tree-view / refresh SVG icons and `card-no-preview.png` are
taken from [AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui),
which is licensed **AGPL-3.0**. This repository is therefore also licensed **AGPL-3.0** — see
`LICENSE` and `NOTICE`.

Anima itself is distributed by CircleStone Labs under its own licence; see the model card.
