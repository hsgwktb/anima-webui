#!/usr/bin/env bash
# Clone or update the Anima WebUI from GitHub and (re)start it.
#
#   ANIMA_REPO=https://github.com/hsgwktb/anima-webui.git bash colab_setup.sh
#
# Expects a ComfyUI already running on 127.0.0.1:8188 with the Anima files in place
# (models/diffusion_models, models/text_encoders, models/vae).
set -euo pipefail

REPO="${ANIMA_REPO:-https://github.com/hsgwktb/anima-webui.git}"
DEST="${ANIMA_DEST:-/content/anima-webui}"
PORT="${ANIMA_PORT:-8001}"
LOGDIR="${ANIMA_LOGDIR:-/content/anima_webui}"

mkdir -p "$LOGDIR"

if [ -d "$DEST/.git" ]; then
  echo "updating $DEST"
  git -C "$DEST" fetch --depth 1 origin
  git -C "$DEST" reset --hard origin/HEAD
else
  echo "cloning into $DEST"
  git clone --depth 1 "$REPO" "$DEST"
fi

pkill -f 'uvicorn app:app' 2>/dev/null || true
sleep 1

cd "$DEST/webui"
nohup python3 -u -m uvicorn app:app --host 127.0.0.1 --port "$PORT" \
  > "$LOGDIR/webui.log" 2>&1 &
sleep 4

echo -n "health: "
curl -s "http://127.0.0.1:${PORT}/internal/health" || echo "(no response yet - check $LOGDIR/webui.log)"
echo
echo "serving $DEST/webui on 127.0.0.1:${PORT}"
