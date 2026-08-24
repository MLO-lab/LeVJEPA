#!/bin/bash
# Download the 10 Walking Tours videos (Venkataramanan et al., "Is ImageNet
# worth 1 video?", ICLR 2024). The HuggingFace dataset
# huggingface.co/datasets/shawshankvkt/Walking_Tours ships only YouTube URLs
# (WTour.txt), not video, so the videos are pulled from YouTube with yt-dlp
# (upstream's download_WTours.py uses pytube, which no longer works against
# current YouTube).
#
# Format 298 = 1280x720 @ 60 fps, avc1 in mp4, video-only. We deliberately do
# NOT take the 4K stream: the Lance store is short-edge 384, so 2160p would be
# ~28 GiB per video decoded and thrown away. Audio is skipped entirely -- the
# loader only reads frames. ~25 GB total.
#
# Usage:
#   bash scripts/download_walking_tours.sh [output_dir]
# Default output dir: data/walking_tours/videos
#
# Requires yt-dlp (installed by `uv sync --extra data`).

set -euo pipefail

OUT="${1:-data/walking_tours/videos}"
YTDLP="${YTDLP:-.venv/bin/yt-dlp}"
[ -x "$YTDLP" ] || YTDLP=yt-dlp

mkdir -p "$OUT"

# city|youtube_id, from WTour.txt
VIDEOS="
Amsterdam|E07rTPgIvn0
Bangkok|hUGcHvN4mME
Chiang_Mai|jyxIRzlkO_g
Istanbul|mA9lYWyXMYU
Kuala_Lampur|yLTycPjm0nk
Singapore|aUJl46bEWYo
Stockholm|Jr9x-RB4E1U
Wildlife|Q0ML5oAjX_w
Venice|fGX0Te6pFvk
Zurich|_NmYvuEILw4
"

fail=0
for entry in $VIDEOS; do
  city="${entry%%|*}"
  vid="${entry##*|}"
  dst="$OUT/${city}.mp4"
  if [ -f "$dst" ]; then
    echo "=== $city already present ($(du -h "$dst" | cut -f1)), skipping"
    continue
  fi
  echo "=== downloading $city ($vid)"
  # 298 is 720p60 mp4; fall back to any <=720p mp4 stream if it is missing.
  "$YTDLP" \
    --no-warnings --no-playlist --retries 10 --fragment-retries 10 \
    --format '298/bestvideo[height<=720][ext=mp4]/bestvideo[height<=720]' \
    --output "$dst" \
    "https://www.youtube.com/watch?v=${vid}" || { echo "FAILED $city"; fail=1; }
done

echo "=== inventory"
ls -la "$OUT"
du -sh "$OUT"
if [ "$fail" -ne 0 ]; then
  echo "one or more downloads FAILED"
  exit 1
fi
echo "all downloads done"
