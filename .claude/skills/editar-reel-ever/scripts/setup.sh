#!/usr/bin/env bash
# Prepara una carpeta de trabajo para editar reels.
# Uso: bash setup.sh <carpeta_de_trabajo>
# Crea:  <carpeta>/dl      video y audio del stream
#        <carpeta>/work    scripts (aquí se corre todo)
#        <carpeta>/fonts/ttf, <carpeta>/models, <carpeta>/work/music, <carpeta>/work/sfx
set -euo pipefail
ROOT="${1:?falta la carpeta de trabajo}"
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$ROOT"/{dl,work,fonts/ttf,models}
cp "$HERE"/*.py "$ROOT/work/"

# Dependencias. opencv 4.10 trae el detector de caras YuNet que usa render4.py.
pip install -q yt-dlp curl_cffi faster-whisper fonttools brotli "opencv-python-headless==4.10.0.84" pillow numpy

# Tipografía de los subtítulos: Poppins (Bold y ExtraBold), desde npm.
( cd "$ROOT/fonts" && npm pack -q @fontsource/poppins >/dev/null && tar xzf fontsource-poppins-*.tgz \
  && python3 -c "
from fontTools.ttLib import TTFont
for w in (700, 800):
    f = TTFont(f'package/files/poppins-latin-{w}-normal.woff2'); f.flavor = None; f.save(f'ttf/Poppins-{w}.ttf')" )

# Detector de caras (zoom a quien habla).
HF_HUB_DISABLE_XET=1 python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download('opencv/face_detection_yunet', 'face_detection_yunet_2023mar.onnx', local_dir='$ROOT/models')"

# Música de fondo propia (sin derechos) y SFX sintéticos de respaldo.
( cd "$ROOT/work" && mkdir -p music sfx && python3 music.py && python3 sfx.py )
echo "Listo: $ROOT"
