#!/usr/bin/env bash
# Baja SFX de una carpeta de Drive compartida por link y los deja listos en <work>/sfx2/<nombre>.wav
# Uso: bash fetch_drive_sfx.sh <carpeta_work> <lista.txt>
# lista.txt: una línea por efecto ->  <fileId> <nombre> <inicio_s> <fin_s|->
#   (los IDs salen de buscar en Drive con  parentId = '<id de la carpeta>';
#    inicio/fin recortan solo la parte que suena: mírala antes con silencedetect)
# Requiere permitir en el entorno los hosts drive.google.com y drive.usercontent.google.com.
set -euo pipefail
WORK="${1:?falta la carpeta work}"; LIST="${2:?falta la lista}"
mkdir -p "$WORK/../sfx_drive" "$WORK/sfx2"
while read -r id name a b; do
  [ -z "${id:-}" ] && continue
  mp3="$WORK/../sfx_drive/$name.mp3"
  curl -sS -L -m 60 -o "$mp3" "https://drive.usercontent.google.com/download?id=$id&export=download"
  to=""; [ "$b" != "-" ] && to="-to $b"
  # -nostdin: si no, ffmpeg se come el resto de la lista
  ffmpeg -nostdin -v error -y -ss "$a" $to -i "$mp3" \
    -af "areverse,afade=t=in:d=0.03,areverse,loudnorm=I=-16:TP=-1.5,aresample=48000" -ac 2 "$WORK/sfx2/$name.wav"
  echo "$name $(ffprobe -v error -show_entries format=duration -of csv=p=0 "$WORK/sfx2/$name.wav") s"
done < "$LIST"
