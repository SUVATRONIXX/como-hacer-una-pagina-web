---
name: editar-reel-ever
description: Edita reels/shorts verticales a partir de un stream de Kick de Ever, con el estilo que ya aprobó el usuario (panel de arriba con zoom a quien habla, video completo abajo, subtítulos palabra por palabra solo en frases clave, SFX de su carpeta de Drive, música suave). Úsala cuando el usuario diga "edítame un reel de ever", "sácame clips del stream de ever" o algo equivalente.
---

# Editar reels de Ever

Flujo completo para sacar clips verticales de un stream de Kick con el estilo que el usuario aprobó.
Todo se habla con el usuario en español. Los scripts están en `scripts/` y un ejemplo de configuración por clip en `referencia/ejemplo_spec.json`.

## 0. Antes de empezar

Pide (solo lo que falte en el mensaje):
- El link del stream de Kick.
- El reparto por partes del stream y cuántos clips (ejemplo que ya usó: compras 2-3, McDonald's 2-4, final 4-5; 15 para escoger).
- El link de su carpeta de SFX en Drive (no se guarda en el repo porque es público).
- Si hay un video de referencia nuevo de estilo.

Red: el entorno de nube debe permitir `kick.com`, `www.kick.com`, `web.kick.com`, `stream.kick.com`, `files.kick.com`, el host de CloudFront del VOD (sale en el error del primer intento, p. ej. `xxxx.cloudfront.net`), `huggingface.co`, `us.aws.cdn.hf.co`, `cas-bridge.xethub.hf.co`, `i.ytimg.com`, `drive.google.com` y `drive.usercontent.google.com`. Si algo falta, propón el cambio con las herramientas de entorno.

## 1. Preparar y descargar

```bash
bash scripts/setup.sh <scratchpad>/reel        # deja <scratchpad>/reel/{dl,work,fonts,models}
cd <scratchpad>/reel/dl
yt-dlp -f audio_only -o 'audio.%(ext)s' -N 8 "<link de kick>"
ffmpeg -i audio.mp4 -ac 1 -ar 16000 audio16.wav
yt-dlp -f 720p -o 'video720.%(ext)s' -N 8 "<link de kick>"     # en segundo plano, tarda
```

## 2. Encontrar los mejores momentos

1. Transcripción rápida de todo el stream: `python3 -I transcribe.py ../dl/audio16.wav transcript.jsonl base` (desde `work/`). Si falla con `metadata_errors`, ya está resuelto en el script (carga el wav con numpy).
2. Picos de risa/gritos: volumen por segundo (`rms.npy`) comparado con la mediana de 2 minutos alrededor; los picos altos casi siempre son lo más divertido.
3. Miniaturas cada minuto (`peek.py`) para ubicar las partes del stream (compras, comida, final, carro).
4. Transcripción precisa (modelo `medium`) solo de las ventanas candidatas: `python3 -I tr_windows.py medium '[[a,b],...]' "nombres y contexto del stream"` → `ventanas_medium.jsonl`. Junta todo en un solo jsonl para los subtítulos.
5. Lee lo transcrito y escoge momentos con remate claro. Criterios del usuario:
   - **Divertidos de verdad**, que "peguen en visitas", tipo clips del Maestro Shifu: un momento, remate rápido.
   - **Sin relleno**: solo lo que pertenece al momento. Contexto sí, plática que no viene al caso no.
   - Duración: lo ideal 50 s o más si el momento lo aguanta; nunca menos de ~35 s; hasta 2 min si es una secuencia que vale la pena completa (p. ej. toda una carrera).
   - Nada de textos que insulten (ni en títulos ni en textos de reacción).
   - Que se vean caras: evita tramos donde la cámara apunta al techo, a la mesa o a una chamarra.

## 3. Formato visual (ya aprobado)

- 1080×1920. Arriba un panel de 1080×1396 con zoom; línea negra de 6 px; abajo el video completo en 16:9 (1080×518).
- **Nunca debe verse el logo de PRISM Live** (esquina inferior derecha del original, x≈995-1245, y≈652-688): la franja de abajo recorta 1138×546 desde (71,47) y el panel de arriba no baja de y=645 cuando cruza x>990. `render4.py` ya lo hace.
- **Zoom a quien habla**: `render4.py` detecta caras (YuNet), mide el movimiento de boca y lo compara con el audio; enfoca a quien habla y cambia cuando cambia quien habla. Si nadie habla, plano normal. Nunca se queda más de 6.5 s en lo mismo (intercala otro plano). Acercamiento lento y pequeño golpe de zoom en cada corte.
- Silencios muertos se recortan solos (pausas >0.75 s sin voz y sin ruido), salvo en `no_trim_groups`.
- Color: más contraste/saturación; escenas oscuras con `gamma` 1.32 y `bright` 0.03.

## 4. Subtítulos (estilo del Short de referencia de Archie2)

- Poppins Bold, **minúsculas**, blancos con sombra suave, centrados en el panel de arriba.
- **Palabra por palabra mientras la dicen**, entrada con rebote y salida con desvanecido.
- **Solo en las frases clave** (2 a 6 rangos por clip, campo `subs`); el resto del clip sin texto.
- Palabras importantes en amarillo (`emph`). Corrige errores de transcripción con `fix`.
- Título fijo arriba opcional (`title`); quítalo si el usuario lo pide.

## 5. Sonido

- **SFX: pocos** (2-3 por clip, solo en remates; 4-5 en un clip de 2 min). Úsalos de la carpeta de Drive del usuario, no sintéticos: lista sus archivos con la herramienta de Drive (`parentId = '<id>'`), mide dónde suena cada uno con `silencedetect`, y bájalos/recórtalos con `scripts/fetch_drive_sfx.sh`. Ver `referencia/sfx.md` para qué efecto va con qué momento.
- Si el usuario pide un clip sin SFX, `"no_sfx": true`.
- **Música de fondo suave** (`music`: `fun` o `chill`, generada por `music.py`, sin derechos), unos 19 dB debajo de las voces.
- Voces niveladas a -15 LUFS, pico máximo -3 dBTP.

## 6. Render

Un JSON por clip (ver `referencia/ejemplo_spec.json`) y:

```bash
cd <scratchpad>/reel/work
python3 -I render4.py specs/<clip>.json            # primera vez
python3 -I render4.py specs/<clip>.json --reuse    # si solo cambian textos, SFX o música (no rehace el video)
```

Corre dos a la vez como máximo (`xargs -P 2`). Lanza los lotes en segundo plano y vigílalos; un render de 50 s tarda ~4 min.

## 7. Revisión antes de entregar (obligatoria)

- **Duración del audio = duración del video** en cada clip (`ffprobe -show_entries stream=duration`). Antes se cortaba el audio al final; ya está corregido en `render4.py` (`apad` antes de `loudnorm` y `atrim` después).
- **Sin velo blanco**: mide el brillo medio del panel de arriba en ~12 puntos; un valor cercano a 200 en una escena normal es el error del flash (ya corregido con `max(t-t0,0)` en las fórmulas).
- Hojas de contacto (`sheet.py`) para ver que el zoom esté en la persona correcta. Si se pega a alguien equivocado: `opts.exclude_x`, `wide_cx` o `fixed` para ese tramo.
- `diag.py` muestra, por frase, qué cara mueve más la boca (sirve para depurar el zoom).

## 8. Entrega

- Página de previsualización (Artifact) con versiones ligeras (540×960, ~700 kbps), ordenadas **del mejor al peor** con el motivo de cada lugar. Publica en dos tandas si pasa de ~60 MB.
- Ranking del mejor al peor en la respuesta, con duración y parte del stream de cada clip.
- Al aprobar: archivos finales 1080×1920 con nombre `NN - Título.mp4` en orden del ranking.
- **Drive**: el conector solo sube ~3 MB por envío, así que los clips no caben. Crea la carpeta en Drive y manda los archivos por aquí (`SendUserFile`); si se piden en .zip, pártelos en .zip de menos de 400 MB (`zip -0`, el video ya va comprimido). Si un archivo pesa demasiado para la vista previa, manda también una versión ligera.
- Avisa siempre de lo que no pudiste ver (no puedes escuchar los clips ni verlos en movimiento) y de lo que quedó pendiente.
