import sys, json, time
from faster_whisper import WhisperModel
src, out, model = sys.argv[1], sys.argv[2], sys.argv[3]
m = WhisperModel(model, device="cpu", compute_type="int8", cpu_threads=4)
import wave, numpy as np
w=wave.open(src); audio=np.frombuffer(w.readframes(w.getnframes()),dtype=np.int16).astype(np.float32)/32768
segs, info = m.transcribe(audio, language="es", beam_size=1, vad_filter=True, word_timestamps=True)
t0=time.time()
with open(out, "w") as f:
    for s in segs:
        f.write(json.dumps({"s":round(s.start,2),"e":round(s.end,2),"t":s.text.strip(),
            "w":[[round(w.start,2),round(w.end,2),w.word] for w in (s.words or [])]}, ensure_ascii=False)+"\n"); f.flush()
print("done", time.time()-t0)
