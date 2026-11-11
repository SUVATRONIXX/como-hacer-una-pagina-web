import sys, json, wave, numpy as np
from faster_whisper import WhisperModel
wins=json.loads(sys.argv[2]); model=sys.argv[1]
w=wave.open("../dl/audio16.wav"); sr=16000
m=WhisperModel(model, device="cpu", compute_type="int8", cpu_threads=4)
out=open(f"ventanas_{model}.jsonl","a")
for a,b in wins:
    w.setpos(int(a*sr)); x=np.frombuffer(w.readframes(int((b-a)*sr)),dtype=np.int16).astype(np.float32)/32768
    segs,_=m.transcribe(x,language="es",beam_size=2,vad_filter=True,word_timestamps=True,condition_on_previous_text=False,
        initial_prompt=(sys.argv[3] if len(sys.argv) > 3 else None))
    for s in segs:
        out.write(json.dumps({"s":round(a+s.start,2),"e":round(a+s.end,2),"t":s.text.strip(),
          "w":[[round(a+x.start,2),round(a+x.end,2),x.word] for x in (s.words or [])]},ensure_ascii=False)+"\n"); out.flush()
    print("win",a,b,flush=True)
print("done")
