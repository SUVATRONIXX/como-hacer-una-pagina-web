import json,sys,numpy as np,wave
name,gi,a,b=sys.argv[1],int(sys.argv[2]),float(sys.argv[3]),float(sys.argv[4])
tr=json.load(open(f"tmp4_{name}/an{gi}.json"))
w=wave.open("../dl/audio16.wav"); sr=16000; w.setpos(int(a*sr)); x=np.frombuffer(w.readframes(int((b-a)*sr)),dtype=np.int16).astype(np.float32)/32768
hop=int(sr/30); k=len(x)//hop; env=np.sqrt((x[:k*hop].reshape(k,hop)**2).mean(1)); et=a+np.arange(k)/30
big=[t for t in tr if len(t["s"])>60]
for l in open("all_words3.jsonl"):
    j=json.loads(l)
    if not (a<=j["s"]<b): continue
    row=[]
    for t in big:
        s=np.array([q for q in t["s"] if j["s"]-0.3<=q[0]<=j["e"]+0.3])
        if len(s)<5: row.append("   -   "); continue
        e=np.interp(s[:,0],et,env)
        c=np.corrcoef(s[:,4],e)[0,1] if s[:,4].std()>0 and e.std()>0 else 0
        row.append(f"cx{int(np.median(s[:,1])):4d} m{s[:,4].mean():4.1f} c{c:+.2f}")
    print(f"{j['s']:.1f}-{j['e']:.1f} | "+" | ".join(row)+" | "+j["t"][:60])
