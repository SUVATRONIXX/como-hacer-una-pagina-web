import json, sys, numpy as np
rms=np.load("rms.npy"); a,b=int(sys.argv[1]),int(sys.argv[2])
for l in open('transcript.jsonl'):
    j=json.loads(l)
    if a<=j['s']<=b:
        s=int(j['s']); loud=rms[s:int(j['e'])+1].max()
        mark="!!" if loud>-20 else ("! " if loud>-25 else "  ")
        print(f"{s//60}:{s%60:02d}{mark}{j['t']}")
