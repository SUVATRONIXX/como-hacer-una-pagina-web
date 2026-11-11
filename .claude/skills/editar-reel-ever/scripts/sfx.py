import numpy as np, wave
SR=48000; rng=np.random.default_rng(1)
def save(name,x):
    x=x/ (np.abs(x).max()+1e-9)*0.9
    st=np.stack([x,x],1); d=(st*32767).astype(np.int16)
    w=wave.open(f"sfx/{name}.wav","wb"); w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes(d.tobytes()); w.close()
def t(d): return np.arange(int(SR*d))/SR
def lp(x,a):  # one-pole lowpass with time-varying coeff
    y=np.zeros_like(x); s=0
    a=np.broadcast_to(a,x.shape)
    for i in range(len(x)): s+=a[i]*(x[i]-s); y[i]=s
    return y
# whoosh: noise with swept bandpass + swell
d=0.55; tt=t(d); n=rng.standard_normal(len(tt))
f=np.interp(tt,[0,d*0.6,d],[0.02,0.35,0.05])
b=lp(n,f)-lp(lp(n,f),f*0.3); env=np.sin(np.pi*tt/d)**2
save("whoosh",b*env)
# whoosh fast
d=0.3; tt=t(d); n=rng.standard_normal(len(tt)); f=np.interp(tt,[0,d*0.5,d],[0.05,0.5,0.08])
save("swish",(lp(n,f)-lp(lp(n,f),f*0.3))*np.sin(np.pi*tt/d)**3)
# boom (bass drop / impact)
d=1.6; tt=t(d); fr=np.interp(tt,[0,0.08,d],[120,55,38]); ph=2*np.pi*np.cumsum(fr)/SR
x=np.sin(ph)*np.exp(-tt*2.2)+0.4*np.tanh(3*np.sin(ph))*np.exp(-tt*3)
x+=lp(rng.standard_normal(len(tt)),0.05)*np.exp(-tt*25)*3
save("boom",np.tanh(1.5*x))
# hit / punch
d=0.35; tt=t(d); fr=np.interp(tt,[0,0.05,d],[180,70,50]); ph=2*np.pi*np.cumsum(fr)/SR
x=np.sin(ph)*np.exp(-tt*10)+lp(rng.standard_normal(len(tt)),0.3)*np.exp(-tt*40)*1.5
save("punch",np.tanh(2*x))
# pop
d=0.12; tt=t(d); fr=np.interp(tt,[0,d],[900,300]); x=np.sin(2*np.pi*np.cumsum(fr)/SR)*np.exp(-tt*45); save("pop",x)
# ding (bell)
d=1.4; tt=t(d); x=sum(a*np.sin(2*np.pi*f*tt)*np.exp(-tt*k) for f,a,k in [(1318,1,3),(2637,.5,5),(3950,.25,7),(1975,.3,4)]); save("ding",x)
# riser
d=1.5; tt=t(d); fr=np.interp(tt,[0,d],[200,1600]); x=np.sin(2*np.pi*np.cumsum(fr)/SR)*0.4
n=rng.standard_normal(len(tt)); x+= (lp(n,np.interp(tt,[0,d],[0.02,0.5]))) * 0.8
save("riser",x*(tt/d)**2)
# record scratch
d=0.5; tt=t(d); n=rng.standard_normal(len(tt)); fm=np.abs(np.sin(2*np.pi*7*tt))
x=lp(n,0.05+0.4*fm)*fm*np.exp(-tt*2); save("scratch",x)
# glitch (bitcrushed stutter)
d=0.4; tt=t(d); x=np.sign(np.sin(2*np.pi*np.where((tt*20).astype(int)%2,220,440)*tt))*0.5+rng.standard_normal(len(tt))*0.3
x*= ((tt*25).astype(int)%3!=0); x=np.round(x*4)/4; save("glitch",x*np.exp(-tt*3))
# camera shutter / click
d=0.25; tt=t(d); n=rng.standard_normal(len(tt)); x=n*np.exp(-tt*80)+np.roll(n*np.exp(-tt*60),int(0.07*SR))*0.8; save("shutter",x)
# cash register-ish (ka-ching): two bells
d=1.2; tt=t(d); x=np.sin(2*np.pi*2093*tt)*np.exp(-tt*4)+np.sin(2*np.pi*2637*np.clip(tt-0.08,0,None))*np.exp(-np.clip(tt-0.08,0,None)*4)*(tt>0.08)
x+=rng.standard_normal(len(tt))*np.exp(-tt*50)*0.7; save("kaching",x)
# bruh-ish low "dun dun dun" dramatic
d=1.8; tt=t(d); x=np.zeros_like(tt)
for st,f in [(0,98),(0.35,92.5),(0.7,73.4)]:
    m=tt>=st; u=tt[m]-st; x[m]+=np.tanh(2*(np.sin(2*np.pi*f*u)+0.5*np.sin(4*np.pi*f*u)))*np.exp(-u*(1.2 if st==0.7 else 4))
save("dramatic",x)
# tick-tock / suspense tick
d=0.08; tt=t(d); save("tick",np.sin(2*np.pi*2500*tt)*np.exp(-tt*120))
print("ok")
