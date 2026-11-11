"""Two soft background loops (original, synthesized): 'chill' and 'fun'. 80 s each, 48 kHz stereo."""
import numpy as np, wave
SR=48000; rng=np.random.default_rng(7)
def note(f,d,decay=2.2,harm=(1,0.35,0.12),trem=0.0):
    t=np.arange(int(SR*d))/SR
    x=sum(a*np.sin(2*np.pi*f*(k+1)*t) for k,a in enumerate(harm))
    x*=np.exp(-t*decay)*(1+trem*np.sin(2*np.pi*5*t))
    a=int(0.006*SR); x[:a]*=np.linspace(0,1,a); r=int(0.03*SR); x[-r:]*=np.linspace(1,0,r)
    return x
def midi(n): return 440*2**((n-69)/12)
def add(buf,x,t,g=1.0,pan=0.0):
    i=int(t*SR); n=min(len(x),buf.shape[0]-i)
    if n<=0: return
    buf[i:i+n,0]+=x[:n]*g*(1-max(pan,0)); buf[i:i+n,1]+=x[:n]*g*(1+min(pan,0))
def kick(): 
    t=np.arange(int(SR*0.22))/SR; f=45+70*np.exp(-t*28); return np.sin(2*np.pi*np.cumsum(f)/SR)*np.exp(-t*14)
def hat(d=0.04):
    n=rng.standard_normal(int(SR*d)); n=np.diff(n,prepend=0); return n*np.exp(-np.arange(len(n))/SR*90)
def snare():
    t=np.arange(int(SR*0.16))/SR; n=rng.standard_normal(len(t)); k=np.ones(6)/6; n=np.convolve(n,k,mode="same")
    return (n*0.8+0.4*np.sin(2*np.pi*190*t))*np.exp(-t*24)
def lowpass(x,k):
    ker=np.hanning(k); ker/=ker.sum()
    return np.stack([np.convolve(x[:,c],ker,mode="same") for c in (0,1)],1)
def render(name,bpm,chords,style):
    beat=60/bpm; L=80.0; buf=np.zeros((int(SR*L),2))
    bar=4*beat; nb=int(L/bar)
    for b in range(nb):
        ch=chords[b%len(chords)]; t0=b*bar
        if style=="chill":
            for hit,g in ((0,1.0),(1.5,0.6),(2.5,0.75)):
                for j,n in enumerate(ch[1:]): add(buf,note(midi(n),bar*0.8,2.0,trem=0.03),t0+hit*beat+j*0.012,0.11*g,pan=(j-1.5)*0.18)
            add(buf,note(midi(ch[0]-12),beat*1.6,2.6,(1,0.2)),t0,0.30); add(buf,note(midi(ch[0]-12),beat*1.2,3.0,(1,0.2)),t0+2.5*beat,0.22)
            add(buf,kick(),t0,0.42); add(buf,kick(),t0+2.5*beat,0.30); add(buf,snare(),t0+beat,0.10); add(buf,snare(),t0+3*beat,0.11)
            for e in range(8): add(buf,hat(),t0+e*beat/2+(0.03 if e%2 else 0),0.035 if e%2 else 0.02,pan=0.3)
        else:
            # bouncy plucks: arpeggio in 8ths + off-beat stabs
            arp=[ch[1],ch[2],ch[3],ch[2],ch[1]+12,ch[2],ch[3],ch[2]]
            for e,n in enumerate(arp): add(buf,note(midi(n+12),beat*0.45,7.0,(1,0.5,0.2)),t0+e*beat/2,0.085,pan=0.35 if e%2 else -0.35)
            for hit in (0.5,1.5,2.5,3.5):
                for j,n in enumerate(ch[1:]): add(buf,note(midi(n),beat*0.3,9.0),t0+hit*beat,0.06,pan=(j-1)*0.2)
            add(buf,note(midi(ch[0]-12),beat*0.9,4.0,(1,0.25)),t0,0.30); add(buf,note(midi(ch[0]-12),beat*0.9,4.0,(1,0.25)),t0+2*beat,0.26)
            add(buf,note(midi(ch[0]),beat*0.4,6.0,(1,0.25)),t0+1.5*beat,0.14); add(buf,note(midi(ch[0]-5),beat*0.4,6.0,(1,0.25)),t0+3.5*beat,0.14)
            for k_ in (0,2): add(buf,kick(),t0+k_*beat,0.40)
            for s_ in (1,3): add(buf,snare(),t0+s_*beat,0.11)
            for e in range(8): add(buf,hat(0.03),t0+e*beat/2,0.03 if e%2 else 0.018,pan=-0.3)
    buf=lowpass(buf,9 if style=="chill" else 5)
    buf/=np.abs(buf).max()*1.15
    f=int(SR*0.5); buf[:f]*=np.linspace(0,1,f)[:,None]; buf[-f*2:]*=np.linspace(1,0,f*2)[:,None]
    w=wave.open(f"music/{name}.wav","wb"); w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes((buf*32767).astype(np.int16).tobytes()); w.close()
# chords as [root, third, fifth, seventh] midi notes
render("chill",88,[[53,57,60,64],[52,55,59,62],[50,53,57,60],[48,52,55,59]],"chill")          # Fmaj7 Em7 Dm7 Cmaj7
render("fun",112,[[48,52,55,60],[55,59,62,67],[57,60,64,69],[53,57,60,65]],"fun")              # C G Am F
print("ok")
