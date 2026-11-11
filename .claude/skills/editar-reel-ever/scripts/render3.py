"""Vertical short renderer v2 (1080x1920).

Layout: zoomed, face-centred top panel + thin black line + full-width strip at the bottom.
Subtitles: lowercase Poppins Bold, white with a soft shadow, 1-2 words at a time, centred,
with pop-in / fade-out animation.

Spec (json): name, title (use *word* to highlight), segments [[a,b] or [a,b,{"x":px}]],
transitions, events [{at, fx, dur, sfx, vol, flash, glitch, text}], emph [words], fix {}, words
"""
import json, os, re, subprocess, sys
import cv2, numpy as np

W, H = 1080, 1920
TOP_H, SEP, BOT_H = 1396, 6, 518
CW = 495                     # source crop width for the top panel (640 * 1080/1396); rows below 640 hold the watermark
CH = 640
D = 0.30                     # transition length
SRC = "../dl/video720.mp4"
HERE = os.path.dirname(os.path.abspath(__file__))
SFX = os.path.join(HERE, "sfx")
FONTS = os.path.join(HERE, "..", "fonts", "ttf")
SUB_Y = 1000
YEL = "&H0000D4FF&"


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        print(r.stderr[-3000:]); raise SystemExit(1)


def ts(t):
    t = max(t, 0); return f"{int(t//3600)}:{int(t%3600//60):02d}:{t%60:05.2f}"


_casc = [cv2.CascadeClassifier(cv2.data.haarcascades + n) for n in
         ("haarcascade_frontalface_default.xml", "haarcascade_profileface.xml")]


def face_x(a, b):
    """Pick the top-panel crop offset that keeps the most face area fully in frame."""
    cap = cv2.VideoCapture(SRC); dets = []
    t = a + 0.3
    while t < b:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000); ok, fr = cap.read()
        if not ok: break
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        for c in _casc:
            for (x, y, w, h) in c.detectMultiScale(g, 1.15, 6, minSize=(70, 70)):
                dets.append((x, x + w, w * h, y + h / 2))
        t += max((b - a) / 14, 0.5)
    cap.release()
    center = (1280 - CW) // 2
    if len(dets) < 3: return center, None
    best, bx = -1, center
    for x0 in range(0, 1280 - CW + 1, 12):
        s = sum(ar for l, r, ar, _ in dets if l >= x0 - 10 and r <= x0 + CW + 10)
        s -= abs(x0 - center) * 6          # mild preference for the centre
        if s > best: best, bx = s, x0
    ys = [cy for l, r, ar, cy in dets if l >= bx - 10 and r <= bx + CW + 10]
    return bx, (float(np.median(ys)) if ys else None)


def build_ass(spec, mapt, total, path):
    head = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {W}
PlayResY: {H}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Sub,Poppins,{spec.get('fs', 132)},&H00FFFFFF,&H00FFFFFF,&H00101010,&H00000000,-1,0,0,0,100,100,-1.5,0,1,0,0,5,40,40,0,1
Style: Title,Poppins,84,&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,-1,0,0,0,100,100,-0.5,0,1,5,0,8,60,60,120,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    L = []
    # persistent title at the top, pops in
    if spec.get("title"):
        t = re.sub(r"\*(.+?)\*", lambda m: "{\\c" + YEL + "}" + m.group(1) + "{\\c&HFFFFFF&}", spec["title"]).replace("|", "\\N")
        anim = "\\fscx60\\fscy60\\t(0,110,\\fscx106\\fscy106)\\t(110,190,\\fscx100\\fscy100)\\fad(60,120)"
        L.append(f"Dialogue: 2,{ts(0.12)},{ts(total)},Title,,0,0,0,,{{\\blur0.6{anim}}}{t}")
    words = []
    for l in open(spec["words"]):
        for s, e, w in json.loads(l)["w"]:
            o = mapt(s)
            if o is None: continue
            w = w.strip().rstrip(".,")
            if w: words.append([o, mapt(e) or o + 0.3, w])
    words.sort()
    fixes = spec.get("fix", {})
    out = []
    for w in words:
        k = w[2].lower().strip(".,¿?¡!")
        if k in fixes:
            if fixes[k] == "": continue
            w[2] = fixes[k]
        out.append(w)
    words = out
    emph = {e.lower() for e in spec.get("emph", [])}
    ranges = [(mapt(a), mapt(b)) for a, b in spec.get("subs", [])]
    ranges = [(a, b) for a, b in ranges if a is not None and b is not None]
    shown = [w for w in words if any(a - 0.05 <= w[0] <= b for a, b in ranges)] if "subs" in spec else words
    for i, w in enumerate(shown):
        st = w[0]
        nxt = shown[i + 1][0] if i + 1 < len(shown) else total
        en = min(max(w[1], st + 0.16), nxt) if nxt - w[1] < 0.3 else min(max(w[1] + 0.12, st + 0.2), nxt)
        en = min(nxt, max(en, st + 0.12)) if nxt > st else st + 0.2
        if en - st < 0.06: continue
        dur = int((en - st) * 1000)
        txt = w[2].lower()
        hot = w[2].lower().strip(".,¿?¡!") in emph
        col = YEL if hot else "&H00FFFFFF&"; big = 114 if hot else 100
        o = max(dur - 60, 0)
        anim = (f"\\pos({W//2},{SUB_Y})\\fad(30,50)\\fscx72\\fscy72\\t(0,70,\\fscx{big+8}\\fscy{big+8})"
                f"\\t(70,130,\\fscx{big}\\fscy{big})\\t({o},{dur},\\fscx{big-8}\\fscy{big-8})")
        L.append(f"Dialogue: 0,{ts(st)},{ts(en)},Sub,,0,0,0,,{{{anim}\\1a&HFF&\\3c&H000000&\\3a&H30&\\bord9\\blur9}}{txt}")
        L.append(f"Dialogue: 1,{ts(st)},{ts(en)},Sub,,0,0,0,,{{{anim}\\c{col}\\3c&H000000&\\3a&H40&\\bord2\\blur1.2\\shad3\\4a&H70&}}{txt}")
    for ev in spec.get("events", []):
        if not ev.get("text"): continue
        t = mapt(ev["at"])
        if t is None: continue
        d = ev.get("tdur", 1.2)
        tag = (f"\\pos({W//2},{ev.get('ty', 520)})\\fs{ev.get('fs', 170)}\\frz{ev.get('rot', -5)}\\fad(40,160)"
               f"\\fscx30\\fscy30\\t(0,100,\\fscx116\\fscy116)\\t(100,180,\\fscx100\\fscy100)")
        L.append(f"Dialogue: 3,{ts(t)},{ts(t+d)},Sub,,0,0,0,,{{{tag}\\1a&HFF&\\3c&H000000&\\3a&H20&\\bord10\\blur8}}{ev['text']}")
        L.append(f"Dialogue: 4,{ts(t)},{ts(t+d)},Sub,,0,0,0,,{{{tag}\\c{ev.get('tcol', YEL)}\\3c&H000000&\\bord3\\blur1}}{ev['text']}")
    open(path, "w").write(head + "\n".join(L) + "\n")


def main():
    spec = json.load(open(sys.argv[1]))
    name = spec["name"]; out_dir = spec.get("out", "out2"); os.makedirs(out_dir, exist_ok=True)
    tmp = f"tmp2_{name}"; os.makedirs(tmp, exist_ok=True)
    raw = [(s[0], s[1], s[2] if len(s) > 2 else {}) for s in spec["segments"]]
    groups = []                                   # parts flagged "hard" are hard-cut onto the previous one (same take, new framing)
    for r in raw:
        if r[2].get("hard") and groups: groups[-1].append(r)
        else: groups.append([r])
    segs = []; cutpops = []; parts = []
    reuse = "--reuse" in sys.argv and os.path.exists(f"{tmp}/base.mkv")
    for i, g in enumerate(groups):
        a0 = g[0][0]; snap = lambda t: a0 + round((t - a0) * 30) / 30
        files = []
        for j, (a, b, o) in enumerate(g):
            a, b = snap(a), snap(b)
            parts.append((i, a - a0, b - a0))
            if reuse: continue
            if j and o.get("z", 1.0) > 1.05: cutpops.append(a)
            x = o.get("x", None); fy = None
            if x is None: x, fy = face_x(a, b)
            z = o.get("z", 1.0)                       # extra zoom on the top panel
            cw, chh = int(CW / z), int(CH / z)
            x = int(min(max(x + (CW - cw) / 2, 0), 1280 - cw))
            # the watermark sits at x>990, y>640: crops left of it may use the full frame height
            ymax = (720 - chh) if x + cw <= 990 else max(CH - chh, 0)
            if "y" in o: y = o["y"]
            elif fy is not None: y = fy - chh * 0.40      # face a bit above the middle of the panel
            else: y = ymax                                 # no face found: handheld vlog shots tilt up, so favour the lower part
            y = int(min(max(y, 0), ymax))
            fc = (f"[0:v]eq=contrast=1.07:saturation=1.22:gamma={spec.get('gamma', 1.03)}:brightness={spec.get('bright', 0)},unsharp=5:5:0.55,split[a][b];"
                  f"[a]crop={cw}:{chh}:{x}:{y},scale={W}:{TOP_H}:flags=lanczos,setsar=1,pad={W}:{TOP_H+SEP}:0:0:black[t];"
                  f"[b]crop=1138:546:71:{o.get('by', 47)},scale={W}:{BOT_H}:flags=lanczos,setsar=1[d];"
                  f"[t][d]vstack,fps=30,format=yuv420p[v]")
            fn = f"{tmp}/p{i}_{j}.mkv"; files.append(fn)
            run(["ffmpeg", "-y", "-v", "error", "-ss", f"{a:.4f}", "-t", f"{b - a:.4f}", "-i", SRC, "-filter_complex", fc,
                 "-map", "[v]", "-map", "0:a", "-c:v", "libx264", "-preset", "veryfast", "-crf", "15",
                 "-c:a", "pcm_s16le", "-ar", "48000", "-ac", "2", fn])
        if reuse: pass
        elif len(files) == 1: os.replace(files[0], f"{tmp}/s{i}.mkv")
        else:
            open(f"{tmp}/l{i}.txt", "w").write("".join(f"file '{os.path.abspath(f)}'\n" for f in files))
            run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", f"{tmp}/l{i}.txt", "-c", "copy", f"{tmp}/s{i}.mkv"])
        segs.append((g[0][0], g[-1][1], {}))
    lens = [b - a for a, b, _ in segs]
    offs, acc = [], 0
    for Ln in lens:
        offs.append(acc); acc += Ln - D
    total = acc + D
    trans = spec.get("transitions") or ["slideleft", "zoomin", "slideup", "smoothright", "fadewhite", "wipeleft"]
    if reuse: pass
    elif len(segs) > 1:
        ins = []; fc = ""; v, a = "[0:v]", "[0:a]"
        for i in range(len(segs)): ins += ["-i", f"{tmp}/s{i}.mkv"]
        for i in range(1, len(segs)):
            fc += f"{v}[{i}:v]xfade=transition={trans[(i-1) % len(trans)]}:duration={D}:offset={offs[i]:.3f}[v{i}];"
            fc += f"{a}[{i}:a]acrossfade=d={D}[a{i}];"
            v, a = f"[v{i}]", f"[a{i}]"
        run(["ffmpeg", "-y", "-v", "error", *ins, "-filter_complex", fc.rstrip(";"), "-map", v, "-map", a,
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "15", "-c:a", "pcm_s16le", f"{tmp}/base.mkv"])
    else:
        os.replace(f"{tmp}/s0.mkv", f"{tmp}/base.mkv")

    def mapt(s):
        for i, (a, b, _) in enumerate(segs):
            if a - 0.05 <= s <= b + 0.05:
                return offs[i] + min(max(s - a, 0), b - a)
        return None

    # top-panel motion: intro snap, slow push-in per segment, punch zooms, shakes
    zt = ["0.20*clip(1-t/0.25,0,1)"]; pan = []; tilt = []
    # every shot gets its own move: push-in, pan, pull-out, pan back, fast push with ease-out, diagonal drift
    for k, (gi, pa, pb) in enumerate(parts):
        ta, tb = offs[gi] + pa, offs[gi] + pb
        if tb - ta < 0.4: continue
        p = f"clip((t-{ta:.3f})/{tb-ta:.3f},0,1)"; on = f"between(t,{ta:.3f},{tb-0.001:.3f})"
        ease = f"(1-pow(1-{p},3))"; sm = f"({p}*{p}*(3-2*{p}))"
        mode = (k + spec.get("motion_seed", 0)) % 6
        if mode == 0: zt.append(f"{on}*(0.02+0.13*{sm})")
        elif mode == 1: zt.append(f"{on}*0.18"); pan.append(f"{on}*(-0.85+1.7*{sm})")
        elif mode == 2: zt.append(f"{on}*(0.17-0.14*{sm})")
        elif mode == 3: zt.append(f"{on}*0.18"); pan.append(f"{on}*(0.85-1.7*{sm})")
        elif mode == 4: zt.append(f"{on}*(0.16*{ease})")
        else: zt.append(f"{on}*(0.10+0.06*{sm})"); pan.append(f"{on}*(0.6-1.2*{sm})"); tilt.append(f"{on}*(-0.7+1.4*{sm})")
    shakes, flashes, glitch = [], [], []
    rot = []
    sfx = [("whoosh", max(offs[i] - 0.12, 0), 0.28) for i in range(1, len(segs))]
    for ev in spec.get("events", []):
        t = mapt(ev["at"])
        if t is None: print("event outside segments", ev); continue
        d = ev.get("dur", 0.8); fx = ev.get("fx", "zoom")
        if fx in ("zoom", "bigzoom", "shake"):
            amp = {"zoom": 0.13, "bigzoom": 0.28, "shake": 0.09}[fx] * ev.get("amp", 1)
            zt.append(f"{amp}*clip((t-{t:.3f})/0.07,0,1)*clip(({t+d:.3f}-t)/0.2+1,0,1)")
        if fx == "shake": shakes.append((t, t + d))
        if fx == "bigzoom": rot.append(f"0.03*sin((t-{t:.3f})*17)*exp(-(t-{t:.3f})*5)*gte(t,{t:.3f})")
        if ev.get("flash"): flashes.append(t)
        if fx == "glitch" or ev.get("glitch"): glitch.append((t, t + ev.get("gdur", 0.25)))
        if ev.get("sfx"): sfx.append((ev["sfx"], max(t + ev.get("sfx_off", 0), 0), ev.get("vol", 0.55)))
    z = "1+" + "+".join(zt)
    sh = "+".join(f"between(t,{a:.3f},{b:.3f})" for a, b in shakes) or "0"
    fl = "+".join(f"0.5*exp(-(t-{t:.3f})*9)*gte(t,{t:.3f})" for t in flashes) or "0"
    build_ass(spec, mapt, total, f"{tmp}/subs.ass")
    pn = "+".join(pan) or "0"; tl = "+".join(tilt) or "0"; rt = "+".join(rot) or "0"
    vf = (f"[0:v]split[base][tp];"
          f"[tp]crop={W}:{TOP_H}:0:0,rotate=a='{rt}':fillcolor=black,"
          f"scale=w='trunc({W}*({z})/2)*2':h='trunc({TOP_H}*({z})/2)*2':eval=frame,"
          f"crop={W}:{TOP_H}:x='(iw-{W})/2*(1+clip({pn},-1,1))+({sh})*min((iw-{W})/2,16)*sin(t*67)'"
          f":y='(ih-{TOP_H})/2*(1+clip({tl},-1,1))+({sh})*min((ih-{TOP_H})/2,16)*cos(t*53)'[tz];"
          f"[base][tz]overlay=0:0,eq=brightness='min({fl},0.55)':eval=frame")
    for a, b in glitch:
        vf += f",rgbashift=rh=-14:bh=14:gv=6:enable='between(t,{a:.3f},{b:.3f})'"
    vf += f",ass={tmp}/subs.ass:fontsdir={FONTS}[v]"
    ins = ["-i", f"{tmp}/base.mkv"]; af = "[0:a]volume=1.0[m];"; mix = ["[m]"]
    for k, (nm, t, vol) in enumerate(sfx):
        ins += ["-i", f"{SFX}/{nm}.wav"]; ms = int(t * 1000)
        af += f"[{k+1}:a]volume={vol},adelay={ms}|{ms}[s{k}];"; mix.append(f"[s{k}]")
    af += "".join(mix) + f"amix=inputs={len(mix)}:normalize=0:duration=first,alimiter=limit=0.95,loudnorm=I=-14:TP=-1.5:LRA=11[a]"
    run(["ffmpeg", "-y", "-v", "error", *ins, "-filter_complex", vf + ";" + af, "-map", "[v]", "-map", "[a]",
         "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-profile:v", "high", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-movflags", "+faststart", f"{out_dir}/{name}.mp4"])
    print(f"{out_dir}/{name}.mp4", round(total, 2), flush=True)


main()
