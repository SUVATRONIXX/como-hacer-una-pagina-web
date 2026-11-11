"""Vertical short renderer v4.

New in this version
  * speaker-aware framing: faces are detected and tracked (YuNet), mouth movement is measured per face,
    and the top panel zooms onto whoever is talking; with nobody talking it goes back to a normal wide framing
  * dead air (no speech and low volume) is cut out
  * soft background music under the voices
Subtitles (key phrases only, one word at a time), title, punch zooms and the few SFX work as in v3.
"""
import json, os, subprocess, sys, wave
import cv2, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_src = open(os.path.join(HERE, "render3.py")).read()
_ns = {"__file__": os.path.join(HERE, "render3.py"), "__name__": "r3"}
exec(_src[:_src.index("def main():")], _ns)            # reuse ts(), run(), build_ass() and the layout constants
ts, run, build_ass = _ns["ts"], _ns["run"], _ns["build_ass"]
W, H, TOP_H, SEP, BOT_H, SFX, FONTS = (_ns[k] for k in ("W", "H", "TOP_H", "SEP", "BOT_H", "SFX", "FONTS"))
SRC = os.path.join(HERE, "..", "dl", "video720.mp4")
WAV = os.path.join(HERE, "..", "dl", "audio16.wav")
YUNET = os.path.join(HERE, "..", "models", "face_detection_yunet_2023mar.onnx")
FPS = 30; ASP = W / TOP_H; D = 0.30
OPTS = {}                                   # per-clip overrides: exclude_x (never zoom on faces in that x range), wide_cx
SW, SH = 1280, 720


# ---------------------------------------------------------------- audio helpers
def rms_db(a, b, hop=0.05):
    w = wave.open(WAV); sr = w.getframerate(); w.setpos(int(a * sr))
    x = np.frombuffer(w.readframes(int((b - a) * sr)), dtype=np.int16).astype(np.float32) / 32768
    n = int(sr * hop); k = len(x) // n
    return 20 * np.log10(np.sqrt((x[:k * n].reshape(k, n) ** 2).mean(1)) + 1e-6)


def load_words(path, a, b):
    words, lines = [], []
    for l in open(path):
        j = json.loads(l)
        if j["e"] < a or j["s"] > b: continue
        lines.append((max(j["s"], a), min(j["e"], b)))
        words += [(s, e) for s, e, _ in j["w"] if a - 0.2 <= s <= b + 0.2]
    return sorted(words), sorted(lines)


def keep_intervals(a, b, words):
    """Drop stretches with no speech and low volume (dead air). Laughter and shouting stay."""
    hop = 0.05; db = rms_db(a, b, hop); n = len(db)
    speech = np.zeros(n, bool)
    for s, e in words:
        i0, i1 = int(max(s - 0.18 - a, 0) / hop), int(max(e + 0.18 - a, 0) / hop) + 1
        speech[i0:i1] = True
    ref = np.median(db[speech]) if speech.any() else np.median(db)
    active = speech | (db > ref - 9)
    keep = np.ones(n, bool); i = 0
    while i < n:
        if not active[i]:
            j = i
            while j < n and not active[j]: j += 1
            if (j - i) * hop > 0.75:
                m = int(0.22 / hop); keep[i + m:j - m] = False
            i = j
        else: i += 1
    out, i = [], 0
    while i < n:
        if keep[i]:
            j = i
            while j < n and keep[j]: j += 1
            out.append((a + i * hop, min(a + j * hop, b))); i = j
        else: i += 1
    out = [(round(x * FPS) / FPS, round(y * FPS) / FPS) for x, y in out]
    return [(x, y) for x, y in out if y - x >= 0.3] or [(a, b)]


# ---------------------------------------------------------------- face tracks + mouth movement
def analyse(a, b, cache):
    if os.path.exists(cache): return json.load(open(cache))
    det = cv2.FaceDetectorYN.create(YUNET, "", (SW, SH), 0.6, 0.3, 50)
    cap = cv2.VideoCapture(SRC); cap.set(cv2.CAP_PROP_POS_MSEC, a * 1000)
    tracks, live, k = [], [], 0

    def roi(g, cx, cy, w, h):
        x0, y0, x1, y1 = int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2)
        x0, y0 = max(x0, 0), max(y0, 0)
        p = g[y0:y1, x0:x1]
        if p.size < 40: return None
        return cv2.resize(p, (24, 16), interpolation=cv2.INTER_AREA).astype(np.float32)

    while True:
        ok, fr = cap.read()
        t = a + k / FPS; k += 1
        if not ok or t > b: break
        if k % 2: continue                                     # 15 fps is enough
        _, faces = det.detect(fr)
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        used = set()
        for f in (faces if faces is not None else []):
            x, y, w, h = f[:4]
            if w < 34: continue
            cx, cy = x + w / 2, y + h / 2
            mx, my = (f[10] + f[12]) / 2, (f[11] + f[13]) / 2  # mouth corners
            ex, ey = (f[4] + f[6]) / 2, (f[5] + f[7]) / 2      # eyes (control region for camera shake)
            mr, er = roi(g, mx, my + 0.06 * h, 0.55 * w, 0.36 * h), roi(g, ex, ey, 0.55 * w, 0.36 * h)
            best, bd = None, 1e9
            for tr in live:
                if tr["id"] in used or t - tr["t"] > 0.7: continue
                d = np.hypot(cx - tr["cx"], cy - tr["cy"])
                if d < 0.9 * max(w, tr["w"]) and d < bd: best, bd = tr, d
            if best is None:
                best = {"id": len(tracks), "s": [], "pm": None, "pe": None}
                tracks.append(best); live.append(best)
            used.add(best["id"])
            sc = 0.0
            if mr is not None and er is not None and best["pm"] is not None and t - best["t"] < 0.2:
                m = np.abs(mr - mr.mean() - best["pm"]).mean(); e = np.abs(er - er.mean() - best["pe"]).mean()
                sc = float(max(m - 0.8 * e, 0.0))
            if mr is not None and er is not None: best["pm"], best["pe"] = mr - mr.mean(), er - er.mean()
            best.update(t=t, cx=cx, cy=cy, w=w)
            best["s"].append([round(t, 3), round(float(cx), 1), round(float(cy), 1), round(float(h), 1), round(sc, 3)])
    cap.release()
    out = [{"id": tr["id"], "s": tr["s"]} for tr in tracks if len(tr["s"]) >= 8]
    json.dump(out, open(cache, "w"))
    return out


def envelope(a, b):
    """Loudness of the voice track at 30 Hz, to compare against each mouth's movement."""
    w = wave.open(WAV); sr = w.getframerate(); w.setpos(int(a * sr))
    x = np.frombuffer(w.readframes(int((b - a) * sr)), dtype=np.int16).astype(np.float32) / 32768
    hop = sr // 30; k = len(x) // hop
    return a + np.arange(k) / 30, np.sqrt((x[:k * hop].reshape(k, hop) ** 2).mean(1))


def pick_shots(pieces, tracks, lines, words):
    """One unit per spoken line (long ones split at pauses); each unit goes to the face whose mouth moves most."""
    units = []
    for pa, pb in pieces:
        cuts = {pa, pb}
        for s, e in lines:
            if e <= pa or s >= pb: continue
            s, e = max(s, pa), min(e, pb); cuts |= {s, e}
            if e - s > 3.5:                                     # split long lines at the pause nearest each 2.5 s
                for target in np.arange(s + 2.5, e - 1.2, 2.5):
                    gaps = [((w0[1] + w1[0]) / 2) for w0, w1 in zip(words, words[1:])
                            if abs((w0[1] + w1[0]) / 2 - target) < 1.2 and w1[0] - w0[1] > 0.1]
                    cuts.add(min(gaps, key=lambda x: abs(x - target)) if gaps else target)
        cs = sorted(cuts)
        for u0, u1 in zip(cs, cs[1:]):
            if u1 - u0 < 0.05: continue
            talking = any(s < u1 - 0.05 and e > u0 + 0.05 for s, e in words)
            units.append([u0, u1, talking])
    shots, prev = [], None
    et, env = envelope(pieces[0][0] - 0.5, pieces[-1][1] + 0.5)
    for u0, u1, talking in units:
        who = None
        if talking:
            cand = []
            for tr in tracks:
                s = [x for x in tr["s"] if u0 - 0.05 <= x[0] <= u1 + 0.05]
                if len(s) < max(3, 0.45 * (u1 - u0) * 15): continue
                ex = OPTS.get("exclude_x")
                if ex and ex[0] <= float(np.median([x[1] for x in s])) <= ex[1]: continue
                sv = np.array([x[4] for x in s]); ev = np.interp([x[0] for x in s], et, env)
                c = float(np.corrcoef(sv, ev)[0, 1]) if len(s) >= 8 and sv.std() > 0 and ev.std() > 0 else 0.0
                # a mouth that moves in time with the sound is talking; a hand over the face is not
                cand.append((float(sv.mean()) * (1 + 3 * min(max(c, 0.0), 0.6)), tr["id"]))
            if cand:
                cand.sort(reverse=True); sc, who = cand[0]
                pv = [c for c in cand if c[1] == prev]
                if pv and pv[0][0] >= 0.88 * sc: who = prev                 # stay on the same person unless clearly beaten
                elif sc < 0.9: who = prev if pv else None                   # nobody on screen is visibly talking
        if shots and shots[-1][2] == who and abs(shots[-1][1] - u0) < 0.01: shots[-1][1] = u1
        else: shots.append([u0, u1, who])
        prev = who if who is not None else prev
    # no flicker: fold very short shots into a neighbour inside the same piece
    i = 0
    while i < len(shots):
        if shots[i][1] - shots[i][0] < 1.0 and len(shots) > 1:
            if i > 0 and abs(shots[i - 1][1] - shots[i][0]) < 0.01: shots[i - 1][1] = shots[i][1]; shots.pop(i); continue
            if i + 1 < len(shots) and abs(shots[i][1] - shots[i + 1][0]) < 0.01: shots[i + 1][0] = shots[i][0]; shots.pop(i); continue
        i += 1
    merged = []
    for sh in shots:
        if merged and merged[-1][2] == sh[2] and abs(merged[-1][1] - sh[0]) < 0.01: merged[-1][1] = sh[1]
        else: merged.append(sh)
    # never sit on one framing for long: break long shots with a cutaway (wide view, or the most present face)
    out = []
    for t0, t1, who in merged:
        if t1 - t0 <= 6.5: out.append([t0, t1, who]); continue
        alt = None
        if who is None:
            ex = OPTS.get("exclude_x"); ok = lambda tr: not ex or not any(ex[0] <= x[1] <= ex[1] for x in tr["s"] if t0 <= x[0] <= t1)
            best = max([tr for tr in tracks if ok(tr)], key=lambda tr: sum(1 for x in tr["s"] if t0 <= x[0] <= t1), default=None)
            if best is not None and sum(1 for x in best["s"] if t0 <= x[0] <= t1) > 0.5 * (t1 - t0) * 15: alt = best["id"]
            if alt is None: out.append([t0, t1, who]); continue
        t, main = t0, True
        while t < t1 - 0.01:
            step = 4.2 if main else 2.4
            e = t1 if t1 - (t + step) < 1.3 else t + step
            out.append([t, e, who if main else alt]); t = e; main = not main
    return out


def rect_for(t, shot, tracks, state):
    """Crop rectangle (x, y, h) of the source for the top panel at time t."""
    t0, t1, who = shot
    if state.get("shot") is not shot:
        state.clear(); state["shot"] = shot
        s = np.zeros((0, 5))
        if who is not None:
            s = np.array([x for x in next(tr for tr in tracks if tr["id"] == who)["s"] if t0 - 0.3 <= x[0] <= t1 + 0.3]).reshape(-1, 5)
        if len(s) < 3: who = None; shot[2] = None                # the tracked face is not there in this shot: use the normal framing
        if who is not None:
            state["tt"], state["cx"], state["cy"] = s[:, 0], s[:, 1], s[:, 2]
            state["h"] = float(np.clip(2.7 * np.median(s[:, 3]), 400, 640))
        else:
            pts = np.array([x for tr in tracks for x in tr["s"] if t0 - 0.2 <= x[0] <= t1 + 0.2] or [[t0, SW / 2, 300, 0, 0]])
            state["tt"] = np.array([t0, t1]); state["cx"] = np.full(2, OPTS.get("wide_cx", np.median(pts[:, 1]))); state["cy"] = np.full(2, np.median(pts[:, 2]))
            state["h"] = 640.0
        state["sx"] = float(np.interp(min(t0 + 0.2, t1), state["tt"], state["cx"]))
        state["sy"] = float(np.interp(min(t0 + 0.2, t1), state["tt"], state["cy"]))
    al = 0.10
    state["sx"] += al * (float(np.interp(t, state["tt"], state["cx"])) - state["sx"])
    state["sy"] += al * (float(np.interp(t, state["tt"], state["cy"])) - state["sy"])
    dt = min(max(t - t0, 0.0), max(t1 - t0, 0.01)); p = dt / max(t1 - t0, 0.01)
    h = state["h"] * (1 - 0.05 * p) / (1 + 0.07 * np.exp(-dt / 0.12))        # slow push-in + small punch on every cut
    w = h * ASP
    if not np.isfinite([state["sx"], state["sy"], h]).all(): state["sx"], state["sy"], h = SW / 2, 300.0, 640.0
    w = h * ASP
    x = min(max(state["sx"] - w / 2, 0), SW - w)
    y = state["sy"] - (0.40 if who is not None else 0.45) * h
    ymax = SH - h if x + w <= 990 else max(645 - h, 0)                                # never show the watermark corner
    y = min(max(y, 0), ymax)
    return x, y, h


# ---------------------------------------------------------------- main
def main():
    spec = json.load(open(sys.argv[1]))
    OPTS.update(spec.get("opts", {}))
    name = spec["name"]; out_dir = spec.get("out", "out6"); os.makedirs(out_dir, exist_ok=True)
    tmp = f"tmp4_{name}"; os.makedirs(tmp, exist_ok=True)
    groups = []
    for s in spec["segments"]:
        o = s[2] if len(s) > 2 else {}
        if o.get("hard") and groups: groups[-1][1] = s[1]
        else: groups.append([s[0], s[1]])
    grade = f"eq=contrast=1.07:saturation=1.22:gamma={spec.get('gamma', 1.03)}:brightness={spec.get('bright', 0)},unsharp=5:5:0.55"
    allpieces, stats = [], []
    for i, (a, b) in enumerate(groups):
        words, lines = load_words(spec["words"], a, b)
        pieces = [(round(a * FPS) / FPS, round(b * FPS) / FPS)] if (spec.get("no_trim") or i in spec.get("no_trim_groups", [])) else keep_intervals(a, b, words)
        tracks = analyse(a, b, f"{tmp}/an{i}.json")
        shots = pick_shots(pieces, tracks, lines, words)
        allpieces.append(pieces)
        stats.append((round(b - a, 1), round(sum(y - x for x, y in pieces), 1), len(shots), sum(1 for s in shots if s[2] is not None)))
        json.dump(shots, open(f"{tmp}/shots{i}.json", "w"))
        if os.path.exists(f"{tmp}/s{i}.mkv") and "--reuse" in sys.argv: continue
        enc = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS),
                                "-i", "-", "-vf", grade + ",format=yuv420p", "-c:v", "libx264", "-preset", "veryfast", "-crf", "15",
                                f"{tmp}/v{i}.mkv"], stdin=subprocess.PIPE)
        cap = cv2.VideoCapture(SRC); canvas = np.zeros((H, W, 3), np.uint8); state = {}
        for pa, pb in pieces:
            cap.set(cv2.CAP_PROP_POS_MSEC, pa * 1000)
            for k in range(round((pb - pa) * FPS)):
                ok, fr = cap.read()
                if not ok: break
                t = pa + k / FPS
                shot = next((s for s in shots if s[0] - 1e-3 <= t < s[1]), None)
                if shot is None: shot = min(shots, key=lambda s: min(abs(t - s[0]), abs(t - s[1])))
                fx_ = next((f for f in spec.get("fixed", []) if f[0] <= t < f[1]), None)
                if fx_: x, y, h = fx_[2], fx_[3], fx_[4]; state.clear()
                else: x, y, h = rect_for(t, shot, tracks, state)
                sc = TOP_H / h
                M = np.float32([[sc, 0, -x * sc], [0, sc, -y * sc]])
                canvas[:TOP_H] = cv2.warpAffine(fr, M, (W, TOP_H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                canvas[TOP_H + SEP:] = cv2.resize(fr[47:593, 71:1209], (W, BOT_H), interpolation=cv2.INTER_AREA)
                enc.stdin.write(canvas.tobytes())
        cap.release(); enc.stdin.close(); enc.wait()
        # matching audio: the same pieces, with tiny fades so the jump cuts do not click
        af = "".join(f"[0:a]atrim={pa:.4f}:{pb:.4f},asetpts=PTS-STARTPTS,afade=t=in:d=0.012,afade=t=out:st={pb-pa-0.012:.4f}:d=0.012[a{j}];"
                     for j, (pa, pb) in enumerate(pieces))
        af += "".join(f"[a{j}]" for j in range(len(pieces))) + f"concat=n={len(pieces)}:v=0:a=1[a]"
        run(["ffmpeg", "-y", "-v", "error", "-i", SRC, "-filter_complex", af, "-map", "[a]", "-ar", "48000", "-ac", "2", f"{tmp}/a{i}.wav"])
        run(["ffmpeg", "-y", "-v", "error", "-i", f"{tmp}/v{i}.mkv", "-i", f"{tmp}/a{i}.wav", "-c:v", "copy", "-c:a", "pcm_s16le", "-shortest", f"{tmp}/s{i}.mkv"])
    lens = [sum(round((y - x) * FPS) for x, y in p) / FPS for p in allpieces]
    offs, acc = [], 0
    for Ln in lens: offs.append(acc); acc += Ln - D
    total = acc + D
    trans = spec.get("transitions") or ["slideleft", "zoomin", "slideup", "smoothright", "fadewhite", "wipeleft"]
    if len(groups) > 1:
        ins = []; fc = ""; v, a = "[0:v]", "[0:a]"
        for i in range(len(groups)): ins += ["-i", f"{tmp}/s{i}.mkv"]
        for i in range(1, len(groups)):
            fc += f"{v}[{i}:v]xfade=transition={trans[(i-1) % len(trans)]}:duration={D}:offset={offs[i]:.3f}[v{i}];"
            fc += f"{a}[{i}:a]acrossfade=d={D}[a{i}];"
            v, a = f"[v{i}]", f"[a{i}]"
        run(["ffmpeg", "-y", "-v", "error", *ins, "-filter_complex", fc.rstrip(";"), "-map", v, "-map", a,
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "15", "-c:a", "pcm_s16le", f"{tmp}/base.mkv"])
    else:
        run(["ffmpeg", "-y", "-v", "error", "-i", f"{tmp}/s0.mkv", "-c", "copy", f"{tmp}/base.mkv"])

    def mapt(s):
        for i, pieces in enumerate(allpieces):
            if not (groups[i][0] - 0.05 <= s <= groups[i][1] + 0.05): continue
            o = offs[i]
            for pa, pb in pieces:
                if s < pa: return o                             # fell in a removed pause: snap to the next kept frame
                if s <= pb: return o + (s - pa)
                o += round((pb - pa) * FPS) / FPS
            return o
        return None

    zt = ["0.18*clip(1-t/0.25,0,1)"]; shakes, flashes, glitch, rot = [], [], [], []
    SD = os.path.join(HERE, spec.get("sfx_dir", "sfx"))
    sfx = [] if spec.get("no_sfx") else [(spec.get("trans_sfx", "whoosh"), max(offs[i] - 0.12, 0), 0.26) for i in range(1, len(groups))]
    for ev in spec.get("events", []):
        t = mapt(ev["at"])
        if t is None: continue
        d = ev.get("dur", 0.8); fx = ev.get("fx", "zoom")
        if fx in ("zoom", "bigzoom", "shake"):
            amp = {"zoom": 0.10, "bigzoom": 0.22, "shake": 0.08}[fx]
            zt.append(f"{amp}*clip((t-{t:.3f})/0.07,0,1)*clip(({t+d:.3f}-t)/0.2+1,0,1)")
        if fx == "shake": shakes.append((t, t + d))
        if fx == "bigzoom": rot.append(f"0.03*sin((t-{t:.3f})*17)*exp(-max(t-{t:.3f},0)*5)*gte(t,{t:.3f})")
        if ev.get("flash"): flashes.append(t)
        if fx == "glitch" or ev.get("glitch"): glitch.append((t, t + ev.get("gdur", 0.25)))
        if ev.get("sfx") and not spec.get("no_sfx"): sfx.append((ev["sfx"], max(t + ev.get("sfx_off", 0), 0), ev.get("vol", 0.45)))
    z = "1+" + "+".join(zt)
    sh = "+".join(f"between(t,{a:.3f},{b:.3f})" for a, b in shakes) or "0"
    fl = "+".join(f"0.5*exp(-max(t-{t:.3f},0)*9)*gte(t,{t:.3f})" for t in flashes) or "0"
    rt = "+".join(rot) or "0"
    build_ass(spec, mapt, total, f"{tmp}/subs.ass")
    vf = (f"[0:v]split[base][tp];"
          f"[tp]crop={W}:{TOP_H}:0:0,rotate=a='{rt}':fillcolor=black,"
          f"scale=w='trunc({W}*({z})/2)*2':h='trunc({TOP_H}*({z})/2)*2':eval=frame,"
          f"crop={W}:{TOP_H}:x='(iw-{W})/2+({sh})*min((iw-{W})/2,16)*sin(t*67)':y='(ih-{TOP_H})/2+({sh})*min((ih-{TOP_H})/2,16)*cos(t*53)'[tz];"
          f"[base][tz]overlay=0:0,eq=brightness='min({fl},0.55)':eval=frame")
    for a, b in glitch:
        vf += f",rgbashift=rh=-14:bh=14:gv=6:enable='between(t,{a:.3f},{b:.3f})'"
    vf += f",ass={tmp}/subs.ass:fontsdir={FONTS}[v]"
    ins = ["-i", f"{tmp}/base.mkv"]; af = "[0:a]volume=1.0[m];"; mix = ["[m]"]
    for k, (nm, t, vol) in enumerate(sfx):
        ins += ["-i", f"{SD}/{nm}.wav"]; ms = int(t * 1000)
        af += f"[{k+1}:a]volume={vol},adelay={ms}|{ms}[s{k}];"; mix.append(f"[s{k}]")
    # voices + effects are levelled first; the music goes underneath at a fixed low level so it never fights the dialogue
    mi = len(sfx) + 1
    ins += ["-stream_loop", "-1", "-i", os.path.join(HERE, "music", spec.get("music", "chill") + ".wav")]
    mv = spec.get("music_vol", 0.17)
    af += ("".join(mix) + f"amix=inputs={len(mix)}:normalize=0:duration=first,alimiter=limit=0.9:level=0,apad=pad_dur=5,loudnorm=I=-15:TP=-3:LRA=11,atrim=0:{total:.3f},asetpts=PTS-STARTPTS[vo];"
           f"[{mi}:a]atrim=0:{total:.3f},volume={mv},afade=t=in:d=0.6,afade=t=out:st={max(total-1.0,0):.3f}:d=1.0[mu];"
           f"[vo][mu]amix=inputs=2:normalize=0:duration=longest,alimiter=limit=0.7:level=0:attack=2:release=40[a]")
    run(["ffmpeg", "-y", "-v", "error", *ins, "-filter_complex", vf + ";" + af, "-map", "[v]", "-map", "[a]", "-t", f"{total:.3f}",
         "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-profile:v", "high", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-movflags", "+faststart", f"{out_dir}/{name}.mp4"])
    print(f"{out_dir}/{name}.mp4", round(total, 2), "| grupos (antes, después, tomas, tomas a quien habla):", stats, flush=True)


main()
