#!/usr/bin/env python3
"""Motor de montaje THC: convierte un proyecto JSON (EDL) en un vídeo vertical 1080x1920.

Uso:
  python render.py proyecto.json --src CARPETA_ORIGINALES --analysis CARPETA_ANALISIS \
                   --clips CARPETA_PLANOS --out FINAL.mp4 [--only 03,04] [--preview]

Cada plano ("clip") del proyecto admite:
  src            archivo original
  pieces         lista de tramos, en orden:
                   {"in": s, "out": s, "speed": 1.0}          velocidad constante
                   {"in": s, "out": s, "speed": [1.0, 0.3]}   speed ramp suave
                   {"in": s, "out": s, "speed": 1, "reverse": true}
                   {"freeze": s}                               congela el último fotograma
  cx, cy         centro del encuadre (0-1). cx="auto" sigue las caras detectadas.
                 También lista de claves [[t, valor], ...] (t = segundos de salida del plano).
  zoom           claves [[t, z, curva]]; curva: "linear" | "ease" | "out" | "in" | "hold"
  punch          lista de [t, z_extra, dur]: punch zoom rápido con desenfoque de movimiento
  shake          lista de [t, intensidad_px, dur]: sacudida de cámara breve
  flash          lista de tiempos t: destello blanco breve
  whip_out       "left" | "right": whip pan de salida (el siguiente plano entra con whip_in)
  whip_in        "left" | "right"
  zoom_out_trans true: sale con zoom rápido + blur (el siguiente puede usar zoom_in_trans)
  zoom_in_trans  true: entra desde zoom fuerte con blur
  stab           true: estabiliza (deshake) el plano
  exposure       ganancia manual; si falta se normaliza el brillo automáticamente
  saturation     multiplicador de saturación del plano (por defecto 1)
"""
import argparse, json, math, os, subprocess, sys, tempfile
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import hdr_prefix, stream_info

OW, OH, FPS = 1080, 1920, 30
WHIP_N = 5          # fotogramas de cada mitad del whip pan
ZT_N = 5            # fotogramas de cada mitad de la transición de zoom


# ---------------------------------------------------------------- curvas
def ease(u, kind):
    u = min(max(u, 0.0), 1.0)
    if kind == "linear":
        return u
    if kind == "in":
        return u ** 3
    if kind == "out":
        return 1 - (1 - u) ** 3
    if kind == "hold":
        return 0.0 if u < 1 else 1.0
    return u * u * (3 - 2 * u) if kind == "smooth" else (
        4 * u ** 3 if u < 0.5 else 1 - (-2 * u + 2) ** 3 / 2)  # ease in-out cúbica


def keyed(keys, t, default):
    """Interpola claves [[t, v, curva?], ...]."""
    if keys is None:
        return default
    if isinstance(keys, (int, float)):
        return float(keys)
    if not keys:
        return default
    if t <= keys[0][0]:
        return keys[0][1]
    for a, b in zip(keys, keys[1:]):
        if a[0] <= t <= b[0]:
            kind = b[2] if len(b) > 2 else "ease"
            u = (t - a[0]) / max(b[0] - a[0], 1e-6)
            return a[1] + (b[1] - a[1]) * ease(u, kind)
    return keys[-1][1]


# ---------------------------------------------------------------- tiempo
def time_map(pieces):
    """Devuelve lista de (t_fuente, velocidad, es_freeze) por fotograma de salida."""
    out = []
    last_t = None
    for p in pieces:
        if "freeze" in p:
            n = max(1, round(p["freeze"] * FPS))
            out += [(last_t, 0.0, True)] * n
            continue
        a, b = p["in"], p["out"]
        sp = p.get("speed", 1.0)
        rev = p.get("reverse", False)
        span = b - a
        if isinstance(sp, list):
            s0, s1 = sp
            n = max(1, round(span / ((s0 + s1) / 2) * FPS))
            sp_curve = [s0 + (s1 - s0) * ease((k + 0.5) / n, "ease") for k in range(n)]
            cum = np.cumsum([0] + sp_curve[:-1])
            cum = cum / (cum[-1] + sp_curve[-1]) * span if n > 1 else np.zeros(1)
            ts = [a + c for c in cum]
            sps = sp_curve
        else:
            n = max(1, round(span / sp * FPS))
            ts = [a + k * span / n for k in range(n)]
            sps = [sp] * n
        if rev:
            ts = [b - (t - a) for t in ts]
        out += [(t, s, False) for t, s in zip(ts, sps)]
        last_t = ts[-1]
    return out


# ---------------------------------------------------------------- fuente
class Source:
    def __init__(self, path, t0, t1, need_w, stab=False, slow=False):
        info = stream_info(path)
        v = next(s for s in info if s["codec_type"] == "video")
        num, den = v.get("avg_frame_rate", "30/1").split("/")
        sfps = float(num) / float(den) if float(den) else 30.0
        self.fps = 60.0 if (slow and sfps >= 50) else min(sfps, 30.0)
        w, h = int(v["width"]), int(v["height"])
        rot = 0
        for sd in v.get("side_data_list", []):
            if "rotation" in sd:
                rot = int(float(sd["rotation"]))
        if "rotate" in v.get("tags", {}):
            rot = int(v["tags"]["rotate"])
        if abs(rot) % 180 == 90:
            w, h = h, w
        # ancho de la ventana 9:16 completa en píxeles nativos
        win_w = h * 9 / 16 if w / h > 9 / 16 else w
        sc = min(1.0, need_w / win_w)
        self.W, self.H = int(w * sc) // 2 * 2, int(h * sc) // 2 * 2
        self.t0 = max(0.0, t0 - 0.15)
        dur = t1 - self.t0 + 0.3
        vf = hdr_prefix(path) + (f"deshake=rx=32:ry=32," if stab else "") + \
            f"fps={self.fps},scale={self.W}:{self.H}:flags=lanczos"
        cmd = ["ffmpeg", "-v", "error", "-ss", f"{self.t0:.3f}", "-i", path, "-t", f"{dur:.3f}",
               "-vf", vf, "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
        raw = subprocess.run(cmd, capture_output=True, check=True).stdout
        n = self.W * self.H * 3
        self.frames = [np.frombuffer(raw[i:i + n], np.uint8).reshape(self.H, self.W, 3)
                       for i in range(0, len(raw) - n + 1, n)]
        if not self.frames:
            raise RuntimeError(f"Sin fotogramas: {path} {t0}-{t1}")

    def get(self, t, blend):
        x = (t - self.t0) * self.fps
        i = int(math.floor(x))
        i = min(max(i, 0), len(self.frames) - 1)
        j = min(i + 1, len(self.frames) - 1)
        fr = x - math.floor(x)
        if blend and 0.15 < fr < 0.85 and i != j:
            return cv2.addWeighted(self.frames[i], 1 - fr, self.frames[j], fr, 0)
        return self.frames[j] if fr >= 0.5 else self.frames[i]


def face_track(analysis_dir, src, tmap, default=0.5):
    name = os.path.splitext(src)[0]
    p = os.path.join(analysis_dir or "", name + ".json")
    if not analysis_dir or not os.path.exists(p):
        return [default] * len(tmap)
    bins = json.load(open(p))["bins"]
    pts = [(b["t"] + 0.25, b["face_x"]) for b in bins if b.get("face_x") is not None]
    if not pts:
        return [default] * len(tmap)
    xs = np.array([q[0] for q in pts]); ys = np.array([q[1] for q in pts])
    raw = [float(np.interp(t, xs, ys)) if t is not None else default for (t, _, _) in tmap]
    # suavizado fuerte (cámara virtual estable)
    k = 15
    pad = np.pad(raw, k, mode="edge")
    sm = np.convolve(pad, np.ones(2 * k + 1) / (2 * k + 1), mode="same")[k:-k]
    return list(sm)


# ---------------------------------------------------------------- render de un plano
def render_clip(c, prev_c, next_c, args, out_path):
    tmap = time_map(c["pieces"])
    n = len(tmap)
    real_ts = [t for t, _, _ in tmap if t is not None]
    zmax = 1.0
    zk = c.get("zoom")
    if isinstance(zk, list):
        zmax = max(k[1] for k in zk)
    elif isinstance(zk, (int, float)):
        zmax = zk
    zmax += max([p[1] for p in c.get("punch", [])] or [0])
    need_w = OW * min(max(zmax, 1.0), 1.7)
    slow = any(s < 0.9 for _, s, f in tmap if not f)
    src = Source(os.path.join(args.src, c["src"]), min(real_ts), max(real_ts), need_w,
                 stab=c.get("stab", False), slow=slow)
    W, H = src.W, src.H
    w0, h0 = (H * 9 / 16, H) if W / H > 9 / 16 else (W, W * 16 / 9)

    # exposición automática
    if "exposure" in c:
        gain = c["exposure"]
    else:
        sample = [src.get(t, False) for t in real_ts[::max(1, len(real_ts) // 6)]]
        lum = np.mean([cv2.cvtColor(s, cv2.COLOR_BGR2GRAY).mean() for s in sample])
        gain = float(np.clip(125 / max(lum, 1), 0.92, 1.35))
    gamma = 1 / (1 + (gain - 1) * 0.9) if gain != 1 else 1.0
    lut = np.clip(((np.arange(256) / 255.0) ** gamma) * 255, 0, 255).astype(np.uint8)
    sat = c.get("saturation", 1.0)

    cxs = face_track(args.analysis, c["src"], tmap) if c.get("cx", 0.5) == "auto" else None

    total = n / FPS
    whip_out = c.get("whip_out")
    whip_in = c.get("whip_in")

    def geom(k_f):
        """Parámetros geométricos en el fotograma fraccionario k_f."""
        t = k_f / FPS
        z = keyed(zk, t, 1.0)
        for (pt, pz, pd) in c.get("punch", []):
            if t >= pt:
                u = (t - pt) / max(pd, 1e-3)
                z += pz * ease(u * 3, "out") if u < 1 else pz  # punch: llega rápido y se queda
        if c.get("zoom_out_trans"):
            u = (k_f - (n - ZT_N)) / ZT_N
            if u > 0:
                z *= 1 + 1.2 * ease(u, "in")
        if c.get("zoom_in_trans"):
            u = k_f / ZT_N
            if u < 1:
                z *= 1 + 1.2 * (1 - ease(u, "out"))
        if cxs is not None:
            ki = min(max(int(round(k_f)), 0), n - 1)
            cx = cxs[ki]
        else:
            cx = keyed(c.get("cx", 0.5), t, 0.5)
        cy = keyed(c.get("cy", 0.5), t, 0.5)
        dx = dy = 0.0
        for (st, amp, sd) in c.get("shake", []):
            if st <= t <= st + sd:
                dec = 1 - (t - st) / sd
                dx += amp * dec * math.sin(t * 97.0)
                dy += amp * dec * math.cos(t * 83.0)
        if whip_out:
            u = (k_f - (n - WHIP_N)) / WHIP_N
            if u > 0:
                dx += (-1 if whip_out == "left" else 1) * -OW * 1.3 * ease(u, "in")
        if whip_in:
            u = k_f / WHIP_N
            if u < 1:
                dx += (-1 if whip_in == "left" else 1) * OW * 1.3 * (1 - ease(u, "out"))
        return max(z, 1.0), cx, cy, dx, dy

    def warp(img, z, cx, cy, dx, dy, flags=cv2.INTER_LINEAR):
        w, h = w0 / z, h0 / z
        x0 = min(max(cx * W - w / 2, 0), W - w)
        y0 = min(max(cy * H - h / 2, 0), H - h)
        s = OW / w
        M = np.array([[s, 0, -x0 * s + dx], [0, s, -y0 * s + dy]], np.float32)
        return cv2.warpAffine(img, M, (OW, OH), flags=flags, borderMode=cv2.BORDER_REFLECT)

    enc = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{OW}x{OH}",
           "-r", str(FPS), "-i", "-", "-i", out_path + ".wav", "-map", "0:v", "-map", "1:a",
           "-c:v", "libx264", "-preset", "veryfast" if args.preview else "slow",
           "-crf", "26" if args.preview else "15", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", out_path]
    build_audio(c, tmap, args, out_path + ".wav")
    p = subprocess.Popen(enc, stdin=subprocess.PIPE)
    prev_dx = None
    flashes = c.get("flash", [])
    for k in range(n):
        t_src, sp, frz = tmap[k]
        img = src.get(t_src, blend=(sp < 0.8 and not frz))
        g = geom(k)
        g_prev, g_next = geom(k - 1), geom(k + 1)
        dz = max(abs(g[0] - g_prev[0]), abs(g_next[0] - g[0])) / g[0]
        ddx = max(abs(g[3] - g_prev[3]), abs(g_next[3] - g[3]))
        if dz > 0.025:  # zoom rápido -> desenfoque de movimiento por sub-muestreo
            subs = [warp(img, *geom(k - j / 6)) for j in range(6)]
            fr = np.mean(subs, axis=0).astype(np.uint8)
        else:
            fr = warp(img, *g)
        if ddx > 20:  # whip pan -> desenfoque direccional
            ks = int(min(ddx * 0.9, 401)) | 1
            fr = cv2.blur(fr, (ks, 1))
        if gamma != 1.0:
            fr = cv2.LUT(fr, lut)
        if sat != 1.0:
            hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[..., 1] = np.clip(hsv[..., 1] * sat, 0, 255)
            fr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        t = k / FPS
        for ft in flashes:
            d = t - ft
            if 0 <= d < 0.15:
                a = 0.85 * (1 - d / 0.15) ** 1.5
                fr = cv2.addWeighted(fr, 1 - a, np.full_like(fr, 255), a, 0)
        p.stdin.write(fr.tobytes())
    p.stdin.close()
    p.wait()
    os.remove(out_path + ".wav")
    import shutil; [shutil.rmtree(os.path.join(os.path.dirname(os.path.abspath(out_path)), d), ignore_errors=True) for d in os.listdir(os.path.dirname(os.path.abspath(out_path))) if d.startswith("tmp")]
    return n


def atempo_chain(s):
    f = []
    while s > 2.0:
        f.append("atempo=2.0"); s /= 2.0
    while s < 0.5:
        f.append("atempo=0.5"); s /= 0.5
    f.append(f"atempo={s:.5f}")
    return ",".join(f)


def build_audio(c, tmap, args, wav):
    """Audio original del plano siguiendo los cambios de velocidad (sin sonidos añadidos)."""
    path = os.path.join(args.src, c["src"])
    has_a = any(s["codec_type"] == "audio" for s in stream_info(path))
    tmp = tempfile.mkdtemp(dir=os.path.dirname(os.path.abspath(wav)))
    parts = []
    for i, pc in enumerate(c["pieces"]):
        o = os.path.join(tmp, f"{i}.wav")
        if "freeze" in pc or not has_a:
            d = pc["freeze"] if "freeze" in pc else None
            if d is None:
                sp = pc.get("speed", 1)
                avg = (sp[0] + sp[1]) / 2 if isinstance(sp, list) else sp
                d = (pc["out"] - pc["in"]) / avg
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                            "anullsrc=r=48000:cl=stereo", "-t", f"{d:.4f}", o], check=True)
        else:
            sp = pc.get("speed", 1)
            avg = (sp[0] + sp[1]) / 2 if isinstance(sp, list) else sp
            af = []
            if pc.get("reverse"):
                af.append("areverse")
            if abs(avg - 1) > 1e-3:
                af.append(atempo_chain(avg))
            af.append("afade=t=in:d=0.01")
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{pc['in']:.4f}", "-i", path,
                            "-t", f"{pc['out'] - pc['in']:.4f}", "-map", "0:a:0", "-vn", "-ac", "2", "-ar", "48000",
                            "-af", ",".join(af), o], check=True)
        parts.append(o)
    lst = os.path.join(tmp, "l.txt")
    open(lst, "w").write("".join(f"file '{q}'\n" for q in parts))
    dur = len(tmap) / FPS
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst,
                    "-af", f"apad,atrim=0:{dur:.4f},afade=t=out:st={max(dur - 0.02, 0):.4f}:d=0.02",
                    "-ar", "48000", "-ac", "2", wav], check=True)


# ---------------------------------------------------------------- principal
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("project")
    ap.add_argument("--src", required=True)
    ap.add_argument("--analysis")
    ap.add_argument("--clips", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only")
    ap.add_argument("--preview", action="store_true")
    args = ap.parse_args()
    proj = json.load(open(args.project))
    clips = proj["clips"]
    os.makedirs(args.clips, exist_ok=True)
    # enlaza automáticamente whips / transiciones de zoom
    for a, b in zip(clips, clips[1:]):
        if a.get("whip_out") and not b.get("whip_in"):
            b["whip_in"] = a["whip_out"]
        if a.get("zoom_out_trans") and "zoom_in_trans" not in b:
            b["zoom_in_trans"] = True
    only = set(args.only.split(",")) if args.only else None
    files, total = [], 0
    for i, c in enumerate(clips):
        cid = c.get("id", f"{i + 1:02d}")
        f = os.path.join(args.clips, f"{cid}.mp4")
        files.append(f)
        if only and cid not in only:
            continue
        n = render_clip(c, clips[i - 1] if i else None, clips[i + 1] if i + 1 < len(clips) else None, args, f)
        total += n
        print(f"plano {cid}: {n / FPS:.2f}s  ({c['src']})", flush=True)
    if only:
        return
    lst = os.path.join(args.clips, "concat.txt")
    open(lst, "w").write("".join(f"file '{os.path.abspath(q)}'\n" for q in files))
    g = proj.get("grade", {})
    vf = (f"eq=contrast={g.get('contrast', 1.04)}:saturation={g.get('saturation', 1.08)}:"
          f"gamma={g.get('gamma', 1.0)},unsharp=5:5:0.35:5:5:0,format=yuv420p")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst,
                    "-vf", vf, "-r", str(FPS), "-c:v", "libx264", "-profile:v", "high", "-level", "4.2",
                    "-preset", "veryfast" if args.preview else "slow", "-crf", "24" if args.preview else "17",
                    "-maxrate", "18M", "-bufsize", "36M", "-c:a", "aac", "-b:a", "192k",
                    "-af", "aresample=async=1", "-movflags", "+faststart", args.out], check=True)
    print("FINAL:", args.out)


if __name__ == "__main__":
    main()
