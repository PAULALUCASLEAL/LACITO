#!/usr/bin/env python3
"""Analiza todos los vídeos de una carpeta.

Por cada vídeo genera:
  - <out>/<nombre>.json : metadatos + estadísticas cada 0.5 s
        (movimiento, brillo, volumen de audio, caras y su posición horizontal)
  - <out>/<nombre>.jpg  : hoja de contactos con fotogramas y marca de tiempo
Y un resumen <out>/summary.json con los mejores momentos por movimiento+audio.

Uso: python analyze.py CARPETA_VIDEOS CARPETA_SALIDA
"""
import json, os, subprocess, sys
import numpy as np
import cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import hdr_prefix

EXTS = (".mp4", ".mov", ".m4v", ".3gp", ".mkv", ".webm", ".avi", ".mts")
STEP = 0.5  # resolución temporal de las estadísticas (s)


def probe(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams",
         "-show_format", path], capture_output=True, text=True, check=True).stdout
    j = json.loads(out)
    v = next(s for s in j["streams"] if s["codec_type"] == "video")
    a = [s for s in j["streams"] if s["codec_type"] == "audio"]
    w, h = int(v["width"]), int(v["height"])
    rot = 0
    for sd in v.get("side_data_list", []):
        if "rotation" in sd:
            rot = int(float(sd["rotation"]))
    if "rotate" in v.get("tags", {}):
        rot = int(v["tags"]["rotate"])
    if abs(rot) % 180 == 90:
        w, h = h, w
    num, den = v.get("avg_frame_rate", "30/1").split("/")
    fps = float(num) / float(den) if float(den) else 30.0
    dur = float(v.get("duration") or j["format"].get("duration") or 0)
    return dict(width=w, height=h, rotation=rot, fps=round(fps, 3),
                duration=round(dur, 3), has_audio=bool(a),
                codec=v.get("codec_name"), size_mb=round(int(j["format"].get("size", 0)) / 1e6, 1))


def frames(path, fps, w, h):
    """Decodifica a baja resolución (respeta la rotación del móvil)."""
    cmd = ["ffmpeg", "-v", "error", "-i", path, "-vf",
           hdr_prefix(path) + f"fps={fps},scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    n = w * h * 3
    while True:
        b = p.stdout.read(n)
        if len(b) < n:
            break
        yield np.frombuffer(b, np.uint8).reshape(h, w, 3)
    p.wait()


def audio_rms(path, dur):
    nbins = int(np.ceil(dur / STEP)) or 1
    try:
        raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vn", "-ac", "1",
                              "-ar", "8000", "-f", "s16le", "-"], capture_output=True, check=True).stdout
    except subprocess.CalledProcessError:
        return [0.0] * nbins
    a = np.frombuffer(raw, np.int16).astype(np.float32) / 32768
    per = int(8000 * STEP)
    res = []
    for i in range(nbins):
        seg = a[i * per:(i + 1) * per]
        res.append(float(20 * np.log10(np.sqrt(np.mean(seg ** 2)) + 1e-6)) if len(seg) else -120.0)
    return res


def analyze(path, outdir):
    name = os.path.splitext(os.path.basename(path))[0]
    meta = probe(path)
    W, H, dur = meta["width"], meta["height"], meta["duration"]
    # baja resolución: lado largo 640
    sc = 640 / max(W, H)
    lw, lh = int(W * sc) // 2 * 2, int(H * sc) // 2 * 2
    sfps = 1 / STEP * 2  # 4 fps para movimiento
    face = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    prof = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")

    stats = []
    prev = None
    thumbs = []
    for i, f in enumerate(frames(path, sfps, lw, lh)):
        t = i / sfps
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        gs = cv2.GaussianBlur(g, (5, 5), 0)
        motion = float(np.mean(cv2.absdiff(gs, prev))) if prev is not None else 0.0
        prev = gs
        rec = dict(t=round(t, 2), motion=round(motion, 2), luma=round(float(g.mean()), 1))
        if i % 2 == 0:  # caras a 2 fps
            fs = list(face.detectMultiScale(g, 1.1, 5, minSize=(int(lh * 0.05),) * 2))
            fs += list(prof.detectMultiScale(g, 1.1, 6, minSize=(int(lh * 0.05),) * 2))
            rec["faces"] = [[round((x + w / 2) / lw, 3), round((y + h / 2) / lh, 3), round(w / lw, 3)]
                            for (x, y, w, h) in fs]
        stats.append(rec)
        thumbs.append((t, f))

    rms = audio_rms(path, dur) if meta["has_audio"] else []
    # agregación cada STEP
    bins = []
    nb = int(np.ceil(dur / STEP)) or 1
    for b in range(nb):
        rs = [s for s in stats if b * STEP <= s["t"] < (b + 1) * STEP]
        if not rs:
            continue
        fc = [fc for s in rs for fc in s.get("faces", [])]
        bins.append(dict(
            t=round(b * STEP, 2),
            motion=round(float(np.mean([s["motion"] for s in rs])), 2),
            luma=round(float(np.mean([s["luma"] for s in rs])), 1),
            audio_db=round(rms[b], 1) if b < len(rms) else None,
            n_faces=len(fc),
            face_x=round(float(np.median([c[0] for c in fc])), 3) if fc else None,
            faces=fc))
    meta["name"] = name
    meta["file"] = os.path.basename(path)
    meta["bins"] = bins
    with open(os.path.join(outdir, name + ".json"), "w") as fh:
        json.dump(meta, fh)

    # hoja de contactos: ≤ 30 miniaturas
    n = len(thumbs)
    k = max(1, int(np.ceil(n / 30)))
    sel = thumbs[::k]
    th = 300
    tw = int(th * lw / lh)
    cols = 6 if tw < th else 4
    rows = int(np.ceil(len(sel) / cols))
    sheet = np.zeros((rows * th, cols * tw, 3), np.uint8)
    for j, (t, f) in enumerate(sel):
        im = cv2.resize(f, (tw, th))
        cv2.rectangle(im, (0, 0), (90, 26), (0, 0, 0), -1)
        cv2.putText(im, f"{t:5.1f}s", (3, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        r, c = divmod(j, cols)
        sheet[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = im
    cv2.putText(sheet, name, (5, sheet.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imwrite(os.path.join(outdir, name + ".jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return meta


def best_windows(meta, win=2.0, top=3):
    bins = meta["bins"]
    if not bins:
        return []
    m = np.array([b["motion"] for b in bins])
    a = np.array([b["audio_db"] if b["audio_db"] is not None else -60 for b in bins])
    f = np.array([min(b["n_faces"], 3) for b in bins])
    mz = (m - m.mean()) / (m.std() + 1e-6)
    az = (a - a.mean()) / (a.std() + 1e-6)
    score = mz + 0.5 * az + 0.3 * f
    k = max(1, int(win / STEP))
    s = np.convolve(score, np.ones(k) / k, mode="valid")
    out, used = [], np.zeros(len(s), bool)
    for idx in np.argsort(-s):
        if used[max(0, idx - k):idx + k].any():
            continue
        used[idx] = True
        out.append(dict(start=bins[idx]["t"], end=round(bins[idx]["t"] + win, 2), score=round(float(s[idx]), 2)))
        if len(out) >= top:
            break
    return out


if __name__ == "__main__":
    src, outdir = sys.argv[1], sys.argv[2]
    os.makedirs(outdir, exist_ok=True)
    files = sorted(f for f in os.listdir(src) if f.lower().endswith(EXTS))
    summary = []
    for f in files:
        try:
            m = analyze(os.path.join(src, f), outdir)
        except Exception as e:  # noqa
            print("ERROR", f, e, file=sys.stderr)
            continue
        bw = best_windows(m)
        summary.append(dict(file=f, w=m["width"], h=m["height"], fps=m["fps"], dur=m["duration"],
                            audio=m["has_audio"], best=bw))
        print(f"{f:40s} {m['width']}x{m['height']} {m['fps']}fps {m['duration']:6.1f}s  best={[(b['start'], b['score']) for b in bw]}", flush=True)
    with open(os.path.join(outdir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
