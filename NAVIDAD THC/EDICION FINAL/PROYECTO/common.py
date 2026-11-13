import json, subprocess


def stream_info(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", path],
                         capture_output=True, text=True, check=True).stdout
    return json.loads(out)["streams"]


def hdr_prefix(path):
    """Filtros para pasar vídeos HDR del móvil (HLG / PQ, típico de iPhone) a SDR natural."""
    v = next(s for s in stream_info(path) if s["codec_type"] == "video")
    trc = v.get("color_transfer", "")
    if trc in ("arib-std-b67", "smpte2084"):
        return ("zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,"
                "tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,format=yuv420p,")
    return ""
