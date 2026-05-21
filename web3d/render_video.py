"""Record a smooth scroll fly-through of index.html to an MP4 (dev/verification only)."""

import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg
from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
PAGE = (HERE / "index.html").as_uri()
CHROME = "/opt/pw-browsers/chromium-1194/chrome-linux/chrome"
OUT_MP4 = HERE / "flythrough.mp4"
W, H = 1280, 720

# Smoothly drive the scroll in-browser so the page's own easing is captured live.
SCROLL_JS = """
() => new Promise(res => {
  const max = document.documentElement.scrollHeight - window.innerHeight;
  const dur = 6500, start = performance.now();
  const ease = t => t < .5 ? 2*t*t : 1 - Math.pow(-2*t+2, 2)/2;
  function step(now){
    const t = Math.min(1, (now - start) / dur);
    window.scrollTo(0, max * ease(t));
    if (t < 1) requestAnimationFrame(step); else setTimeout(res, 900);
  }
  requestAnimationFrame(step);
});
"""


def main() -> int:
    rec_dir = HERE / "_vid"
    rec_dir.mkdir(exist_ok=True)
    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            executable_path=CHROME,
            args=["--use-gl=swiftshader", "--enable-webgl", "--no-sandbox", "--hide-scrollbars"],
        )
        ctx = browser.new_context(
            viewport={"width": W, "height": H},
            record_video_dir=str(rec_dir),
            record_video_size={"width": W, "height": H},
        )
        page = ctx.new_page()
        page.goto(PAGE)
        page.wait_for_timeout(900)          # hold on the opening frame
        page.evaluate(SCROLL_JS)            # smooth fly-through (+ tail hold)
        ctx.close()                         # finalizes the .webm
        browser.close()

    webm = sorted(rec_dir.glob("*.webm"), key=lambda p: p.stat().st_mtime)[-1]
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg, "-y", "-i", str(webm),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
        "-movflags", "+faststart", "-an", str(OUT_MP4),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(res.stderr[-1500:])
        return 1
    print(f"wrote {OUT_MP4}  ({OUT_MP4.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
