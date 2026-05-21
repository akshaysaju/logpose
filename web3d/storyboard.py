"""Compose a single PNG storyboard of the scroll fly-through (reliable image delivery)."""

import sys
from pathlib import Path

from PIL import Image, ImageDraw
from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
PAGE = (HERE / "index.html").as_uri()
CHROME = "/opt/pw-browsers/chromium-1194/chrome-linux/chrome"
OUT = HERE / "storyboard.png"

FRACS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
CW, CH = 760, 460          # per-cell capture size
COLS, PAD = 2, 18
BG = (8, 10, 22)
ACCENT = (94, 234, 212)


def capture():
    imgs = []
    with sync_playwright() as pw:
        b = pw.chromium.launch(executable_path=CHROME,
                               args=["--use-gl=swiftshader", "--no-sandbox"])
        page = b.new_page(viewport={"width": CW, "height": CH}, device_scale_factor=2)
        page.goto(PAGE)
        page.wait_for_timeout(700)
        mx = page.evaluate("document.documentElement.scrollHeight - window.innerHeight")
        for f in FRACS:
            page.evaluate(f"window.scrollTo(0, {mx} * {f})")
            page.wait_for_timeout(1500)
            png = page.screenshot()
            imgs.append((f, Image.open(__import__("io").BytesIO(png)).convert("RGB")))
        b.close()
    return imgs


def main() -> int:
    imgs = capture()
    rows = (len(imgs) + COLS - 1) // COLS
    cell_w = imgs[0][1].width
    cell_h = imgs[0][1].height
    W = COLS * cell_w + (COLS + 1) * PAD
    H = rows * cell_h + (rows + 1) * PAD
    board = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(board)
    for i, (f, im) in enumerate(imgs):
        r, c = divmod(i, COLS)
        x = PAD + c * (cell_w + PAD)
        y = PAD + r * (cell_h + PAD)
        board.paste(im, (x, y))
        draw.rectangle([x, y, x + cell_w - 1, y + cell_h - 1], outline=ACCENT, width=2)
        label = f"SCROLL {int(f * 100)}%"
        draw.rectangle([x + 10, y + 10, x + 10 + 9 * len(label) + 12, y + 34], fill=(4, 6, 14))
        draw.text((x + 18, y + 16), label, fill=ACCENT)
    board.save(OUT)
    print(f"wrote {OUT}  ({OUT.stat().st_size // 1024} KB, {W}x{H})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
