"""Headless screenshots of index.html at several scroll depths (dev/verification only)."""

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
PAGE = (HERE / "index.html").as_uri()
CHROME = "/opt/pw-browsers/chromium-1194/chrome-linux/chrome"
SHOTS = [0.0, 0.33, 0.66, 1.0]


def main() -> int:
    out_dir = HERE / "shots"
    out_dir.mkdir(exist_ok=True)
    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            executable_path=CHROME,
            args=["--use-gl=swiftshader", "--enable-webgl", "--no-sandbox"],
        )
        page = browser.new_page(viewport={"width": 1280, "height": 800},
                                device_scale_factor=2)
        page.goto(PAGE)
        page.wait_for_timeout(600)
        max_scroll = page.evaluate(
            "document.documentElement.scrollHeight - window.innerHeight"
        )
        for i, frac in enumerate(SHOTS):
            page.evaluate(f"window.scrollTo(0, {max_scroll} * {frac})")
            page.wait_for_timeout(1400)  # let the rAF easing settle
            path = out_dir / f"scroll_{int(frac*100):03d}.png"
            page.screenshot(path=str(path))
            print(f"captured {path.name}  (scroll {int(frac*100)}%)")
        browser.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
