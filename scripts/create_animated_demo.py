"""Generate high-quality animated GIF terminal demo for PromptDiff using Pillow and Rich."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def create_demo_gif() -> None:
    assets_dir = Path(__file__).resolve().parent.parent / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    demo_png = assets_dir / "demo.png"
    target_gif = assets_dir / "demo.gif"

    # Base image from demo.png if available
    if demo_png.exists():
        base_img = Image.open(demo_png).convert("RGBA")
        width, height = base_img.size

        # Create animated sequence:
        # 1. Typing command
        # 2. Scanning / Progressing
        # 3. Final rich terminal diff view
        frames: list[Image.Image] = []
        durations: list[int] = []

        # Dark terminal background
        bg_color = (15, 23, 42, 255)  # Slate 900

        # Frame 1: Terminal with prompt
        f1 = Image.new("RGBA", (width, height), bg_color)
        draw = ImageDraw.Draw(f1)
        # Window bar dots
        draw.ellipse([20, 18, 34, 32], fill=(239, 68, 68, 255))
        draw.ellipse([42, 18, 56, 32], fill=(245, 158, 11, 255))
        draw.ellipse([64, 18, 78, 32], fill=(16, 185, 129, 255))

        try:
            font = ImageFont.truetype("arial.ttf", 22)
            font_code = ImageFont.truetype("consolas.ttf", 20)
        except Exception:
            font = ImageFont.load_default()
            font_code = font

        draw.text((100, 14), "promptdiff - zsh", fill=(148, 163, 184, 255), font=font)
        draw.text((30, 80), "$ promptdiff run prompts/system_v1.txt prompts/system_v2.txt --inputs testcases.jsonl --forecast 1M", fill=(56, 189, 248, 255), font=font_code)
        frames.append(f1.convert("RGB"))
        durations.append(1200)

        # Frame 2: Progress running
        f2 = f1.copy()
        draw2 = ImageDraw.Draw(f2)
        draw2.text((30, 130), "⚡ Running regression test on 3 test cases...", fill=(203, 213, 225, 255), font=font_code)
        draw2.rectangle([30, 170, 400, 185], fill=(51, 65, 85, 255))
        draw2.rectangle([30, 170, 260, 185], fill=(16, 185, 129, 255))
        draw2.text((420, 166), "66% [2/3] [00:01<00:00]", fill=(148, 163, 184, 255), font=font_code)
        frames.append(f2.convert("RGB"))
        durations.append(1000)

        # Frame 3: Full results
        f3 = base_img.convert("RGB")
        frames.append(f3)
        durations.append(5000) # hold final view

        # Save animated GIF
        frames[0].save(
            target_gif,
            save_all=True,
            append_images=frames[1:],
            optimize=True,
            duration=durations,
            loop=0,
        )
        print(f"[+] Successfully generated animated demo GIF: {target_gif} ({target_gif.stat().st_size / 1024:.1f} KB)")
    else:
        # Create from scratch if demo.png not found
        img = Image.new("RGB", (1200, 750), (15, 23, 42))
        img.save(target_gif)
        print(f"[+] Created starter GIF: {target_gif}")


if __name__ == "__main__":
    create_demo_gif()
