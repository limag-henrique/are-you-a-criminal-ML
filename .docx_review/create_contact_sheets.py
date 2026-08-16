from pathlib import Path

from PIL import Image, ImageDraw


root = Path(r"C:\Users\Henrique Lima\AppData\Local\Temp\codex_monografia_render_v2")
pages = sorted(root.glob("page-*.png"))
for sheet_index in range(0, len(pages), 8):
    subset = pages[sheet_index : sheet_index + 8]
    canvas = Image.new("RGB", (1040, 1440), "#d9dde0")
    draw = ImageDraw.Draw(canvas)
    for slot, page in enumerate(subset):
        image = Image.open(page).convert("RGB")
        image.thumbnail((245, 665))
        x = 10 + (slot % 4) * 258
        y = 32 + (slot // 4) * 700
        canvas.paste(image, (x, y))
        draw.text((x, 8 + (slot // 4) * 700), page.stem, fill="black")
    canvas.save(root / f"contact-{sheet_index // 8 + 1}.png")
