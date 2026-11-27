#!/usr/bin/env python3
"""Render HearME's vector mark and native macOS icon sizes. Requires Pillow."""
import json
from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
BG, WHITE, ACCENT = '#168C82', '#FFFFFF', '#BDF4DD'
# Cubic paths in a 1024-unit design space. Rounded strokes echo the sibling apps.
EAR = [('M', 280, 430), ('C', 280, 316, 350, 240, 454, 240),
       ('C', 564, 240, 630, 318, 630, 420),
       ('C', 630, 514, 590, 551, 532, 606),
       ('C', 501, 636, 512, 700, 463, 746),
       ('C', 421, 788, 344, 770, 328, 708)]
INNER = [('M', 374, 430), ('C', 374, 372, 404, 336, 454, 336),
         ('C', 510, 336, 540, 374, 540, 424),
         ('C', 540, 474, 503, 496, 466, 520),
         ('C', 442, 536, 426, 556, 426, 586)]
WAVE = [('M', 726, 342), ('C', 790, 397, 790, 485, 726, 540)]
PATHS = [(EAR, WHITE, 52), (INNER, WHITE, 48), (WAVE, ACCENT, 48)]
MARK_SCALE = 1.25


def points(path):
    result = []
    for command in path:
        if command[0] == 'M':
            current = (command[1], command[2])
            result.append(current)
        else:
            x0, y0 = current
            x1, y1, x2, y2, x3, y3 = command[1:]
            for i in range(1, 65):
                t = i / 64
                u = 1 - t
                result.append((u**3*x0 + 3*u*u*t*x1 + 3*u*t*t*x2 + t**3*x3,
                               u**3*y0 + 3*u*u*t*y1 + 3*u*t*t*y2 + t**3*y3))
            current = (x3, y3)
    return result


def render(size, inset=0, tile=True):
    factor = 4
    canvas = Image.new('RGBA', (size*factor, size*factor))
    draw = ImageDraw.Draw(canvas)
    scale = (size - inset*2) * factor / 1024
    offset = inset * factor
    if tile:
        draw.rounded_rectangle((offset, offset, size*factor-offset, size*factor-offset),
                               radius=225*scale, fill=BG)
    for path, color, width in PATHS:
        coords = [(offset+(512+(x-512)*MARK_SCALE)*scale,
                   offset+(512+(y-512)*MARK_SCALE)*scale) for x, y in points(path)]
        draw.line(coords, fill=color, width=round(width*scale*MARK_SCALE), joint='curve')
        radius = width*scale*MARK_SCALE/2
        for x, y in coords:
            draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=color)
    return canvas.resize((size, size), Image.Resampling.LANCZOS)


def main():
    brand = ROOT / 'Brand'
    assets = ROOT / 'Resources/Assets.xcassets'
    icons = assets / 'AppIcon.appiconset'
    icons.mkdir(parents=True, exist_ok=True)
    brand.mkdir(parents=True, exist_ok=True)
    info = {'author': 'xcode', 'version': 1}
    (assets / 'Contents.json').write_text(json.dumps({'info': info}, indent=2)+'\n')
    entries = []
    for size in (16, 32, 128, 256, 512):
        for scale in (1, 2):
            pixels = size*scale
            name = f'AppIcon-{size}@{scale}x.png'
            render(pixels, inset=pixels*92/1024).save(icons / name)
            entries.append({'idiom': 'mac', 'size': f'{size}x{size}',
                            'scale': f'{scale}x', 'filename': name})
    (icons / 'Contents.json').write_text(json.dumps({'images': entries, 'info': info}, indent=2)+'\n')
    render(1024).save(brand / 'HearME-1024.png')
    render(1024, inset=92).save(brand / 'HearME-macOS-1024.png')
    paths = '\n'.join(f'<path d="{" ".join(c[0]+" "+" ".join(map(str,c[1:])) for c in path)}" stroke="{color}" stroke-width="{width}"/>'
                      for path, color, width in PATHS)
    (brand / 'HearME.svg').write_text(f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1024 1024">
<rect width="1024" height="1024" rx="225" fill="{BG}"/>
<g fill="none" stroke-linecap="round" stroke-linejoin="round" transform="translate(512 512) scale({MARK_SCALE}) translate(-512 -512)">{paths}</g>
</svg>
''')
    mark = assets / 'BrandMark.imageset'
    mark.mkdir(exist_ok=True)
    render(256).save(mark / 'BrandMark.png')
    (mark / 'Contents.json').write_text(json.dumps({'images': [{'idiom':'universal', 'filename':'BrandMark.png'}], 'info':info}, indent=2)+'\n')

if __name__ == '__main__':
    main()
