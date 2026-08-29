#!/usr/bin/env python3
"""Generate deterministic, realistic elevator-button simulation textures."""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = (
    PROJECT_ROOT
    / 'ros2_ws'
    / 'src'
    / 'piper_elevator_gazebo'
    / 'models'
    / 'elevator_button'
)
TEXTURE_ROOT = MODEL_ROOT / 'textures'
BUTTONS = ('1', '2', '3', '4', 'up', 'down', 'open', 'close', 'alarm')
FONT_REGULAR = Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
FONT_BOLD = Path('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf')


def _brushed_steel(size, seed=7):
    """Return a subtle brushed stainless-steel RGB texture."""
    rng = np.random.default_rng(seed)
    height, width = size[1], size[0]
    horizontal = rng.normal(0.0, 5.0, (1, width, 1))
    grain = rng.normal(0.0, 1.4, (height, width, 1))
    vertical_light = np.linspace(-10.0, 12.0, width)[None, :, None]
    base = np.full((height, width, 3), [92.0, 98.0, 103.0])
    tint = np.asarray([0.92, 1.0, 1.06])[None, None, :]
    pixels = base + horizontal * tint + grain + vertical_light
    return Image.fromarray(np.uint8(np.clip(pixels, 0, 255)), 'RGB')


def _font(path, size):
    return ImageFont.truetype(str(path), size=size)


def _center_text(draw, text, box, font, fill, stroke_width=0):
    bounds = draw.textbbox((0, 0), text, font=font, stroke_width=stroke_width)
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    left, top, right, bottom = box
    position = (
        left + (right - left - width) / 2.0 - bounds[0],
        top + (bottom - top - height) / 2.0 - bounds[1],
    )
    draw.text(
        position,
        text,
        font=font,
        fill=fill,
        stroke_width=stroke_width,
        stroke_fill=(15, 17, 18),
    )


def _draw_arrow(draw, direction):
    if direction == 'up':
        points = [(256, 122), (143, 259), (211, 259), (211, 363),
                  (301, 363), (301, 259), (369, 259)]
    else:
        points = [(211, 149), (301, 149), (301, 253), (369, 253),
                  (256, 390), (143, 253), (211, 253)]
    draw.polygon(points, fill=(245, 247, 246))


def _draw_door(draw, opening, ink):
    line = 17
    if opening:
        # This is the real-panel convention learned most reliably by the
        # runtime model: two door leaves with arrows moving away from center.
        draw.line((220, 151, 220, 361), fill=ink, width=18)
        draw.line((292, 151, 292, 361), fill=ink, width=18)
        draw.polygon([(123, 256), (202, 202), (202, 310)], fill=ink)
        draw.polygon([(389, 256), (310, 202), (310, 310)], fill=ink)
        return
    draw.rounded_rectangle((126, 123, 386, 389), radius=8,
                           outline=ink, width=11)
    draw.line((256, 132, 256, 380), fill=ink, width=8)
    draw.line((160, 256, 217, 256), fill=ink, width=line)
    draw.polygon([(243, 256), (201, 221), (201, 291)], fill=ink)
    draw.line((352, 256, 295, 256), fill=ink, width=line)
    draw.polygon([(269, 256), (311, 221), (311, 291)], fill=ink)


def _draw_alarm(draw):
    white = (245, 247, 246)
    draw.arc((150, 125, 362, 351), 190, 350, fill=white, width=20)
    draw.line((151, 264, 126, 330), fill=white, width=20)
    draw.line((361, 264, 386, 330), fill=white, width=20)
    draw.line((126, 330, 386, 330), fill=white, width=20)
    draw.ellipse((229, 344, 283, 398), fill=white)
    draw.ellipse((237, 111, 275, 149), fill=white)


def render_button(label):
    """Render one high-contrast real-elevator-style button face."""
    image = _brushed_steel((512, 512), seed=31 + BUTTONS.index(label))
    image = image.filter(ImageFilter.GaussianBlur(radius=0.45))
    draw = ImageDraw.Draw(image)

    # Layered rings give the button a machined bezel and a real recessed face.
    if label in {'4', 'open'}:
        draw.rounded_rectangle((22, 22, 490, 490), radius=62,
                               fill=(36, 39, 42),
                               outline=(185, 191, 194), width=8)
    else:
        draw.ellipse((22, 22, 490, 490), fill=(36, 39, 42),
                     outline=(185, 191, 194), width=8)
    draw.ellipse((48, 48, 464, 464), fill=(181, 187, 190),
                 outline=(235, 238, 239), width=10)
    light_face = False
    face_fill = (226, 228, 224) if light_face else (21, 24, 27)
    glyph_fill = (18, 20, 22) if light_face else (245, 247, 246)
    draw.ellipse((78, 78, 434, 434), fill=face_fill,
                 outline=(79, 84, 88), width=7)
    highlight = (255, 255, 252) if light_face else (132, 137, 140)
    draw.arc((89, 89, 423, 423), 205, 335, fill=highlight, width=7)

    if label in {'1', '2', '3', '4'}:
        number_font = FONT_BOLD if label == '4' else FONT_REGULAR
        _center_text(
            draw,
            label,
            (86, 74, 426, 430),
            _font(number_font, 245),
            glyph_fill,
            stroke_width=0,
        )
    elif label in {'up', 'down'}:
        _draw_arrow(draw, label)
    elif label in {'open', 'close'}:
        _draw_door(draw, opening=label == 'open', ink=glyph_fill)
    else:
        _draw_alarm(draw)
    return image


def render_panel(button_images):
    """Render an orthographic reference of the complete panel."""
    panel = _brushed_steel((864, 1024), seed=11)
    draw = ImageDraw.Draw(panel)
    draw.rounded_rectangle((92, 38, 772, 182), radius=16,
                           fill=(5, 9, 12), outline=(21, 25, 27), width=7)
    _center_text(
        draw,
        '3',
        (92, 38, 772, 182),
        _font(FONT_REGULAR, 116),
        (255, 52, 31),
    )
    positions = {
        '1': (192, 304), '2': (432, 304), '3': (672, 304),
        '4': (192, 512), 'up': (432, 512), 'down': (672, 512),
        'open': (192, 720), 'close': (432, 720), 'alarm': (672, 720),
    }
    face_size = 166
    for label, center in positions.items():
        face = button_images[label].resize(
            (face_size, face_size), Image.Resampling.LANCZOS
        )
        panel.paste(
            face,
            (center[0] - face_size // 2, center[1] - face_size // 2),
        )
    return panel


def render_home_composite(panel):
    """Perspective-project the panel into the saved home camera image."""
    source = PROJECT_ROOT / 'test_logs' / 'home_panel_named_home_view.png'
    if not source.is_file():
        return None
    home = cv2.imread(str(source), cv2.IMREAD_COLOR)
    texture = cv2.cvtColor(np.asarray(panel), cv2.COLOR_RGB2BGR)
    source_quad = np.float32([
        [0, 0],
        [texture.shape[1] - 1, 0],
        [texture.shape[1] - 1, texture.shape[0] - 1],
        [0, texture.shape[0] - 1],
    ])
    target_quad = np.float32([
        [287, 106],
        [563, 99],
        [580, 438],
        [287, 444],
    ])
    transform = cv2.getPerspectiveTransform(source_quad, target_quad)
    warped = cv2.warpPerspective(
        texture,
        transform,
        (home.shape[1], home.shape[0]),
    )
    mask = cv2.warpPerspective(
        np.full(texture.shape[:2], 255, dtype=np.uint8),
        transform,
        (home.shape[1], home.shape[0]),
    )
    home[mask > 0] = warped[mask > 0]
    output = PROJECT_ROOT / 'test_logs' / 'home_panel_texture_preview.png'
    cv2.imwrite(str(output), home)
    return output


def main():
    """Generate the runtime textures and the home-view regression preview."""
    TEXTURE_ROOT.mkdir(parents=True, exist_ok=True)
    images = {}
    for label in BUTTONS:
        image = render_button(label)
        image.save(TEXTURE_ROOT / f'button_{label}.png', optimize=True)
        images[label] = image
    panel = render_panel(images)
    panel.save(TEXTURE_ROOT / 'panel_reference.png', optimize=True)
    preview = render_home_composite(panel)
    print(f'generated {len(images)} button textures in {TEXTURE_ROOT}')
    if preview is not None:
        print(f'generated detector preview at {preview}')


if __name__ == '__main__':
    main()
