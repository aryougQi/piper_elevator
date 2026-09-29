#!/usr/bin/env python3
"""Generate deterministic, realistic elevator-button simulation textures.

The layout mirrors the reference cabin panel photo:

    [ bell ]  [ handset ]     alarm / intercom service pair
                [  3  ]
                [  2  ]
                [  1  ]
    [ open ]  [ close ]       door pair
                [  ^  ]
                [  v  ]

Every control is a black machined tile with a recessed near-black face, so the
button texture also carries the bezel that the reference photo shows. Run on
the host (needs Pillow/numpy); outputs are committed so the container never
renders textures at launch time.
"""

from pathlib import Path

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
WALL_MODEL_ROOT = (
    PROJECT_ROOT
    / 'ros2_ws'
    / 'src'
    / 'piper_elevator_gazebo'
    / 'models'
    / 'cabin_wall'
)
WALL_TEXTURE_ROOT = WALL_MODEL_ROOT / 'textures'

# Reading order of the reference panel, top row first.
BUTTONS = (
    'alarm', 'intercom', '3', '2', '1', 'open', 'close', 'up', 'down',
)
FONT_REGULAR = Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
FONT_BOLD = Path('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf')

# Panel geometry, kept in sync with models/elevator_button/model.sdf.
#
# Two measured constraints set these numbers:
#   1. the home camera sees z = 0.150..0.570 m at the panel plane, so the
#      plate top must stay at or below 0.570 m;
#   2. the coarse approach leaves less joint3 margin the lower a control sits
#      (measured: 0.729 rad for `3` down to 0.342 rad for `up`, with `down`
#      failing outright), so the lowest control should be as high as possible.
# 0.28 m tall at a 0.43 m centre spans 0.290..0.570 m: the whole plate is in
# frame and `down` sits at 0.313 m instead of 0.269 m.
#
# The physical control faces stay at 0.033 m for the press collision and the
# detector's field of view.  The visible tile is drawn with the smaller ratio
# from the reference photo, while the surrounding texture matches the steel
# plate instead of producing a grey square around every button.
# Seven 0.033 m rows need 0.231 m and six 6 mm gaps need 0.036 m: that fits
# the 0.28 m plate with 6.5 mm top and bottom margins, so row centres are
# uniform at +-0.117 / +-0.078 / +-0.039 / 0.
# The plate width/height ratio stays 0.467, taken from the photo.
PANEL_WIDTH_M = 0.1308
PANEL_HEIGHT_M = 0.280
BUTTON_FACE_M = 0.033
PAIR_OFFSET_M = 0.0298
BUTTON_LOCAL_Z = {
    'alarm': 0.117,
    'intercom': 0.117,
    '3': 0.078,
    '2': 0.039,
    '1': 0.0,
    'open': -0.039,
    'close': -0.039,
    'up': -0.078,
    'down': -0.117,
}

# Button face palette taken from the photo.
TILE_FILL = (27, 28, 30)
TILE_EDGE_LIGHT = (104, 107, 110)
TILE_EDGE_DARK = (9, 9, 10)
FACE_FILL = (11, 11, 12)
FACE_EDGE = (62, 64, 67)
INK = (238, 239, 241)
INK_SOFT_OUTLINE = (92, 94, 97)
BELL_INK = (250, 198, 40)
OPEN_INK = (56, 166, 62)


def _brushed_steel(size, seed=7, base=(96.0, 101.0, 106.0), grain=1.4):
    """Return a subtle brushed stainless-steel RGB texture."""
    rng = np.random.default_rng(seed)
    height, width = size[1], size[0]
    horizontal = rng.normal(0.0, 4.5, (1, width, 1))
    noise = rng.normal(0.0, grain, (height, width, 1))
    vertical_light = np.linspace(-9.0, 11.0, width)[None, :, None]
    tint = np.asarray([0.93, 1.0, 1.05])[None, None, :]
    pixels = np.asarray(base)[None, None, :] + horizontal * tint + noise
    pixels = pixels + vertical_light
    return Image.fromarray(np.uint8(np.clip(pixels, 0, 255)), 'RGB')


def _render_marble(size=(1024, 1024), seed=5):
    """Return a mid-grey marble slab with soft light veins.

    Gazebo has no image-based lighting here, so diffuse albedo is the only
    thing the camera sees: a near-black base renders as a black wall.  Keep
    the slab around mid grey and blur the veins so it reads as polished stone
    instead of speckled noise.
    """
    rng = np.random.default_rng(seed)
    width, height = size

    coarse = rng.normal(0.0, 1.0, (height // 16, width // 16))
    coarse = np.uint8(np.clip(128.0 + coarse * 44.0, 0, 255))
    cloud = np.asarray(
        Image.fromarray(coarse, 'L').resize(size, Image.Resampling.BICUBIC),
        dtype=np.float64,
    )
    base = 104.0 + (cloud - 128.0) * 0.18
    grain = rng.normal(0.0, 1.1, (height, width))
    pixels = np.stack(
        [base * 0.98 + grain, base + grain, base * 1.03 + grain],
        axis=-1,
    )
    image = Image.fromarray(np.uint8(np.clip(pixels, 0, 255)), 'RGB')

    veins = Image.new('L', size, 0)
    vein_draw = ImageDraw.Draw(veins)
    for _ in range(16):
        x = float(rng.uniform(0.0, width))
        y = float(rng.uniform(0.0, height))
        heading = float(rng.uniform(0.0, 2.0 * np.pi))
        stroke = int(rng.integers(2, 5))
        for _step in range(int(rng.integers(50, 110))):
            heading += float(rng.normal(0.0, 0.30))
            nx = x + float(np.cos(heading)) * float(rng.uniform(8.0, 24.0))
            ny = y + float(np.sin(heading)) * float(rng.uniform(8.0, 24.0))
            vein_draw.line(
                (x, y, nx, ny),
                fill=int(rng.integers(120, 210)),
                width=stroke,
            )
            x, y = nx, ny
    veins = veins.filter(ImageFilter.GaussianBlur(radius=3.0))
    mask = veins.point(lambda value: min(255, int(value * 1.15)))
    return Image.composite(
        Image.new('RGB', size, (226, 229, 233)),
        image,
        mask,
    )


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


def _outlined_polygon(draw, points, fill, outline, width=7):
    """Fill a polygon and stroke it with a width Pillow always supports."""
    draw.polygon(points, fill=fill)
    draw.line(list(points) + [points[0]], fill=outline, width=width,
              joint='curve')


def _draw_arrow(draw, direction):
    """Light grey block arrow with a dark rim, as on the reference panel."""
    if direction == 'up':
        points = [(256, 124), (378, 250), (304, 250), (304, 382),
                  (208, 382), (208, 250), (134, 250)]
    else:
        points = [(256, 388), (134, 262), (208, 262), (208, 130),
                  (304, 130), (304, 262), (378, 262)]
    _outlined_polygon(draw, points, INK, INK_SOFT_OUTLINE, width=7)


def _draw_door(draw, opening):
    """Two leaves plus outward (open) or inward (close) travel arrows."""
    ink = OPEN_INK if opening else INK
    if opening:
        left = [(104, 256), (240, 164), (240, 348)]
        right = [(408, 256), (272, 164), (272, 348)]
    else:
        left = [(104, 164), (104, 348), (240, 256)]
        right = [(408, 164), (408, 348), (272, 256)]
    for points in (left, right):
        draw.polygon(points, fill=ink)
    draw.rounded_rectangle((241, 148, 271, 364), radius=12, fill=ink)


def _draw_bell(draw):
    """Amber alarm bell."""
    draw.pieslice((152, 142, 360, 350), 180, 360, fill=BELL_INK)
    draw.polygon(
        [(152, 246), (360, 246), (386, 322), (126, 322)],
        fill=BELL_INK,
    )
    draw.rounded_rectangle((118, 318, 394, 348), radius=15, fill=BELL_INK)
    draw.ellipse((234, 354, 278, 398), fill=BELL_INK)
    draw.ellipse((238, 116, 274, 152), fill=BELL_INK)


def _draw_intercom(draw):
    """White handset resting on its cradle."""
    draw.arc((126, 158, 386, 418), 180, 360, fill=INK, width=36)
    draw.rounded_rectangle((108, 264, 192, 312), radius=20, fill=INK)
    draw.rounded_rectangle((320, 264, 404, 312), radius=20, fill=INK)


def _button_panel_background(panel, label):
    """Crop the matching panel steel behind one physical button face."""
    width, height = panel.size
    scale = height / PANEL_HEIGHT_M
    if label in {'alarm', 'open'}:
        offset_y = -PAIR_OFFSET_M
    elif label in {'intercom', 'close'}:
        offset_y = PAIR_OFFSET_M
    else:
        offset_y = 0.0
    center_x = width / 2.0 + offset_y * scale
    center_y = height / 2.0 - BUTTON_LOCAL_Z[label] * scale
    face = int(round(BUTTON_FACE_M * scale))
    left = int(round(center_x - face / 2.0))
    top = int(round(center_y - face / 2.0))
    crop = panel.crop((left, top, left + face, top + face))
    return crop.resize((512, 512), Image.Resampling.BICUBIC)


def render_button(label, background=None):
    """Render one black machined button tile with its centered glyph."""
    if background is None:
        image = _brushed_steel(
            (512, 512),
            seed=31 + BUTTONS.index(label),
            base=(212.0, 216.0, 221.0),
            grain=2.6,
        )
    else:
        image = background.copy()
    image = image.filter(ImageFilter.GaussianBlur(radius=0.4))
    draw = ImageDraw.Draw(image)

    # Contact shadow, raised tile, then the recessed glyph face.
    draw.rounded_rectangle((36, 40, 484, 488), radius=68,
                           fill=(46, 48, 50))
    draw.rounded_rectangle((28, 28, 476, 476), radius=70, fill=TILE_FILL)
    draw.rounded_rectangle((28, 28, 476, 476), radius=70,
                           outline=TILE_EDGE_DARK, width=6)
    draw.arc((34, 34, 470, 470), 190, 350, fill=TILE_EDGE_LIGHT, width=6)
    draw.rounded_rectangle((64, 64, 440, 440), radius=54, fill=FACE_FILL)
    draw.rounded_rectangle((64, 64, 440, 440), radius=54,
                           outline=FACE_EDGE, width=4)
    draw.arc((70, 70, 434, 434), 200, 340, fill=(38, 40, 42), width=5)

    if label in {'1', '2', '3'}:
        _center_text(
            draw,
            label,
            (58, 46, 454, 466),
            _font(FONT_REGULAR, 292),
            INK,
        )
    elif label in {'up', 'down'}:
        _draw_arrow(draw, label)
    elif label in {'open', 'close'}:
        _draw_door(draw, opening=label == 'open')
    elif label == 'intercom':
        _draw_intercom(draw)
    else:
        _draw_bell(draw)
    return image


def render_panel_plate(size=(470, 1006), seed=11):
    """Render the brushed stainless plate applied to the panel body.

    The world's key light comes from behind the panel (direction
    ``-0.5 0 -1`` while the plate faces ``-x``), so the plate is only lit by
    ambient and the fill light.  A high-metalness material would darken it
    further and Gazebo has no environment map here.  Compensate by baking a
    bright base, vertical brush grain and a soft vertical sheen band into the
    albedo map, which is what the reference photo shows.
    """
    width, height = size
    plate = _brushed_steel(size, seed=seed, base=(212.0, 216.0, 221.0),
                           grain=2.6)
    pixels = np.asarray(plate, dtype=np.float64)
    across = np.linspace(0.0, 1.0, width)[None, :, None]
    # Broad sheen running top-to-bottom, slightly left of centre.
    sheen = 0.93 + 0.11 * np.exp(-((across - 0.42) / 0.30) ** 2)
    return Image.fromarray(
        np.uint8(np.clip(pixels * sheen, 0, 255)), 'RGB'
    )


def render_panel(button_images):
    """Render an orthographic reference of the complete panel."""
    width, height = 480, 1024
    panel = render_panel_plate((width, height))
    scale = height / PANEL_HEIGHT_M
    draw = ImageDraw.Draw(panel)
    # The reference has a broad dark outer bezel and a thin bright inner lip.
    # The live model draws the same frame with panel_rim_* geometry, because a
    # box albedo map cannot be relied on for the plate's side faces.
    outer_inset = max(10, int(round(min(width, height) * 0.032)))
    inner_inset = outer_inset + max(3, outer_inset // 4)
    draw.rectangle(
        (0, 0, width - 1, height - 1),
        outline=(45, 48, 51),
        width=outer_inset,
    )
    draw.rectangle(
        (inner_inset, inner_inset, width - 1 - inner_inset,
         height - 1 - inner_inset),
        outline=(232, 235, 238),
        width=max(2, outer_inset // 5),
    )

    face = int(round(BUTTON_FACE_M * scale))
    for label, z in BUTTON_LOCAL_Z.items():
        if label in {'alarm', 'open'}:
            offset_y = -PAIR_OFFSET_M
        elif label in {'intercom', 'close'}:
            offset_y = PAIR_OFFSET_M
        else:
            offset_y = 0.0
        center_x = int(round(width / 2.0 + offset_y * scale))
        center_y = int(round(height / 2.0 - z * scale))
        image = button_images[label].resize(
            (face, face), Image.Resampling.LANCZOS
        )
        panel.paste(image, (center_x - face // 2, center_y - face // 2))
    return panel


def main():
    """Generate the runtime textures and the panel reference image."""
    TEXTURE_ROOT.mkdir(parents=True, exist_ok=True)
    WALL_TEXTURE_ROOT.mkdir(parents=True, exist_ok=True)
    plate = render_panel_plate()
    images = {}
    for label in BUTTONS:
        image = render_button(label, _button_panel_background(plate, label))
        image.save(TEXTURE_ROOT / f'button_{label}.png', optimize=True)
        images[label] = image
    panel = render_panel(images)
    panel.save(TEXTURE_ROOT / 'panel_reference.png', optimize=True)
    plate.save(TEXTURE_ROOT / 'panel_steel.png', optimize=True)
    wall = _render_marble()
    wall.save(WALL_TEXTURE_ROOT / 'wall_marble.png', optimize=True)
    print(f'generated {len(images)} button textures in {TEXTURE_ROOT}')
    print(f'generated panel plate texture in {TEXTURE_ROOT}')
    print(f'generated marble wall texture in {WALL_TEXTURE_ROOT}')


if __name__ == '__main__':
    main()
