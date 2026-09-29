#!/usr/bin/env python3
"""Generate simulation panel model variants and lighting world variants.

Outputs (relative to sim_variants/):
  models/elevator_button_v{0..4}/   panel model variants (textures + sdf)
  worlds/light_{L0..L5}.sdf         lighting scenario worlds

Deterministic: all randomness is seeded. Run on the host with the yolo env
(requires PIL/numpy). The container sees the results via the ros2_ws bind
mount.
"""

import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

SIM_VARIANTS = Path(__file__).resolve().parents[1]
GAZEBO_SRC = SIM_VARIANTS.parent
ORIGINAL_MODEL = GAZEBO_SRC / 'models' / 'elevator_button'
WORLDS_OUT = SIM_VARIANTS / 'worlds'
MODELS_OUT = SIM_VARIANTS / 'models'

FONT_REGULAR = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
FONT_BOLD = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
# Reading order of the vertical cabin panel, top row first.
BUTTONS = ('alarm', 'intercom', '3', '2', '1', 'open', 'close', 'up', 'down')


# --------------------------------------------------------------------------
# Button texture rendering (mirrors scripts/generate_elevator_panel_assets.py)
# --------------------------------------------------------------------------

BASE_STYLE = dict(
    tile_fill=(27, 28, 30),
    face_fill=(11, 11, 12),
    ink=(238, 239, 241),
    ink_outline=(92, 94, 97),
    edge_light=(104, 107, 110),
    edge_dark=(9, 9, 10),
    steel_base=(212.0, 216.0, 221.0),
    grain=2.6,
    blur=0.4,
    font='regular',
    font_size=292,
    glyph_scale=1.0,
)
BELL_INK = (250, 198, 40)
OPEN_INK = (56, 166, 62)


def _brushed_steel(size, seed, base=(96.0, 101.0, 106.0), grain_sigma=1.4):
    rng = np.random.default_rng(seed)
    height, width = size[1], size[0]
    horizontal = rng.normal(0.0, 4.5, (1, width, 1))
    noise = rng.normal(0.0, grain_sigma, (height, width, 1))
    vertical_light = np.linspace(-9.0, 11.0, width)[None, :, None]
    tint = np.asarray([0.93, 1.0, 1.05])[None, None, :]
    pixels = np.asarray(base)[None, None, :] + horizontal * tint + noise
    pixels = pixels + vertical_light
    return Image.fromarray(np.uint8(np.clip(pixels, 0, 255)), 'RGB')


def _center_text(draw, text, box, font, fill, stroke_width=0):
    bounds = draw.textbbox((0, 0), text, font=font, stroke_width=stroke_width)
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    left, top, right, bottom = box
    position = (
        left + (right - left - width) / 2.0 - bounds[0],
        top + (bottom - top - height) / 2.0 - bounds[1],
    )
    draw.text(position, text, font=font, fill=fill,
              stroke_width=stroke_width, stroke_fill=(15, 17, 18))


def _outlined_polygon(draw, points, fill, outline, width=7):
    draw.polygon(points, fill=fill)
    draw.line(list(points) + [points[0]], fill=outline, width=width,
              joint='curve')


def _draw_arrow(draw, direction, ink, outline):
    if direction == 'up':
        points = [(256, 124), (378, 250), (304, 250), (304, 382),
                  (208, 382), (208, 250), (134, 250)]
    else:
        points = [(256, 388), (134, 262), (208, 262), (208, 130),
                  (304, 130), (304, 262), (378, 262)]
    _outlined_polygon(draw, points, ink, outline)


def _draw_door(draw, opening, ink):
    colour = OPEN_INK if opening else ink
    if opening:
        left = [(104, 256), (240, 164), (240, 348)]
        right = [(408, 256), (272, 164), (272, 348)]
    else:
        left = [(104, 164), (104, 348), (240, 256)]
        right = [(408, 164), (408, 348), (272, 256)]
    for points in (left, right):
        draw.polygon(points, fill=colour)
    draw.rounded_rectangle((241, 148, 271, 364), radius=12, fill=colour)


def _draw_bell(draw):
    draw.pieslice((152, 142, 360, 350), 180, 360, fill=BELL_INK)
    draw.polygon([(152, 246), (360, 246), (386, 322), (126, 322)],
                 fill=BELL_INK)
    draw.rounded_rectangle((118, 318, 394, 348), radius=15, fill=BELL_INK)
    draw.ellipse((234, 354, 278, 398), fill=BELL_INK)
    draw.ellipse((238, 116, 274, 152), fill=BELL_INK)


def _draw_intercom(draw, ink):
    draw.arc((126, 158, 386, 418), 180, 360, fill=ink, width=36)
    draw.rounded_rectangle((108, 264, 192, 312), radius=20, fill=ink)
    draw.rounded_rectangle((320, 264, 404, 312), radius=20, fill=ink)


def _render_glyph(label, cfg):
    """Draw the glyph on its own layer so variants can scale it cheaply."""
    layer = Image.new('RGBA', (512, 512), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    ink = tuple(cfg['ink'])
    outline = tuple(cfg['ink_outline'])
    if label in {'1', '2', '3'}:
        font_path = FONT_BOLD if cfg['font'] == 'bold' else FONT_REGULAR
        _center_text(draw, label, (58, 46, 454, 466),
                     ImageFont.truetype(font_path, cfg['font_size']), ink)
    elif label in {'up', 'down'}:
        _draw_arrow(draw, label, ink, outline)
    elif label in {'open', 'close'}:
        _draw_door(draw, opening=label == 'open', ink=ink)
    elif label == 'intercom':
        _draw_intercom(draw, ink)
    else:
        _draw_bell(draw)
    return layer


def _scale_layer(layer, scale):
    if abs(scale - 1.0) < 1e-6:
        return layer
    size = max(1, int(round(512 * scale)))
    resized = layer.resize((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new('RGBA', (512, 512), (0, 0, 0, 0))
    offset = (512 - size) // 2
    canvas.paste(resized, (offset, offset))
    return canvas


def _button_panel_background(panel, label):
    """Crop the matching panel steel behind one physical button face."""
    width, height = panel.size
    scale = height / 0.280
    if label in {'alarm', 'open'}:
        offset_y = -0.0298
    elif label in {'intercom', 'close'}:
        offset_y = 0.0298
    else:
        offset_y = 0.0
    centers = {
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
    center_x = width / 2.0 + offset_y * scale
    center_y = height / 2.0 - centers[label] * scale
    face = int(round(0.033 * scale))
    left = int(round(center_x - face / 2.0))
    top = int(round(center_y - face / 2.0))
    crop = panel.crop((left, top, left + face, top + face))
    return crop.resize((512, 512), Image.Resampling.BICUBIC)


def render_button(label, style, background=None):
    """Render one black machined button tile with a per-variant style."""
    cfg = dict(BASE_STYLE)
    cfg.update(style)

    if background is None:
        image = _brushed_steel(
            (512, 512), seed=31 + BUTTONS.index(label),
            base=cfg['steel_base'], grain_sigma=cfg['grain'])
    else:
        image = background.copy()
    image = image.filter(ImageFilter.GaussianBlur(radius=cfg['blur']))
    draw = ImageDraw.Draw(image)

    tile = tuple(cfg['tile_fill'])
    face = tuple(cfg['face_fill'])
    # Contact shadow, raised tile, then the recessed glyph face.
    draw.rounded_rectangle((36, 40, 484, 488), radius=68, fill=(46, 48, 50))
    draw.rounded_rectangle((28, 28, 476, 476), radius=70, fill=tile)
    draw.rounded_rectangle((28, 28, 476, 476), radius=70,
                           outline=tuple(cfg['edge_dark']), width=6)
    draw.arc((34, 34, 470, 470), 190, 350,
             fill=tuple(cfg['edge_light']), width=6)
    draw.rounded_rectangle((64, 64, 440, 440), radius=54, fill=face)
    draw.rounded_rectangle((64, 64, 440, 440), radius=54,
                           outline=(62, 64, 67), width=4)
    draw.arc((70, 70, 434, 434), 200, 340, fill=(38, 40, 42), width=5)

    glyph = _scale_layer(_render_glyph(label, cfg), cfg['glyph_scale'])
    image = image.convert('RGBA')
    image.alpha_composite(glyph)
    return image.convert('RGB')


# --------------------------------------------------------------------------
# Panel model variants
# --------------------------------------------------------------------------

# Panel body material (wall_panel visual) per variant:
#   (diffuse tint rgb, metalness, roughness)
# The plate now carries a bright stainless albedo map, so the variant only
# tints it.  Keep metalness low: Gazebo has no environment map in this
# workcell and a high-metalness plate renders almost black.
PANEL_BODIES = {
    'v0': ((1.0, 1.0, 1.0), 0.12, 0.28),
    'v1': ((0.22, 0.23, 0.25), 0.40, 0.45),
    'v2': ((0.86, 0.88, 0.90), 0.20, 0.18),
    'v3': ((0.80, 0.74, 0.66), 0.05, 0.55),
    'v4': ((1.0, 1.0, 1.0), 0.22, 0.22),
}

# Button glyph style per variant
BUTTON_STYLES = {
    'v0': {},
    'v1': dict(
        tile_fill=(214, 216, 218), face_fill=(232, 234, 235),
        ink=(22, 23, 25), ink_outline=(120, 122, 125),
        edge_light=(250, 250, 250), edge_dark=(150, 152, 155),
    ),
    'v2': dict(
        steel_base=(70.0, 74.0, 78.0), grain=3.2, blur=0.9,
        tile_fill=(33, 32, 30), face_fill=(18, 17, 16),
        ink=(214, 212, 205), font_size=250,
    ),
    'v3': dict(font='bold', font_size=310, glyph_scale=1.12),
    'v4': dict(font_size=236, glyph_scale=0.84,
               steel_base=(112.0, 116.0, 121.0), ink=(226, 228, 230)),
}

VARIANT_DESCRIPTIONS = {
    'v0': '原始外观(与部署仿真一致)',
    'v1': '浅色按钮面+深色字形(反色风格), 面板更深',
    'v2': '老化磨损: 重纹理噪声, 小字号, 更粗糙',
    'v3': '粗体大字形, 暖色哑光面板',
    'v4': '小字形细图标, 亮钢底板',
}


def _patch_panel_body(sdf_text, body):
    """Tint the plate material.  The albedo_map line makes the block unique;
    every button visual also uses <ambient>1 1 1 1</ambient>, so a bare
    ambient/diffuse replacement would corrupt all nine buttons."""
    diffuse, metalness, roughness = body
    old_block = (
        '        <material>\n'
        '          <ambient>1 1 1 1</ambient>\n'
        '          <diffuse>1 1 1 1</diffuse>\n'
        '          <specular>0.30 0.30 0.30 1</specular>\n'
        '          <pbr><metal><albedo_map>'
        'model://elevator_button/textures/panel_steel.png</albedo_map>'
        '<metalness>0.12</metalness><roughness>0.28</roughness>'
        '</metal></pbr>\n'
        '        </material>'
    )
    new_block = (
        '        <material>\n'
        '          <ambient>1 1 1 1</ambient>\n'
        f'          <diffuse>{diffuse[0]} {diffuse[1]} {diffuse[2]} 1</diffuse>\n'
        '          <specular>0.30 0.30 0.30 1</specular>\n'
        '          <pbr><metal><albedo_map>'
        'model://elevator_button/textures/panel_steel.png</albedo_map>'
        f'<metalness>{metalness}</metalness><roughness>{roughness}</roughness>'
        '</metal></pbr>\n'
        '        </material>'
    )
    if old_block not in sdf_text:
        raise RuntimeError('panel plate material block not found in model.sdf')
    return sdf_text.replace(old_block, new_block)


def _rename_model(sdf_text, name):
    return sdf_text.replace('<model name="elevator_button">',
                            f'<model name="{name}">')


def generate_models():
    original_sdf = (ORIGINAL_MODEL / 'model.sdf').read_text()
    original_config = (ORIGINAL_MODEL / 'model.config').read_text()
    for vid in ('v0', 'v1', 'v2', 'v3', 'v4'):
        out = MODELS_OUT / f'elevator_button_{vid}'
        (out / 'textures').mkdir(parents=True, exist_ok=True)
        (out / 'meshes').mkdir(parents=True, exist_ok=True)
        sdf = _rename_model(_patch_panel_body(original_sdf, PANEL_BODIES[vid]),
                            f'elevator_button_{vid}')
        sdf = sdf.replace('model://elevator_button/',
                          f'model://elevator_button_{vid}/')
        (out / 'model.sdf').write_text(sdf)
        (out / 'model.config').write_text(original_config)
        for mesh in (ORIGINAL_MODEL / 'meshes').iterdir():
            shutil.copy(mesh, out / 'meshes' / mesh.name)
        # Drop textures of controls that the vertical panel no longer has.
        for stale in (out / 'textures').glob('button_*.png'):
            if stale.stem.removeprefix('button_') not in BUTTONS:
                stale.unlink()
        # The plate texture is shared: the variant tints it with <diffuse>.
        shutil.copy(ORIGINAL_MODEL / 'textures' / 'panel_steel.png',
                    out / 'textures' / 'panel_steel.png')
        panel = Image.open(
            ORIGINAL_MODEL / 'textures' / 'panel_steel.png'
        ).convert('RGB')
        for label in BUTTONS:
            img = render_button(
                label,
                BUTTON_STYLES[vid],
                _button_panel_background(panel, label),
            )
            img.save(out / 'textures' / f'button_{label}.png', optimize=True)
        print(f'generated variant {vid}: {out}')


# --------------------------------------------------------------------------
# Lighting world variants
# --------------------------------------------------------------------------

LIGHT_SCENARIOS = {
    # (key intensity, key diffuse, key direction, fill intensity, fill diffuse, fill pose)
    'L0': dict(name='基准(原世界)', key_i=1.0, key_rgb=(0.95, 0.95, 0.95),
               key_dir=(-0.5, 0, -1), fill_i=0.7, fill_rgb=(0.8, 0.85, 1.0),
               fill_pose=(0.3, -0.5, 0.8)),
    'L1': dict(name='昏暗', key_i=0.45, key_rgb=(0.9, 0.9, 0.9),
               key_dir=(-0.5, 0, -1), fill_i=0.0, fill_rgb=(0.8, 0.85, 1.0),
               fill_pose=(0.3, -0.5, 0.8)),
    'L2': dict(name='明亮暖光', key_i=1.5, key_rgb=(1.0, 0.9, 0.72),
               key_dir=(-0.5, 0, -1), fill_i=0.9, fill_rgb=(1.0, 0.95, 0.85),
               fill_pose=(0.3, -0.5, 0.8)),
    'L3': dict(name='冷蓝', key_i=0.85, key_rgb=(0.72, 0.82, 1.1),
               key_dir=(-0.5, 0, -1), fill_i=0.5, fill_rgb=(0.6, 0.7, 1.0),
               fill_pose=(0.3, 0.5, 0.8)),
    'L4': dict(name='侧向光', key_i=1.2, key_rgb=(0.95, 0.95, 0.95),
               key_dir=(-0.7, -0.55, -0.8), fill_i=0.35, fill_rgb=(0.8, 0.85, 1.0),
               fill_pose=(-0.4, -0.5, 0.9)),
    'L5': dict(name='夜间(强点光)', key_i=0.28, key_rgb=(0.85, 0.88, 0.95),
               key_dir=(-0.5, 0, -1), fill_i=1.35, fill_rgb=(1.0, 0.92, 0.75),
               fill_pose=(0.35, 0.15, 0.62)),
}


def generate_worlds():
    base_world = (GAZEBO_SRC / 'worlds' / 'button_press.sdf').read_text()
    WORLDS_OUT.mkdir(parents=True, exist_ok=True)
    for lid, cfg in LIGHT_SCENARIOS.items():
        old_key = (
            '    <light name="key_light" type="directional">\n'
            '      <pose>0 0 3 0 0 0</pose>\n'
            '      <cast_shadows>true</cast_shadows>\n'
            '      <intensity>1.0</intensity>\n'
            '      <direction>-0.5 0 -1</direction>\n'
            '      <diffuse>0.95 0.95 0.95 1</diffuse>\n'
            '      <specular>0.2 0.2 0.2 1</specular>\n'
            '    </light>'
        )
        new_key = (
            '    <light name="key_light" type="directional">\n'
            '      <pose>0 0 3 0 0 0</pose>\n'
            '      <cast_shadows>true</cast_shadows>\n'
            f'      <intensity>{cfg["key_i"]}</intensity>\n'
            f'      <direction>{cfg["key_dir"][0]} {cfg["key_dir"][1]} {cfg["key_dir"][2]}</direction>\n'
            f'      <diffuse>{cfg["key_rgb"][0]} {cfg["key_rgb"][1]} {cfg["key_rgb"][2]} 1</diffuse>\n'
            '      <specular>0.2 0.2 0.2 1</specular>\n'
            '    </light>'
        )
        old_fill = (
            '    <light name="fill_light" type="point">\n'
            '      <pose>0.3 -0.5 0.8 0 0 0</pose>\n'
            '      <intensity>0.7</intensity>\n'
            '      <diffuse>0.8 0.85 1.0 1</diffuse>'
        )
        fill_pose = ' '.join(str(v) for v in cfg['fill_pose'])
        new_fill = (
            '    <light name="fill_light" type="point">\n'
            f'      <pose>{fill_pose} 0 0 0</pose>\n'
            f'      <intensity>{cfg["fill_i"]}</intensity>\n'
            f'      <diffuse>{cfg["fill_rgb"][0]} {cfg["fill_rgb"][1]} {cfg["fill_rgb"][2]} 1</diffuse>'
        )
        world = base_world
        if old_key not in world or old_fill not in world:
            raise RuntimeError(f'light block not found for {lid}')
        world = world.replace(old_key, new_key).replace(old_fill, new_fill)
        path = WORLDS_OUT / f'light_{lid}.sdf'
        path.write_text(world)
        print(f'generated world {lid} ({cfg["name"]}): {path}')


def write_manifest():
    manifest = {
        'panel_variants': {
            vid: {'description': desc, 'body': PANEL_BODIES[vid]}
            for vid, desc in VARIANT_DESCRIPTIONS.items()
        },
        'light_scenarios': {
            lid: {'name': cfg['name'],
                  'key_intensity': cfg['key_i'],
                  'key_rgb': list(cfg['key_rgb']),
                  'key_direction': list(cfg['key_dir']),
                  'fill_intensity': cfg['fill_i'],
                  'fill_rgb': list(cfg['fill_rgb']),
                  'fill_pose': list(cfg['fill_pose'])}
            for lid, cfg in LIGHT_SCENARIOS.items()
        },
        # Vertical cabin panel: reading order is top row first, then the three
        # centered floor tiles, the door pair, up and down.
        'button_local_poses': {
            'alarm': [0.010, -0.0298, 0.117],
            'intercom': [0.010, 0.0298, 0.117],
            '3': [0.010, 0.0, 0.078],
            '2': [0.010, 0.0, 0.039],
            '1': [0.010, 0.0, 0.0],
            'open': [0.010, -0.0298, -0.039],
            'close': [0.010, 0.0298, -0.039],
            'up': [0.010, 0.0, -0.078],
            'down': [0.010, 0.0, -0.117],
        },
        'button_face_size_m': [0.033, 0.033],
        'panel_size_m': [0.012, 0.1308, 0.280],
        'panel_base_pose': {'position': [0.55, 0.03, 0.43],
                            'yaw_deg': 180.0},
    }
    path = SIM_VARIANTS / 'variant_manifest.json'
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f'wrote {path}')


if __name__ == '__main__':
    generate_models()
    generate_worlds()
    write_manifest()
