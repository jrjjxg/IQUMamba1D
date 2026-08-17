"""Draw the compact, code-faithful Stage 377 architecture figure."""

from __future__ import annotations

from html import escape
from pathlib import Path


WIDTH = 1920
HEIGHT = 1080
OUT = Path(__file__).with_name("stage377_architecture_optimized.svg")

BLUE = "#2F67C7"
BLUE_FILL = "#F3F8FF"
ORANGE = "#E66A2C"
ORANGE_FILL = "#FFF8F2"
GREEN = "#2F7D42"
GREEN_FILL = "#F3FAF4"
VIOLET = "#6652B8"
VIOLET_FILL = "#F7F4FF"
INK = "#20252B"
MUTED = "#5E6875"
LINE = "#3A4149"


def rect(x, y, w, h, fill, stroke, *, rx=8, sw=2, dash=None):
    dashed = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{dashed}/>'
    )


def text(x, y, lines, *, size=20, weight=400, fill=INK, anchor="middle", gap=None, family="Arial"):
    if isinstance(lines, str):
        lines = [lines]
    gap = gap or int(size * 1.25)
    spans = []
    for index, line in enumerate(lines):
        dy = 0 if index == 0 else gap
        spans.append(f'<tspan x="{x}" dy="{dy}">{escape(line)}</tspan>')
    return (
        f'<text x="{x}" y="{y}" text-anchor="{anchor}" fill="{fill}" '
        f'font-family="{family}" font-size="{size}" font-weight="{weight}">'
        + "".join(spans)
        + "</text>"
    )


def math_text(x, y, content, *, size=21, weight=400, fill=INK, anchor="middle"):
    return (
        f'<text x="{x}" y="{y}" text-anchor="{anchor}" fill="{fill}" '
        f'font-family="Cambria Math, Times New Roman" font-size="{size}" '
        f'font-style="italic" font-weight="{weight}">{content}</text>'
    )


def path(d, *, stroke=LINE, sw=2, dash=None, marker=True, fill="none"):
    dashed = f' stroke-dasharray="{dash}"' if dash else ""
    arrow = ' marker-end="url(#arrow)"' if marker else ""
    return f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{dashed}{arrow}/>'


def line(x1, y1, x2, y2, **kwargs):
    return path(f"M {x1} {y1} L {x2} {y2}", **kwargs)


def box_label(parts, x, y, w, h, *, fill, stroke, title, detail, title_color=None):
    parts.append(rect(x, y, w, h, fill, stroke, rx=7, sw=1.8))
    parts.append(text(x + w / 2, y + 27, title, size=18, weight=700, fill=title_color or stroke))
    parts.append(text(x + w / 2, y + 55, detail, size=16, weight=400, fill=INK, gap=20))


def main():
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
        "<defs>",
        '<marker id="arrow" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto" markerUnits="strokeWidth">',
        f'<path d="M 0 0 L 9 4.5 L 0 9 z" fill="{LINE}"/>',
        "</marker>",
        '<filter id="softShadow" x="-10%" y="-10%" width="120%" height="120%">',
        '<feDropShadow dx="0" dy="2" stdDeviation="3" flood-color="#000000" flood-opacity="0.08"/>',
        "</filter>",
        "</defs>",
        '<rect width="1920" height="1080" fill="#FFFFFF"/>',
    ]

    parts.append(text(960, 42, "Stage 377: Complex-State BiMamba + UniRepLK + Multi-Scale Latent Mask", size=30, weight=700))
    parts.append(text(960, 70, "No ASC  |  real simplex masks  |  one shared decoder call", size=17, fill=MUTED))

    # Input.
    parts.append(rect(20, 305, 125, 92, "#FFFFFF", BLUE, rx=7, sw=1.8))
    parts.append(text(82, 333, ["I/Q mixture", "y  [B, 2, L]"], size=17, weight=700, gap=27))
    parts.append(line(145, 351, 180, 351))

    # Encoder container and stages.
    parts.append(rect(180, 92, 405, 708, BLUE_FILL, BLUE, rx=14, sw=2.5))
    parts.append(text(382, 122, "Encoder", size=24, weight=700, fill=BLUE))
    parts.append(text(382, 147, "four-scale residual feature pyramid", size=15, fill=MUTED))

    stages = [
        (0, 165, "F0  [B, 32, L]", "Residual Conv x2", "UniRepLK residual delta"),
        (1, 325, "F1  [B, 64, L/2]", "Residual Conv x2", "Independent Complex-State BiMamba", "UniRepLK residual delta"),
        (2, 485, "F2  [B, 128, L/4]", "Residual Conv x2", "UniRepLK residual delta"),
        (3, 645, "F3  [B, 256, L/8]", "Residual Conv x2", "Independent Complex-State BiMamba"),
    ]
    centers = []
    for stage in stages:
        idx, y, shape, conv, *mods = stage
        h = 112
        x = 218
        w = 330
        centers.append(y + h / 2)
        parts.append(rect(x, y, w, h, "#FFFFFF", BLUE, rx=7, sw=1.7))
        parts.append(text(x + 14, y + 24, f"Stage {idx}" + (" bottleneck" if idx == 3 else ""), size=17, weight=700, fill=BLUE, anchor="start"))
        parts.append(text(x + w - 14, y + 24, shape, size=16, weight=700, anchor="end"))
        parts.append(rect(x + 35, y + 37, w - 70, 28, "#FFFDF7", "#C99A34", rx=5, sw=1.3))
        parts.append(text(x + w / 2, y + 57, conv, size=15, weight=700))
        if len(mods) == 1:
            color = ORANGE if "UniRepLK" in mods[0] else VIOLET
            parts.append(rect(x + 35, y + 73, w - 70, 28, "#FFFFFF", color, rx=5, sw=1.3, dash="6 4"))
            parts.append(text(x + w / 2, y + 93, mods[0], size=14, weight=700, fill=color))
        else:
            parts.append(rect(x + 18, y + 73, 176, 28, "#FFFFFF", VIOLET, rx=5, sw=1.3, dash="6 4"))
            parts.append(text(x + 106, y + 93, "Independent BiMamba", size=13, weight=700, fill=VIOLET))
            parts.append(rect(x + 202, y + 73, 110, 28, "#FFFFFF", ORANGE, rx=5, sw=1.3, dash="6 4"))
            parts.append(text(x + 257, y + 93, "UniRepLK delta", size=12, weight=700, fill=ORANGE))

    for y1, y2 in zip(centers[:-1], centers[1:]):
        parts.append(line(382, y1 + 56, 382, y2 - 56))
        parts.append(text(400, (y1 + y2) / 2 + 4, "downsample x2", size=13, fill=MUTED, anchor="start"))

    # Compact multi-scale mask routing.
    parts.append(rect(650, 112, 450, 666, ORANGE_FILL, ORANGE, rx=14, sw=2.5))
    parts.append(text(875, 142, "Multi-Scale Latent Mask & Slot Packing", size=22, weight=700, fill=ORANGE))
    parts.append(text(875, 168, "same operator form at all scales; H0-H3 have independent weights", size=14, fill=MUTED))

    shapes_out = ["[2B, 32, L]", "[2B, 64, L/2]", "[2B, 128, L/4]", "[2B, 256, L/8]"]
    for idx, (cy, out_shape) in enumerate(zip(centers, shapes_out)):
        parts.append(line(585, cy, 684, cy))
        parts.append(math_text(616, cy - 9, f"F{sub(idx)}", size=18))
        parts.append(rect(684, cy - 34, 225, 68, "#FFFFFF", ORANGE, rx=7, sw=1.6))
        parts.append(text(703, cy - 7, f"H{idx}", size=17, weight=700, fill=ORANGE, anchor="start"))
        parts.append(text(758, cy - 7, f"1x1 Conv: C{idx} -> 2C{idx}", size=15, weight=700, anchor="start"))
        parts.append(text(796, cy + 19, "softmax-slot mask", size=14, fill=MUTED))
        parts.append(line(909, cy, 946, cy))

    parts.append(rect(946, 184, 115, 535, VIOLET_FILL, VIOLET, rx=12, sw=2.0))
    parts.append(text(1003, 307, ["element-wise", "mask", "then", "slot-to-batch", "view"], size=17, weight=700, fill=VIOLET, gap=29))
    parts.append(text(1003, 480, ["[B, 2, C_i, L_i]", "->", "[2B, C_i, L_i]"], size=14, weight=700, fill=INK, gap=24))
    parts.append(text(1003, 585, ["one utility", "reshape/view only", "no parameters"], size=13, fill=MUTED, gap=21))
    for idx, cy in enumerate(centers):
        parts.append(line(1061, cy, 1120, cy))
        parts.append(math_text(1084, cy - 9, f"Z{sub(idx)}", size=18))
        parts.append(text(1112, cy + 22, shapes_out[idx], size=13, fill=MUTED, anchor="end"))

    # Decoder container, with the true bottom-up plain-skip sequence.
    parts.append(rect(1160, 92, 440, 708, GREEN_FILL, GREEN, rx=14, sw=2.5))
    parts.append(text(1380, 122, "Decoder", size=24, weight=700, fill=GREEN))
    parts.append(text(1380, 147, "3-stage shared plain-skip decoder", size=15, fill=MUTED))

    decoder_boxes = [
        (1255, 690, 250, 36, "masked bottleneck Z3"),
        (1280, 630, 200, 34, "Upsample x2"),
        (1255, 570, 250, 36, "Concat masked skip Z2"),
        (1255, 515, 250, 36, "Residual Conv Block"),
        (1280, 455, 200, 34, "Upsample x2"),
        (1255, 395, 250, 36, "Concat masked skip Z1"),
        (1255, 340, 250, 36, "Residual Conv Block"),
        (1280, 280, 200, 34, "Upsample x2"),
        (1255, 220, 250, 36, "Concat masked skip Z0"),
        (1230, 165, 220, 36, "Residual Conv Block"),
    ]
    for x, y, w, h, label in decoder_boxes:
        parts.append(rect(x, y, w, h, "#FFFFFF", GREEN, rx=6, sw=1.6))
        parts.append(text(x + w / 2, y + h / 2 + 6, label, size=15, weight=700, fill=INK))
    parts.append(rect(1470, 165, 105, 36, "#FFFFFF", GREEN, rx=6, sw=1.6))
    parts.append(text(1522, 188, "1x1 out", size=14, weight=700, fill=GREEN))
    parts.append(line(1450, 183, 1470, 183))
    # Explicit bottom-up links keep the decoder flow easy to audit.
    links = [(690, 664), (630, 606), (570, 551), (515, 489), (455, 431), (395, 376), (340, 314), (280, 256), (220, 201)]
    for y1, y2 in links:
        parts.append(line(1380, y1, 1380, y2))

    # Routed multi-scale skips into matching decoder levels.
    route_targets = [238, 413, 588, 708]
    for cy, target in zip(centers, route_targets):
        parts.append(path(f"M 1120 {cy} L 1138 {cy} L 1138 {target} L 1255 {target}"))

    parts.append(text(1380, 766, "one decoder invocation  |  effective batch = 2B  |  shared weights", size=14, weight=700, fill=GREEN))
    parts.append(text(1180, 786, "No ASC / no skip processors", size=13, fill=MUTED, anchor="start"))

    # Output.
    parts.append(rect(1640, 225, 250, 150, "#FFFFFF", GREEN, rx=8, sw=2.0))
    parts.append(text(1765, 253, "Output", size=20, weight=700, fill=GREEN))
    parts.append(text(1765, 283, ["[2B, 2, L]", "reshape [B, 2, 2, L]", "-> [B, 4, L]"], size=15, weight=700, gap=24))
    parts.append(text(1765, 357, "[S1_I, S1_Q, S2_I, S2_Q]", size=14, weight=700, fill=GREEN))
    parts.append(path("M 1575 183 L 1608 183 L 1608 300 L 1640 300"))

    # Generic mechanism expanded exactly once.
    parts.append(rect(180, 842, 1420, 196, "#FFFFFF", "#7A828C", rx=10, sw=1.7, dash="8 5"))
    parts.append(text(205, 873, "Generic latent-mask head (expanded once; applied independently for i = 0, 1, 2, 3)", size=18, weight=700, fill=ORANGE, anchor="start"))

    pipeline = [
        (220, 900, 170, 82, "F^(i)", "[B, C_i, L_i]", BLUE, BLUE_FILL),
        (435, 900, 190, 82, "H_i: 1x1 Conv", "C_i -> 2C_i", ORANGE, ORANGE_FILL),
        (670, 900, 210, 82, "reshape", "[B, 2, C_i, L_i]", ORANGE, ORANGE_FILL),
        (925, 900, 210, 82, "softmax", "over source slots", ORANGE, ORANGE_FILL),
        (1180, 900, 170, 82, "element-wise mask", "F^(i) .* M^(i)", ORANGE, ORANGE_FILL),
        (1395, 900, 170, 82, "view", "[2B, C_i, L_i]", VIOLET, VIOLET_FILL),
    ]
    for x, y, w, h, title_label, detail, color, fill_color in pipeline:
        box_label(parts, x, y, w, h, fill=fill_color, stroke=color, title=title_label, detail=detail)
    for first, second in zip(pipeline[:-1], pipeline[1:]):
        x1 = first[0] + first[2]
        x2 = second[0]
        parts.append(line(x1, 941, x2, 941))

    parts.append(text(440, 1015, "H0-H3: same structure, different channel widths, independent parameters", size=14, weight=700, fill=ORANGE))
    parts.append(math_text(935, 1017, "M1^(i)(c,t) + M2^(i)(c,t) = 1", size=18, weight=700, fill=ORANGE))
    parts.append(text(1435, 1015, "slot-to-batch: parameter-free reshape/view", size=14, weight=700, fill=VIOLET))

    parts.append(text(382, 824, "(a) Encoder", size=16, weight=700, fill=BLUE))
    parts.append(text(1380, 824, "(b) Decoder", size=16, weight=700, fill=GREEN))

    parts.append("</svg>")
    OUT.write_text("\n".join(parts), encoding="utf-8")
    print(OUT)


def sub(index: int) -> str:
    return str(index)


if __name__ == "__main__":
    main()
