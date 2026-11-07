from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
CASE_KEYS = ["Boeing_737", "Cessna_172"]


def latest_figure_dir(case_key: str) -> Path:
    case_dir = ROOT / "Cases" / case_key
    matches = sorted(p for p in case_dir.glob("8*") if p.is_dir())
    if not matches:
        raise FileNotFoundError(f"No latest figure directory found under {case_dir}")
    return matches[0]


def max_run(mask: np.ndarray) -> tuple[int, int, int]:
    if mask.size == 0:
        return 0, -1, -1
    values = mask.astype(np.int8)
    diff = np.diff(np.concatenate(([0], values, [0])))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    if len(starts) == 0:
        return 0, -1, -1
    lengths = ends - starts
    idx = int(np.argmax(lengths))
    return int(lengths[idx]), int(starts[idx]), int(ends[idx] - 1)


def detect_axes(image: Image.Image) -> tuple[int, int, int, int, int, int] | None:
    w, h = image.size
    scale = max(w, h) / 1600
    sw = max(1, int(w / scale))
    sh = max(1, int(h / scale))
    sample = image.convert("RGB").resize((sw, sh))
    arr = np.asarray(sample)
    dark = (arr[:, :, 0] < 95) & (arr[:, :, 1] < 95) & (arr[:, :, 2] < 95)

    min_horizontal = max(120, int(sw * 0.30))
    horizontal_candidates: list[tuple[int, int, int, int]] = []
    for y in range(int(sh * 0.25), int(sh * 0.98)):
        length, start, end = max_run(dark[y, :])
        if length >= min_horizontal:
            horizontal_candidates.append((length, y, start, end))
    if not horizontal_candidates:
        return None

    # Choose the lowest long line; for these figures that is the x axis, not data marks.
    horizontal_candidates.sort(key=lambda item: (item[1], item[0]))
    _, y_axis, x_left, x_right = horizontal_candidates[-1]

    min_vertical = max(100, int(sh * 0.22))
    vertical_candidates: list[tuple[int, int, int, int, int]] = []
    for x in range(0, int(sw * 0.60)):
        length, top, bottom = max_run(dark[:, x])
        if length >= min_vertical:
            distance_to_origin = abs(x - x_left)
            vertical_candidates.append((length, -distance_to_origin, x, top, bottom))
    if not vertical_candidates:
        return None

    _, _, x_axis, y_top, y_bottom = sorted(vertical_candidates, reverse=True)[0]
    return (
        int(round(x_axis / sw * w)),
        int(round(y_axis / sh * h)),
        int(round(x_left / sw * w)),
        int(round(x_right / sw * w)),
        int(round(y_top / sh * h)),
        int(round(y_bottom / sh * h)),
    )


def arrow_scores(image: Image.Image, axes: tuple[int, int, int, int, int, int]) -> tuple[float, float]:
    """Estimate whether x/y arrowheads already exist near the detected axes.

    The score subtracts the axis-line core from the dark-pixel mass near an
    expected arrowhead. Existing filled arrowheads produce a clearly positive
    score, while plain spines stay near zero.
    """

    w, h = image.size
    scale = max(w, h) / 1600
    sw = max(1, int(w / scale))
    sh = max(1, int(h / scale))
    sample = image.convert("RGB").resize((sw, sh))
    arr = np.asarray(sample)
    dark = (arr[:, :, 0] < 95) & (arr[:, :, 1] < 95) & (arr[:, :, 2] < 95)

    x_axis, y_axis, _x_left, x_right, y_top, _y_bottom = axes
    x_axis = int(round(x_axis / w * sw))
    y_axis = int(round(y_axis / h * sh))
    x_right = int(round(x_right / w * sw))
    y_top = int(round(y_top / h * sh))

    base = min(sw, sh)
    head_len = max(14, int(base * 0.026))
    head_half = max(9, int(head_len * 0.52))
    line_width = max(2, int(base * 0.0032))

    x0 = max(0, x_right - head_len)
    x1 = min(sw, x_right + 2)
    y0 = max(0, y_axis - head_half)
    y1 = min(sh, y_axis + head_half)
    x_area = max(1, (x1 - x0) * (y1 - y0))
    x_total = int(dark[y0:y1, x0:x1].sum())
    line_y0 = max(0, y_axis - line_width)
    line_y1 = min(sh, y_axis + line_width + 1)
    x_line = int(dark[line_y0:line_y1, x0:x1].sum())
    x_score = (x_total - x_line) / x_area

    x0 = max(0, x_axis - head_half)
    x1 = min(sw, x_axis + head_half)
    y0 = max(0, y_top - 2)
    y1 = min(sh, y_top + head_len)
    y_area = max(1, (x1 - x0) * (y1 - y0))
    y_total = int(dark[y0:y1, x0:x1].sum())
    line_x0 = max(0, x_axis - line_width)
    line_x1 = min(sw, x_axis + line_width + 1)
    y_line = int(dark[y0:y1, line_x0:line_x1].sum())
    y_score = (y_total - y_line) / y_area

    return x_score, y_score


def draw_axis_arrows(
    image: Image.Image,
    axes: tuple[int, int, int, int, int, int],
    *,
    draw_x: bool,
    draw_y: bool,
) -> Image.Image:
    x_axis, y_axis, _x_left, x_right, y_top, _y_bottom = axes
    w, h = image.size
    out = image.convert("RGBA")
    draw = ImageDraw.Draw(out)

    base = min(w, h)
    line_width = max(6, int(base * 0.0032))
    head_len = max(38, int(base * 0.026))
    head_half = max(24, int(head_len * 0.52))
    margin = max(16, int(base * 0.006))
    color = (35, 35, 35, 255)

    x_tip = min(w - margin, max(x_right, x_axis + head_len * 2) + head_len)
    y_tip = max(margin, min(y_top, y_axis - head_len * 2) - head_len)

    if draw_x:
        draw.line([(x_axis, y_axis), (x_tip - head_len, y_axis)], fill=color, width=line_width)
        draw.polygon(
            [
                (x_tip, y_axis),
                (x_tip - head_len, y_axis - head_half),
                (x_tip - head_len, y_axis + head_half),
            ],
            fill=color,
        )

    if draw_y:
        draw.line([(x_axis, y_axis), (x_axis, y_tip + head_len)], fill=color, width=line_width)
        draw.polygon(
            [
                (x_axis, y_tip),
                (x_axis - head_half, y_tip + head_len),
                (x_axis + head_half, y_tip + head_len),
            ],
            fill=color,
        )
    return out


def process_file(path: Path, backup_dir: Path) -> bool:
    image = Image.open(path)
    axes = detect_axes(image)
    if axes is None:
        return False
    x_score, y_score = arrow_scores(image, axes)
    threshold = 0.055
    draw_x = x_score < threshold
    draw_y = y_score < threshold
    if not draw_x and not draw_y:
        return False

    backup_dir.mkdir(parents=True, exist_ok=True)
    backup = backup_dir / path.name
    if not backup.exists():
        shutil.copy2(path, backup)

    updated = draw_axis_arrows(image, axes, draw_x=draw_x, draw_y=draw_y)
    dpi = image.info.get("dpi", (600, 600))
    updated.convert("RGB").save(path, dpi=dpi)
    return True


def main() -> None:
    total_processed = 0
    total_skipped = 0
    for case_key in CASE_KEYS:
        figure_dir = latest_figure_dir(case_key)
        backup_dir = figure_dir / "_support_files" / "axis_arrow_originals"
        processed = 0
        skipped = 0
        for path in sorted(figure_dir.glob("*.png")):
            if process_file(path, backup_dir):
                processed += 1
            else:
                skipped += 1
        total_processed += processed
        total_skipped += skipped
        print(f"{case_key}: processed={processed}, skipped={skipped}, dir={figure_dir}")
    print(f"TOTAL: processed={total_processed}, skipped={total_skipped}")


if __name__ == "__main__":
    main()
