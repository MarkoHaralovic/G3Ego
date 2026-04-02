# GENERATED with GPT 5.2

import os
import textwrap

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ACTION_LABELS_FILES = {
    "test": "/home/s3758869/egocentric_video_graph_framework_ar/MECCANO/dataset/MECCANO_test_actions.csv",
    "train": "/home/s3758869/egocentric_video_graph_framework_ar/MECCANO/dataset/MECCANO_train_actions.csv",
    "val": "/home/s3758869/egocentric_video_graph_framework_ar/MECCANO/dataset/MECCANO_val_actions.csv",
}

OUTPUT_DIRS = [
    "/home/s3758869/egocentric_video_graph_framework_ar/auxilliary",
    "/home/s3758869/egocentric_video_graph_framework_ar/auxilliary/meccano_visuals",
]
FPS = 12.0
TOP_K_ACTIONS = 18
WIDTH = 1800
HEIGHT = 1200
DURATION_BIN_LABELS = ["0.0-0.5s", "0.5-1.0s", "1.0-2.0s", "2.0s+"]

BG = "#F7F4EE"
PANEL = "#FFFDF9"
TEXT = "#1F2933"
MUTED = "#52606D"
GRID = "#D9E2EC"
BAR = "#2F6FED"
BAR_EDGE = "#1E4DB7"
HIST = "#F59E0B"
AXIS = "#334E68"


def load_action_labels(split):
    return pd.read_csv(ACTION_LABELS_FILES[split])


def prepare_dataframe(df):
    df = df.copy()
    start = df["start_frame"].str.replace(".jpg", "", regex=False).astype(int)
    end = df["end_frame"].str.replace(".jpg", "", regex=False).astype(int)
    df["duration_frames"] = end - start + 1
    df["duration_seconds"] = df["duration_frames"] / FPS
    return df


def _load_font(size, bold=False):
    candidates = []
    if bold:
        candidates.extend(
            [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
            ]
        )
    else:
        candidates.extend(
            [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/dejavu/DejaVuSans.ttf",
            ]
        )
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


FONT_TITLE = _load_font(42, bold=True)
FONT_SUBTITLE = _load_font(24)
FONT_AXIS = _load_font(24, bold=True)
FONT_LABEL = _load_font(24)
FONT_SMALL = _load_font(20)
FONT_TICK = _load_font(20)
FONT_VALUE = _load_font(22, bold=True)
FONT_TABLE = _load_font(16)
FONT_TABLE_BOLD = _load_font(16, bold=True)


def _text_size(draw, text, font):
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    return right - left, bottom - top


def _draw_rotated_text(image, xy, text, font, fill, angle=90):
    dummy = Image.new("RGBA", (10, 10), (0, 0, 0, 0))
    draw_dummy = ImageDraw.Draw(dummy)
    w, h = _text_size(draw_dummy, text, font)
    txt = Image.new("RGBA", (w + 10, h + 10), (0, 0, 0, 0))
    draw_txt = ImageDraw.Draw(txt)
    draw_txt.text((5, 5), text, font=font, fill=fill)
    rotated = txt.rotate(angle, expand=True)
    image.alpha_composite(rotated, dest=xy)


def _draw_title(draw, title, subtitle):
    draw.text((90, 55), title, font=FONT_TITLE, fill=TEXT)
    draw.text((90, 110), subtitle, font=FONT_SUBTITLE, fill=MUTED)


def _draw_card(draw, bounds):
    draw.rounded_rectangle(bounds, radius=28, fill=PANEL, outline="#E5E7EB", width=2)


def _format_seconds(value):
    return "{:.1f}s".format(value)

def _ensure_output_dirs():
    for d in OUTPUT_DIRS:
        os.makedirs(d, exist_ok=True)


def _save_png(image, filename):
    rgb = image.convert("RGB")
    for d in OUTPUT_DIRS:
        rgb.save(os.path.join(d, filename), quality=95)


def _compute_duration_bin_counts(values):
    counts = [0, 0, 0, 0]
    for value in values:
        if value < 0.5:
            counts[0] += 1
        elif value < 1.0:
            counts[1] += 1
        elif value < 2.0:
            counts[2] += 1
        else:
            counts[3] += 1
    return counts


def create_action_distribution_chart(df, split):
    counts = df["action_name"].value_counts().sort_values(ascending=False)
    top_counts = counts.head(TOP_K_ACTIONS)
    if len(counts) > TOP_K_ACTIONS:
        top_counts.loc["all other classes"] = counts.iloc[TOP_K_ACTIONS:].sum()
    top_counts = top_counts.sort_values(ascending=True)

    image = Image.new("RGBA", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(image)
    _draw_title(
        draw,
        "MECCANO {} split: action distribution".format(split.upper()),
        "Top classes by number of annotated clips. Each bar shows count and share of the split.",
    )

    card = (70, 170, WIDTH - 70, HEIGHT - 80)
    _draw_card(draw, card)

    plot_left = 510
    plot_top = 260
    plot_right = WIDTH - 180
    plot_bottom = HEIGHT - 180
    plot_width = plot_right - plot_left
    plot_height = plot_bottom - plot_top

    max_count = max(top_counts.max(), 1)
    total_count = float(len(df))
    y_step = plot_height / float(len(top_counts))
    bar_height = max(18, int(y_step * 0.62))

    tick_count = 6
    for i in range(tick_count + 1):
        x = plot_left + int((plot_width * i) / float(tick_count))
        value = int(round((max_count * i) / float(tick_count)))
        draw.line((x, plot_top, x, plot_bottom), fill=GRID, width=1)
        label = str(value)
        tw, th = _text_size(draw, label, FONT_TICK)
        draw.text((x - tw / 2, plot_bottom + 14), label, font=FONT_TICK, fill=MUTED)

    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill=AXIS, width=3)
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill=AXIS, width=3)

    _draw_rotated_text(
        image,
        (145, int((plot_top + plot_bottom) / 2) - 120),
        "Action class",
        FONT_AXIS,
        TEXT,
        angle=90,
    )
    xlabel = "Number of annotated clips"
    tw, _ = _text_size(draw, xlabel, FONT_AXIS)
    draw.text((plot_left + (plot_width - tw) / 2, HEIGHT - 120), xlabel, font=FONT_AXIS, fill=TEXT)

    dur_min = df["duration_seconds"].min()
    dur_mean = df["duration_seconds"].mean()
    dur_max = df["duration_seconds"].max()

    summary = "Total clips: {}    Unique classes: {}".format(
        len(df), df["action_name"].nunique()
    )
    dur_summary = "Duration (s): min {}    mean {}    max {}".format(
        _format_seconds(dur_min), _format_seconds(dur_mean), _format_seconds(dur_max)
    )
    draw.text((plot_left, 200), summary, font=FONT_SUBTITLE, fill=MUTED)
    draw.text((plot_left, 232), dur_summary, font=FONT_SUBTITLE, fill=MUTED)

    for idx, (label, value) in enumerate(top_counts.items()):
        y_center = plot_top + int((idx + 0.5) * y_step)
        y0 = int(y_center - bar_height / 2)
        y1 = int(y_center + bar_height / 2)
        bar_len = int((value / float(max_count)) * (plot_width - 40))
        draw.rounded_rectangle(
            (plot_left, y0, plot_left + bar_len, y1),
            radius=10,
            fill=BAR,
            outline=BAR_EDGE,
            width=2,
        )
        wrapped = textwrap.shorten(label.replace("_", " "), width=30, placeholder="...")
        tw, th = _text_size(draw, wrapped, FONT_LABEL)
        draw.text((plot_left - 20 - tw, y_center - th / 2), wrapped, font=FONT_LABEL, fill=TEXT)

        percentage = (value / total_count) * 100.0
        value_text = "{} ({:.1f}%)".format(int(value), percentage)
        vtw, vth = _text_size(draw, value_text, FONT_VALUE)
        draw.text((plot_left + bar_len + 14, y_center - vth / 2), value_text, font=FONT_VALUE, fill=BAR_EDGE)

    _save_png(image, "action_distribution_{}.png".format(split))


def create_duration_distribution_chart(df, split):
    durations = df["duration_seconds"].tolist()
    counts = _compute_duration_bin_counts(durations)
    max_count = max(max(counts), 1)
    mean_value = df["duration_seconds"].mean()
    median_value = df["duration_seconds"].median()
    under_one = (df["duration_seconds"] < 1.0).mean() * 100.0

    image = Image.new("RGBA", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(image)
    _draw_title(
        draw,
        "MECCANO {} split: action duration".format(split.upper()),
        "Histogram using bins 0.0-0.5s, 0.5-1.0s, 1.0-2.0s, and 2.0s+.",
    )

    card = (70, 170, WIDTH - 70, HEIGHT - 80)
    _draw_card(draw, card)

    plot_left = 180
    plot_top = 260
    plot_right = WIDTH - 430
    plot_bottom = HEIGHT - 190
    plot_width = plot_right - plot_left
    plot_height = plot_bottom - plot_top

    for i in range(6):
        y = plot_bottom - int((plot_height * i) / 5.0)
        draw.line((plot_left, y, plot_right, y), fill=GRID, width=1)
        value = int(round((max_count * i) / 5.0))
        label = str(value)
        tw, th = _text_size(draw, label, FONT_TICK)
        draw.text((plot_left - 18 - tw, y - th / 2), label, font=FONT_TICK, fill=MUTED)

    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill=AXIS, width=3)
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill=AXIS, width=3)

    _draw_rotated_text(
        image,
        (95, int((plot_top + plot_bottom) / 2) - 170),
        "Number of annotated clips",
        FONT_AXIS,
        TEXT,
        angle=90,
    )
    xlabel = "Duration bins"
    tw, _ = _text_size(draw, xlabel, FONT_AXIS)
    draw.text((plot_left + (plot_width - tw) / 2, HEIGHT - 130), xlabel, font=FONT_AXIS, fill=TEXT)

    bin_width_px = plot_width / float(len(counts))
    for idx, count in enumerate(counts):
        x0 = plot_left + int(idx * bin_width_px) + 24
        x1 = plot_left + int((idx + 1) * bin_width_px) - 24
        y1 = plot_bottom
        y0 = plot_bottom - int((count / float(max_count)) * (plot_height - 8))
        draw.rounded_rectangle((x0, y0, x1, y1), radius=10, fill=HIST, outline="#B45309", width=2)

        label = DURATION_BIN_LABELS[idx]
        tw, th = _text_size(draw, label, FONT_TICK)
        x_center = int((x0 + x1) / 2)
        draw.text((x_center - tw / 2, plot_bottom + 18), label, font=FONT_TICK, fill=MUTED)

        pct = (count / float(len(durations))) * 100.0 if durations else 0.0
        value_label = "{} ({:.1f}%)".format(count, pct)
        vtw, vth = _text_size(draw, value_label, FONT_SMALL)
        draw.text((x_center - vtw / 2, max(y0 - vth - 10, plot_top + 10)), value_label, font=FONT_SMALL, fill="#9A3412")

    legend_x = plot_right + 40
    legend_y = plot_top + 40
    draw.rounded_rectangle((legend_x, legend_y, WIDTH - 120, legend_y + 230), radius=18, fill="#FFF7ED", outline="#FCD34D", width=2)
    draw.text((legend_x + 24, legend_y + 18), "Summary", font=FONT_AXIS, fill=TEXT)
    summary_lines = [
        "Min: {}".format(_format_seconds(df["duration_seconds"].min())),
        "Median: {}".format(_format_seconds(median_value)),
        "Mean: {}".format(_format_seconds(mean_value)),
        "Max: {}".format(_format_seconds(df["duration_seconds"].max())),
        "Under 1 second: {:.1f}%".format(under_one),
    ]
    for idx, line in enumerate(summary_lines):
        draw.text((legend_x + 24, legend_y + 60 + idx * 34), line, font=FONT_LABEL, fill=TEXT)

    _save_png(image, "duration_distribution_{}.png".format(split))


def create_duration_by_action_chart(df, split):
    image = Image.new("RGBA", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(image)
    _draw_title(
        draw,
        "MECCANO {} split: duration by action".format(split.upper()),
        "All action classes (sorted by median duration).",
    )

    card = (70, 170, WIDTH - 70, HEIGHT - 80)
    _draw_card(draw, card)

    grouped = (
        df.groupby("action_name")["duration_seconds"]
        .agg(["count", "mean", "median", "min", "max"])
        .sort_values("median", ascending=False)
    )

    # Table layout (multi-column) so we can fit all classes in one image.
    table_left = 110
    table_top = 235
    table_right = WIDTH - 110
    table_bottom = HEIGHT - 120

    n_rows = len(grouped)
    # Use 2 wide columns for readability; 61 rows fits comfortably at our row height.
    n_cols = 2 if n_rows > 32 else 1
    rows_per_col = (n_rows + n_cols - 1) // n_cols
    col_width = int((table_right - table_left) / float(max(n_cols, 1)))

    headers = ["action", "med", "mean", "min", "max", "n"]
    header_y = table_top + 10
    row_start_y = header_y + 28
    row_h = 26

    # Draw column headers for each table column.
    for col_idx in range(n_cols):
        x0 = table_left + col_idx * col_width
        x_action = x0 + 10
        # numeric columns are right-aligned to keep a clean table edge
        x_median_r = x0 + int(col_width * 0.72)
        x_mean_r = x0 + int(col_width * 0.82)
        x_min_r = x0 + int(col_width * 0.90)
        x_max_r = x0 + int(col_width * 0.96)
        x_n_r = x0 + int(col_width * 0.99)

        draw.text((x_action, header_y), headers[0], font=FONT_TABLE_BOLD, fill=MUTED)
        for header, xr in [
            (headers[1], x_median_r),
            (headers[2], x_mean_r),
            (headers[3], x_min_r),
            (headers[4], x_max_r),
            (headers[5], x_n_r),
        ]:
            tw, _ = _text_size(draw, header, FONT_TABLE_BOLD)
            draw.text((xr - tw, header_y), header, font=FONT_TABLE_BOLD, fill=MUTED)

        draw.line((x0 + 6, header_y + 28, x0 + col_width - 6, header_y + 28), fill=GRID, width=2)

    # Render rows, split across columns.
    for row_idx, (action, row) in enumerate(grouped.iterrows()):
        col_idx = row_idx // rows_per_col
        pos_in_col = row_idx % rows_per_col
        if col_idx >= n_cols:
            break

        x0 = table_left + col_idx * col_width
        y = row_start_y + pos_in_col * row_h

        # Alternating row background for scanability.
        if pos_in_col % 2 == 0:
            draw.rounded_rectangle(
                (x0 + 6, y - 2, x0 + col_width - 6, y + row_h - 6),
                radius=8,
                fill="#FAF8F3",
                outline=None,
            )

        x_action = x0 + 10
        x_median_r = x0 + int(col_width * 0.72)
        x_mean_r = x0 + int(col_width * 0.82)
        x_min_r = x0 + int(col_width * 0.90)
        x_max_r = x0 + int(col_width * 0.96)
        x_n_r = x0 + int(col_width * 0.99)

        # Render full action name (no ellipses); action names are short enough to fit in the widened table.
        action_txt = action.replace("_", " ")
        draw.text((x_action, y), action_txt, font=FONT_TABLE, fill=TEXT)
        for value_txt, xr in [
            (_format_seconds(row["median"]), x_median_r),
            (_format_seconds(row["mean"]), x_mean_r),
            (_format_seconds(row["min"]), x_min_r),
            (_format_seconds(row["max"]), x_max_r),
            (str(int(row["count"])), x_n_r),
        ]:
            tw, _ = _text_size(draw, value_txt, FONT_TABLE)
            draw.text((xr - tw, y), value_txt, font=FONT_TABLE, fill=TEXT)

    _save_png(image, "duration_by_action_{}.png".format(split))


def action_stats(split):
    df = prepare_dataframe(load_action_labels(split))
    create_action_distribution_chart(df, split)
    create_duration_distribution_chart(df, split)
    create_duration_by_action_chart(df, split)


def main():
    _ensure_output_dirs()
    for mode in ["train", "val", "test"]:
        action_stats(mode)


if __name__ == "__main__":
    main()
