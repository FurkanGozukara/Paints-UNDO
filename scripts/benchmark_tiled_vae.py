import argparse
import csv
import glob
import json
import math
import os
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageStat


ROOT = Path(__file__).resolve().parents[1]
BENCH_DIR = ROOT / "results" / "benchmarks" / "full_default_tiled_vae"


SETTINGS = {
    "image_width": 1024,
    "image_height": 1024,
    "undo_steps": [400, 600, 800, 900, 950, 999],
    "keyframe_steps": 50,
    "keyframe_cfg": 3.0,
    "video_steps": 50,
    "video_cfg": 7.5,
    "fps": 4,
    "fixed_seeds": {
        "keyframe": 12345,
        "video": 123,
    },
}


PROMPT = "1girl, masterpiece, best quality"
NEGATIVE_PROMPT = "lowres, bad anatomy, bad hands, cropped, worst quality"


class ConsoleProgress:
    def tqdm(self, iterable, **kwargs):
        try:
            from tqdm.auto import tqdm

            return tqdm(iterable, **kwargs)
        except Exception:
            return iterable


def bool_arg(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}")


def sync_cuda(torch_module):
    if torch_module.cuda.is_available():
        torch_module.cuda.synchronize()


def reset_cuda_peak(torch_module):
    if torch_module.cuda.is_available():
        torch_module.cuda.empty_cache()
        torch_module.cuda.reset_peak_memory_stats()
        torch_module.cuda.synchronize()


def cuda_peak_gb(torch_module):
    if not torch_module.cuda.is_available():
        return 0.0
    torch_module.cuda.synchronize()
    return torch_module.cuda.max_memory_allocated() / (1024 ** 3)


def frame_stats(path):
    image = Image.open(path).convert("RGB")
    stat = ImageStat.Stat(image)
    return {
        "mean": [float(x) for x in stat.mean],
        "extrema": [[int(lo), int(hi)] for lo, hi in image.getextrema()],
        "bytes": int(Path(path).stat().st_size),
    }


def latest_video_folder(app_module, input_name):
    pattern = os.path.join(app_module.result_dir, "video_frames", f"{input_name}_final_frames_*")
    folders = [Path(p) for p in glob.glob(pattern) if Path(p).is_dir()]
    if not folders:
        return None
    return max(folders, key=lambda p: p.stat().st_mtime)


def run_combo(keyframe_tiled_vae, video_tiled_vae):
    os.environ.setdefault("PAINTS_UNDO_UNET_XFORMERS_OP", "triton_splitk,auto,sdpa")
    os.environ.setdefault("PAINTS_UNDO_ATTENTION_VALIDATE", "first")

    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))

    import torch
    import gradio_app as app

    input_path = (ROOT.parent / "Test_Image.png").resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Missing benchmark input image: {input_path}")

    combo = f"k{int(keyframe_tiled_vae)}_v{int(video_tiled_vae)}"
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    progress = ConsoleProgress()

    print(f"[BENCH] Starting {combo}")
    print(f"[BENCH] Input: {input_path}")
    print(f"[BENCH] Key-frame tiled VAE: {keyframe_tiled_vae}")
    print(f"[BENCH] Video tiled VAE: {video_tiled_vae}")

    reset_cuda_peak(torch)
    step2_start = time.perf_counter()
    keyframes, keyframe_folder, keyframe_seed = app.process(
        str(input_path),
        PROMPT,
        SETTINGS["undo_steps"],
        SETTINGS["image_width"],
        SETTINGS["image_height"],
        SETTINGS["fixed_seeds"]["keyframe"],
        SETTINGS["keyframe_steps"],
        NEGATIVE_PROMPT,
        SETTINGS["keyframe_cfg"],
        False,
        False,
        keyframe_tiled_vae,
        progress=progress,
    )
    sync_cuda(torch)
    step2_seconds = time.perf_counter() - step2_start
    step2_peak_gb = cuda_peak_gb(torch)

    gallery_keyframes = [(path,) for path in keyframes]
    reset_cuda_peak(torch)
    step3_start = time.perf_counter()
    mp4_path, video = app.process_video(
        gallery_keyframes,
        PROMPT,
        SETTINGS["video_steps"],
        SETTINGS["video_cfg"],
        SETTINGS["fps"],
        SETTINGS["fixed_seeds"]["video"],
        str(input_path),
        False,
        False,
        video_tiled_vae,
        progress=progress,
    )
    sync_cuda(torch)
    step3_seconds = time.perf_counter() - step3_start
    step3_peak_gb = cuda_peak_gb(torch)

    input_name = input_path.stem
    video_folder = latest_video_folder(app, input_name)
    video_frame_paths = []
    if video_folder is not None:
        video_frame_paths = sorted(video_folder.glob("*.png"))

    result = {
        "combo": combo,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
        "keyframe_tiled_vae": bool(keyframe_tiled_vae),
        "video_tiled_vae": bool(video_tiled_vae),
        "settings": dict(SETTINGS),
        "step2_seconds": step2_seconds,
        "step2_peak_gb": step2_peak_gb,
        "step3_seconds": step3_seconds,
        "step3_peak_gb": step3_peak_gb,
        "total_seconds": step2_seconds + step3_seconds,
        "peak_gb_max": max(step2_peak_gb, step3_peak_gb),
        "keyframe_folder": str(keyframe_folder),
        "keyframe_count": len(keyframes),
        "keyframe_seed": keyframe_seed,
        "mp4_path": str(mp4_path),
        "mp4_bytes": int(Path(mp4_path).stat().st_size) if Path(mp4_path).exists() else 0,
        "video_folder": str(video_folder) if video_folder else None,
        "video_frame_count": len(video_frame_paths) if video_frame_paths else len(video),
    }

    if video_frame_paths:
        result["first_video_frame"] = frame_stats(video_frame_paths[0])
        result["last_video_frame"] = frame_stats(video_frame_paths[-1])

    output_path = BENCH_DIR / f"{combo}.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"[BENCH] Wrote {output_path}")
    return result


def load_results():
    results = []
    for path in sorted(BENCH_DIR.glob("k*_v*.json")):
        results.append(json.loads(path.read_text(encoding="utf-8")))
    expected = {"k0_v0", "k0_v1", "k1_v0", "k1_v1"}
    found = {row["combo"] for row in results}
    missing = sorted(expected - found)
    if missing:
        raise RuntimeError(f"Missing benchmark result JSONs: {', '.join(missing)}")
    return sorted(results, key=lambda row: (row["keyframe_tiled_vae"], row["video_tiled_vae"]))


def combo_label(row):
    key = "Key ON" if row["keyframe_tiled_vae"] else "Key OFF"
    video = "Video ON" if row["video_tiled_vae"] else "Video OFF"
    return f"{key}\n{video}"


def load_font(size, bold=False):
    names = ["arialbd.ttf" if bold else "arial.ttf", "segoeuib.ttf" if bold else "segoeui.ttf"]
    for name in names:
        try:
            return ImageFont.truetype(name, size=size)
        except Exception:
            pass
    return ImageFont.load_default()


def draw_multiline_center(draw, xy, text, font, fill):
    lines = text.split("\n")
    line_heights = []
    widths = []
    for line in lines:
        box = draw.textbbox((0, 0), line, font=font)
        widths.append(box[2] - box[0])
        line_heights.append(box[3] - box[1] + 4)
    x, y = xy
    y_cursor = y
    for line, width, height in zip(lines, widths, line_heights):
        draw.text((x - width / 2, y_cursor), line, font=font, fill=fill)
        y_cursor += height


def draw_runtime_panel(draw, rows, box, fonts):
    x0, y0, x1, y1 = box
    axis_color = (60, 66, 76)
    grid_color = (224, 229, 236)
    step2_color = (57, 115, 198)
    step3_color = (232, 143, 51)
    text_color = (29, 34, 43)

    draw.text((x0, y0), "Runtime by stage (seconds)", font=fonts["subtitle"], fill=text_color)
    chart_top = y0 + 76
    chart_bottom = y1 - 72
    chart_left = x0 + 72
    chart_right = x1 - 26
    max_value = math.ceil(max(row["total_seconds"] for row in rows) * 1.12 / 40) * 40

    for tick in range(0, int(max_value) + 1, 40):
        y = chart_bottom - (tick / max_value) * (chart_bottom - chart_top)
        draw.line((chart_left, y, chart_right, y), fill=grid_color)
        draw.text((chart_left - 54, y - 8), str(tick), font=fonts["small"], fill=(82, 88, 98))

    draw.line((chart_left, chart_top, chart_left, chart_bottom), fill=axis_color, width=2)
    draw.line((chart_left, chart_bottom, chart_right, chart_bottom), fill=axis_color, width=2)

    group_width = (chart_right - chart_left) / len(rows)
    bar_width = min(92, group_width * 0.46)
    for index, row in enumerate(rows):
        cx = chart_left + group_width * (index + 0.5)
        base = chart_bottom
        step2_h = (row["step2_seconds"] / max_value) * (chart_bottom - chart_top)
        step3_h = (row["step3_seconds"] / max_value) * (chart_bottom - chart_top)
        draw.rectangle((cx - bar_width / 2, base - step2_h, cx + bar_width / 2, base), fill=step2_color)
        draw.rectangle(
            (cx - bar_width / 2, base - step2_h - step3_h, cx + bar_width / 2, base - step2_h),
            fill=step3_color,
        )
        draw.text(
            (cx - 34, base - step2_h - step3_h - 23),
            f"{row['total_seconds']:.1f}s",
            font=fonts["small_bold"],
            fill=text_color,
        )
        draw_multiline_center(draw, (cx, chart_bottom + 12), combo_label(row), fonts["small"], text_color)

    legend_y = y0 + 8
    draw.rectangle((x1 - 250, legend_y, x1 - 234, legend_y + 16), fill=step2_color)
    draw.text((x1 - 226, legend_y - 1), "Step 2 key frames", font=fonts["small"], fill=text_color)
    draw.rectangle((x1 - 250, legend_y + 26, x1 - 234, legend_y + 42), fill=step3_color)
    draw.text((x1 - 226, legend_y + 25), "Step 3 video frames", font=fonts["small"], fill=text_color)


def draw_vram_panel(draw, rows, box, fonts):
    x0, y0, x1, y1 = box
    axis_color = (60, 66, 76)
    grid_color = (224, 229, 236)
    step2_color = (57, 115, 198)
    step3_color = (38, 166, 125)
    text_color = (29, 34, 43)

    draw.text((x0, y0), "Peak CUDA memory by stage (GB)", font=fonts["subtitle"], fill=text_color)
    chart_top = y0 + 76
    chart_bottom = y1 - 72
    chart_left = x0 + 72
    chart_right = x1 - 26
    tick_step = 5
    max_value = math.ceil(max(row["peak_gb_max"] for row in rows) * 1.12 / tick_step) * tick_step

    for tick in range(0, int(max_value) + tick_step, tick_step):
        y = chart_bottom - (tick / max_value) * (chart_bottom - chart_top)
        draw.line((chart_left, y, chart_right, y), fill=grid_color)
        draw.text((chart_left - 44, y - 8), str(tick), font=fonts["small"], fill=(82, 88, 98))

    draw.line((chart_left, chart_top, chart_left, chart_bottom), fill=axis_color, width=2)
    draw.line((chart_left, chart_bottom, chart_right, chart_bottom), fill=axis_color, width=2)

    group_width = (chart_right - chart_left) / len(rows)
    bar_width = min(42, group_width * 0.22)
    for index, row in enumerate(rows):
        cx = chart_left + group_width * (index + 0.5)
        values = [
            ("S2", row["step2_peak_gb"], step2_color, -bar_width * 0.62),
            ("S3", row["step3_peak_gb"], step3_color, bar_width * 0.62),
        ]
        for short_label, value, color, offset in values:
            left = cx + offset - bar_width / 2
            right = cx + offset + bar_width / 2
            top = chart_bottom - (value / max_value) * (chart_bottom - chart_top)
            draw.rectangle((left, top, right, chart_bottom), fill=color)
            draw.text((left - 4, top - 20), f"{value:.1f}", font=fonts["small"], fill=text_color)
            draw.text((left + 7, chart_bottom + 8), short_label, font=fonts["small"], fill=text_color)
        draw_multiline_center(draw, (cx, chart_bottom + 30), combo_label(row), fonts["small"], text_color)

    legend_y = y0 + 8
    draw.rectangle((x1 - 250, legend_y, x1 - 234, legend_y + 16), fill=step2_color)
    draw.text((x1 - 226, legend_y - 1), "Step 2 peak", font=fonts["small"], fill=text_color)
    draw.rectangle((x1 - 250, legend_y + 26, x1 - 234, legend_y + 42), fill=step3_color)
    draw.text((x1 - 226, legend_y + 25), "Step 3 peak", font=fonts["small"], fill=text_color)


def write_summary_files(rows):
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = BENCH_DIR / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "combo",
                "keyframe_tiled_vae",
                "video_tiled_vae",
                "step2_seconds",
                "step3_seconds",
                "total_seconds",
                "step2_peak_gb",
                "step3_peak_gb",
                "peak_gb_max",
                "keyframe_count",
                "video_frame_count",
                "mp4_path",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["combo"],
                    row["keyframe_tiled_vae"],
                    row["video_tiled_vae"],
                    f"{row['step2_seconds']:.6f}",
                    f"{row['step3_seconds']:.6f}",
                    f"{row['total_seconds']:.6f}",
                    f"{row['step2_peak_gb']:.6f}",
                    f"{row['step3_peak_gb']:.6f}",
                    f"{row['peak_gb_max']:.6f}",
                    row.get("keyframe_count", ""),
                    row.get("video_frame_count", ""),
                    row.get("mp4_path", ""),
                ]
            )

    md_path = BENCH_DIR / "summary.md"
    lines = [
        "# Tiled VAE Benchmark",
        "",
        "| Combo | Key-frame tiled | Video tiled | Step 2 sec | Step 3 sec | Total sec | Step 2 GB | Step 3 GB | Max GB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {combo} | {key} | {video} | {s2:.2f} | {s3:.2f} | {total:.2f} | {g2:.2f} | {g3:.2f} | {gmax:.2f} |".format(
                combo=row["combo"],
                key=row["keyframe_tiled_vae"],
                video=row["video_tiled_vae"],
                s2=row["step2_seconds"],
                s3=row["step3_seconds"],
                total=row["total_seconds"],
                g2=row["step2_peak_gb"],
                g3=row["step3_peak_gb"],
                gmax=row["peak_gb_max"],
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, md_path


def render_chart(rows):
    width, height = 1500, 980
    image = Image.new("RGB", (width, height), (248, 250, 252))
    draw = ImageDraw.Draw(image)
    fonts = {
        "title": load_font(34, bold=True),
        "subtitle": load_font(24, bold=True),
        "small": load_font(17),
        "small_bold": load_font(17, bold=True),
    }
    text_color = (29, 34, 43)

    draw.text((52, 34), "Paints-UNDO default generation benchmark", font=fonts["title"], fill=text_color)
    gpu = rows[0].get("gpu", "GPU")
    settings = rows[0].get("settings", {})
    sub = (
        f"{gpu} | Step 2: {settings.get('keyframe_steps', '?')} steps at "
        f"{settings.get('image_width', '?')}x{settings.get('image_height', '?')} | "
        f"Step 3: {settings.get('video_steps', '?')} steps, fps {settings.get('fps', '?')}"
    )
    draw.text((54, 78), sub, font=fonts["small"], fill=(82, 88, 98))

    panel_fill = (255, 255, 255)
    panel_outline = (222, 227, 234)
    runtime_box = (44, 124, width - 44, 500)
    vram_box = (44, 548, width - 44, 930)
    draw.rounded_rectangle(runtime_box, radius=8, fill=panel_fill, outline=panel_outline, width=1)
    draw.rounded_rectangle(vram_box, radius=8, fill=panel_fill, outline=panel_outline, width=1)

    draw_runtime_panel(draw, rows, (72, 150, width - 72, 476), fonts)
    draw_vram_panel(draw, rows, (72, 574, width - 72, 906), fonts)

    output_path = BENCH_DIR / "tiled_vae_benchmark_chart.png"
    image.save(output_path)
    return output_path


def chart():
    rows = load_results()
    csv_path, md_path = write_summary_files(rows)
    chart_path = render_chart(rows)
    print(f"[BENCH] Wrote {csv_path}")
    print(f"[BENCH] Wrote {md_path}")
    print(f"[BENCH] Wrote {chart_path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark tiled VAE combinations.")
    parser.add_argument("--key-tiled", type=bool_arg, help="Enable Step 2 key-frame tiled VAE.")
    parser.add_argument("--video-tiled", type=bool_arg, help="Enable Step 3 video tiled VAE.")
    parser.add_argument("--chart", action="store_true", help="Render summary CSV, Markdown, and chart.")
    args = parser.parse_args()

    if args.chart:
        chart()
        return

    if args.key_tiled is None or args.video_tiled is None:
        parser.error("--key-tiled and --video-tiled are required unless --chart is used")

    run_combo(args.key_tiled, args.video_tiled)


if __name__ == "__main__":
    main()
