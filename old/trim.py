import os
import json
import subprocess
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from pipeline_config import COMP_OUTPUT_DIR, VIDEO_DIR, TRIM_OUTPUT_DIR, TRIM_PADDING, TRIM_MAX_WORKERS

def process_segment(segment, idx, video_file, video_name, video_length, video_trim_dir):
    start = max(0, segment["start"] - TRIM_PADDING)
    end = min(video_length, segment["end"] + TRIM_PADDING)
    duration = end - start

    out_name = f"{idx + 1}_{video_name}.mp4"
    out_path = os.path.join(video_trim_dir, out_name)
    abs_out_path = os.path.abspath(out_path)

    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel", "error",
        "-ss", str(start),
        "-i", video_file,
        "-t", str(duration),
        "-c", "copy",
        out_path
    ]

    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    segment["segment_path"] = abs_out_path
    return segment

def process_manifest(file):
    try:
        with open(os.path.join(COMP_OUTPUT_DIR, file), "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Failed to load {file}: {e}")
        return

    video_path = data["video_path"]
    segments = data["segments"]
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_file = os.path.join(VIDEO_DIR, f"{video_name}.mp4")

    if not os.path.exists(video_file):
        print(f"Video file not found: {video_file}")
        return

    video_length = segments[-1]["end"]
    video_trim_dir = os.path.join(TRIM_OUTPUT_DIR, video_name)
    os.makedirs(video_trim_dir, exist_ok=True)

    # Multithreaded segment trimming
    updated_segments = []
    with ThreadPoolExecutor(max_workers=TRIM_MAX_WORKERS) as executor:
        futures = [
            executor.submit(process_segment, segment, idx, video_file, video_name, video_length, video_trim_dir)
            for idx, segment in enumerate(segments)
        ]

        for future in tqdm(futures, total=len(futures), desc=video_name, unit="segment", leave=False):
            updated_segments.append(future.result())

    # Sort segments to preserve original order
    updated_segments.sort(key=lambda seg: seg["segment_path"])

    manifest = {
        "video_path": video_path,
        "text": data.get("text", ""),
        "segments": updated_segments
    }

    manifest_path = os.path.join(TRIM_OUTPUT_DIR, f"{video_name}.json")
    with open(manifest_path, "w", encoding="utf-8") as mf:
        json.dump(manifest, mf, indent=4, ensure_ascii=False)

def trim_stage():
    os.makedirs(TRIM_OUTPUT_DIR, exist_ok=True)
    manifest_files = [f for f in os.listdir(COMP_OUTPUT_DIR) if f.endswith(".json")]

    with ThreadPoolExecutor() as executor:
        list(tqdm(executor.map(process_manifest, manifest_files), total=len(manifest_files), desc="Trimming stage", unit="file"))

    try:
        os.system("stty sane")
    except Exception as e:
        print(f"Error resetting terminal: {e}")

if __name__ == "__main__":
    trim_stage()
