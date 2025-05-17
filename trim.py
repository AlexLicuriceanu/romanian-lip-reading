import os
import json
import subprocess
from tqdm import tqdm
from pipeline_config import COMP_OUTPUT_DIR, VIDEO_DIR, TRIM_OUTPUT_DIR, PADDING

def trim_stage():
    os.makedirs(TRIM_OUTPUT_DIR, exist_ok=True)

    comp_files = [f for f in os.listdir(COMP_OUTPUT_DIR) if f.endswith(".json")]
    total_files = len(comp_files)

    for i, file in enumerate(comp_files):
        with open(os.path.join(COMP_OUTPUT_DIR, file), "r", encoding="utf-8") as f:
            data = json.load(f)

        video_path = data["video_path"]
        segments = data["segments"]

        video_name = os.path.splitext(os.path.basename(video_path))[0]
        video_file = os.path.join(VIDEO_DIR, f"{video_name}.mp4")

        if not os.path.exists(video_file):
            print(f"Video file not found: {video_file}")
            continue

        video_trim_dir = os.path.join(TRIM_OUTPUT_DIR, video_name)
        os.makedirs(video_trim_dir, exist_ok=True)

        print(f"Processing: {video_file} ({i + 1}/{total_files})")

        manifest = {
            "video_path": video_path,
            "segments": []
        }

        for idx, segment in tqdm(enumerate(segments), total=len(segments), desc=f"Trimming"):

            start = max(0, segment["start"] - PADDING)
            end = segment["end"] + PADDING

            if idx == 0:
                start = segment["start"]
            elif idx == len(segments) - 1:
                end = segment["end"]

            duration = end - start

            out_name = f"{idx+1:06d}_{video_name}.mp4"
            out_path = os.path.join(video_trim_dir, out_name)

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

            subprocess.run(cmd)

            manifest["segments"].append({
                "clip": out_name,
                "text": segment["text"]
            })

        manifest["video_path"] = video_path

        manifest_path = os.path.join(video_trim_dir, f"trimmed_{video_name}.json")
        with open(manifest_path, "w", encoding="utf-8") as mf:
            json.dump(manifest, mf, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    print("Trim stage started.")
    trim_stage()
    print("Trim stage finished.\n")