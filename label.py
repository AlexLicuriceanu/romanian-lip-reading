import os
import json
import shutil
import subprocess
import uuid
from tqdm import tqdm
from pipeline_config import ASD_OUTPUT_DIR
from concurrent.futures import ThreadPoolExecutor
import time

# Constants
INPUT_SIZE = 192
OUTPUT_MINIMUM_SIZE = 256
CROP_OUTPUT_DIR = "./cropped_outputs/"
LABELING_DIR = "./labeled_outputs"
CROP_LIPS = True
MAX_THREADS = 8
UID_LENGTH = 8

def process_manifest_file(file):
    manifest_path = os.path.join(CROP_OUTPUT_DIR, file)
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Failed to load {file}: {e}")
        return

    video_path = data["video_path"]
    segments = data["segments"]
    subsegments_count = data.get("subsegments", {})

    for idx, segment in enumerate(segments):
        start = round(segment["start"], 3)
        end = round(segment["end"], 3)
        duration = round(end - start, 3)

        label = segment.get("label")
        if not label:
            continue  # skip empty labels

        output_dir = os.path.join(LABELING_DIR, label)
        os.makedirs(output_dir, exist_ok=True)

        # Thread-safe unique filename using uid
        unique_id = uuid.uuid4().hex[:UID_LENGTH]
        output_name = f"{unique_id}_{label.replace(' ', '_')}.mp4"
        output_path = os.path.join(output_dir, output_name)

        try:
            segment_path = segment.get("segment_path")
            cropped_segment_path = segment.get("cropped_segment_path")
            segment_name = segment.get("segment_name")
            subsegment_count = subsegments_count.get(segment_name)
        except KeyError as e:
            print(f"Missing key in segment {idx} of {file}: {e}")
            continue
            


        if CROP_LIPS and cropped_segment_path:
            segment_path = cropped_segment_path

        if subsegment_count == 1:
            try:
                shutil.copy2(segment_path, output_path)
            except Exception as e:
                print(f"Error copying segment {segment_name} of {file}: {e}")
        elif subsegment_count > 1:
            cmd = [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel", "error",
                "-ss", str(start),
                "-i", segment_path,
                "-t", str(duration),
                "-c", "copy",
                output_path
            ]
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"FFmpeg error for {segment_name} of {file}: {e}")
        else:
            print(f"Invalid subsegment count for {segment_name} in {file}, skipping.")

def labeling_stage():
    os.makedirs(LABELING_DIR, exist_ok=True)

    manifest_files = [f for f in os.listdir(CROP_OUTPUT_DIR) if f.endswith(".json")]

    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        list(tqdm(executor.map(process_manifest_file, manifest_files),
                  total=len(manifest_files), desc="Labeling stage"))

if __name__ == "__main__":
    labeling_stage()
