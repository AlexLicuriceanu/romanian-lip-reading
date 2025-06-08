import os
import json
import shutil
import subprocess
import uuid
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat

UID_LENGTH = 8

def process_manifest_file(file, crop_output_dir, labeling_dir):
    """Process a single manifest file to put segments into labeled directories."""

    manifest_path = os.path.join(crop_output_dir, file)
    
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    segments = data["segments"]
    subsegments_count = data.get("subsegments", {})

    for idx, segment in enumerate(segments):
        start = round(segment["start"], 3)
        end = round(segment["end"], 3)
        duration = round(end - start, 3)

        label = segment.get("label")
        if not label:
            continue  # skip empty labels

        output_dir = os.path.join(labeling_dir, label)
        os.makedirs(output_dir, exist_ok=True)

        # Thread-safe unique filename using uid
        unique_id = uuid.uuid4().hex[:UID_LENGTH]
        output_name = f"{unique_id}_{label.replace(' ', '_')}.mp4"
        output_path = os.path.join(output_dir, output_name)

        cropped_segment_path = segment.get("cropped_segment_path", None)
        segment_name = segment["segment_name"]
        subsegment_count = subsegments_count.get(segment_name, 1)

        if subsegment_count == 1:
            try:
                shutil.copy2(cropped_segment_path, output_path)
            except Exception as e:
                print(f"Error copying segment {segment_name} of {file}: {e}")
        elif subsegment_count > 1:
            cmd = [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel", "error",
                "-ss", str(start),
                "-i", cropped_segment_path,
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

def label_stage(crop_output_dir, labeling_dir, max_workers=1):
    """Main function to handle the labeling stage of the pipeline"""

    os.makedirs(labeling_dir, exist_ok=True)

    manifest_files = [f for f in os.listdir(crop_output_dir) if f.endswith(".json")]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(tqdm(executor.map(process_manifest_file, manifest_files, repeat(crop_output_dir), repeat(labeling_dir)),
                  total=len(manifest_files), desc="Labeling stage"))

if __name__ == "__main__":
    from pipeline_config import LABELING_DIR, CROP_OUTPUT_DIR, LABELING_MAX_WORKERS
    
    label_stage(
        crop_output_dir=CROP_OUTPUT_DIR,
        labeling_dir=LABELING_DIR,
        max_workers=LABELING_MAX_WORKERS
    )
