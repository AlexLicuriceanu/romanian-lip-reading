import os
import json
import subprocess
from tqdm import tqdm
from pipeline_config import ASD_OUTPUT_DIR
import time 

# Constants
INPUT_SIZE = 192
OUTPUT_MINIMUM_SIZE = 256
CROP_OUTPUT_DIR = "./cropped_outputs/"
LABELING_DIR = "./labeled_outputs"
CROP_LIPS = True

def labeling_stage():
    os.makedirs(LABELING_DIR, exist_ok=True)

    manifest_files = [f for f in os.listdir(CROP_OUTPUT_DIR) if f.endswith(".json")]
    total_files = len(manifest_files)

    for file in tqdm(manifest_files, desc="Total", total=total_files, colour="cyan"):
        with open(os.path.join(CROP_OUTPUT_DIR, file), "r", encoding="utf-8") as f:
            data = json.load(f)

        video_path = data["video_path"]
        segments = data["segments"]
        subsegments_count = data.get("subsegments", {})

        for idx, segment in tqdm(enumerate(segments), total=len(subsegments_count), desc=f"{file}", leave=False):

            start = round(segment["start"], 3)
            end = round(segment["end"], 3)
            duration = round(end - start, 3)

            label = segment.get("label")
            output_dir = os.path.join(LABELING_DIR, label)
            os.makedirs(output_dir, exist_ok=True)

            # Count files in the output directory
            file_count = len([f for f in os.listdir(output_dir) if f.endswith(".mp4")])
            output_name = f"{file_count + 1}_{str.replace(label, " ", "_")}.mp4"

            segment_path = segment.get("segment_path") #FIXME
            cropped_segment_path = segment.get("cropped_segment_path")
            segment_name = segment.get("segment_name")
            subsegment_count = subsegments_count.get(segment_name, 1)

            output_path = os.path.join(output_dir, output_name)

            # Use the cropped segment in CROP_LIPS == True
            if CROP_LIPS and cropped_segment_path:
                segment_path = cropped_segment_path

            if subsegment_count == 1:
                # If there is only one subsegment, copy it directly
                cmd = [
                    "cp",
                    segment_path,
                    output_path
                ]

                try:
                    subprocess.run(cmd)
                except subprocess.CalledProcessError as e:
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
                    subprocess.run(cmd)
                except subprocess.CalledProcessError as e:
                    print(f"Error processing segment {idx+1} of {file}: {e}")
                    continue
            else:
                print(f"Subsegment count for {segment_name} is zero or negative, skipping.")
                continue


if __name__ == "__main__":
    print("Labeling stage started")
    start_time = time.time()
    labeling_stage()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Labeling stage completed. Time: {elapsed_time:.2f}")
