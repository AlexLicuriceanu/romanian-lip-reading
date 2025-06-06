import os
import json
import subprocess
from tqdm import tqdm
from pipeline_config import ASD_OUTPUT_DIR
import time 

CROPPING_DIR = "./cropped_outputs"

def cropping_stage():
    os.makedirs(CROPPING_DIR, exist_ok=True)

    manifest_files = [f for f in os.listdir(ASD_OUTPUT_DIR) if f.endswith(".json")]
    total_files = len(manifest_files)

    for file in manifest_files:
        with open(os.path.join(ASD_OUTPUT_DIR, file), "r", encoding="utf-8") as f:
            data = json.load(f)

        video_path = data["video_path"]
        segments = data["segments"]

        for idx, segment in tqdm(enumerate(segments), total=len(segments), desc=f"Processing {file}"):

            start = round(segment["start"], 3)
            end = round(segment["end"], 3)
            duration = round(end - start, 3)

            label = segment.get("label")
            output_dir = os.path.join(CROPPING_DIR, label)
            os.makedirs(output_dir, exist_ok=True)

            # Count files in the output directory
            file_count = len([f for f in os.listdir(output_dir) if f.endswith(".mp4")])
            output_name = f"{file_count + 1}_{str.replace(label, " ", "_")}.mp4"

            segment_path = segment.get("segment_path")
            cmd = [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel", "error",
                "-ss", str(start),
                "-i", segment_path,
                "-t", str(duration),
                "-c", "copy",
                os.path.join(output_dir, output_name)
            ]

            try:
                subprocess.run(cmd)
            except subprocess.CalledProcessError as e:
                print(f"Error processing segment {idx+1} of {file}: {e}")
                continue

            print(os.path.join(LABELING_DIR, f"{idx+1}_{os.path.basename(video_path)}"))
            break
            # Here you would typically run your labeling tool or process
            # For example, you might call a command-line tool or a function that processes the segment
            # This is a placeholder for the actual labeling logic
            print(f"Labeling segment from {start} to {end} seconds (duration: {duration}s)")

            # Example command (replace with actual labeling command):
            # cmd = ["labeling_tool", "--input", video_path, "--output", segment_path, "--start", str(start), "--end", str(end)]
            # subprocess.run(cmd)
        # for idx, segment in tqdm(enumerate(segments), desc=f"Processing {file}", total=len(segments)):
        #     start = round(segment["start"], 3)
        #     end = round(segment["end"], 3)

        #     duration = round(end - start, 3)


if __name__ == "__main__":
    print("Cropping stage started.")
    start_time = time.time()
    cropping_stage()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Cropping stage completed in {elapsed_time:.2f} seconds.")
    print("Cropping stage finished.\n")
