import os
import json
from talknet.main import run_talknet_batch
from tqdm import tqdm
import multiprocessing as mp
from pipeline_config import (
        TRIM_OUTPUT_DIR, ASD_OUTPUT_DIR, ASD_MAX_WORKERS,
        DELETE_INTERMEDIATE_FILES, ASD_IN_MEMORY_THRESHOLD
    )

def asd_stage(trim_output_dir=TRIM_OUTPUT_DIR, asd_output_dir=ASD_OUTPUT_DIR):
    """
    Runs the ASD stage of the pipeline, processing each segment from the trimmed output directory.
    Arguments:
        trim_output_dir (str): Directory containing the trimmed output files.
        asd_output_dir (str): Directory to save the ASD output files.
    Returns:
        None
    """

    os.makedirs(asd_output_dir, exist_ok=True)

    # Recursively find all .json files in TRIM_OUTPUT_DIR
    trim_files = []
    for root, dirs, files in os.walk(trim_output_dir):
        for f in files:
            if f.endswith(".json"):
                trim_files.append(os.path.join(root, f))

    # Run the ASD stage
    for file in tqdm(trim_files, desc="ASD stage", unit="file", total=len(trim_files), leave=False):
        try:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"[ASD] Error decoding JSON from {file}: {e}")
            continue

        try:
            segments = data.get("segments")
            video_path = data.get("video_path")
        except KeyError as e:
            print(f"[ASD] Missing key in JSON from {file}: {e}")
            continue

        video_name = os.path.splitext(os.path.basename(video_path))[0]
        segments_dir = os.path.join(trim_output_dir, video_name)

        # Run TalkNet in parallel on all segment from the current video
        run_talknet_batch(
            clip_dir=segments_dir,
            save_dir=asd_output_dir,
            segments=segments,
            max_workers=ASD_MAX_WORKERS,
            in_memory_threshold=ASD_IN_MEMORY_THRESHOLD
        )
            
        # Compute the path to each segment for this video
        segment_paths = [os.path.join(asd_output_dir, video_name, segment_name) for segment_name in segments]
        
        # Set up manifest
        manifest = {
            "video_name": video_name,
            "segments": {}
        }

        # Merge all individual segment manifests into a single manifest
        for segment_path in segment_paths:
            segment_name = os.path.basename(segment_path)
            segment_manifest_path = os.path.join(segment_path, f"{segment_name}.json")

            if not os.path.exists(segment_manifest_path):
                print(f"[ASD] Manifest file not found for segment {segment_name}, skipping.")
                continue

            try:
                with open(segment_manifest_path, "r", encoding="utf-8") as f:
                    segment_data = json.load(f)
            except json.JSONDecodeError as e:
                print(f"[ASD] Error decoding JSON from {segment_manifest_path}: {e}")
                continue

            # Add segment data to the main manifest
            manifest["segments"][segment_name] = segment_data

        # Save the final manifest for the video
        manifest_path = os.path.join(asd_output_dir, f"{video_name}.json")
        try:
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False)
        except IOError as e:
            print(f"[ASD] Error writing manifest for {video_name}: {e}")
            continue


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    asd_stage(
        trim_output_dir=TRIM_OUTPUT_DIR,
        asd_output_dir=ASD_OUTPUT_DIR
    )