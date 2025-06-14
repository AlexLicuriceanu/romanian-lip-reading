import os
import json
from talknet.main import run_talknet_batch
import numpy as np
import glob
from tqdm import tqdm
from shutil import rmtree

def asd_stage(trim_output_dir, asd_output_dir, max_word_gap, debug, delete_intermediate_files):
    """Main function to run the ASD stage of the pipeline"""

    os.makedirs(asd_output_dir, exist_ok=True)

    # Recursively find all .json files in TRIM_OUTPUT_DIR
    trim_files = []
    for root, dirs, files in os.walk(trim_output_dir):
        for f in files:
            if f.endswith(".json"):
                trim_files.append(os.path.join(root, f))

    total_files = len(trim_files)

    # Run the ASD stage
    for file in tqdm(trim_files, desc="ASD stage", unit="file", total=total_files):
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)

        segments = data.get("segments", [])

        if not segments:
            print(f"No segments found in {file}, skipping.")
            continue

        video_path = data.get("video_path")
        if not video_path:
            print(f"No video path found in {file}, skipping.")
            continue

        video_name = os.path.splitext(os.path.basename(video_path))[0]
        segments_dir = os.path.join(trim_output_dir, video_name)

        # Run TalkNet on all segment from the current video
        run_talknet_batch(
            clip_dir=segments_dir,
            save_dir=asd_output_dir,
            segments=segments,
            max_workers=2,
            in_memory_threshold=500
        )
            
        segment_paths = [os.path.join(asd_output_dir, video_name, segment_name) for segment_name in segments]
        
        manifest = {}

        for segment_path in segment_paths:
            segment_name = os.path.basename(segment_path)
            segment_manifest_path = os.path.join(segment_path, f"{segment_name}.json")

            if not os.path.exists(segment_manifest_path):
                print(f"Manifest file not found for segment {segment_name}, skipping.")
                continue

            with open(segment_manifest_path, "r", encoding="utf-8") as f:
                segment_data = json.load(f)

            # Add segment data to the main manifest
            manifest[segment_name] = segment_data

        # Save the final manifest for the video
        manifest_path = os.path.join(asd_output_dir, f"{video_name}.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    import multiprocessing as mp

    

    # your normal entry point
    from pipeline_config import (
        TRIM_OUTPUT_DIR, ASD_OUTPUT_DIR, ASD_MAX_WORKERS_DATALOADER, ASD_FACEDET_SCALE, 
        ASD_MIN_TRACK, ASD_NUM_FAILED_DET, ASD_MIN_FACE_SIZE, ASD_CROP_SCALE, ASD_START_TIME, ASD_DURATION_SET,
        ASD_DURATION, ASD_MAX_WORD_GAP, ASD_DEBUG, TRIM_PADDING, ASD_YOLO_MODEL_PATH, ASD_TALKNET_MODEL_PATH,
        DELETE_INTERMEDIATE_FILES, ASD_MIN_AREA_RATIO
    )

    mp.set_start_method('spawn', force=True)
    asd_stage(
        trim_output_dir=TRIM_OUTPUT_DIR,
        asd_output_dir=ASD_OUTPUT_DIR,
        max_word_gap=ASD_MAX_WORD_GAP,
        debug=ASD_DEBUG,
        delete_intermediate_files=DELETE_INTERMEDIATE_FILES
    )