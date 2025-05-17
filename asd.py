import os
import json
from pipeline_config import ASD_OUTPUT_DIR, TRIM_OUTPUT_DIR
from talknet_asd.runTalkNet import run_talknet

def asd_stage():
    os.makedirs(ASD_OUTPUT_DIR, exist_ok=True)

    # Recursively find all trimmed_*.json files in subdirectories
    trimmed_files = []
    for root, dirs, files in os.walk(TRIM_OUTPUT_DIR):
        for f in files:
            if f.endswith(".mp4"):
                trimmed_files.append(os.path.join(root, f))
    
    total_files = len(trimmed_files)

    for i, trimmed_file in enumerate(trimmed_files):

        trimmed_file_name = os.path.basename(trimmed_file)
        print(f"Processing: {trimmed_file_name} ({i + 1}/{total_files})")
        
        # Set up paths
        trimmed_file_path = os.path.abspath(trimmed_file)
        output_filename = os.path.splitext(trimmed_file_name)[0]
        output_path = os.path.abspath(os.path.join(ASD_OUTPUT_DIR, output_filename))
        
        # Run ASD
        run_talknet(video_path=trimmed_file_path, output_dir=output_path)
        break



if __name__ == "__main__":
    print("ASD stage started.")
    asd_stage()
    print("ASD stage finished.\n")