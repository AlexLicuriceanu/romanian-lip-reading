import os
import json
import subprocess
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from pipeline_config import VIDEO_DIR, COMP_OUTPUT_DIR, TRIM_OUTPUT_DIR, TRIM_PADDING, TRIM_MAX_WORKERS

def process_segment(trim_padding, segment, idx, video_file, video_name, video_length, video_trim_dir):
    """
    Trim a single segment from the video file.
    Arguments:
        trim_padding (int): Padding to apply before and after the segment.
        segment (dict): Segment information containing 'start' and 'end'.
        idx (int): Index of the segment in the list.
        video_file (str): Path to the original video file.
        video_name (str): Name of the video file without extension.
        video_length (int): Total length of the video in seconds.
        video_trim_dir (str): Directory to save trimmed segments.
    Returns:
        tuple: A tuple containing the segment name and updated segment information.
    """
    start = max(0, segment["start"] - trim_padding)
    end = min(video_length, segment["end"] + trim_padding)
    duration = end - start

    out_name = f"{idx + 1}_{video_name}.mp4"
    out_path = os.path.join(video_trim_dir, out_name)
    abs_out_path = os.path.abspath(out_path)

    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-ss", str(start),
        "-i", video_file,
        "-t", str(duration),
        "-c", "copy",
        out_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    segment["segment_path"] = abs_out_path

    return f"{idx + 1}_{video_name}", segment

def process_manifest(file, video_dir, comp_output_dir, trim_output_dir, trim_padding, trim_max_workers):
    """
    Process a single manifest file to trim segments from the corresponding video.
    Arguments:
        file (str): Name of the manifest file to process.
        video_dir (str): Directory containing the original video files.
        comp_output_dir (str): Directory containing the compiled manifest files.
        trim_output_dir (str): Directory to save trimmed segments.
        trim_padding (int): Padding to apply before and after each segment.
        trim_max_workers (int): Maximum number of threads to use for trimming.
    Returns:
        None
    """
    try:
        with open(os.path.join(comp_output_dir, file), "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[TRIM] Failed to load {file}: {e}")
        return

    video_path = data["video_path"]
    segments = data["segments"]
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_file = os.path.join(video_dir, f"{video_name}.mp4")

    if not os.path.exists(video_file):
        print(f"[TRIM] Video file not found: {video_file}")
        return

    video_length = segments[-1]["end"]
    video_trim_dir = os.path.join(trim_output_dir, video_name)
    os.makedirs(video_trim_dir, exist_ok=True)

    # Thread-based segment parallelism
    updated_segments = {}
    with ThreadPoolExecutor(max_workers=trim_max_workers) as executor:
        futures = [
            executor.submit(
                process_segment,
                trim_padding, segment, idx,
                video_file, video_name,
                video_length, video_trim_dir
            )
            for idx, segment in enumerate(segments)
        ]

        for future in tqdm(futures, total=len(futures), desc=video_name, unit="segment", leave=False):
            name, seg = future.result()
            updated_segments[name] = seg

    manifest = {
        "video_path": video_path,
        "text": data.get("text", ""),
        "segments": updated_segments
    }

    manifest_path = os.path.join(trim_output_dir, f"{video_name}.json")
    try:
        with open(manifest_path, "w", encoding="utf-8") as mf:
            json.dump(manifest, mf, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"[TRIM] Failed to write manifest for {video_name}: {e}")
        return
    
def trim_stage(video_dir, comp_output_dir, trim_output_dir, trim_padding, trim_max_workers):
    """
    Sequentially process each manifest file, parallelizing per-file segment trimming.
    Arguments:
        video_dir (str): Directory containing the original video files.
        comp_output_dir (str): Directory containing the compiled manifest files.
        trim_output_dir (str): Directory to save trimmed segments.
        trim_padding (int): Padding to apply before and after each segment.
        trim_max_workers (int): Maximum number of threads to use for trimming.
    Returns:
        None
    """
    os.makedirs(trim_output_dir, exist_ok=True)
    manifest_files = [f for f in os.listdir(comp_output_dir) if f.endswith(".json")]

    for file in tqdm(manifest_files, desc="Trimming stage", unit="file"):
        process_manifest(
            file,
            video_dir,
            comp_output_dir,
            trim_output_dir,
            trim_padding,
            trim_max_workers
        )

    # Reset terminal - temporary solution FIXME
    try:
        os.system("stty sane")
    except Exception as e:
        print(f"Error resetting terminal: {e}")

if __name__ == "__main__":
    trim_stage(
        video_dir=VIDEO_DIR,
        comp_output_dir=COMP_OUTPUT_DIR,
        trim_output_dir=TRIM_OUTPUT_DIR,
        trim_padding=TRIM_PADDING,
        trim_max_workers=TRIM_MAX_WORKERS
    )
