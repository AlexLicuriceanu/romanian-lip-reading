import os
import tqdm
import json
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np


def run_talknet(video_path, save_path, in_memory_threshold=0, return_visualization=False, start_seconds=0, end_seconds=-1):
    try:
        from .demoTalkNet import setup, main
        import gc
        gc.collect()

        s, DET = setup()
        faces = main(
            s, DET, video_path,
            start_seconds=start_seconds,
            end_seconds=end_seconds,
            return_visualization=return_visualization,
            face_boxes="",
            in_memory_threshold=in_memory_threshold,
            save_path=save_path
        )
        
        return faces
    
    except Exception as e:
        raise RuntimeError(f"Error: {str(e)}")
    
def run_talknet_batch(clip_dir, save_dir, segments, max_workers, in_memory_threshold=0):
    all_files = sorted(f for f in os.listdir(clip_dir) if f.endswith(".mp4"))
    full_paths = [os.path.join(clip_dir, f) for f in all_files]

    video_name = os.path.basename(clip_dir)
    os.makedirs(save_dir, exist_ok=True)
    save_paths = [os.path.join(save_dir, video_name, os.path.splitext(os.path.basename(path))[0]) for path in full_paths]

    for save_path in save_paths:
        os.makedirs(save_path, exist_ok=True)


    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(run_talknet, video_path, save_path, in_memory_threshold=in_memory_threshold): video_path
            for video_path, save_path in zip(full_paths, save_paths)
        }

        video_name = os.path.basename(clip_dir)
        for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc=video_name):
            try:
                tracks, scores, files, faces = future.result()

                # Map back to segment name
                video_path = futures[future]
                segment_name = os.path.splitext(os.path.basename(video_path))[0]

                if segment_name not in segments:
                    print(f"Segment '{segment_name}' not found in segments dict, skipping")
                    continue

                word_timestamps = segments[segment_name].get("word_timestamps", [])
                if not word_timestamps:
                    print(f"No word_timestamps for {segment_name}, skipping")
                    continue

                manifests = []
                fps = 25.0
                padding = 0.2
                offset = word_timestamps[0]["start"]

                # Compute the spoken words for each track
                for i, (track, score) in enumerate(zip(tracks, scores)):
                    if len(track['track']['frame']) == 0:
                        continue

                    avg_score = np.mean(score)
                    if avg_score < 0.4:
                        continue  # skip weak tracks

                    # Track-level start and end time in seconds
                    frames = track['track']['frame']
                    seg_start = frames[0] / fps
                    seg_end = frames[-1] / fps

                    abs_start = offset + seg_start
                    abs_end = offset + seg_end

                    # Collect words overlapping with this track's duration
                    spoken_words = [
                        w["word"]
                        for w in word_timestamps
                        if w["end"] >= abs_start - padding and w["start"] <= abs_end + padding
                    ]

                    if not spoken_words:
                        continue

                    sentence = " ".join(spoken_words).strip()
                    segment_path = os.path.join(save_dir, video_name, segment_name, "pycrop", f"{i}.avi")
                    manifests.append({
                        "track_id": i,
                        "label": sentence,
                        "segment_path": segment_path
                    })

                output_file = os.path.join(save_dir, video_name, segment_name, f"{segment_name}.json")
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(manifests, f, indent=2, ensure_ascii=False)


            except Exception as e:
                print(f"Error processing {futures[future]}: {e}")
    