import os
import json
from talknet_asd.runTalkNet import run_talknet
from talknet_asd.runTalkNet import load_yolo_model, load_talknet_model
import numpy as np
import glob
from tqdm import tqdm
from shutil import rmtree

def asd_stage(trim_output_dir, asd_output_dir, yolo_model, talknet_model, max_workers_dataloader, facedet_scale,
              min_track, num_failed_det, min_face_size, duration_set, crop_scale, start_time, duration,
              padding, max_word_gap, asd_debug, delete_intermediate_files, min_area_ratio):
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

        segments_length = len(segments)

        video_path = data.get("video_path")
        if not video_path:
            print(f"No video path found in {file}, skipping.")
            continue

        video_name = os.path.splitext(os.path.basename(video_path))[0]

        # Run TalkNet on all segment from the current video
        for i, segment in tqdm(enumerate(segments), desc=f"{video_name}", total=segments_length, leave=False):

            segment_path = segment.get("segment_path")

            if not segment_path:
                print(f"No segment path found in {segment}, skipping.")
                continue
            
            segment_name = os.path.splitext(os.path.basename(segment_path))[0]
            output_dir = os.path.join(asd_output_dir, video_name, segment_name)
            os.makedirs(output_dir, exist_ok=True)

            word_timestamps = segment.get("word_timestamps", [])

            # Run TalkNet on the current segment
            tracks, scores, args, files = run_talknet(
                yolo_model=yolo_model,
                talknet_model=talknet_model,
                video_path=segment_path,
                output_dir=output_dir,
                n_data_loader_thread=max_workers_dataloader,
                facedet_scale=facedet_scale,
                min_track=min_track,
                duration_set=duration_set,
                num_failed_det=num_failed_det,
                min_face_size=min_face_size,
                crop_scale=crop_scale,
                start_time=start_time,
                duration=duration,
                debug=asd_debug,
                visualize=False,
                word_timestamps=word_timestamps
            )

            # If no face was detected, skip this segment
            if scores is None or len(scores) == 0:
                # print("No valid scores found. Exiting.")

                if delete_intermediate_files:
                    os.remove(os.path.join(args.pyaviPath, "video.avi"))
                    os.remove(os.path.join(args.pyaviPath, "audio.wav"))
                    rmtree(args.pyframesPath, ignore_errors=True)
                continue
            
            # Identify best track
            best_index = max(range(len(scores)), key=lambda i: np.mean(scores[i]))
            first_frame = tracks[best_index]['track']['frame'][0]
            fps = 25.0
            cropped_video_start = word_timestamps[0]['start'] + (first_frame / fps)
            best_score = np.mean(scores[best_index])

            generate_manifest(tracks, scores, args, cropped_video_start, padding, max_word_gap)

            # Clean up unused crops
            for i, crop_path in enumerate(files):
                if i != best_index:
                    os.remove(crop_path)


            # Clean up intermediate files if specified
            if delete_intermediate_files:
                os.remove(os.path.join(args.pyaviPath, "video.avi"))
                os.remove(os.path.join(args.pyaviPath, "audio.wav"))
                rmtree(args.pyframesPath, ignore_errors=True)

        # Get all json files inside ./asd_outputs/<number>_<video_name>/
        segment_names = [os.path.splitext(os.path.basename(seg.get("segment_path", "")))[0] for seg in segments if seg.get("segment_path")]

        # Merge all json files from ./asd_outputs/segment_name/pyavi/
        asd_output_files = []

        for segment_name in segment_names:
            segment_manifest_dir = os.path.join(asd_output_dir,video_name, segment_name, "pyavi")

            if os.path.exists(segment_manifest_dir):
                asd_output_files.extend([os.path.join(segment_manifest_dir, f) for f in os.listdir(segment_manifest_dir) if f.endswith(".json")])

        # The contents of each json file is a list of dictionaries
        final_manifest = {"video_path": video_path, "segments": []}  # Key to hold list of all segment manifests

        for asd_file in asd_output_files:
            with open(asd_file, "r", encoding="utf-8") as f:
                segment_data = json.load(f)  # List of dicts from ASD output

            # Infer segment name from parent folder of pyavi directory
            segment_dir = os.path.dirname(asd_file)  # .../pyavi
            segment_name = os.path.basename(os.path.dirname(segment_dir))  # e.g., 0001_video_name

            # Rebuild full path to segment video
            segment_video_path = os.path.abspath(os.path.join(asd_output_dir, video_name, segment_name, "pycrop", "00000.avi"))

            for entry in segment_data:
                entry["segment_path"] = segment_video_path
                entry["segment_name"] = segment_name
                final_manifest["segments"].append(entry)

        # Save the manifest to a JSON file
        output_manifest_path = os.path.join(asd_output_dir, f"{os.path.splitext(os.path.basename(file))[0]}.json")
        with open(output_manifest_path, "w", encoding="utf-8") as f:
            json.dump(final_manifest, f, indent=2, ensure_ascii=False)


def generate_manifest(tracks, scores, args, segment_start, padding, max_word_gap):
    """Generates a manifest containing the active sepeaker segments and the speaker's face crop"""

    word_timestamps = args.wordTimestamps
    flist = glob.glob(os.path.join(args.pyframesPath, '*.jpg'))
    flist.sort()
    faces = [[] for _ in range(len(flist))]

    for tidx, track in enumerate(tracks):
        score = scores[tidx]
        for fidx, frame in enumerate(track['track']['frame'].tolist()):
            s = score[max(fidx - 2, 0): min(fidx + 3, len(score) - 1)]  # average smoothing
            s = np.mean(s)
            faces[frame].append({
                'track': tidx,
                'score': float(s),
                's': track['proc_track']['s'][fidx],
                'x': track['proc_track']['x'][fidx],
                'y': track['proc_track']['y'][fidx]
            })

    manifest = _generate_manifest(
        faces=faces,
        word_timestamps=args.wordTimestamps,
        start_time=word_timestamps[0]['start'],
        segment_start=segment_start,
        padding=padding,
        max_word_gap=max_word_gap
    )

    with open(os.path.join(args.pyaviPath, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    return manifest


def _generate_manifest(faces, word_timestamps, start_time, segment_start, padding, max_word_gap, fps=25.0, threshold=0.5, min_duration=0.2):
    """Helper function to generate the manifest from face detections and word timestamps"""

    active_segments = []
    current_segment = None

    for fidx, frame_faces in enumerate(faces):
        timestamp = start_time + (fidx / fps)
        active = any(face['score'] >= threshold for face in frame_faces)

        if active:
            if current_segment is None:
                current_segment = [timestamp, timestamp, [fidx]]
            else:
                current_segment[1] = timestamp
                current_segment[2].append(fidx)
        else:
            if current_segment:
                if current_segment[1] - current_segment[0] >= min_duration:
                    active_segments.append(tuple(current_segment))
                current_segment = None

    if current_segment and (current_segment[1] - current_segment[0] >= min_duration):
        active_segments.append(tuple(current_segment))

    # Merge adjacent segments if gap is small
    merged_segments = []
    for seg in active_segments:
        if not merged_segments:
            merged_segments.append(list(seg))
        else:
            prev_seg = merged_segments[-1]
            if seg[0] - prev_seg[1] <= max_word_gap:
                prev_seg[1] = seg[1]
                prev_seg[2].extend(seg[2])
            else:
                merged_segments.append(list(seg))

    active_segments = [tuple(seg) for seg in merged_segments]

    manifest = []

    for seg_start, seg_end, frame_indices in active_segments:
        words_in_segment = [
            word['word'] for word in word_timestamps
            if word['end'] >= seg_start - padding and word['start'] <= seg_end + padding
        ]

        if not words_in_segment:
            continue

        sentence = " ".join(words_in_segment).strip()
        entry = {
            "start": round(seg_start - segment_start, 3),
            "end": round(seg_end - segment_start, 3),
            "label": sentence,
        }

        manifest.append(entry)

    return manifest

if __name__ == "__main__":
    from pipeline_config import (
        TRIM_OUTPUT_DIR, ASD_OUTPUT_DIR, ASD_MAX_WORKERS_DATALOADER, ASD_FACEDET_SCALE, 
        ASD_MIN_TRACK, ASD_NUM_FAILED_DET, ASD_MIN_FACE_SIZE, ASD_CROP_SCALE, ASD_START_TIME, ASD_DURATION_SET,
        ASD_DURATION, ASD_MAX_WORD_GAP, ASD_DEBUG, TRIM_PADDING, ASD_YOLO_MODEL_PATH, ASD_TALKNET_MODEL_PATH,
        DELETE_INTERMEDIATE_FILES, ASD_MIN_AREA_RATIO
    )

    asd_yolo_model = load_yolo_model(model_path=ASD_YOLO_MODEL_PATH)
    asd_talknet_model = load_talknet_model(model_path=ASD_TALKNET_MODEL_PATH)

    asd_stage(
        trim_output_dir=TRIM_OUTPUT_DIR,
        asd_output_dir=ASD_OUTPUT_DIR,
        yolo_model=asd_yolo_model,
        talknet_model=asd_talknet_model,
        max_workers_dataloader=ASD_MAX_WORKERS_DATALOADER,
        facedet_scale=ASD_FACEDET_SCALE,
        min_track=ASD_MIN_TRACK,
        duration_set=ASD_DURATION_SET,
        num_failed_det=ASD_NUM_FAILED_DET,
        min_face_size=ASD_MIN_FACE_SIZE,
        crop_scale=ASD_CROP_SCALE,
        start_time=ASD_START_TIME,
        duration=ASD_DURATION,
        padding=TRIM_PADDING,
        max_word_gap=ASD_MAX_WORD_GAP,
        asd_debug=ASD_DEBUG,
        delete_intermediate_files=DELETE_INTERMEDIATE_FILES,
        min_area_ratio=ASD_MIN_AREA_RATIO
    )
    