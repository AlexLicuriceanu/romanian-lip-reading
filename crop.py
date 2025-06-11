import cv2
import numpy as np
import onnxruntime as ort
import os
import json
from tqdm import tqdm
import subprocess
from pipeline_config import (
    CROP_FACEMESH_ONNX_MODEL_PATH,
    CROP_LIP_LANDMARKS,
    CROP_ONNX_PROVIDERS
)

# Load the ONNX model for face mesh
session = ort.InferenceSession(CROP_FACEMESH_ONNX_MODEL_PATH, providers=CROP_ONNX_PROVIDERS)
input_name = session.get_inputs()[0].name
output_names = [o.name for o in session.get_outputs()]
print(f"Loaded FaceMesh model from {CROP_FACEMESH_ONNX_MODEL_PATH}")

def process_batch(crop_output_dir, output_minimum_size, model_input_size, frames, input_batch,
                  out_writer_dict, width, height, padding, fps, video_name, segment_name):
    """Process a batch of frames to detect lip landmarks and crop the lips"""

    input_tensor = np.stack(input_batch)
    batch_size = input_tensor.shape[0]

    # Dummy tensors for cropping parameters - faces are already centered and cropped
    crop_x1 = np.zeros((batch_size, 1), dtype=np.int32)
    crop_y1 = np.zeros((batch_size, 1), dtype=np.int32)
    crop_width = np.full((batch_size, 1), model_input_size, dtype=np.int32)
    crop_height = np.full((batch_size, 1), model_input_size, dtype=np.int32)

    scores_batch, landmarks_batch = session.run(
        output_names,
        {
            input_name: input_tensor,
            'crop_x1': crop_x1,
            'crop_y1': crop_y1,
            'crop_width': crop_width,
            'crop_height': crop_height,
        }
    )

    for frame, score, landmarks in zip(frames, scores_batch, landmarks_batch):
        if score < 0.95:
            continue

        points = []

        # Convert landmarks to pixel coordinates
        for idx in CROP_LIP_LANDMARKS:
            x = int(landmarks[idx][0] / model_input_size * width)
            y = int(landmarks[idx][1] / model_input_size * height)
            points.append((x, y))

        # Calculate bounding box around the lip landmarks
        xs, ys = zip(*points)
        x1, y1 = max(min(xs) - padding, 0), max(min(ys) - padding, 0)
        x2, y2 = min(max(xs) + padding, width - 1), min(max(ys) + padding, height - 1)

        # Ensure the bounding box is square
        box_size = max(x2 - x1, y2 - y1)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        half = box_size // 2

        # Calculate square coordinates
        sq_x1 = max(cx - half, 0)
        sq_y1 = max(cy - half, 0)
        sq_x2 = min(sq_x1 + box_size, width)
        sq_y2 = min(sq_y1 + box_size, height)

        # Crop the lip region
        lip_crop = frame[sq_y1:sq_y2, sq_x1:sq_x2]
        if lip_crop.shape[0] < 5 or lip_crop.shape[1] < 5:
            continue

        out_path = os.path.join(crop_output_dir, video_name, f"{segment_name}.mp4")

        # Write the cropped lip region to the video file
        if segment_name not in out_writer_dict:
            output_size = (lip_crop.shape[1], lip_crop.shape[0])

            if output_minimum_size and output_minimum_size > output_size[0]:
                output_size = (output_minimum_size, output_minimum_size)

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_path, fourcc, fps, output_size)
            out_writer_dict[segment_name] = (writer, output_size)

        writer, output_size = out_writer_dict[segment_name]
        lip_crop_resized = cv2.resize(lip_crop, output_size)
        writer.write(lip_crop_resized)


def crop_stage(asd_output_dir, crop_output_dir, output_minimum_size, model_input_size, padding, batch_size, crop_with_audio, crop_lips):
    if not os.path.exists(crop_output_dir):
        os.makedirs(crop_output_dir)

    asd_files = [f for f in os.listdir(asd_output_dir) if f.endswith(".json")]
    total_files = len(asd_files)

    for file in tqdm(asd_files, desc="Cropping stage", total=total_files, unit="file"):
        with open(os.path.join(asd_output_dir, file), "r", encoding="utf-8") as f:
            data = json.load(f)

        video_path = data.get("video_path")
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        video_output_dir = os.path.join(crop_output_dir, video_name)

        os.makedirs(video_output_dir, exist_ok=True)

        segments = data.get("segments", [])
        if not segments:
            continue

        crop_manifest = {
            "video_path": video_path,
            "segments": [],
            "subsegments": {}
        }

        for segment in tqdm(segments, desc=f"{file}", leave=False):
            segment_path = segment.get("segment_path")
            segment_name = segment.get("segment_name")
            if not segment_path:
                continue
            
            # Count how many subsegments each segment has
            crop_manifest["subsegments"].setdefault(segment_name, 0)
            crop_manifest["subsegments"][segment_name] += 1

            cap = cv2.VideoCapture(segment_path)
            if not cap.isOpened():
                continue
            
            # Read the first frame to determine the video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            input_batch = []
            frames_batch = []
            out_writer_dict = {}

            # Read frames in batches
            while True:

                if not crop_lips:
                    break

                ret, frame = cap.read()
                if not ret:
                    break

                input_frame = cv2.resize(frame, (model_input_size, model_input_size))
                input_rgb = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                input_tensor = input_rgb.transpose(2, 0, 1)
                input_batch.append(input_tensor)
                frames_batch.append(frame)

                if len(input_batch) == batch_size:
                    process_batch(
                        crop_output_dir=crop_output_dir,
                        output_minimum_size=output_minimum_size,
                        model_input_size=model_input_size,
                        frames=frames_batch,
                        input_batch=input_batch,
                        out_writer_dict=out_writer_dict,
                        width=width, height=height,
                        padding=padding,
                        fps=fps,
                        video_name=video_name,
                        segment_name=segment_name
                    )

                    input_batch.clear()
                    frames_batch.clear()

            # Process leftover frames
            if input_batch and crop_lips:
                process_batch(
                    crop_output_dir=crop_output_dir,
                    output_minimum_size=output_minimum_size,
                    model_input_size=model_input_size,
                    frames=frames_batch,
                    input_batch=input_batch,
                    out_writer_dict=out_writer_dict,
                    width=width, height=height,
                    padding=padding,
                    fps=fps,
                    video_name=video_name,
                    segment_name=segment_name
                )

            cap.release()

            for writer, _ in out_writer_dict.values():
                writer.release()

            cropped_path = os.path.abspath(os.path.join(crop_output_dir, video_name, f"{segment_name}.mp4"))

            if crop_with_audio:
                final_path = cropped_path.replace(".mp4", "_with_audio.mp4")
                cmd = [
                    "ffmpeg", "-y", "-i", cropped_path, "-i", segment_path,
                    "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-shortest", final_path
                ]

                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                os.replace(final_path, cropped_path)

            crop_manifest["segments"].append({
                "start": segment["start"],
                "end": segment["end"],
                "label": segment.get("label", ""),
                "segment_path": segment_path,
                "segment_name": segment_name,
                "cropped_segment_path": cropped_path if crop_lips else segment_path
            })

        # Save the manifest to a JSON file
        manifest_path = os.path.join(crop_output_dir, f"{video_name}.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(crop_manifest, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    from pipeline_config import (
        ASD_OUTPUT_DIR,
        CROP_OUTPUT_DIR,
        CROP_OUTPUT_MINIMUM_SIZE,
        CROP_MODEL_INPUT_SIZE,
        CROP_PADDING,
        CROP_BATCH_SIZE,
        CROP_WITH_AUDIO,
        CROP_LIPS
    )

    crop_stage(
        asd_output_dir=ASD_OUTPUT_DIR,
        crop_output_dir=CROP_OUTPUT_DIR,
        output_minimum_size=CROP_OUTPUT_MINIMUM_SIZE,
        model_input_size=CROP_MODEL_INPUT_SIZE,
        padding=CROP_PADDING,
        batch_size=CROP_BATCH_SIZE,
        crop_with_audio=CROP_WITH_AUDIO,
        crop_lips=CROP_LIPS
    )
