import cv2
import numpy as np
import onnxruntime as ort
import os
import json
from tqdm import tqdm
import subprocess
import shutil
from pipeline_config import (
    CROP_FACEMESH_ONNX_MODEL_PATH,
    CROP_LIP_LANDMARKS,
    CROP_ONNX_PROVIDERS,
    ASD_OUTPUT_DIR, CROP_OUTPUT_DIR, CROP_OUTPUT_MINIMUM_SIZE,
    CROP_MODEL_INPUT_SIZE, CROP_PADDING, CROP_BATCH_SIZE,
    CROP_WITH_AUDIO, CROP_LIPS
)

# Load ONNX model
sess_options = ort.SessionOptions()
sess_options.intra_op_num_threads = os.cpu_count()
sess_options.inter_op_num_threads = 1
session = ort.InferenceSession(CROP_FACEMESH_ONNX_MODEL_PATH, sess_options=sess_options, providers=CROP_ONNX_PROVIDERS)
input_name = session.get_inputs()[0].name
output_names = [o.name for o in session.get_outputs()]

def process_batch(crop_output_dir, output_minimum_size, model_input_size, frames, input_batch,
                  out_writer_dict, width, height, padding, fps, label, segment_key):
    """
    Process a batch of frames to detect lip landmarks and crop the lip region.
    Arguments:
        crop_output_dir (str): Directory to save cropped lip segments.
        output_minimum_size (int): Minimum size for the output video.
        model_input_size (int): Input size for the model.
        frames (list): List of frames from the video.
        input_batch (list): List of input tensors for the model.
        out_writer_dict (dict): Dictionary to hold video writers for each segment.
        width (int): Width of the video frames.
        height (int): Height of the video frames.
        padding (int): Padding to apply around the cropped lip region.
        fps (float): Frames per second of the video.
        label (str): Label for the segment.
        segment_key (str): Unique key for the segment.
    Returns:
        None
    """
    input_tensor = np.stack(input_batch)
    batch_size = input_tensor.shape[0]

    # Dummy values, video is already face-cropped
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

    label_dir = os.path.join(crop_output_dir, label.replace(" ", "_"))
    os.makedirs(label_dir, exist_ok=True)
    out_path = os.path.join(label_dir, f"{segment_key}.mp4")

    for frame, score, landmarks in zip(frames, scores_batch, landmarks_batch):
        if score < 0.95:
            continue

        # Convert landmarks to pixel coordinates
        points = [(int(landmarks[idx][0] / model_input_size * width),
                   int(landmarks[idx][1] / model_input_size * height)) for idx in CROP_LIP_LANDMARKS]
        xs, ys = zip(*points)
        x1, y1 = max(min(xs) - padding, 0), max(min(ys) - padding, 0)
        x2, y2 = min(max(xs) + padding, width - 1), min(max(ys) + padding, height - 1)

        # Ensure the box is square
        box_size = max(x2 - x1, y2 - y1)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        half = box_size // 2
        sq_x1, sq_y1 = max(cx - half, 0), max(cy - half, 0)
        sq_x2, sq_y2 = min(sq_x1 + box_size, width), min(sq_y1 + box_size, height)

        # Crop the lip region
        lip_crop = frame[sq_y1:sq_y2, sq_x1:sq_x2]
        if lip_crop.shape[0] < 5 or lip_crop.shape[1] < 5:
            continue

        # Resize the cropped lip region to the minimum size
        if segment_key not in out_writer_dict:
            output_size = (lip_crop.shape[1], lip_crop.shape[0])
            if output_minimum_size and output_minimum_size > output_size[0]:
                output_size = (output_minimum_size, output_minimum_size)
            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, output_size)
            out_writer_dict[segment_key] = (writer, output_size)

        writer, output_size = out_writer_dict[segment_key]
        writer.write(cv2.resize(lip_crop, output_size))

def crop_stage(asd_output_dir, crop_output_dir, output_minimum_size, model_input_size,
               padding, batch_size, crop_with_audio, crop_lips):
    """
    Crop the lip region from the ASD outputs.
    Arguments:
        asd_output_dir (str): Directory containing the ASD output files.
        crop_output_dir (str): Directory to save cropped lip segments.
        output_minimum_size (int): Minimum size for the output video.
        model_input_size (int): Input size for the model.
        padding (int): Padding to apply around the cropped lip region.
        batch_size (int): Number of frames to process in a batch.
        crop_with_audio (bool): Whether to crop with audio.
        crop_lips (bool): Whether to crop the lip region.
    Returns:
        None
    """
    os.makedirs(crop_output_dir, exist_ok=True)
    asd_files = [f for f in os.listdir(asd_output_dir) if f.endswith(".json")]

    for file in tqdm(asd_files, desc="Cropping stage"):
        with open(os.path.join(asd_output_dir, file), "r", encoding="utf-8") as f:
            data = json.load(f)

        video_name = data.get("video_name", os.path.splitext(file)[0])
        segments = data.get("segments", [])

        # Flatten segments
        flattened_segments = []
        for segment_name, segment_data in segments.items():
            for segment in segment_data:
                flattened_segments.append({
                    "segment_path": segment.get("segment_path"),
                    "segment_name": segment_name,
                    "track_id": segment.get("track_id", 0),
                    "label": segment.get("label", "").strip()
                })

        segments = flattened_segments

        for segment in tqdm(segments, desc=f"{file}", leave=False, total=len(segments)):
            segment_path = segment.get("segment_path")
            segment_name = segment.get("segment_name", "unknown")
            track_id = segment.get("track_id", 0)
            label = segment.get("label", "").strip()

            if not segment_path or not os.path.exists(segment_path) or not label:
                continue
            
            segment_key = f"{segment_name}_track{track_id}"

            if not crop_lips:
                # Copy the file directly if not cropping lips
                label_dir = os.path.join(crop_output_dir, label.replace(" ", "_"))
                os.makedirs(label_dir, exist_ok=True)
                out_path = os.path.join(label_dir, f"{segment_key}.mp4")

                if not os.path.exists(out_path):
                    shutil.copy(segment_path, out_path)

                continue
            
            cap = cv2.VideoCapture(segment_path)
            if not cap.isOpened():
                continue

            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            input_batch, frames_batch, out_writer_dict = [], [], {}

            while True:
            
                ret, frame = cap.read()
                if not ret:
                    break
                resized = cv2.resize(frame, (model_input_size, model_input_size))
                input_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                input_tensor = input_rgb.transpose(2, 0, 1)
                input_batch.append(input_tensor)
                frames_batch.append(frame)

                if len(input_batch) == batch_size:
                    process_batch(crop_output_dir, output_minimum_size, model_input_size,
                                  frames_batch, input_batch, out_writer_dict,
                                  width, height, padding, fps, label, segment_key)
                    input_batch.clear()
                    frames_batch.clear()

            if input_batch and crop_lips:
                process_batch(crop_output_dir, output_minimum_size, model_input_size,
                              frames_batch, input_batch, out_writer_dict,
                              width, height, padding, fps, label, segment_key)

            cap.release()
            for writer, _ in out_writer_dict.values():
                writer.release()

            cropped_path = os.path.abspath(os.path.join(crop_output_dir, label.replace(" ", "_"), f"{segment_key}.mp4"))

            if crop_with_audio:
                temp_path = cropped_path.replace(".mp4", "_with_audio.mp4")

                cmd = [
                    "ffmpeg", "-y",
                    "-i", cropped_path,
                    "-i", segment_path,
                    "-map", "0:v:0", "-map", "1:a:0",
                    "-c:v", "copy", "-c:a", "aac",
                    "-shortest", temp_path
                ]

                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"[CROP] Failed to merge audio for {cropped_path}")
                    print(result.stderr)
                elif os.path.exists(temp_path):
                    os.replace(temp_path, cropped_path)
                else:
                    print(f"[CROP] ffmpeg did not produce expected output: {temp_path}")

if __name__ == "__main__":
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
