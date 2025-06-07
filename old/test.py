import cv2
import numpy as np
import onnxruntime as ort
import time
import os
import json
from tqdm import tqdm
from pipeline_config import ASD_OUTPUT_DIR
import subprocess

# Constants
MODEL_PATH = "face_mesh_Nx3x192x192_post.onnx"
INPUT_SIZE = 192
OUTPUT_MINIMUM_SIZE = 256
CROP_OUTPUT_DIR = "./cropped_outputs/"
CROP_WITH_AUDIO = True
CROP_PADDING = 10
BATCH_SIZE = 2048

LIP_LANDMARKS = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
    146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308,
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308
]

PROVIDERS = ['CUDAExecutionProvider', 'CPUExecutionProvider']

so = ort.SessionOptions()
so.intra_op_num_threads = 4 
session = ort.InferenceSession(MODEL_PATH, sess_options=so, providers=PROVIDERS)
input_name = session.get_inputs()[0].name
output_names = [o.name for o in session.get_outputs()]

def process_batch(frames, input_batch, out_writer_dict, width, height, fps, video_name, segment_name):
    input_tensor = np.stack(input_batch)
    batch_size = input_tensor.shape[0]

    crop_x1 = np.zeros((batch_size, 1), dtype=np.int32)
    crop_y1 = np.zeros((batch_size, 1), dtype=np.int32)
    crop_width = np.full((batch_size, 1), INPUT_SIZE, dtype=np.int32)
    crop_height = np.full((batch_size, 1), INPUT_SIZE, dtype=np.int32)

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
        for idx in LIP_LANDMARKS:
            x = int(landmarks[idx][0] / INPUT_SIZE * width)
            y = int(landmarks[idx][1] / INPUT_SIZE * height)
            points.append((x, y))

        xs, ys = zip(*points)
        x1, y1 = max(min(xs) - CROP_PADDING, 0), max(min(ys) - CROP_PADDING, 0)
        x2, y2 = min(max(xs) + CROP_PADDING, width - 1), min(max(ys) + CROP_PADDING, height - 1)

        box_size = max(x2 - x1, y2 - y1)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        half = box_size // 2

        sq_x1 = max(cx - half, 0)
        sq_y1 = max(cy - half, 0)
        sq_x2 = min(sq_x1 + box_size, width)
        sq_y2 = min(sq_y1 + box_size, height)

        lip_crop = frame[sq_y1:sq_y2, sq_x1:sq_x2]
        if lip_crop.shape[0] < 5 or lip_crop.shape[1] < 5:
            continue

        out_path = os.path.join(CROP_OUTPUT_DIR, video_name, f"{segment_name}.mp4")

        if segment_name not in out_writer_dict:
            output_size = (lip_crop.shape[1], lip_crop.shape[0])
            if OUTPUT_MINIMUM_SIZE and OUTPUT_MINIMUM_SIZE > output_size[0]:
                output_size = (OUTPUT_MINIMUM_SIZE, OUTPUT_MINIMUM_SIZE)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_path, fourcc, fps, output_size)
            out_writer_dict[segment_name] = (writer, output_size)

        writer, output_size = out_writer_dict[segment_name]
        lip_crop_resized = cv2.resize(lip_crop, output_size)
        writer.write(lip_crop_resized)

def crop_stage():
    if not os.path.exists(CROP_OUTPUT_DIR):
        os.makedirs(CROP_OUTPUT_DIR)

    manifests = [f for f in os.listdir(ASD_OUTPUT_DIR) if f.endswith(".json")]
    for file in tqdm(manifests, desc="Cropping stage"):
        with open(os.path.join(ASD_OUTPUT_DIR, file), "r", encoding="utf-8") as f:
            data = json.load(f)

        video_path = data.get("video_path")
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        video_output_dir = os.path.join(CROP_OUTPUT_DIR, video_name)
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

            crop_manifest["subsegments"].setdefault(segment_name, 0)
            crop_manifest["subsegments"][segment_name] += 1

            cap = cv2.VideoCapture(segment_path)
            if not cap.isOpened():
                continue

            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            input_batch = []
            frames_batch = []
            out_writer_dict = {}

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                input_frame = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
                input_rgb = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                input_tensor = input_rgb.transpose(2, 0, 1)
                input_batch.append(input_tensor)
                frames_batch.append(frame)

                if len(input_batch) == BATCH_SIZE:
                    process_batch(frames_batch, input_batch, out_writer_dict, width, height, fps, video_name, segment_name)
                    input_batch.clear()
                    frames_batch.clear()

            # Process leftover frames
            if input_batch:
                process_batch(frames_batch, input_batch, out_writer_dict, width, height, fps, video_name, segment_name)

            cap.release()

            for writer, _ in out_writer_dict.values():
                writer.release()

            cropped_path = os.path.abspath(os.path.join(CROP_OUTPUT_DIR, video_name, f"{segment_name}.mp4"))

            if CROP_WITH_AUDIO:
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
                "cropped_segment_path": cropped_path
            })

        manifest_path = os.path.join(CROP_OUTPUT_DIR, f"{video_name}.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(crop_manifest, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    crop_stage()
