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
OUTPUT_VIDEO = "output_facemesh.mp4"
OUTPUT_MINIMUM_SIZE = 256
CROP_OUTPUT_DIR = "./cropped_outputs/"
CROP_WITH_AUDIO = True  # Whether to keep audio in cropped videos
CROP_PADDING = 10

# https://github.com/tensorflow/tfjs-models/blob/838611c02f51159afdd77469ce67f0e26b7bbb23/face-landmarks-detection/src/mediapipe-facemesh/keypoints.ts
LIP_LANDMARKS = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, # Lips upper, outer
    146, 91, 181, 84, 17, 314, 405, 321, 375, 291, # Lips lower, outer
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, # Lips upper, inner
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308 # Lips lower, inner
]

PROVIDERS = [
    # (
    #     'TensorrtExecutionProvider',
    #     {
    #         'trt_engine_cache_enable': True,
    #         'trt_engine_cache_path': '.',
    #         'trt_fp16_enable': True,
    #     }
    # ),
    'CUDAExecutionProvider',
    'CPUExecutionProvider'
]


session = ort.InferenceSession(MODEL_PATH, providers=PROVIDERS)
input_name = session.get_inputs()[0].name
output_names = [o.name for o in session.get_outputs()]


def crop_stage():
    """Stage to crop the lip region from the face mesh landmarks."""
    if not os.path.exists(CROP_OUTPUT_DIR):
        os.makedirs(CROP_OUTPUT_DIR)

    manifests = [f for f in os.listdir(ASD_OUTPUT_DIR) if f.endswith(".json")]
    total_files = len(manifests)

    for file in tqdm(manifests, desc="Total cropping progress", total=total_files, colour="cyan"):
        
        # Open the manifest
        with open(os.path.join(ASD_OUTPUT_DIR, file), "r", encoding="utf-8") as f:
            data = json.load(f)

        # Get the video name
        video_path = data.get("video_path")
        video_name = os.path.splitext(os.path.basename(video_path))[0]

        # Create the output directory for this video
        video_output_dir = os.path.join(CROP_OUTPUT_DIR, video_name)
        os.makedirs(video_output_dir, exist_ok=True)

        # Get the video's segments
        segments = data.get("segments", [])
        if not segments:
            print(f"No segments found in {file}, skipping.")
            continue

        crop_manifest = {}
        crop_manifest["video_path"] = video_path
        crop_manifest["segments"] = []
        crop_manifest["subsegments"] = {}

        for idx, segment in tqdm(enumerate(segments), desc=f"{file}", total=len(segments), leave=False):
            
            # Get the segment path and name
            segment_path = segment.get("segment_path")
            segment_name = segment.get("segment_name")

            if not segment_path:
                print(f"No segment path found in segment {idx+1} of {file}, skipping.")
                continue

            # Count each segment_name to crop_manifest["subsegments"]
            crop_manifest["subsegments"].setdefault(segment_name, 0)
            crop_manifest["subsegments"][segment_name] += 1


            # Open the video
            cap = cv2.VideoCapture(segment_path)
            if not cap.isOpened():
                print(f"Could not open video file: {segment_path}")
                continue

            # Get video info
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out_writer = None  # Delay initialization
            output_size = None
            frame_idx = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                input_frame = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
                input_rgb = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                input_tensor = input_rgb.transpose(2, 0, 1)[np.newaxis, :]

                scores, landmarks = session.run(
                    output_names,
                    {
                        input_name: input_tensor,
                        'crop_x1': np.array([[0]], dtype=np.int32),
                        'crop_y1': np.array([[0]], dtype=np.int32),
                        'crop_width': np.array([[INPUT_SIZE]], dtype=np.int32),
                        'crop_height': np.array([[INPUT_SIZE]], dtype=np.int32),
                    }
                )

                for face, score in zip(landmarks, scores):
                    if score > 0.95:
                        points = []
                        for idx in LIP_LANDMARKS:
                            x = int(face[idx][0] / INPUT_SIZE * width)
                            y = int(face[idx][1] / INPUT_SIZE * height)
                            points.append((x, y))

                        # Get initial padded bounding box
                        padding = CROP_PADDING
                        xs, ys = zip(*points)
                        x1, y1 = max(min(xs) - padding, 0), max(min(ys) - padding, 0)
                        x2, y2 = min(max(xs) + padding, width - 1), min(max(ys) + padding, height - 1)

                        # Compute center and size
                        box_w = x2 - x1
                        box_h = y2 - y1
                        box_size = max(box_w, box_h)

                        # Center coordinates
                        cx = (x1 + x2) // 2
                        cy = (y1 + y2) // 2

                        # Make square box centered at (cx, cy)
                        half_size = box_size // 2
                        sq_x1 = max(cx - half_size, 0)
                        sq_y1 = max(cy - half_size, 0)
                        sq_x2 = min(sq_x1 + box_size, width)
                        sq_y2 = min(sq_y1 + box_size, height)

                        # Adjust again if crop exceeds image borders
                        if sq_x2 - sq_x1 != box_size:
                            sq_x1 = max(sq_x2 - box_size, 0)
                        if sq_y2 - sq_y1 != box_size:
                            sq_y1 = max(sq_y2 - box_size, 0)

                        # Final square crop
                        lip_crop = frame[sq_y1:sq_y2, sq_x1:sq_x2]

                        # Skip if crop is invalid
                        if lip_crop.shape[0] < 5 or lip_crop.shape[1] < 5:
                            continue

                        # Initialize writer on first valid crop
                        if out_writer is None:
                            output_size = (lip_crop.shape[1], lip_crop.shape[0])

                            if OUTPUT_MINIMUM_SIZE is not None and OUTPUT_MINIMUM_SIZE > output_size[0]:
                                output_size = (OUTPUT_MINIMUM_SIZE, OUTPUT_MINIMUM_SIZE)

                            out_writer = cv2.VideoWriter(
                                os.path.join(CROP_OUTPUT_DIR, video_name, f"{segment_name}.mp4"),
                                fourcc, fps, output_size
                            )

                        # Resize all subsequent crops to this size
                        lip_crop_resized = cv2.resize(lip_crop, output_size)
                        out_writer.write(lip_crop_resized)
                        break  # Only use the first face per frame if there are multiple

                frame_idx += 1

            cap.release()

            entry = {
                "start": segment["start"],
                "end": segment["end"],
                "label": segment.get("label", ""),
                "segment_path": segment_path,
                "segment_name": segment_name,
                "cropped_segment_path": os.path.abspath(os.path.join(CROP_OUTPUT_DIR, video_name, f"{segment_name}.mp4"))
            }

            crop_manifest["segments"].append(entry)

            if out_writer:
                out_writer.release()


                if CROP_WITH_AUDIO:
                    cropped_path = os.path.join(CROP_OUTPUT_DIR, video_name, f"{segment_name}.mp4")
                    final_path = os.path.join(CROP_OUTPUT_DIR, video_name, f"{segment_name}_with_audio.mp4")

                    cmd = [
                        "ffmpeg",
                        "-y",  # overwrite
                        "-i", cropped_path,
                        "-i", segment_path,
                        "-map", "0:v:0",
                        "-map", "1:a:0",
                        "-c:v", "copy",
                        "-c:a", "aac",
                        "-shortest",
                        final_path
                    ]

                    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    os.replace(final_path, cropped_path)

        # Save the updated manifest
        crop_manifest_path = os.path.join(CROP_OUTPUT_DIR, f"{video_name}.json")
        with open(crop_manifest_path, "w", encoding="utf-8") as f:
            json.dump(crop_manifest, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    print("Crop stage started")
    start_time = time.time()
    crop_stage()
    end_time = time.time()
    print(f"Crop stage finished. Time: {end_time - start_time:.2f}\n")
