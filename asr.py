import whisper_s2t
import json
import os
from pipeline_config import VIDEO_DIR, AUDIO_DIR, ASR_OUTPUT_DIR
from pipeline_config import MODEL_IDENTIFIER, BACKEND, COMPUTE_TYPE, DEVICE
from pipeline_config import ASR_OPTIONS, LANG_CODES, TASKS, INITIAL_PROMPTS, BATCH_SIZE
from typing import List

# Load the model
model = whisper_s2t.load_model(
    model_identifier=MODEL_IDENTIFIER,
    backend=BACKEND,
    compute_type=COMPUTE_TYPE,
    device=DEVICE,
    asr_options=ASR_OPTIONS,
)

def asr(file_path: List[str]):
    # Run transcription with word-level timestamps
    results = model.transcribe_with_vad(
        file_path,
        lang_codes=LANG_CODES,
        tasks=TASKS,
        initial_prompts=INITIAL_PROMPTS,
        batch_size=BATCH_SIZE
    )

    return results

def extract_audio(video_path, audio_path):
    os.system(f"ffmpeg -y -hide_banner -loglevel error -i \"{video_path}\" -ar 16000 -ac 1 -vn \"{audio_path}\"")

def asr_stage():

    os.makedirs(AUDIO_DIR, exist_ok=True)
    os.makedirs(ASR_OUTPUT_DIR, exist_ok=True)

    # Extract audio from video files, then run ASR on the audio files
    video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith(".mp4")]
    total_files = len(video_files)

    for i, video_file in enumerate(video_files):
        video_path = os.path.join(VIDEO_DIR, video_file)
        audio_path = os.path.join(AUDIO_DIR, f"{os.path.splitext(video_file)[0]}.wav")

        extract_audio(video_path, audio_path)

        # Check if audio extraction was successful
        if not os.path.exists(audio_path):
            print(f"Failed to extract audio for: {video_file}, skipping.")
            continue

        print(f"Processing: {video_file} ({i + 1}/{total_files})")
        results = asr([audio_path])

        result = results[0]

        output = {
            "video_path": video_path,
            "segments": result
        } 

        # Save results to JSON file
        output_file = os.path.join(ASR_OUTPUT_DIR, f"asr_{os.path.splitext(video_file)[0]}.json")

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    print("ASR stage started.")
    asr_stage()
    print("ASR stage finished.\n")
    
