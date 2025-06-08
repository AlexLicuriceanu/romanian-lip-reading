import whisper_s2t
import json
import os
from tqdm import tqdm

def asr_load_model(model_identifier, backend, compute_type, device, asr_options):
    """Load the ASR model with specified parameters"""
    return whisper_s2t.load_model(
        model_identifier=model_identifier,
        backend=backend,
        compute_type=compute_type,
        device=device,
        asr_options=asr_options
    )

def asr(model, file_path, lang_codes, tasks, initial_prompts, batch_size):
    """Perform ASR on the audio files using the given parameters"""
    results = model.transcribe_with_vad(
        file_path,
        lang_codes=lang_codes,
        tasks=tasks,
        initial_prompts=initial_prompts,
        batch_size=batch_size,
        tqdm_leave=False,
        tqdm_desc=os.path.basename(file_path[0])
    )

    return results

def extract_audio(video_path, audio_path):
    """Extract audio from the video file and save it as a WAV file"""
    os.system(f"ffmpeg -y -hide_banner -loglevel error -i \"{video_path}\" -ar 16000 -ac 1 -vn \"{audio_path}\"")


def asr_stage(video_dir, audio_dir, asr_output_dir, model, lang_codes, tasks, initial_prompts, batch_size):
    """Main ASR stage to process video files, extract audio, and perform ASR"""
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(asr_output_dir, exist_ok=True)

    # Ensure the model is loaded
    if model is None:
        raise ValueError("[ASR] ASR model is not loaded")
    
    # Get all video files
    video_files = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]
    total_files = len(video_files)

    # Check if there are any video files to process
    if total_files == 0:
        raise ValueError("[ASR] No video files found in the specified directory")

    # Process each video file
    for video_file in tqdm(video_files, desc="ASR stage", total=total_files, unit="file"):
        video_path = os.path.abspath(os.path.join(video_dir, video_file))
        audio_path = os.path.join(audio_dir, f"{os.path.splitext(video_file)[0]}.wav")

        # Extract audio from the video file
        extract_audio(video_path, audio_path)

        # Check if the audio was extracted successfully
        if not os.path.exists(audio_path):
            print(f"[ASR] Failed to extract audio for: {video_file}, skipping.")
            continue

        # Perform ASR on the extracted audio
        results = asr(model=model,
                      file_path=[audio_path], 
                      lang_codes=lang_codes,
                      tasks=tasks,
                      initial_prompts=initial_prompts,
                      batch_size=batch_size)
        
        # WhisperS2T returns List[List[Dict]] 
        result = results[0]

        # Flatten word_timestamps from all segments
        word_timestamps = []
        for segment in result:
            if "word_timestamps" in segment:
                word_timestamps.extend(segment["word_timestamps"])

        # Create the output manifest
        output_manifest = {
            "video_path": video_path,
            "word_timestamps": word_timestamps
        }

        # Save the manifest to JSON
        output_path = os.path.join(asr_output_dir, f"{os.path.splitext(video_file)[0]}.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_manifest, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    # In case the script is run directly, not from pipeline.py
    from pipeline_config import (
        VIDEO_DIR, AUDIO_DIR, ASR_OUTPUT_DIR,
        ASR_MODEL_IDENTIFIER, ASR_BACKEND, ASR_COMPUTE_TYPE, DEVICE,
        ASR_OPTIONS, ASR_LANG_CODES, ASR_TASKS, ASR_INITIAL_PROMPTS,
        ASR_BATCH_SIZE
    )

    model = asr_load_model(
        model_identifier=ASR_MODEL_IDENTIFIER,
        backend=ASR_BACKEND,
        compute_type=ASR_COMPUTE_TYPE,
        device=DEVICE,
        asr_options=ASR_OPTIONS
    )

    asr_stage(
        video_dir=VIDEO_DIR,
        audio_dir=AUDIO_DIR,
        asr_output_dir=ASR_OUTPUT_DIR,
        model=model,
        lang_codes=ASR_LANG_CODES,
        tasks=ASR_TASKS,
        initial_prompts=ASR_INITIAL_PROMPTS,
        batch_size=ASR_BATCH_SIZE
    )
