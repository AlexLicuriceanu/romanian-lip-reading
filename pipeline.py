import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import time
from datetime import datetime, timedelta
from pipeline_config import *

from asr import asr_load_model, asr_stage
from composition import composition_stage
from trim import trim_stage
from talknet_asd.runTalkNet import load_yolo_model, load_talknet_model
from asd import asd_stage
from crop import crop_stage

if __name__ == "__main__":
    """Run all the stages of the dataset creation pipeline"""
    start_time = time.time()

    # # 1. ASR stage: Load the ASR model
    # asr_model = asr_load_model(
    #     model_identifier=ASR_MODEL_IDENTIFIER,
    #     backend=ASR_BACKEND,
    #     compute_type=ASR_COMPUTE_TYPE,
    #     device=DEVICE,
    #     asr_options=ASR_OPTIONS
    # )

    # # 2. ASR stage: Run ASR on the video files
    # asr_stage(
    #     video_dir=VIDEO_DIR,
    #     audio_dir=AUDIO_DIR,
    #     asr_output_dir=ASR_OUTPUT_DIR,
    #     model=asr_model,
    #     lang_codes=ASR_LANG_CODES,
    #     tasks=ASR_TASKS,
    #     initial_prompts=ASR_INITIAL_PROMPTS,
    #     batch_size=ASR_BATCH_SIZE
    # )

    # # 3. Composition stage: Compose the ASR outputs into words or sentences
    # composition_stage(
    #     asr_output_dir=ASR_OUTPUT_DIR,
    #     comp_output_dir=COMP_OUTPUT_DIR,
    #     punctuation_endings=COMP_PUNCTUATION_ENDINGS,
    #     conjunctions=COMP_CONJUNCTIONS,
    #     max_words=COMP_MAX_WORDS,
    #     mode=COMP_MODE,
    #     remove_punctuation=COMP_REMOVE_PUNCTUATION,
    #     max_workers=COMP_MAX_WORKERS
    # )

    # # 4. Trimming stage: Trim the video into clips based on the composed outputs
    # trim_stage(
    #     video_dir=VIDEO_DIR,
    #     comp_output_dir=COMP_OUTPUT_DIR,
    #     trim_output_dir=TRIM_OUTPUT_DIR,
    #     trim_padding=TRIM_PADDING,
    #     trim_max_workers=TRIM_MAX_WORKERS
    # )

    # # 5. ASD stage: Load the YOLO model for face detection
    # asd_yolo_model = load_yolo_model(model_path=ASD_YOLO_MODEL_PATH)

    # # 6. ASD stage: Load the TalkNet model for ASD
    # asd_talknet_model = load_talknet_model(model_path=ASD_TALKNET_MODEL_PATH)

    # # 7. ASD stage: Run the ASD model on the trimmed clips
    # asd_stage(
    #     trim_output_dir=TRIM_OUTPUT_DIR,
    #     asd_output_dir=ASD_OUTPUT_DIR,
    #     yolo_model=asd_yolo_model,
    #     talknet_model=asd_talknet_model,
    #     max_workers_dataloader=ASD_MAX_WORKERS_DATALOADER,
    #     facedet_scale=ASD_FACEDET_SCALE,
    #     min_track=ASD_MIN_TRACK,
    #     duration_set=ASD_DURATION_SET,
    #     num_failed_det=ASD_NUM_FAILED_DET,
    #     min_face_size=ASD_MIN_FACE_SIZE,
    #     crop_scale=ASD_CROP_SCALE,
    #     start_time=ASD_START_TIME,
    #     duration=ASD_DURATION,
    #     padding=TRIM_PADDING,
    #     max_word_gap=ASD_MAX_WORD_GAP,
    #     asd_debug=ASD_DEBUG,
    #     delete_intermediate_files=DELETE_INTERMEDIATE_FILES
    # )

    # 8. Cropping stage: Crop the lip region from the ASD outputs
    crop_stage(
        asd_output_dir=ASD_OUTPUT_DIR,
        crop_output_dir=CROP_OUTPUT_DIR,
        output_minimum_size=CROP_OUTPUT_MINIMUM_SIZE,
        model_input_size=CROP_MODEL_INPUT_SIZE,
        padding=CROP_PADDING,
        batch_size=CROP_BATCH_SIZE,
        crop_with_audio=CROP_WITH_AUDIO
    )

    end_time = time.time()
    print(f"Pipeline completed. Time: {timedelta(seconds=end_time - start_time)}")