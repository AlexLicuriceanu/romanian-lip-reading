import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import time
from datetime import datetime, timedelta
from pipeline_config import *

from asr import asr_load_model, asr_stage
from composition import composition_stage
from trim import trim_stage

if __name__ == "__main__":
    """Run all the stages of the dataset creation pipeline"""
    start_time = time.time()

    # 1. ASR stage: Load the ASR model
    asr_model = asr_load_model(
        model_identifier=ASR_MODEL_IDENTIFIER,
        backend=ASR_BACKEND,
        compute_type=ASR_COMPUTE_TYPE,
        device=DEVICE,
        asr_options=ASR_OPTIONS
    )

    # 2. ASR stage: Run ASR on the video files
    asr_stage(
        video_dir=VIDEO_DIR,
        audio_dir=AUDIO_DIR,
        asr_output_dir=ASR_OUTPUT_DIR,
        model=asr_model,
        lang_codes=ASR_LANG_CODES,
        tasks=ASR_TASKS,
        initial_prompts=ASR_INITIAL_PROMPTS,
        batch_size=ASR_BATCH_SIZE
    )

    # 3. Composition stage: Compose the ASR outputs into words or sentences
    composition_stage(
        asr_output_dir=ASR_OUTPUT_DIR,
        comp_output_dir=COMP_OUTPUT_DIR,
        punctuation_endings=COMP_PUNCTUATION_ENDINGS,
        conjunctions=COMP_CONJUNCTIONS,
        max_words=COMP_MAX_WORDS,
        mode=COMP_MODE,
        remove_punctuation=COMP_REMOVE_PUNCTUATION,
        max_workers=COMP_MAX_WORKERS
    )

    # 4. Trimming stage: Trim the video into clips based on the composed outputs
    trim_stage(
        video_dir=VIDEO_DIR,
        comp_output_dir=COMP_OUTPUT_DIR,
        trim_output_dir=TRIM_OUTPUT_DIR,
        trim_padding=TRIM_PADDING,
        trim_max_workers=TRIM_MAX_WORKERS
    )

    end_time = time.time()
    print(f"Pipeline completed. Time: {timedelta(seconds=end_time - start_time)}")