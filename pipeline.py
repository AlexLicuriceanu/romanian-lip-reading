from pipeline_config import *
from asr import asr_load_model, asr_stage

if __name__ == "__main__":
    """Run all the stages of the dataset creation pipeline"""
    
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
        model=asr_model,
        lang_codes=ASR_LANG_CODES,
        tasks=ASR_TASKS,
        initial_prompts=ASR_INITIAL_PROMPTS,
        batch_size=ASR_BATCH_SIZE
    )