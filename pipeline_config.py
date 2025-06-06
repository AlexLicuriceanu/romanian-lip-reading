# 1. Downloader configuration
VIDEO_DIR = "./video/"  # Directory containing video files

# 2. ASR configuration
AUDIO_DIR = "./audio/"  # Directory containing audio files
ASR_OUTPUT_DIR = "./asr_outputs/"  # Directory to save the ASR output files
MODEL_IDENTIFIER = "large-v2"  # "large-v3" doesn't work with word-level timestamps yet
BACKEND = "CTranslate2" # or "CTranslate2", "TensorRT-LLM"
COMPUTE_TYPE = "float32"  # or "int8", "float32"
DEVICE = "cuda"  # or "cpu"
ASR_OPTIONS = {'word_timestamps': True, 'beam_size': 1}
LANG_CODES = ["ro"]
TASKS = ["transcribe"]
INITIAL_PROMPTS = [None]
BATCH_SIZE = 1

# 3. Composition configuration
COMP_OUTPUT_DIR = "./composed_outputs/"  # Directory to save the composed output files
PUNCTUATION_ENDINGS = {".", "!", "?", ":"} # Punctuation that indicates the end of a sentence
CONJUNCTIONS = {"și", "dar", "iar", "sau", "pentru", "deoarece", "că", "fiindcă", "însă", "decât"} # Sentence split heuristic (most used Romanian conjunctions) 
MAX_WORDS = 25  # Maximum number of words in a sentence
MODES = ["word", "sentence", "heuristic"] # Composition modes
DEFAULT_MODE = "heuristic" # Default composition mode
REMOVE_PUNCTUATION = True  # Remove punctuation from the text

# 4. Trimming configuration
TRIM_OUTPUT_DIR = "./trimmed_outputs/" # Directory to save the trimmed clips
PADDING = 1.0  # Seconds of padding to add to the start and end of each clip

# 5. ASD configuration
ASD_OUTPUT_DIR = "/home/rhc/licenta4/asd_outputs/"  # Directory to save the ASD output files
N_DATA_LOADER_THREAD = 10  # Number of workers
FACEDET_SCALE = 0.25 # Scale factor for face detection, the frames will be scale to 0.25 orig (Only for S3FD)
MIN_TRACK = 10 # Number of min frames for each shot
NUM_FAILED_DET = 10 # Number of missed detections allowed before tracking is stopped
MIN_FACE_SIZE = 1 # Minimum face size in pixels (Only for S3FD)
CROP_SCALE = 0.40 # Scale bounding box
START_TIME = 0 # The start time of the video (Don't modify)
DURATION = 0 # The duration of the video, when set as 0, will extract the whole video (Don't modify)
ASD_DEBUG = False # Show debug messages when running ASD