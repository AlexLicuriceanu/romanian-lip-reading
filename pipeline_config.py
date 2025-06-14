# 0. General configuration for the pipeline
DEVICE = "cuda"  # or "cpu"
DELETE_INTERMEDIATE_FILES = True  # Delete intermediate files where possible

# 1. Downloader configuration
VIDEO_DIR = "./video/"  # Directory containing video files
AUDIO_DIR = "./audio/"  # Directory containing audio files

# 2. ASR configuration
ASR_OUTPUT_DIR = "./asr_outputs/"  # Directory to save the ASR output files
ASR_MODEL_IDENTIFIER = "large-v2"  # "large-v3" doesn't work with word-level timestamps yet
ASR_BACKEND = "CTranslate2" # or "CTranslate2", "TensorRT-LLM"
ASR_COMPUTE_TYPE = "float32"  # or "int8", "float32"
ASR_OPTIONS = {'word_timestamps': True, 'beam_size': 1}
ASR_LANG_CODES = ["ro"]
ASR_TASKS = ["transcribe"]
ASR_INITIAL_PROMPTS = [None]
ASR_BATCH_SIZE = 1

# 3. Composition configuration
COMP_OUTPUT_DIR = "./composed_outputs/"  # Directory to save the composed output files
COMP_PUNCTUATION_ENDINGS = {".", "!", "?", ":"} # Punctuation that indicates the end of a sentence
COMP_CONJUNCTIONS = {"și", "dar", "iar", "sau", "pentru", "deoarece", "că", "fiindcă", "însă", "decât"} # Sentence split heuristic (most used Romanian conjunctions) 
COMP_MAX_WORDS = 25  # Maximum number of words in a sentence
COMP_MODE = "heuristic" # "word", "sentence", "heuristic"
COMP_REMOVE_PUNCTUATION = True  # Remove punctuation from the text
COMP_MAX_WORKERS = 8  # Maximum number of threads for the composition stage

# 4. Trimming configuration
TRIM_OUTPUT_DIR = "./trimmed_outputs/" # Directory to save the trimmed clips
TRIM_MAX_WORKERS = 8  # Maximum number of threads for trimming
TRIM_PADDING = 1.0  # Seconds of padding to add to the start and end of each clip

# 5. ASD configuration
ASD_OUTPUT_DIR = "./asd_outputs/"  # Directory to save the ASD output files
ASD_YOLO_MODEL_PATH = "./models/yolov11n-face.pt" # Path to the YOLO model for face detection
ASD_TALKNET_MODEL_PATH = "./models/pretrain_TalkSet.model" # Path to the TalkNet model for ASD
ASD_MAX_WORKERS_DATALOADER = 10  # Number of workers for the dataloader
ASD_FACEDET_SCALE = 0.25 # Scale factor for face detection, the frames will be scale to 0.25 orig (Only for S3FD - slow, don't use)
ASD_MIN_FACE_SIZE = 1 # Minimum face size in pixels (Only for S3FD - slow, don't use)
ASD_MIN_TRACK = 20 # Number of min frames for each shot
ASD_NUM_FAILED_DET = 10 # Number of missed detections allowed before tracking is stopped
ASD_DURATION_SET = {1, 2, 3} # {1,2,4,6} - Slower, more reliable or {1,1,1,2,2,2,3,3,4,5,6} - Even slower, even more reliable 
ASD_CROP_SCALE = 0.40 # Scale bounding box
ASD_MAX_WORD_GAP = 0.8 # Maximum gap (in seconds) between adjacent segments to be considered part of the same segment
ASD_START_TIME = 0 # The start time of the video (Don't modify)
ASD_DURATION = 0 # The duration of the video, when set as 0, will extract the whole video (Don't modify)
ASD_DEBUG = False # Show debug messages when running ASD
ASD_MIN_AREA_RATIO = 0.001 # Minimum area ratio of the face bounding box to the frame size to consider it a valid face detection

# 6. Cropping configuration
CROP_OUTPUT_DIR = "./cropped_outputs/"  # Directory to save the cropped outputs
CROP_FACEMESH_ONNX_MODEL_PATH = "./models/face_mesh_Nx3x192x192_post.onnx"  # Path to the ONNX model for face mesh
CROP_MODEL_INPUT_SIZE = 192 # Input size for the ONNX model
CROP_OUTPUT_MINIMUM_SIZE = 256 # Minimum size of the cropped region
CROP_WITH_AUDIO = False # Whether to crop the audio along with the video
CROP_PADDING = 10 # Padding around the cropped face
CROP_BATCH_SIZE = 2048 # Batch size for processing frames in the ONNX model
CROP_LIP_LANDMARKS = [  # Lip landmarks for cropping
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
    146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308,
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308
]
CROP_ONNX_PROVIDERS = [ # Providers for ONNX model execution
    'TensorrtExecutionProvider',
    'CUDAExecutionProvider',
    'CPUExecutionProvider']
CROP_LIPS = True  # Whether to crop the lip region

# 7. Labeling configuration
LABELING_DIR = "./dataset"
LABELING_MAX_WORKERS = 8  # Maximum number of threads for labeling