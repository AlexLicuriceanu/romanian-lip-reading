import json
import os
import re
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from pipeline_config import (
    ASR_OUTPUT_DIR, COMP_OUTPUT_DIR,
    COMP_PUNCTUATION_ENDINGS, COMP_CONJUNCTIONS, COMP_MAX_WORDS,
    COMP_MODE, COMP_REMOVE_PUNCTUATION, COMP_MAX_WORKERS
)

def clean_text(text):
    """
    Sanitize text
    Arguments:
        text (str): Input text to be sanitized.
    Returns:
        str: Sanitized text with punctuation and symbols removed and converted to lowercase.
    """
    return re.sub(r"[.,:!?\'\"]", "", str.lower(text))

def split_sentence_by_heuristic(words, conjunctions, max_words):
    """
    Split a list of words into chunks based on conjunctions and maximum word count.
    Arguments:
        words (list): List of word dictionaries, each containing 'word', 'start', and 'end'.
        conjunctions (set): Set of conjunctions to use for splitting.
        max_words (int): Maximum number of words per chunk.
    Returns:
        list: List of chunks, where each chunk is a list of word dictionaries.
    """
    chunks = []
    chunk = []
    last_conjunction_index = -1

    for word in words:
        chunk.append(word)
        lw = word["word"].strip(",.").lower()

        if lw in conjunctions:
            last_conjunction_index = len(chunk) - 1

        if len(chunk) >= max_words:
            if last_conjunction_index != -1:
                split_at = last_conjunction_index + 1
                chunks.append(chunk[:split_at])
                chunk = chunk[split_at:]
            else:
                chunks.append(chunk)
                chunk = []

            last_conjunction_index = -1  # reset after splitting

    if chunk:
        chunks.append(chunk)

    return chunks

def process_segments(word_timestamps, video_path, punctuation_endings, conjunctions, max_words, mode):
    """
    Process word timestamps into segments based on the specified mode.
    Arguments:
        word_timestamps (list): List of word timestamps, each containing 'word', 'start', and 'end'.
        video_path (str): Path to the video file.
        punctuation_endings (set): Set of punctuation characters that indicate sentence endings.
        conjunctions (set): Set of conjunctions to use for splitting sentences.
        max_words (int): Maximum number of words per segment.
        mode (str): Mode of segmentation ('word', 'sentence', or 'heuristic').
    Returns:
        dict: A dictionary containing the video path, full text, and segments.
    """
    all_words = word_timestamps
    full_text = " ".join(w["word"] for w in all_words)

    final_segments = []

    if mode == "word":
        # Create segments for each word
        for word in all_words:
            final_segments.append({
                "text": word["word"],
                "start": word["start"],
                "end": word["end"],
                "word_timestamps": [word]
            })

    else:
        # Sentence or heuristic
        current_sentence = []

        for word in all_words:
            current_sentence.append(word)

            # Check if the word ends with punctuation
            if any(word["word"].strip().endswith(p) for p in punctuation_endings):

                # Split the current sentence using the heuristic or keep it as is
                if mode == "heuristic":
                    sub_sentences = split_sentence_by_heuristic(words=current_sentence, conjunctions=conjunctions, max_words=max_words)
                else:
                    sub_sentences = [current_sentence]

                for chunk in sub_sentences:
                    final_segments.append({
                        "text": " ".join(w["word"] for w in chunk).strip(),
                        "start": chunk[0]["start"],
                        "end": chunk[-1]["end"],
                        "word_timestamps": chunk
                    })

                current_sentence = []

        # Handle any remaining words in the current sentence
        if current_sentence:
            if mode == "heuristic":
                sub_sentences = split_sentence_by_heuristic(words=current_sentence, conjunctions=conjunctions, max_words=max_words)
            else:
                sub_sentences = [current_sentence]

            for chunk in sub_sentences:
                final_segments.append({
                    "text": " ".join(w["word"] for w in chunk).strip(),
                    "start": chunk[0]["start"],
                    "end": chunk[-1]["end"],
                    "word_timestamps": chunk
                })

    return {
        "video_path": video_path,
        "text": full_text.strip(),
        "segments": final_segments
    }

def process_single_file(asr_output_dir, comp_output_dir, file, punctuation_endings, conjunctions, max_words, mode, remove_punctuation):
    """
    Process a single ASR output file to create a composed manifest.
    Arguments:
        asr_output_dir (str): Directory containing ASR output files.
        comp_output_dir (str): Directory to save composed manifests.
        file (str): Name of the ASR output file to process.
        punctuation_endings (set): Set of punctuation characters that indicate sentence endings.
        conjunctions (set): Set of conjunctions to use for splitting sentences.
        max_words (int): Maximum number of words per segment.
        mode (str): Mode of segmentation ('word', 'sentence', or 'heuristic').
        remove_punctuation (bool): Whether to remove punctuation from the text.
    Returns:
        tuple: A tuple containing the output path and the composed manifest.
    """
    try:
        with open(os.path.join(asr_output_dir, file), "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return None, f"[COMP] {file} failed to load: {e}"

    if not isinstance(data, dict) or "word_timestamps" not in data:
        return None, f"[COMP] {file} has invalid format, skipping"

    # Extract word timestamps and video path
    try:
        word_timestamps = data["word_timestamps"]
        video_path = data["video_path"]
    except KeyError as e:
        return None, f"[COMP] Missing key in {file}: {e}, skipping"

    # Check if word_timestamps is a list
    if not isinstance(word_timestamps, list):
        return None, f"[COMP] Invalid word_timestamps format in {file}, expected a list, skipping"

    # Process word timestamps to create segments
    output_manifest = process_segments(word_timestamps=word_timestamps, video_path=video_path, mode=mode,
                                       punctuation_endings=punctuation_endings, conjunctions=conjunctions,
                                       max_words=max_words)

    # Remove punctuation if necessary
    if remove_punctuation:
        for segment in output_manifest["segments"]:
            segment["text"] = clean_text(segment["text"])

            # Clean words inside word_timestamps too
            for word in segment.get("word_timestamps", []):
                word["word"] = clean_text(word["word"])

    output_path = os.path.join(comp_output_dir, f"{mode}_{file}")
    return output_path, output_manifest

def composition_stage(asr_output_dir, comp_output_dir, punctuation_endings, conjunctions, max_words, mode, remove_punctuation, max_workers=1):
    """
    Composition stage to process ASR outputs and compose segments.
    Arguments:
        asr_output_dir (str): Directory containing ASR output files.
        comp_output_dir (str): Directory to save composed manifests.
        punctuation_endings (set): Set of punctuation characters that indicate sentence endings.
        conjunctions (set): Set of conjunctions to use for splitting sentences.
        max_words (int): Maximum number of words per segment.
        mode (str): Mode of segmentation ('word', 'sentence', or 'heuristic').
        remove_punctuation (bool): Whether to remove punctuation from the text.
        max_workers (int): Number of worker processes to use for parallel processing.
    Returns:
        None
    """
    os.makedirs(comp_output_dir, exist_ok=True)

    # Get all ASR manifests
    asr_files = [f for f in os.listdir(asr_output_dir) if f.endswith(".json")]
    total_files = len(asr_files)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                process_single_file,
                asr_output_dir,
                comp_output_dir,
                file,
                punctuation_endings,
                conjunctions,
                max_words,
                mode,
                remove_punctuation
            ): file for file in asr_files
        }

        for future in tqdm(as_completed(futures), total=total_files, desc="Composition stage", unit="file"):
            result = future.result()
            if result is None:
                continue

            output_path, output_manifest = result
            if output_path is None:
                print(output_manifest)  # output_manifest contains the error message
                continue
            
            try:
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(output_manifest, f, indent=2, ensure_ascii=False)
            except Exception as e:
                print(f"[COMP] Failed to write {output_path}: {e}")

if __name__ == "__main__":
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
