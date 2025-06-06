import json
import os
import re
from pipeline_config import ASR_OUTPUT_DIR, COMP_OUTPUT_DIR
from pipeline_config import PUNCTUATION_ENDINGS, CONJUNCTIONS, MAX_WORDS
from pipeline_config import MODES, DEFAULT_MODE, REMOVE_PUNCTUATION


def clean_text(text):
    return re.sub(r"[.,:!?\'\"]", "", str.lower(text))


def split_sentence_by_heuristic(words, max_words=MAX_WORDS):
    chunks = []
    chunk = []
    last_conjunction_index = -1

    for i, word in enumerate(words):
        chunk.append(word)
        lw = word["word"].strip(",.").lower()

        if lw in CONJUNCTIONS:
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


def process_segments(word_timestamps, video_path, mode=DEFAULT_MODE):
    all_words = word_timestamps
    full_text = " ".join(w["word"] for w in all_words)

    final_segments = []

    if mode == "word":
        for word in all_words:
            final_segments.append({
                "text": word["word"],
                "start": word["start"],
                "end": word["end"],
                "word_timestamps": [word]
            })

    else:  # sentence or heuristic
        current_sentence = []

        for word in all_words:
            current_sentence.append(word)

            if any(word["word"].strip().endswith(p) for p in PUNCTUATION_ENDINGS):
                if mode == "heuristic":
                    sub_sentences = split_sentence_by_heuristic(current_sentence)
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

        if current_sentence:
            if mode == "heuristic":
                sub_sentences = split_sentence_by_heuristic(current_sentence)
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


def composition_stage():
    os.makedirs(COMP_OUTPUT_DIR, exist_ok=True)

    asr_files = [f for f in os.listdir(ASR_OUTPUT_DIR) if f.endswith(".json")]
    total_files = len(asr_files)

    for i, file in enumerate(asr_files):
        with open(os.path.join(ASR_OUTPUT_DIR, file), "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict) or "word_timestamps" not in data:
            print(f"Skipping {file} - invalid format")
            continue

        segments = data["word_timestamps"]
        video_path = data.get("video_path", "unknown_video")
        video_name = os.path.basename(video_path)

        print(f"Processing: {video_name} ({i + 1}/{total_files})")

        result = process_segments(segments, video_path, mode=DEFAULT_MODE)

        if REMOVE_PUNCTUATION:
            for segment in result["segments"]:
                segment["text"] = clean_text(segment["text"])
                
                # Clean words inside word_timestamps too
                for word in segment.get("word_timestamps", []):
                    word["word"] = clean_text(word["word"])

        output_file = os.path.join(COMP_OUTPUT_DIR, f"{DEFAULT_MODE}_{file}")

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4, ensure_ascii=False)


def run_composition_stage():
    print("Composition stage started.")
    composition_stage()
    print("Composition stage finished.\n")


if __name__ == "__main__":
    run_composition_stage()
