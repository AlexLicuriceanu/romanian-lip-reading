import yt_dlp
import os
from tqdm import tqdm
from yt_dlp.utils import sanitize_filename
from pipeline_config import VIDEO_DIR, DOWNLOADER_FORMAT, DOWNLOADER_QUIET

def download_videos_from_file(txt_file, output_path='downloads'):
    os.makedirs(output_path, exist_ok=True)

    try:
        with open(txt_file, 'r') as f:
            urls = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"[DL] The file {txt_file} does not exist.")
        return

    for url in tqdm(urls, desc="Download stage", unit="video", total=len(urls)):
        try:
            with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
                info = ydl.extract_info(url, download=False)
                title = info.get('title', 'video')
                sanitized_title = sanitize_filename(title, restricted=True)
                filename_template = os.path.join(output_path, f"{sanitized_title}.%(ext)s")

            ydl_opts = {
                'outtmpl': filename_template,
                'format': DOWNLOADER_FORMAT,
                'merge_output_format': 'mp4',
                'quiet': DOWNLOADER_QUIET,
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
                
        except Exception as e:
            pass

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="YouTube Video Downloader")
    parser.add_argument("--video-list", help="Path to the .txt file with video URLs", required=True)
    args = parser.parse_args()

    download_videos_from_file(args.video_list, VIDEO_DIR)
