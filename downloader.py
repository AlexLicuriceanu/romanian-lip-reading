import yt_dlp
import os
from yt_dlp.utils import sanitize_filename
from pipeline_config import VIDEO_DIR

def download_videos_from_file(txt_file, output_path='downloads'):
    os.makedirs(output_path, exist_ok=True)

    with open(txt_file, 'r') as f:
        urls = [line.strip() for line in f if line.strip()]

    for url in urls:
        try:
            with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
                info = ydl.extract_info(url, download=False)
                title = info.get('title', 'video')
                sanitized_title = sanitize_filename(title, restricted=True)
                filename_template = os.path.join(output_path, f"{sanitized_title}.%(ext)s")

            ydl_opts = {
                'outtmpl': filename_template,
                'format': 'bestvideo+bestaudio/best',
                'merge_output_format': 'mp4',
                'quiet': False,
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
