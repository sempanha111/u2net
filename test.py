import os
import subprocess
from pytubefix import YouTube

# --- IMPORTANT ---
# You must use a real YouTube "watch" URL.
url = 'https://youtu.be/lZWaDmUlRJo?si=4jnQ5XRjj2o8XEf4' # Example: Your Tom & Jerry video

try:
    yt = YouTube(url)
    print(f"Video Title: {yt.title}\n")

    # --- CHANGE 1: REMOVED file_extension='mp4' to find ALL streams (mp4, webm, etc.) ---
    video_streams = yt.streams.filter(adaptive=True, only_video=True).order_by('resolution').desc()
    audio_streams = yt.streams.filter(adaptive=True, only_audio=True).order_by('abr').desc()

    if not video_streams:
        print("No suitable video streams found for this video.")
        exit()

    # --- Let the user choose a video resolution ---
    available_streams = {}
    print("Available video resolutions to download:")
    for i, stream in enumerate(video_streams):
        available_streams[i + 1] = stream
        size_mb = round(stream.filesize / 1048576, 2) if stream.filesize else 'N/A'
        # --- CHANGE 2: Displaying container and codec for clarity ---
        print(f"  {i + 1}. {stream.resolution} ({stream.fps}fps) - Codec: {stream.video_codec} - Container: {stream.mime_type.split('/')[1]} - Size: {size_mb} MB")

    print("-" * 30)

    # --- Get user input for video ---
    while True:
        try:
            choice = int(input("Enter the number of the video resolution you want: "))
            if choice in available_streams:
                video_to_download = available_streams[choice]
                break
            else:
                print("Invalid number. Please choose from the list above.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    # --- Select the best audio stream ---
    # We will choose an audio file that matches the video's container if possible
    audio_to_download = yt.streams.filter(only_audio=True, file_extension=video_to_download.subtype).order_by('abr').desc().first()
    if not audio_to_download: # Fallback if no matching audio container is found
        audio_to_download = yt.streams.filter(only_audio=True).order_by('abr').desc().first()

    print(f"\nSelected Video: {video_to_download.resolution} ({video_to_download.video_codec})")
    print(f"Selected Audio: {audio_to_download.abr} ({audio_to_download.audio_codec})")
    print("\nDownloading... (this may take a moment)")

    # --- Download video and audio to temporary files ---
    safe_title = "".join(c for c in yt.title if c.isalnum() or c in (' ', '.', '_')).rstrip()
    video_filename = f"video_temp.{video_to_download.subtype}"
    audio_filename = f"audio_temp.{audio_to_download.subtype}"
    final_filename = f"{safe_title} - {video_to_download.resolution}.mkv" # Using MKV as a universal container for merging
    
    output_path = 'downloads'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    video_to_download.download(output_path=output_path, filename=video_filename)
    audio_to_download.download(output_path=output_path, filename=audio_filename)
    print("Video and audio parts downloaded successfully.")

    # --- Merge files with FFmpeg ---
    print("Merging files with FFmpeg...")
    video_path = os.path.join(output_path, video_filename)
    audio_path = os.path.join(output_path, audio_filename)
    final_path = os.path.join(output_path, final_filename)
    
    ffmpeg_command = ['ffmpeg', '-i', video_path, '-i', audio_path, '-c', 'copy', '-y', final_path]

    try:
        subprocess.run(ffmpeg_command, check=True, capture_output=True, text=True)
        print(f"\nDownload and merge complete!")
        print(f"Final video saved as: {final_path}")
    except subprocess.CalledProcessError as e:
        print("\nError during FFmpeg merging process.")
        print(f"FFmpeg stderr: {e.stderr}")
    except FileNotFoundError:
        print("\nError: 'ffmpeg' command not found. Please ensure FFmpeg is installed and in your system's PATH.")
    finally:
        # --- Clean up temporary files ---
        print("Cleaning up temporary files...")
        if os.path.exists(video_path):
            os.remove(video_path)
        if os.path.exists(audio_path):
            os.remove(audio_path)

except Exception as e:
    print(f"An unexpected error occurred: {e}")