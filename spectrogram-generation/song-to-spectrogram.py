# TO DO NEXT:
# Download mp3s for valid YT videos
# Create spectrograms from remaining ones (potentially without downloads?)

import numpy as np
import pandas as pd
import requests
import yt_dlp
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt

# Store salami id, yt link, and song title for valid songs
valid_songs = {}

# ----------------------------------------------------------------
# TABLE ADJUSTMENTS

data = pd.read_csv("./spectrogram-generation/SALAMI-data/salami_youtube_pairings.csv")
data2 = pd.read_csv("./spectrogram-generation/SALAMI-data/metadata.csv")

# Merge song name to existing pairings
final_data = pd.merge(data, data2, how='left',left_on='salami_id', right_on='SONG_ID')

# Remove irrelevant columns
final_data = final_data[['salami_id', 'youtube_id', 'SONG_TITLE']]

# Renaming columns
final_data = final_data.rename(columns={'salami_id': 'SALAMI_ID', 'youtube_id': 'YT_ID'})


# ----------------------------------------------------------------
# VALIDATING LINKS

# Grab current row info
cur_row = final_data.loc[final_data["SALAMI_ID"] == 3, ["SALAMI_ID", "YT_ID", "SONG_TITLE"]]
salami_id = cur_row.iloc[0]["SALAMI_ID"]
youtube_id = cur_row.iloc[0]["YT_ID"]
song_title = cur_row.iloc[0]["SONG_TITLE"]

# Confirm that YT video is valid
def try_site(url):
    request = requests.get(url)
    return request.status_code == 200
youtube_link = "https://youtu.be/" + youtube_id
is_valid = try_site(youtube_link)
print(f"The song link for {song_title} is valid: {is_valid}")

# Add to List of Valid Songs
if is_valid:
    valid_songs[int(salami_id)] = [youtube_link, song_title]
print(valid_songs)


# ----------------------------------------------------------------
# MAKING SPECTROGRAMS FROM VALID SONGS

def download_audio(youtube_url, download_dir='./spectrogram-generation/audio-data'):
    # Create download directory if it doesn't exist
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    # Set yt-dlp options
    ydl_opts = {
        'format': 'bestaudio/best',
        'quiet': True,
        'noplaylist': True,
        'extractaudio': True,
        'audioquality': 0,  # highest quality
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3'
        }],
        'outtmpl': os.path.join(download_dir, '%(title)s.%(ext)s')  # Output path format
    }

    # Download the audio (in MP3 format)
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])
    
    # Return the downloaded MP3 file path
    info_dict = ydl.extract_info(youtube_url, download=True)
    mp3_file = os.path.join(download_dir, f"{info_dict['title']}.mp3")
    
    return mp3_file

def create_spectrogram(mp3_file):
    # Load audio using librosa
    y, sr = librosa.load(mp3_file, sr=None)  # y = audio signal, sr = sample rate
    
    # Generate Mel spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)

    # Plot the Mel spectrogram
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.show()

# Main function to download audio and create spectrogram
def process_youtube_audio(youtube_url):
    print("Downloading audio...")
    mp3_file = download_audio(youtube_url)

    print(f"Audio downloaded: {mp3_file}")
    print("Generating spectrogram...")
    create_spectrogram(mp3_file)

process_youtube_audio(youtube_link)
