import numpy as np
import pandas as pd
import requests
import yt_dlp
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt

class SpectrogramGenerator:
    def __init__(self):
        self.data_path = "./spectrogram-generation/SALAMI-data/salami_youtube_pairings.csv" # pairings
        self.metadata_path = "./spectrogram-generation/SALAMI-data/metadata.csv" # dataset
        self.valid_songs = {} # Valid Song: salami id to [yt link, and song title]
        self.final_data = None
        self.load_and_merge_data() # Auto table merging
        self.validate_links() # Auto link validation

    # Confirms that a YT link is valid
    def try_site(self, url):
        request = requests.get(url)
        return request.status_code == 200 and "Video unavailable" not in request.text
    
    def load_and_merge_data(self):
        data = pd.read_csv(self.data_path)
        data2 = pd.read_csv(self.metadata_path)
        
        # Merge song name to existing pairings
        self.final_data = pd.merge(data, data2, how='left', left_on='salami_id', right_on='SONG_ID')
        self.final_data = self.final_data[['salami_id', 'youtube_id', 'SONG_TITLE']]
        self.final_data = self.final_data.rename(columns={'salami_id': 'SALAMI_ID', 'youtube_id': 'YT_ID'})

    def validate_links(self):
        for _, row in self.final_data.iterrows():
            salami_id = row["SALAMI_ID"]
            youtube_id = row["YT_ID"]
            song_title = row["SONG_TITLE"]
            
            youtube_link = f"https://youtu.be/{youtube_id}"
            is_valid = self.try_site(youtube_link)
            print(f"Validity of {song_title}: {is_valid}")
            
            if is_valid:
                self.valid_songs[int(salami_id)] = [youtube_link, song_title]

    def download_mp3(self, youtube_url, output_dir='./spectrogram-generation/audio-data'):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(youtube_url, download=True)
            return ydl.prepare_filename(info_dict).replace('.webm', '.mp3').replace('.m4a', '.mp3')
    
    def create_spectrogram(self, audio_path, output_dir='./spectrogram-generation/spectrograms'):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        y, sr = librosa.load(audio_path)
        plt.figure(figsize=(10, 4))
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Spectrogram')
        
        spectrogram_path = os.path.join(output_dir, os.path.basename(audio_path).replace('.mp3', '.png'))
        plt.savefig(spectrogram_path)
        plt.close()
        return spectrogram_path
    
    def process_songs(self):
        for key, value in self.valid_songs.items():
            # Download MP3
            youtube_link = value[0]
            song_title = value[1]
            mp3_path = self.download_mp3(youtube_link)
            print(f"Downloaded {song_title} MP3 to {mp3_path}")
            
            # Create Spectrogram
            spectrogram_path = self.create_spectrogram(mp3_path)
            print(f"Created {song_title} spectrogram at {spectrogram_path}")
