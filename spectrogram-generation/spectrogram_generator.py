import numpy as np
import pandas as pd
import yt_dlp
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt

class SpectrogramGenerator:
    def __init__(self):
        self.data = pd.read_csv('./spectrogram-generation/SALAMI-data/validated-salami.csv')

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
        
        y, sr = librosa.load(audio_path, sr=None)
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
        for _, row in self.data.iterrows():
            # Download MP3
            youtube_link = row['YT_LINK']
            song_title = row['SONG_TITLE']
            mp3_path = self.download_mp3(youtube_link)
            print(f"Downloaded {song_title} MP3 to {mp3_path}")
            
            # Create Spectrogram
            spectrogram_path = self.create_spectrogram(mp3_path)
            print(f"Created {song_title} spectrogram at {spectrogram_path}")

    def show_data(self):
        print(self.data)
    
    def process_random_songs(self, num):
        sampled_songs = self.data.sample(num)

        for _, row in sampled_songs.iterrows():
            # Download MP3
            youtube_link = row['YT_LINK']
            song_title = row['SONG_TITLE']
            mp3_path = self.download_mp3(youtube_link)
            print(f"Downloaded {song_title} MP3 to {mp3_path}")
            
            # Create Spectrogram
            spectrogram_path = self.create_spectrogram(mp3_path)
            print(f"Created {song_title} spectrogram at {spectrogram_path}")
