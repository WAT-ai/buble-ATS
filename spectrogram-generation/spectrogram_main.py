import sys
from spectrogram_generator import SpectrogramGenerator

def main():
    generator = SpectrogramGenerator()
    
    # Process the songs (download MP3s and create spectrograms)
    generator.process_songs()

if __name__ == "__main__":
    main()
