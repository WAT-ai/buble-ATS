from spectrogram_generator import SpectrogramGenerator

def main():
    generator = SpectrogramGenerator()
    
    # Process the songs at certain indices (download MP3s and create spectrograms)
    arr = [0, 1]
    generator.process_index_songs(arr)

if __name__ == "__main__":
    main()
