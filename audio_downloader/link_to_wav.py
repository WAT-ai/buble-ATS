import requests
from tqdm import tqdm
import math

def next_power_of_2(x):
    return 1 if x == 0 else 2**math.ceil(math.log2(x))

def download_audio(url, output_file, chunk_size=8192):  # Increased chunk size
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            total_size = int(response.headers.get('content-length', 0))
            # Use a context manager for the file writing process
            with open('./audio_files/' + output_file, 'wb') as f:
                with tqdm(
                    desc="Downloading",
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    unit_divisor=chunk_size,
                ) as bar:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            bar.update(len(chunk))
            print(f"Audio downloaded successfully and saved as {output_file}")
        else:
            print(f"Failed to download the audio. Status code: {response.status_code}")
    except Exception as e:
        print(f"An error occurred: {e}")


url = input('URL: ')
output_file = input('Output file name: ')
output_file += '.wav'
chunk_size = input('Download chunk size: ')
chunk_size = next_power_of_2(int(chunk_size))
download_audio(url, output_file, chunk_size)
