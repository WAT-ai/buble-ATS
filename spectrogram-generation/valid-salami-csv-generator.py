import requests
import pandas as pd
import csv

data_path = "./spectrogram-generation/SALAMI-data/salami_youtube_pairings.csv" # pairings
metadata_path = "./spectrogram-generation/SALAMI-data/metadata.csv"
valid_songs = []
final_data = None
csv_file_path = './spectrogram-generation/SALAMI-data/validated-salami.csv'


# Confirms that a YT link is valid
def try_site(url):
    request = requests.get(url)
    return request.status_code == 200 and "Video unavailable" not in request.text

def load_and_merge_data(d, m):
    global final_data
    data = pd.read_csv(d)
    data2 = pd.read_csv(m)
        
    # Merge song name to existing pairings
    final_data = pd.merge(data, data2, how='left', left_on='salami_id', right_on='SONG_ID')
    final_data = final_data[['salami_id', 'youtube_id', 'SONG_TITLE']]
    final_data = final_data.rename(columns={'salami_id': 'SALAMI_ID', 'youtube_id': 'YT_ID'})

def validate_links():
    for _, row in final_data.iterrows():
        salami_id = row["SALAMI_ID"]
        youtube_id = row["YT_ID"]
        song_title = row["SONG_TITLE"]
            
        youtube_link = f"https://youtu.be/{youtube_id}"
        is_valid = try_site(youtube_link)
        print(f"Validity of {song_title}: {is_valid}")
            
        if is_valid:
            valid_songs.append([int(salami_id), youtube_link, song_title])

def write_csv():
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["SALAMI_ID", "YT_LINK", "SONG_TITLE"])  # Add header
        writer.writerows(valid_songs)
    
    print(f"CSV file '{csv_file_path}' created successfully.")

if __name__ == "__main__":
    load_and_merge_data(data_path, metadata_path)
    validate_links()
    write_csv()
