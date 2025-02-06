# TO DO NEXT:
# Download mp3s for valid YT videos
# Create spectrograms from remaining ones (potentially without downloads?)

# import librosa
# import librosa.display
# import matplotlib.pyplot as plt
# import numpy as np
import pandas as pd
import requests

# Store salami id, yt id, and song title for valid songs
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
SALAMI_ID = cur_row.iloc[0]["SALAMI_ID"]
YT_ID = cur_row.iloc[0]["YT_ID"]
SONG_TITLE = cur_row.iloc[0]["SONG_TITLE"]

# Confirm that YT video is valid
def try_site(url):
    request = requests.get(url)
    return request.status_code == 200
is_valid = try_site("https://youtu.be/" + YT_ID)
print(f"The song link for {SONG_TITLE} is valid: {is_valid}")

# Add to List of Valid Songs
if is_valid:
    valid_songs[int(SALAMI_ID)] = [YT_ID, SONG_TITLE]
print(valid_songs)

