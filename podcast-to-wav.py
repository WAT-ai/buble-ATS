# Script to convert a list of podcast links into wav files

# Had to make a virtual environment for dependencies:
# python3 -m venv path/to/venv
# source path/to/venv/bin/activate
# python3 -m pip install xyz
import requests
from pydub import AudioSegment
from io import BytesIO


# Credentials
API_KEY = "ENTER HERE" # DELETE THIS BEFORE YOU GIT PUSH
SPREADSHEET_ID = "1cP0T3PX1FKajqqTcIJMipeqebLmQZ8aLtQ7BXoFXWz4"
SHEET_NAME = "Working Stations"
CELLS = "B:C"


def get_google_sheet_data(spreadsheet_id, sheet_name, cells, api_key):
    # Construct the URL for the Google Sheets API
    url = f'https://sheets.googleapis.com/v4/spreadsheets/{spreadsheet_id}/values/{sheet_name}!{cells}?alt=json&key={api_key}'

    try:
        # Make a GET request to retrieve data from the Google Sheets API
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors

        # Parse the JSON response
        data = response.json()
        return data

    except requests.exceptions.RequestException as e:
        # Handle any errors that occur during the request
        print(f"An error occurred: {e}")
        return None


def print_podcasts(sheet_data, links):
    print("\n----------------------------------- PODCASTS LIST -----------------------------------\n\n")
    for i in range(1, len(sheet_data)):
        print(f'NAME: {sheet_data[i][0]}')
        print(f'LINK: {sheet_data[i][1]}\n')
    print("\n----------------------------------- PODCASTS LIST -----------------------------------\n")


def download_and_convert_to_wav(url, output_filename):
    try:
        # Request the stream
        response = requests.get(url, stream=True, timeout=10)
        print(response.status_code)
        
        if response.status_code == 200:
            audio_data = BytesIO()

            # Read the response in chunks and write to BytesIO
            # CURRENT LIMITER HERE
            total_downloaded = 0
            max_size_mb = 5
            for chunk in response.iter_content(chunk_size=8192):
                audio_data.write(chunk)
                total_downloaded += len(chunk)
                print(total_downloaded)

                # Check if we've exceeded the max size limit
                if total_downloaded > max_size_mb * 1024 * 30:  # Convert MB to bytes
                    print(f"Reached maximum size limit of {max_size_mb} MB for {url}.")
                    break

            audio_data.seek(0)
            audio = AudioSegment.from_file(audio_data)
            
            print(audio)
            # Export as WAV
            wav_filename = f"{output_filename}.wav"
            audio.export(wav_filename, format="wav")
            print(f"Converted and saved {url} to {wav_filename}\n")
        else:
            print(f"Failed to retrieve audio from {output_filename} at {url}\n")
    except Exception as e:
        print(f"Error processing {url}: {e}")


if __name__ == '__main__':
    sheet_data = get_google_sheet_data(SPREADSHEET_ID, SHEET_NAME, CELLS, API_KEY)
    links = []

    if sheet_data:
        # Print Podcasts
        sheet_data = sheet_data["values"]
        print_podcasts(sheet_data, links)

        # Convert to WAV
        for i in range(1, len(sheet_data)):
            name = "PODCAST_WAV: " + sheet_data[i][0]
            link = sheet_data[i][1]
            links.append(sheet_data[i][1])
            
            download_and_convert_to_wav(link, name)

    else:
        print("Failed to fetch data from Google Sheets API.")
