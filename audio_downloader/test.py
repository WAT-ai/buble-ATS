import boto3
import os
from pydub import AudioSegment
import requests
from botocore.exceptions import NoCredentialsError, PartialCredentialsError, BotoCoreError, ClientError
from io import BytesIO
s3 = boto3.client(
    's3',
    aws_access_key_id='',
    aws_secret_access_key='',
    region_name='us-east-1'
)
def check_or_create_bucket(s3_bucket, region='us-east-1'):
    
    try:
        # Check if the bucket exists by calling head_bucket
        s3.head_bucket(Bucket=s3_bucket)
        print(f"Bucket '{s3_bucket}' already exists.")
    except ClientError as e:
        # If the error is not a 404 (bucket not found), then raise it
        error_code = int(e.response['Error']['Code'])
        if error_code == 404:
            try:
                s3.create_bucket(
                    Bucket=s3_bucket,
                    CreateBucketConfiguration={'LocationConstraint': region}
                )
                print(f"Bucket '{s3_bucket}' created successfully.")
            except ClientError as create_error:
                print(f"Error creating bucket: {create_error}")
                raise
        else:
            print(f"Error checking bucket existence: {e}")
            raise

def stream_audio(url):
    try:
        response = requests.get(url, stream=True, timeout=10)
        response.raise_for_status()  # Check if the request was successful
        return response
    except requests.exceptions.RequestException as e:
        print(f"Error streaming audio from {url}: {e}")
        return None

def chunk_and_upload_audio(audio_stream, s3_bucket, folder_name, chunk_duration=180000):
    try:
        # Use BytesIO to process the audio in memory
        audio_data = BytesIO(audio_stream.content)
        audio = AudioSegment.from_file(audio_data, format="wav")  # Adjust format as needed (e.g., mp3, wav)

        # Set up AWS S3 client
        total_chunks = len(audio) // chunk_duration

        for i in range(0, len(audio), chunk_duration):
            chunk = audio[i:i + chunk_duration]

            # Create an in-memory BytesIO buffer for the chunk
            chunk_buffer = BytesIO()
            chunk.export(chunk_buffer, format="wav")  # Export chunk to buffer in same format
            chunk_buffer.seek(0)  # Move the buffer cursor to the start

            # Create the S3 key for the chunk
            chunk_filename = f"{folder_name}/chunk_{i // chunk_duration}.wav"
            try:
                s3.upload_fileobj(chunk_buffer, s3_bucket, chunk_filename)
                print(f"Uploaded chunk {i // chunk_duration + 1}/{total_chunks} to s3://{s3_bucket}/{chunk_filename}")
            except (NoCredentialsError, PartialCredentialsError, BotoCoreError) as e:
                print(f"Error uploading {chunk_filename} to S3: {e}")
            finally:
                chunk_buffer.close()

    except Exception as e:
        print(f"Error processing and uploading audio: {e}")

def process_audio_file(url, s3_bucket, region='us-east-1'):
    # Step 1: Check if the bucket exists or create it
    check_or_create_bucket(s3_bucket, region)

    # Extract file name without extension to use as folder name
    file_name = os.path.splitext(url.split("/")[-1])[0]
    
    # Create the folder structure within 'audio_files' main folder
    folder_name = f"audio_files/{file_name}"

    # Step 2: Stream audio file from URL
    audio_stream = stream_audio(url)
    if not audio_stream:
        print("Audio streaming failed. Aborting process.")
        return

    # Step 3: Chunk and upload directly to S3
    chunk_and_upload_audio(audio_stream, s3_bucket, folder_name)

# Example usage
audio_url = "https://file-examples.com/storage/fe5f9cbfeb6722a469d332b/2017/11/file_example_WAV_1MG.wav"  # Change to .wav or other format if needed
s3_bucket = "antenna-ai-bucket"

process_audio_file(audio_url, s3_bucket, region='us-west-2')
