import streamlit as st
import os
from pydub import AudioSegment
from io import BytesIO
import boto3
from dotenv import load_dotenv
from botocore.exceptions import NoCredentialsError, PartialCredentialsError, BotoCoreError, ClientError

# Set the path to the ffmpeg executable if needed
AudioSegment.converter = "C:/ffmpeg/bin/ffmpeg.exe"
region = 'us-west-1'
load_dotenv()

# Initialize S3 client with environment variables for credentials
s3 = boto3.client(
    's3',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=region
)

def check_bucket_exists(bucket_name):
    try:
        response = s3.list_buckets()
        # Extract the names of all buckets
        bucket_names = [bucket['Name'] for bucket in response['Buckets']]
        if bucket_name in bucket_names:
            print(f"Bucket '{bucket_name}' exists.")
            return True
        else:
            print(f"Bucket '{bucket_name}' does not exist.")
            return False
    except ClientError as e:
        print(f"Error checking bucket existence: {e}")
        return False


def create_bucket(s3_bucket, region=region):
    try:
        s3.create_bucket(
            Bucket=s3_bucket
        )
        # s3.create_bucket(
        #     Bucket=s3_bucket,
        #     CreateBucketConfiguration={'LocationConstraint': region}
        # )
        print(f"Bucket '{s3_bucket}' created successfully.")
    except ClientError as e:
        print(f"Error creating bucket: {e}")
        raise

def get_upload():
    uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])
    if uploaded_file is not None:
        file_type = uploaded_file.name.split('.')[-1].lower()
        st.write(f"Uploaded file: {uploaded_file.name} ({file_type})")
        try:
            audio_data = AudioSegment.from_file(uploaded_file, format=file_type)
            
            st.audio(uploaded_file, format=f"audio/{file_type}")
            
            buffer = BytesIO()
            audio_data.export(buffer, format=file_type)
            buffer.seek(0)
            bucket_name = "buble-ats"
            s3_key= f"audio-data/audio_files/{uploaded_file.name}"
            try:
                s3.upload_fileobj(
                    buffer,
                    bucket_name,
                    s3_key,
                    ExtraArgs={"ContentType": f"audio/{file_type}"}
                )
                st.success("File successfully uploaded to S3!")
            except Exception as e:
                st.error(f"Failed to upload file to S3: {e}")
        except Exception as err:
            st.error(f"Failed to upload file to S3: {e}")
    else:
        st.write("Please upload a .wav or .mp3 file")

get_upload()
