# buble-ATS
Audio Temporal Segmentation &amp; Sentiment Analysis 

## how to upload to aws s3 bucket

1. Install AWS CLI
   1. `pip install awscli`
   2. `aws configure`
   3. Enter your AWS Access Key ID, Secret Access Key, Default region name, Default output format
      1. Access Key ID and Secret Access Key for can be found in discord thread for now
2. Upload files to S3 bucket    
   1. `aws s3 cp <local_file_name> s3://<bucket_filepath>/`
       local_file_name: file name in your local machine
         bucket_filepath: file path in your S3 bucket where you want to upload the file (audio data goes in `antennai/audio-data/`)
