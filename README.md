# buble-ATS
Audio Temporal Segmentation &amp; Sentiment Analysis 

## how to upload to aws s3 bucket

1. Install AWS CLI
   1. `pip install awscli`
   2. `aws configure`
   3. Follow the wizard to enter your AWS Access Key ID, Secret Access Key, region name and output format
      1. Access Key ID can be found in discord thread for now
      2. Secret Access Key can be found in discord thread for now
      3. Default region name: us-west-1
      4. Default output format: json
2. Upload files to S3 bucket    
   1. `aws s3 cp <local_file_name> s3://<bucket_filepath>/`
       local_file_name: file name in your local machine
         bucket_filepath: file path in your S3 bucket where you want to upload the file (audio data goes in `antennai/audio-data/`)
