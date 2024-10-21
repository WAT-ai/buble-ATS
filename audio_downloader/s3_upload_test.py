import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError

def upload_file_to_s3(file_name, bucket_name, s3_file_name=None):
    """
    Uploads a file to an S3 bucket.

    :param file_name: File to upload
    :param bucket_name: Bucket to upload to
    :param s3_file_name: S3 object name (optional). If not specified, file_name is used
    :return: True if the file was uploaded successfully, else False
    """
    s3 = boto3.client('s3')
    try:
        # Default to using the same name in S3 if not specified
        if s3_file_name is None:
            s3_file_name = file_name

        # Upload file
        s3.upload_file(file_name, bucket_name, s3_file_name)
        print(f"File '{file_name}' uploaded successfully to '{bucket_name}/{s3_file_name}'")
        return True
    except FileNotFoundError:
        print(f"The file '{file_name}' was not found")
        return False
    except NoCredentialsError:
        print("Credentials not available")
        return False
    except PartialCredentialsError:
        print("Incomplete credentials found")
        return False
    except Exception as e:
        print(f"An error occurred: {e}")
        return False

if __name__ == "__main__":
    # Replace these with your details
    file_name = input("Enter the file path you want to upload: ")
    bucket_name = "your-bucket-name"  # Replace with your S3 bucket name
    s3_file_name = input("Enter the name for the file in S3 (or press Enter to use the same name): ")

    # Call the upload function
    upload_file_to_s3(file_name, bucket_name, s3_file_name or None)
