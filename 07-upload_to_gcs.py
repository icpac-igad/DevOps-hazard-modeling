import os
import datetime
import logging
from google.cloud import storage
import requests
import certifi

# Set the certificate path to use certifi's built-in certificates
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

# Set up logging
log_dir = r'C:\Users\hkoros\gcs_upload\logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_filename = os.path.join(log_dir, f'upload_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)

# Set environment variable
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'C:\Users\hkoros\gcs_upload\credentials\gcs-key.json'

# Change to the directory where your files are located
data_directory = r'D:\Data\output'
os.chdir(data_directory)
logging.info(f"Changed directory to: {data_directory}")

# Initialize client
try:
    client = storage.Client()
    bucket = client.bucket('imergv8_ea')
    logging.info("Successfully connected to GCS bucket: imergv8_ea")
except Exception as e:
    logging.error(f"Failed to connect to GCS: {e}")
    exit(1)

# List of files to upload
files_to_upload = [
    'rain_imerg_zone1.txt',
    'rain_imerg_zone2.txt',
    'rain_imerg_zone3.txt',
    'rain_imerg_zone4.txt',
    'rain_imerg_zone5.txt',
    'rain_imerg_zone6.txt',
    'streamflow_imerg_zone1.txt',
    'streamflow_imerg_zone2.txt',
    'streamflow_imerg_zone3.txt',
    'streamflow_imerg_zone4.txt',
    'streamflow_imerg_zone5.txt',
    'streamflow_imerg_zone6.txt'
]

# Generate today's date for filename prefixes
today_str = datetime.datetime.now().strftime("%Y%m%d")

# Upload statistics
successful_uploads = 0
failed_uploads = 0

# Upload files
logging.info(f"Starting upload of {len(files_to_upload)} files")
for filename in files_to_upload:
    file_path = os.path.join(data_directory, filename)
    if os.path.exists(file_path):
        try:
            # Create a new filename with date prefix to avoid conflicts
            gcs_filename = f"{today_str}_{filename}"
            
            # Upload file with new name
            blob = bucket.blob(gcs_filename)
            blob.upload_from_filename(file_path)
            logging.info(f'Successfully uploaded {filename} as {gcs_filename}')
            successful_uploads += 1
        except Exception as e:
            logging.error(f'Error uploading {filename}: {e}')
            failed_uploads += 1
    else:
        logging.warning(f'File not found: {filename}')
        failed_uploads += 1

# Summary
logging.info(f"Upload complete. Success: {successful_uploads}, Failed: {failed_uploads}")
logging.info(f"Log saved to: {log_filename}")