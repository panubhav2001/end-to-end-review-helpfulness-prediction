import logging
import os
from datetime import datetime
from google.cloud import storage
import pytz

# Define IST timezone
IST = pytz.timezone('Asia/Kolkata')

# Set up Google Cloud Storage (GCS) bucket configuration
GCS_BUCKET_NAME = "amazon-reviews-project"  # Replace with your GCS bucket name
GCS_LOGS_FOLDER = "app_logs/"  # Folder in the bucket where logs will be saved

# Create the log file with a timestamped name
LOG_FILE = f"{datetime.now(IST).strftime('%m_%d_%Y_%H_%M_%S')}.log"
logs_local_path = os.path.join(os.getcwd(), "logs")
os.makedirs(logs_local_path, exist_ok=True)

LOG_FILE_PATH = os.path.join(logs_local_path, LOG_FILE)

# Configure logging
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def upload_log_to_gcs(local_log_path, bucket_name, destination_blob_name):
    """Uploads a file to Google Cloud Storage."""
    try:
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(local_log_path)
        logger.info(f"Log file {local_log_path} uploaded to {bucket_name}/{destination_blob_name}")
    except Exception as e:
        logger.error(f"Failed to upload log file to GCS: {e}")

# Automatically upload the log to GCS at the end of the session
def upload_log_on_exit():
    gcs_destination = os.path.join(GCS_LOGS_FOLDER, LOG_FILE)
    upload_log_to_gcs(LOG_FILE_PATH, GCS_BUCKET_NAME, gcs_destination)

# Register the upload function to run when the script exits
import atexit
atexit.register(upload_log_on_exit)

if __name__ == '__main__':
    logger.info('Logging has started')