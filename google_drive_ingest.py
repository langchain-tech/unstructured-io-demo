import os
import subprocess
import sys
import json
from dotenv import load_dotenv

load_dotenv()

drive_id = os.getenv("GOOGLE_DRIVE_ID")
pwd = os.getcwd()


def run_unstructured_ingest(drive_id, service_account_key_path, output_dir,extract_image_block_output_dir, num_processes=2, recursive=True, verbose=True):
    additional_partition_args = {
        "extract_images_in_pdf": True,
        "chunking_strategy": "by_title",
        "max_characters": 3000,
        "new_after_n_chars": 2500,
        "extract_image_block_output_dir": extract_image_block_output_dir
    }

    command = [
        "unstructured-ingest", "google-drive",
        "--drive-id", drive_id,
        "--service-account-key", service_account_key_path,
        "--output-dir", output_dir,
        "--num-processes", str(num_processes),
        "--recursive" if recursive else "",
        "--verbose" if verbose else "",
        "--additional-partition-args", json.dumps(additional_partition_args)
    ]
    
    command = [arg for arg in command if arg]
    print(command)
    print()
    process = subprocess.Popen(command, stdout=sys.stdout, stderr=sys.stderr, text=True)
    process.communicate()
    return process.returncode


if __name__ == "__main__":
    drive_id =drive_id
    service_account_key_path = os.path.join(pwd, "service_account_key_for_google_drive.json")
    output_dir = os.path.join(pwd, "google-drive-ingest-output")
    extract_image_block_output_dir=os.path.join(pwd, "google-drive-ingest-output/image")

    run_unstructured_ingest(drive_id, service_account_key_path, output_dir,extract_image_block_output_dir)
