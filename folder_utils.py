# helpers to find and load files from drive

from drive_utils import get_drive_service
import io
import os
from googleapiclient.http import MediaIoBaseDownload
from googleapiclient.discovery import build
from pathlib import Path

# Get folder ID by name
def get_folder_id_by_name(service, folder_name, parent_id=None):
    query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder'"
    if parent_id:
        query += f" and '{parent_id}' in parents"

    results = service.files().list(q=query, fields="files(id, name)").execute()
    folders = results.get('files', [])
    return folders[0]['id'] if folders else None


# Download files from a specific folder in Google Drive
def download_file(service, folder_id, local_base_path='test_files', current_drive_path=""):
    """
    Recursively download all files in a Google Drive folder and its subfolders.

    :param service: Google Drive service object.
    :param folder_id: The starting folder ID.
    :param local_base_path: Local base folder to store files.
    :param current_drive_path: Drive-relative folder path (for recursive metadata).
    """
    # Step 1: Get all items inside current folder
    query = f"'{folder_id}' in parents and trashed = false"
    response = service.files().list(q=query, fields="files(id, name, mimeType)").execute() # mime type is used to differentiate files and folders
    items = response.get("files", [])

    for item in items:
        file_id = item.get('id')
        file_name = item.get('name')
        mime_type = item.get('mimeType') # mime type is used to differentiate files and folders

        if mime_type == 'application/vnd.google-apps.folder':
            # Recurse into subfolder
            new_drive_path = os.path.join(current_drive_path, file_name) # add subfolder to current path
            # Recurse into subfolder
            download_file(
                service,
                folder_id=file_id,
                local_base_path=local_base_path,
                current_drive_path=new_drive_path
            )
        
        else:
            # Create local path reflecting Drive structure
            full_local_folder = os.path.join(local_base_path, current_drive_path) # joins test_files path with current drive path to construct in local machine
            os.makedirs(full_local_folder, exist_ok=True) # ensure the folder exists
            local_file_path = os.path.join(full_local_folder, file_name) # full path to save the file in local machine

            try:
                request = service.files().get_media(fileId=file_id) # request binary content of the file
                fh = io.FileIO(local_file_path, 'wb')
                downloader = MediaIoBaseDownload(fh, request)
                done = False
                while not done:
                    status, done = downloader.next_chunk()
                print(f"Downloaded: {file_name} → {local_file_path}")
            except Exception as e:
                print(f"Failed to download {file_name}: {e}") # too big, network error, etc.


              

        

# if __name__ == "__main__":
#     service = get_drive_service()
#     knowledge_base_id = get_folder_id_by_name(service, "Knowledge Base")
#     regulatory_id = get_folder_id_by_name(service, "Regulatory", parent_id=knowledge_base_id)
#     print("Regulatory folder ID:", regulatory_id)