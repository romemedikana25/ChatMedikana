### Helper Functions for Google Drive Integration ###

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import os

SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

# Initialize Google Drive service
# Google Drive Service is used to interact with Google Drive API to list and download files and more from Drive.
def get_drive_service(token_path="token.json", credentials_path="credentials.json"):
    creds = Credentials.from_authorized_user_file(token_path, SCOPES)
    service = build('drive', 'v3', credentials=creds)
    return service