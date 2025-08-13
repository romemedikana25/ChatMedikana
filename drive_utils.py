from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import os
import streamlit as st
from google.auth.transport.requests import Request


SCOPES = ['https://www.googleapis.com/auth/drive.readonly']


# Initialize Google Drive service
# Google Drive Service is used to interact with Google Drive API to list and download files and more from Drive.
def get_drive_service(token_path="token.json", credentials_path="credentials.json"):
    if "gdrive_token" not in st.secrets:
        raise RuntimeError("Missing [gdrive_token] in st.secrets.")

    t = st.secrets["gdrive_token"]

    # Minimal required fields
    required = ("client_id", "client_secret", "refresh_token")
    missing = [k for k in required if not t.get(k)]
    if missing:
        raise RuntimeError(f"Missing required gdrive_token field(s): {', '.join(missing)}")

    creds = Credentials(
        token=t.get("token"),  # may be None; will be refreshed below
        refresh_token=t["refresh_token"],
        token_uri=t.get("token_uri", "https://oauth2.googleapis.com/token"),
        client_id=t["client_id"],
        client_secret=t["client_secret"],
        scopes=t.get("scopes", list(SCOPES)),
    )

    # Refresh if token is absent/expired and a refresh_token exists
    if (not creds.valid) and creds.refresh_token:
        creds.refresh(Request())

    return build("drive", "v3", credentials=creds)
