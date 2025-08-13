from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import os
import pickle
import json

# Only allow reading files from Google Drive
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

def authenticate():
    creds = None

    # Load token.json if it exists
    if os.path.exists('token.json'):
        with open('token.json', 'r') as token:
            creds_data = json.load(token)
            from google.oauth2.credentials import Credentials
            creds = Credentials.from_authorized_user_info(info=creds_data, scopes=SCOPES)

    # If there are no (valid) credentials, do the OAuth flow
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            # Start browser-based login
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)

        # Save the new credentials
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    print("✅ Google Drive API authenticated and token.json saved!")

if __name__ == '__main__':
    authenticate()