import os
import csv
import google.auth
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]


def authenticate_google_drive():
    """Authenticate and create a Google Drive API client."""
    creds = None
    if os.path.exists("token.json"):
        creds, _ = google.auth.load_credentials_from_file("token.json")
    else:
        flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
        creds = flow.run_local_server(port=0)
        with open("token.json", "w", encoding="utf-8") as token:
            token.write(creds.to_json())

    drive_service = build("drive", "v3", credentials=creds)
    return drive_service


def get_file_links(drive_service, folder_id):
    """Get all file links from the folder."""
    results = (
        drive_service.files()
        .list(q=f"'{folder_id}' in parents", fields="files(id, name)")
        .execute()
    )
    items = results.get("files", [])

    file_data = []

    if not items:
        print("No files found.")
    else:
        for item in items:
            file_id = item["id"]
            file_name = item["name"]
            file_link = f"https://drive.google.com/uc?export=download&id={file_id}"
            file_data.append({"file": file_name, "link": file_link})
            print(f"File: {file_name}, Link: {file_link}")

    return file_data


def save_to_csv(file_data, filename):
    """Save file links to a CSV file."""
    with open(filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["file", "link"])
        writer.writeheader()
        writer.writerows(file_data)
    print(f"CSV file '{filename}' has been created successfully.")


def getlink(folder_id, csv_destination="image_links.csv"):
    drive_service = authenticate_google_drive()
    file_data = get_file_links(drive_service, folder_id)
    save_to_csv(file_data, csv_destination)

    if os.path.exists("token.json"):
        os.remove("token.json")
        print("token.json has been removed.")
