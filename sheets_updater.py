import os
from google.oauth2 import service_account
from googleapiclient.discovery import build

class SheetsUpdater:
    def __init__(self, service_account_json: str):
        credentials = service_account.Credentials.from_service_account_file(
            service_account_json,
            scopes=['https://www.googleapis.com/auth/spreadsheets']
        )
        self.service = build('sheets', 'v4', credentials=credentials)

    def write_to_cell(self, spreadsheet_id: str, sheet_name: str, cell: str, value: str):
        """
        Updates a specific cell in Google Sheets.
        Example usage: write_to_cell("1abc123...", "Sheet1", "D4", "12")
        """
        range_name = f"{sheet_name}!{cell}"
        body = {'values': [[value]]}
        
        try:
            result = self.service.spreadsheets().values().update(
                spreadsheetId=spreadsheet_id,
                range=range_name,
                valueInputOption="USER_ENTERED",
                body=body
            ).execute()
            
            print(f"✅ Successfully updated {result.get('updatedCells')} cell(s).")
            return True
        except Exception as e:
            print(f"❌ Failed to update cell: {e}")
            return False

if __name__ == "__main__":
    # To use this in main.py, you would find the row index from ChromaDB, 
    # then calculate the cell (e.g. Row 4, Col D -> D4), and call this script.
    updater = SheetsUpdater("google-service-account.json")
    # updater.write_to_cell("YOUR_SHEET_ID_HERE", "เดือน2/2569", "B5", "10")