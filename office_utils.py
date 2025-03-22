import os
import win32com.client as client


def download_email_attachment(folder_path: str) -> None:

    outlook = client.Dispatch("Outlook.Application").GetNamespace("MAPI")

    for filename in os.listdir(folder_path):
        if filename.endswith(".msg"):
            msg = outlook.OpenSharedItem(os.path.join(folder_path, filename))
            for attachment in msg.Attachments:
                attachment.SaveAsFile(os.path.join(folder_path, attachment.FileName))
                print(f"Saved {attachment.FileName}")

    print("All email attachments have been downloaded.")
