from google.oauth2 import service_account, credentials
from google.oauth2.service_account import Credentials
from google.auth.transport.requests import Request
from google.auth.exceptions import RefreshError
from typing import Optional

import os
import requests

credentials_path = 'C:\\Users\\Anthony.Wong\\AppData\\Roaming\\gcloud'

has_service_account_credentials = {
    'ul-gs-s-sandbx-02-prj': False,
    'ul-gs-d-901791-11-prj': True
}


def get_credentials_path(project: str) -> str:
    """Return the full path to the <project>'s credentials key file (User/Service Account).
    """
    return f'{credentials_path}\\application_{project}_credentials.json'


def get_user_credentials() -> Credentials:
    """Return the default user credentials. Refresh the credentials 
    token if necessary.
    """
    adc_path = f'{credentials_path}\\application_default_credentials.json'
    creds = credentials.Credentials.from_authorized_user_file(adc_path)

    try:
        if not creds.valid or creds.expired:
            if creds.refresh_token:
                creds.refresh(Request())
            return creds
    except RefreshError:
        print("User credentials is invalid and no longer refreshable. Run 'gcloud auth application-default login' to reauthenticate.")

    
def get_service_account_credentials(project: str) -> Credentials:
    """Return the service account credentials (key) for <project>.
    """
    project_credentials_path = get_credentials_path(project)
    return service_account.Credentials.from_service_account_file(project_credentials_path)


def get_credentials(project: str) -> Credentials:
    """Return the user or service account credentials associated 
    with <project>.
    """
    try:
        if has_service_account_credentials[project]:
            return get_service_account_credentials(project)
        return get_user_credentials()
    except KeyError:
        print(f"Project Undefined: No credentials found for project {project}")


def set_google_app_env(project: str) -> None:
    """Set 'GOOGLE_APPLICATION_CREDENTIALS environment variable
    to the <project>'s service account key path locally.
    """
    if has_service_account_credentials[project]:
        project_credentials_path = get_credentials_path(project)
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = project_credentials_path 
        print("Environment variable 'GOOGLE_APPLICATION_CREDENTIALS' successfully set!")
    print(f"No service account credentials found for project '{project}'")


    





    
