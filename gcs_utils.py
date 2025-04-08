from utils.iam_utils import get_credentials
from google.cloud.storage import Client


def get_storage_client(project: str) -> Client:
    return Client(project=project, credentials=get_credentials(project))


def download_blob(
    project: str, 
    bucket_name: str, 
    source_blob_name: str, 
    destination_file_name: str):
    """Downloads a blob from the bucket.
    
    Example:
    download_blob(
        project='ul-gs-s-sandbx-02-prj',
        bucket_name='icis-reports', 
        source_blob_name='reports/petchem/a-polye-pdf-20110105', 
        destination_file_name='.\\data\\hackathon\\unstrucutured\\a-polye-pdf-20110105.pdf'
    )
    """
    storage_client = get_storage_client(project)
    bucket = storage_client.bucket(bucket_name)

    # Construct a client side representation of a blob.
    # Note `Bucket.blob` differs from `Bucket.get_blob` as it doesn't retrieve
    # any content from Google Cloud Storage. As we don't need additional data,
    # using `Bucket.blob` is preferred here.
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

    print(
        "Downloaded storage object {} from bucket {} to local file {}.".format(
            source_blob_name, bucket_name, destination_file_name
        )
    )
