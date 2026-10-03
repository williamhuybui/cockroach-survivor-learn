# Intial set up to connect to Google Drive

#### 1. Go to [APIs Console](https://console.cloud.google.com/cloud-resource-manager?authuser=2) and make your own project.
#### 2. Search for ‘Google Drive API’, select the entry, and click ‘Enable’.
#### 3. Select ‘Credentials’ from the left menu, click ‘Create Credentials’, select ‘OAuth client ID’
#### 4. Now, the product name and consent screen need to be set -> click ‘Configure consent screen’ and follow the instructions. Once finished:

```
a.Select ‘Application type’ to be Web application.
b.Enter an appropriate name.
c.Input http://localhost:8080/ for ‘Authorized redirect URIs’.
d.Click ‘Create’.
```
#### 5. Click ‘Download JSON’ on the right side of Client ID to download client_secret_<really long ID>.json.

The downloaded file has all authentication information of your application. Rename the file to “client_secrets.json” and place it in your working directory.

## Installation

In Python Notebook, intsall library [PyDrive2](https://pypi.org/project/PyDrive2/) to install toolkit to connect with Google drive programmatically.

```bash
pip install PyDrive2
```

## Run quickstart.ipynb

To connect and extract data/file from Google drive.

## Stay logged in

The first run opens a browser to log in, then saves the token to `credentials.json` (next to `client_secrets.json`, gitignored). Later runs reuse it with no browser. To log in again, delete `credentials.json`.

## Kill port 8080 after use

In terminal, run the follow:

```
lsof -i :8080

kill <PID> 
```

