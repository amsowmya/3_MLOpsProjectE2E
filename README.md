bash init_setup.sh

# Airflow terms
ti : task instance / task information 
xcom_push : cross communication push
xcom_pull : cross communication pull

# DVC 
dvc init

# add the files to local remote storage
dvc remote add -d remote_storage my_remote_storage

# pipeline reproducability
dvc repro

# Deployment
docker build -t youtubelive.azure.io/youtubecomm:latest .
docker login youtubelive.azure.io
username
password

docker push youtubelive.azure.io/youtubecomm:latest