#!/bin/bash

sudo apt-get update

sudo apt-get install python3-venv -y
python3 -m venv venv_name
source venv_name/bin/activate

sudo apt install git-all -y

cd /opt
sudo mkdir inference_module ; cd inference_module
sudo mkdir src ; sudo mkdir bin ; cd src

sudo git clone https://github.com/neurahealth/neuracovid-py.git
cd neuracovid-py
sudo git clone https://github.com/lindawangg/COVID-Net.git
mv inference.py COVID-Net

pip install --upgrade pip
pip install -r requirements.txt

cd COVID-Net ; sudo mkdir models ; cd models ; sudo mkdir COVIDNet-CXR-Large ; cd COVIDNet-CXR-Large

# before below command use copying_files_from_google_drive_to_google_cloud_storage.ipynb file to copy model into GCP storage bucket

sudo gsutil cp gs://bucket_name/checkpoint .
sudo gsutil cp gs://bucket_name/model-8485.data-00000-of-00001 .
sudo gsutil cp gs://bucket_name/model-8485.index .
sudo gsutil cp gs://bucket_name/model.meta .

cd .. ; cd ..

sudo gsutil cp gs://bucket_name/neuracovid_dev.json .
sudo gsutil cp gs://bucket_name/env .

#give grants to /opt path” to successfully run python inference.py command

sudo apt-get install tmux -y

tmux 

cd ; source venv_name/bin/activate
cd /opt/inference_module/src/neuracovid-py/COVID-Net

python inference.py
