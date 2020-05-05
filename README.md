# neuracovid-py has a AI model which will give inferences by using tensorflow.
This file covers all requirments and dependencies from scratch for application Neuracovid-py

## follow the below steps to run model

### Step 1
#### Create pubsub topic and subscription
Go to Google Cloud Platform \
Click on Topics, create a new topic \
create a subscription for the topic \
Click on Subscriptions, select topic and create subscription \
In Google Cloud Platform select storage click on browser and create bucket to store artifacts eg. model , credential file etc \
create another bucket to store logs.
#### Create a vm instance and run commands on SSH terminal window from root directory 
recommendation is to have 2vcpu configuration to process max 3 messages at a time and 8vcpu configuration to process more than 25 messages
### Step 2
#### sudo apt-get update
apt-get update downloads the package lists and "updates" them to get information on the newest versions of packages and their dependencies
#### sudo apt-get install python3-venv -y
#### python3 -m venv venv_name
#### source venv_name/bin/activate
The python3-venv module allows us to create “virtual environments” and source will activate "virtual environment"
#### pip install --upgrade pip
This will upgrade pip to latest available version 
### Step 3
#### sudo apt install git-all -y
Install Git to clone repository

### Step 4
#### cd /opt ; sudo mkdir inference_module ; cd inference_module
#### sudo mkdir src ; sudo mkdir bin ; cd src
create path to clone repository
#### Step 5
#### git clone https://github.com/neurahealth/neuracovid-py.git
It will clone the Git Hub repository dedicated for neuracovid Application
#### cd neuracovid-py

#### pip install -r requirements.txt
###### below dependences will get install 
Tensorflow==1.14.0 \
numpy \
opencv-python \
scikit-Learn \
matplotlib \
pydicom \
pandas \
firebase_admin \
google-cloud-pubsub \
python-dotenv

### Application flow diagram

<img src="Application%20flow.png">

#### NeuraCovid Youtube video link
*https://www.youtube.com/watch?v=9erwBwCPPzU&feature=emb_title*

### Step 6
#### git clone https://github.com/lindawangg/COVID-Net.git
COVID-Net.git is a open source repository to fight againest the pandemic COVID-19, above command will clone it in neuracovid-
py directory

### Step 7
#### [Model] go to https://drive.google.com/drive/folders/1eNidqMyz3isLjGYN1evzQu--A-JVkzbk save in google drive 
#### google_drive_to_google_cloud_storage.ipynb file
This file will help to save model from google drive to storage bucket,run commands in colab to save model in storage bucket 
### Step 8
#### cd COVID-Net ; sudo mkdir models ; cd models ; sudo mkdir COVIDNet-CXR-Large ; cd COVIDNet-CXR-Large
create path to store model
#### gsutil cp gs://bucket_name/checkpoint .
#### gsutil cp gs://bucket_name/model-8485.data-00000-of-00001 .
#### gsutil cp gs://bucket_name/model-8485.index .
#### gsutil cp gs://bucket_name/model.meta .
This will copy model files from storage bucket (these files should be inside models/COVIDNet-CXR-Large/).
#### cd .. ; cd ..
come back to root directory to run script inference.py
#### Copy crerdential file and environment file to working directory(COVID-Net)
##### save credential .json file and .env in storage bucket
gsutil cp gs://bucket_name/credential.json . \
gsutil cp gs://bucket_name/.env .
#### give grants to /opt path” to successfully run python inference.py command
### run
#### python inference.py
## How inference.py work ?
inference.py script has a updated AI model \
This script will get trigger when a new message comes in pubsub topic, message will sent to topic upon the image upload by the user,
findings will be store in firebase-database for every user uniquely
#### Inferences will be:
prediction and sensitivity \
Normal, Pneumonia, COVID-19 and sensitivity will stored in pecentage
## Deployment
application can be deloped to GCP kubernetes engine 

#### Done Cheers!