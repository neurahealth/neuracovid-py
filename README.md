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
#### git clone https://github.com/haydengunraj/COVIDNet-CT.git

### Step 7
####  save models in google drive 
#### [Model-CXR-Large] go to https://bit.ly/CovidNet-CXR-Large
#### [Model-SEV-GEO]   go to https://bit.ly/COVIDNet-SEV-GEO 
#### [Model-SEV-OPC]   go to https://bit.ly/COVIDNet-SEV-OPC
#### [COVIDNet-CT-A]   go to https://bit.ly/2BAPyvM

#### google drive to google cloud storage.ipynb 
This file will help to save model from google drive to storage bucket,run commands in colab to save model in storage bucket 
### Step 8
#### cd COVID-Net ; sudo mkdir models ; cd models 
create path to store model
#### gsutil cp gs://bucket_name/COVIDNet-CXR-Large .
#### gsutil cp gs://bucket_name/COVIDNet-SEV-GEO .
#### gsutil cp gs://bucket_name/COVIDNet-SEV-OPC .

This will copy model files from storage bucket (these files should be inside models/COVIDNet-CXR-Large/).
#### cd ..
#### gsutil cp gs://bucket_name/COVIDNet-CT-A COVIDNet-CT-A/models
#### cd ..

come back to root directory to run script inference.py

#### Copy crerdential file and environment file to working directory(COVID-Net)
##### save credential .json file and .env in storage bucket
gsutil cp gs://bucket_name/credential.json . \
gsutil cp gs://bucket_name/.env .
#### give grants to /opt path” to successfully run python inference.py command
### run
#### for inference only > python inference.py  
#### for inference and heatmap > inference_heatmap.py 
#### for CT innference > COVIDNet-CT-A/covidnet_ct.py
###### open another shell 
#### python inference_severity.py
## How inference.py | inference_heatmap.py  and inference_severity.py work ?
inference.py | inference_heatmap.py script has a updated AI model \
This script will get trigger when a new message comes in pubsub topic, message will sent to topic upon the image upload by the user,
findings will be store in firebase-database and heatmaps to storage for every user uniquely
#### Inferences will be:
prediction, sensitivity and severity \
Normal, Pneumonia, COVID-19 and sensitivity will stored in pecentage
## Deployment
application can be deployed to GCP kubernetes engine 

#### Done Cheers!
