"""
The Original Inference.py file belongs to COVID-Net Repository under GNU Affero General Public License v3.0
and is available here: https://github.com/lindawangg/COVID-Net/blob/master/inference.py
Neura Health made modifications to inference.py to run with neuracovid application.
"""

import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
import sys
import json
import firebase_admin
from firebase_admin import firestore,storage
from firebase_admin import credentials
from google.cloud import pubsub
from google.cloud import storage as store
import logging
import dotenv
dotenv.load_dotenv()
from distutils.dir_util import copy_tree
import shutil

#Covid-net packages
import numpy as np
import tensorflow as tf
import os, argparse
import cv2
from tensorflow.python.framework import ops

#TODO
# do this changes in below code

#credential_json_file = ['file_name']
#databaseURL = ['databaseURL']
#storageBucket = ['storageBucket']
#project_id = ['Project ID']
#subscription_name = ['subscription_name']
#logs_bucket = ['logs_bucket_name']


class PubsubMessageHandler():
    
    def PubsubCallback(self,message):
        #print("start pubsubcallback")
        msg_id =  message.message_id
        #print(msg_id)
        filename = msg_id+'.log' 
        #print(filename)

        my_logger = logging.getLogger(msg_id)
        my_logger.setLevel(logging.INFO)
        handler = logging.handlers.RotatingFileHandler(filename, maxBytes=20) # will create logger file
        my_logger.addHandler(handler)
        my_logger.info('Started with message_id : {}'.format(msg_id))
        
        #print(message)
        sub_data = message.data.decode('utf-8') #decoding Pubsub message 
        d = json.loads(sub_data)
        userId = d['userId'] 
        bucket = d['bucket']
        url = d['url']
        fileName = d['fileName']
        currentTime = d['currentTime']
        date = d['date']
       
        # initialize firebase app to download user's uplaoded image from storage
        credential_json_file = ['file_name']
        databaseURL = ['databaseURL']
        storageBucket = ['storageBucket']
        
        if (not len(firebase_admin._apps)):    
            cred = credentials.Certificate(credential_json_file)
            fa=firebase_admin.initialize_app(cred, {"databaseURL": databaseURL,'storageBucket':storageBucket})
            fc=firebase_admin.firestore.client(fa)
            db = firestore.client()
        blob = storage.bucket(storageBucket).blob(url) # this will download image from url
        my_logger.info('images name after tagging with message_id :{}.jpeg'.format(msg_id))
        blob.download_to_filename('assets/{}.jpeg'.format(msg_id)) # tagging image with unique message id
        
        my_logger.info('Processing model now')
        
        try:
            parser = argparse.ArgumentParser(description='COVID-Net Inference')
            parser.add_argument('--weightspath', default='models/COVIDNet-CXR-Large', type=str, help='Path to output folder')
            parser.add_argument('--metaname', default='model.meta', type=str, help='Name of ckpt meta file')
            parser.add_argument('--ckptname', default='model-8485', type=str, help='Name of model ckpts')
            parser.add_argument('--imagepath', default='assets/{}.jpeg'.format(msg_id), type=str, help='Full path to image to be inferenced')
            args = parser.parse_args()

            mapping = {'normal': 0, 'pneumonia': 1, 'COVID-19': 2}
            inv_mapping = {0: 'normal', 1: 'pneumonia', 2: 'COVID-19'}
            
            ops.reset_default_graph()
            sess = tf.Session() 
            tf.get_default_graph()
            saver = tf.train.import_meta_graph(os.path.join(args.weightspath, args.metaname))
            saver.restore(sess, os.path.join(args.weightspath, args.ckptname))
        
            graph = tf.get_default_graph()
        
            image_tensor = graph.get_tensor_by_name("input_1:0")
            pred_tensor = graph.get_tensor_by_name("dense_3/Softmax:0")
            x = cv2.imread(args.imagepath)
            x = cv2.resize(x, (224, 224))
            x = x.astype('float32') / 255.0
            
            pred = sess.run(pred_tensor, feed_dict={image_tensor: np.expand_dims(x, axis=0)})
            print('Prediction: {}'.format(inv_mapping[pred.argmax(axis=1)[0]]))
            print('Confidence')
            print('Normal: {:.3f}, Pneumonia: {:.3f}, COVID-19: {:.3f}'.format(pred[0][0], pred[0][1], pred[0][2]))
            print('**DISCLAIMER**')
            print('Do not use this prediction for self-diagnosis. You should check with your local authorities for the latest advice on seeking medical assistance.')
            #print(currentTime)
            
            N = pred[0][0] # Normal
            P = pred[0][1] # Pneumonia 
            C = pred[0][2] # COVID-19
            
            result = ('{}'.format(inv_mapping[pred.argmax(axis=1)[0]]))
        
            #Converting into percentage
            confidence = {
                        'normal': str("%.2f"%(N*100)),
                        'pneumonia': str("%.2f"%(P*100)),
                        'covid':str("%.2f"%(C*100))
                    }
            
            detection = {'detections':
                    {'prediction':result,
                        'confidence' :confidence
                        }
                    }
            
            my_logger.info('finish successfully and message acknowledge')
            message.ack()
        except BaseException as error:
            my_logger.error('{}'.format(error))
            my_logger.info('error occurred need to reprocess message NOT acknowledge')
        
        try:
            my_logger.info('Detection : {}'.format(detection))
            
            credential_json_file = ['json_file_name']
            databaseURL = ['databaseURL']
            storageBucket = ['storageBucket']
            
            # might gives error in run-time for initialization to rule out such error checking initialization of firebase app again
            
            if (not len(firebase_admin._apps)):
                cred = credentials.Certificate(credential_json_file)
                fa=firebase_admin.initialize_app(cred, {"databaseURL": databaseURL,'storageBucket':storageBucket})
                fc=firebase_admin.firestore.client(fa)
                db = firestore.client()
                doc_ref = db.collection(u'stripe_customers/{0}/results'.format(userId)).document(currentTime)
                doc_ref.update(detection)
            else:
                print('alredy initialize')
                db = firestore.client()
                doc_ref = db.collection(u'stripe_customers/{0}/results'.format(userId)).document(currentTime)
                doc_ref.update(detection)
        
        except Exception as e :
            my_logger.error('Detection : not done')
        
        logs_bucket = ['logs_bucket_name']
        storage_client = store.Client.from_service_account_json(credential_json_file)
        log_bucket = storage_client.get_bucket(logs_bucket)
        log_blob = log_bucket.blob('{0}/logs/{1}/{2}'.format(userId,date,currentTime))
        log_blob.upload_from_filename('{}.log'.format(msg_id))
        #print(msg_id)
        os.remove('./{}.log'.format(msg_id))
        os.remove('./assets/{}.jpeg'.format(msg_id))
        print("waiting for new message...")

def main():
    project_id = os.getenv("project_id") # from .env file here we'll check for development environment and production environment
    print('Project id : ',project_id)
    
    if project_id=="project_name":
        project_id = ['Project ID']
        subscription_name = ['subscription_name']
        print("processing in development environment")
        subscription_path = "projects/{0}/subscriptions/(1)".format(project_id,subscription_name)
    elif project_id=="project_name":
        project_id = ['Project ID']
        subscription_name = ['subscription_name']        
        print("processing in production environment")
        subscription_path = "projects/{0}/subscriptions/{1}".format(project_id,subscription_name)
    else:
        print("project_id not found")
    # Below three lines would trigger the processing of the message from uploads subscription
    handler = PubsubMessageHandler()
    subscriber = pubsub.SubscriberClient()
    future = subscriber.subscribe(subscription_path,handler.PubsubCallback) # this line will fetch new message from pubsub
    
    try:
        future.result()
        print("message processed",future.result())
    
    except KeyboardInterrupt:
        # User  exits the script early.
        future.cancel()             #to cancel the process with keyboard interrupt Ctrl+c
        print('Received keyboard interrupt, exiting...')
    
    print("inference-module terminated")        

if __name__ == '__main__':
    main()
