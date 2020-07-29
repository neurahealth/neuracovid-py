"""
The Original Inference.py file belongs to COVIDNet-CT Repository under GNU Affero General Public License v3.0
and is available here: https://github.com/haydengunraj/COVIDNet-CT.git
inference script for COVIDNet-CT model for COVID-19 detection in CT images.
"""
import json
import sys
import firebase_admin
from firebase_admin import firestore,storage,credentials
from google.cloud import pubsub
from google.cloud import storage as store
import logging.handlers
import os,argparse
import dotenv
dotenv.load_dotenv()
from numpy import asarray
import shutil

import os
import sys
import cv2
import json
import numpy as np
from math import ceil
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from dataset import COVIDxCTDataset
from data_utils import auto_body_crop
from utils import parse_args
slim = tf.contrib.slim

# Tensor names
IMAGE_INPUT_TENSOR = 'Placeholder:0'
LABEL_INPUT_TENSOR = 'Placeholder_1:0'
CLASS_PRED_TENSOR = 'ArgMax:0'
CLASS_PROB_TENSOR = 'softmax_tensor:0'
TRAINING_PH_TENSOR = 'is_training:0'
LOSS_TENSOR = 'add:0'

# Class names ordered by class index
CLASS_NAMES = ('Normal', 'Pneumonia', 'COVID-19')

class PubsubMessageHandler():
    def PubsubCallback(self,message):
        msg_id =  message.message_id

        filename = msg_id+'.log'
        my_logger = logging.getLogger(msg_id)
        my_logger.setLevel(logging.INFO)
        handler = logging.handlers.RotatingFileHandler(filename, maxBytes=20) # will create logger file
        my_logger.addHandler(handler)
        my_logger.info('Started with message_id : {}'.format(msg_id))

        sub_data = message.data.decode('utf-8') #decoding Pubsub message
        d = json.loads(sub_data)
        userId = d['userId']
        bucket = d['bucket']
        url = d['url']
        fileName = d['fileName']
        currentTime = d['currentTime']
        time = d['time']
        date = d['date']
        path = d['path']
        
        credential_json_file = ['json_file_name']
        databaseURL = ['databaseURL']
        storageBucket = ['storageBucket']
        
        if (not len(firebase_admin._apps)):
            cred = credentials.Certificate(credential_json_file)
            fa=firebase_admin.initialize_app(cred, {"databaseURL": databaseURL,'storageBucket':storageBucket})
            fc=firebase_admin.firestore.client(fa)
            db = firestore.client()
        blob = storage.bucket(storageBucket).blob(url)
        blob.download_to_filename('assets/{}.png'.format(msg_id)) # tagging image with unique message id
        my_logger.info('image name after tagging with message_id :{}.png'.format(msg_id))

        try:
            my_logger.info('Processing model now')
            def create_session():
                """Helper function for session creation"""
                config = tf.ConfigProto()
                config.gpu_options.allow_growth = True
                sess = tf.Session(config=config)
                return sess


            class Metrics:
                """Lightweight class for tracking metrics"""
                def __init__(self):
                    num_classes = len(CLASS_NAMES)
                    self.labels = list(range(num_classes))
                    self.class_names = CLASS_NAMES
                    self.confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.uint32)

            class COVIDNetCTRunner:
                """Primary training/testing/inference class"""
                def __init__(self, meta_file, ckpt=None, data_dir=None, input_height=224, input_width=224, max_bbox_jitter=0.025,
                             max_rotation=10, max_shear=0.15, max_pixel_shift=10, max_pixel_scale_change=0.2):
                    self.meta_file = meta_file
                    self.ckpt = ckpt
                    self.input_height = input_height
                    self.input_width = input_width
                    if data_dir is None:
                        self.dataset = None
                    else:
                        self.dataset = COVIDxCTDataset(
                            data_dir,
                            image_height=input_height,
                            image_width=input_width,
                            max_bbox_jitter=max_bbox_jitter,
                            max_rotation=max_rotation,
                            max_shear=max_shear,
                            max_pixel_shift=max_pixel_shift,
                            max_pixel_scale_change=max_pixel_scale_change
                        )

                def load_graph(self):
                    """Creates new graph and session"""
                    graph = tf.Graph()
                    with graph.as_default():
                        # Create session and load model
                        sess = create_session()

                        # Load meta file
                        print('Loading meta graph from ' + self.meta_file)
                        saver = tf.train.import_meta_graph(self.meta_file)
                    return graph, sess, saver

                def load_ckpt(self, sess, saver):
                    """Helper for loading weights"""
                    # Load weights
                    if self.ckpt is not None:
                        print('Loading weights from ' + self.ckpt)
                        saver.restore(sess, self.ckpt)

                def infer(self, image_file, autocrop=False):
                    """Run inference on the given image"""
                    # Load and preprocess image
                    image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
                    if autocrop:
                        image, _ = auto_body_crop(image)
                    image = cv2.resize(image, (self.input_width, self.input_height), cv2.INTER_CUBIC)
                    image = image.astype(np.float32) / 255.0
                    image = np.expand_dims(np.stack((image, image, image), axis=-1), axis=0)

                    # Create feed dict
                    feed_dict = {IMAGE_INPUT_TENSOR: image, TRAINING_PH_TENSOR: False}

                    # Run inference
                    graph, sess, saver = self.load_graph()
                    with graph.as_default():
                        # Load checkpoint
                        self.load_ckpt(sess, saver)

                        # Run image through model
                        class_, probs = sess.run([CLASS_PRED_TENSOR, CLASS_PROB_TENSOR], feed_dict=feed_dict)
                        print('\nPredicted Class: ' + CLASS_NAMES[class_[0]])
                        print('Confidences:' + ', '.join(
                            '{}: {}'.format(name, conf) for name, conf in zip(CLASS_NAMES, probs[0])))

                        N = probs[0][0]
                        P = probs[0][1]
                        C = probs[0][2]

                        result = CLASS_NAMES[class_[0]]
                        print('**DISCLAIMER**')
                        print('Do not use this prediction for self-diagnosis. '
                              'You should check with your local authorities for '
                              'the latest advice on seeking medical assistance.')
                        confidence = {
                        'normal': str("%.2f"%(N*100)),
                        'pneumonia': str("%.2f"%(P*100)),
                        'covid':str("%.2f"%(C*100))
                        }

                        detection = {'detections':
                        {'prediction':result,
                        'confidence' :confidence,
                        'imageType':'CT Scan'
                        }
                        }
                        my_logger.info('CT Scan Detection : {}'.format(detection))
                    try:
                        if (not len(firebase_admin._apps)):
                            cred = credentials.Certificate(credential_json_file )
                            fa=firebase_admin.initialize_app(cred, {"databaseURL": databaseURL ,'storageBucket':storageBucket })
                            fc=firebase_admin.firestore.client(fa)
                            db = firestore.client
                            doc_ref = db.collection(u'stripe_customers/{0}/results'.format(userId)).document(currentTime)
                            doc_ref.update(detection)
                            my_logger.error('CT Scan Detection : Saved to firestore and message acknowledged')
                        else:
                            print('alredy initialize')
                            db = firestore.client()
                            doc_ref = db.collection(u'stripe_customers/{0}/results'.format(userId)).document(currentTime)
                            doc_ref.update(detection)
                            my_logger.error('CT Scan Detection : Saved to firestore and message acknowledged')
                    except Exception as e :
                        print(e)
                        my_logger.error('CT Scan Detection : NOT Saved',e)


            if __name__ == '__main__':
                # Suppress most console output
                tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

                parser = argparse.ArgumentParser(description='COVIDNet-CT Infer Script')
                parser.add_argument('-md', '--model_dir', type=str, default='models/COVIDNet-CT-A', help='Model directory')
                parser.add_argument('-mn', '--meta_name', type=str, default='model.meta', help='Model meta name')
                parser.add_argument('-ck', '--ckpt_name', type=str, default='model', help='Model checkpoint name')
                parser.add_argument('-dd', '--data_dir', type=str, default=None, help='Data directory')
                parser.add_argument('-ih', '--input_height', type=int, default=512, help='Input image height')
                parser.add_argument('-iw', '--input_width', type=int, default=512, help='Input image width')
                parser.add_argument('-im', '--image_file', type=str, default='assets/{}.png'.format(msg_id), help='Image file')
                parser.add_argument('-ac', '--auto_crop', action='store_true',help='Flag to attempt automatic cropping of the image')
                augmentation_kwargs = {}
                # Create full paths
                args = parser.parse_args()

                meta_file = os.path.join(args.model_dir, args.meta_name)
                ckpt = os.path.join(args.model_dir, args.ckpt_name)

                # Create runner
                runner = COVIDNetCTRunner(
                    meta_file,
                    ckpt=ckpt,
                    data_dir=args.data_dir,
                    input_height=args.input_height,
                    input_width=args.input_width,
                    **augmentation_kwargs
                )
                # Run inference
                runner.infer(args.image_file, args.auto_crop)
        except Exception as e:
            print(e)

        try:
            storage_client = store.Client.from_service_account_json(credential_json_file )
            log_bucket = storage_client.get_bucket('bucket-logs')
            log_blob = log_bucket.blob('{0}/logs/{1}/{2}'.format(userId,date,currentTime))
            log_blob.upload_from_filename('{}.log'.format(msg_id))
            #print(msg_id)
            os.remove('./{}.log'.format(msg_id))
            os.remove('./assets/{}.png'.format(msg_id))
            message.ack()
            print("end of pubsubcallback")
        except Exception as e:
            print(e)

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
    
    print(" CT inference-module terminated")        

if __name__ == '__main__':
    main()
