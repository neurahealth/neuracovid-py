"""
The Original Inference.py file belongs to COVID-Net Repository under GNU Affero General Public License v3.0
and is available here: https://github.com/lindawangg/COVID-Net/blob/master/inference.py
Neura Health made modifications to inference.py to run with neuracovid application.
"""
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

import json
import sys
import firebase_admin
import shutil

import dotenv
dotenv.load_dotenv()
import logging
import logging.handlers
from firebase_admin import firestore,storage
from firebase_admin import credentials
from google.cloud import pubsub
from google.cloud import storage as store
from distutils.dir_util import copy_tree

#Covid-net packages
import numpy as np
import tensorflow as tf
import os, argparse

from tensorflow.python.framework import ops

from data import process_image_file

# import pandas as pd
from data import process_image_file
from collections import defaultdict

from tensorflow.python.framework import ops

class PubsubMessageHandler():
     def PubsubCallback(self,message):

        msg_id =  message.message_id
        filename = 'severity_'+msg_id+'.log'

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
        time = d['time']
        date = d['date']

        credential_json_file = ['file_name']
        databaseURL = ['databaseURL']
        storageBucket = ['storageBucket']

        if (not len(firebase_admin._apps)):
            cred = credentials.Certificate(credential_json_file)
            fa=firebase_admin.initialize_app(cred, {"databaseURL": databaseURL,'storageBucket':storageBucket})
            fc=firebase_admin.firestore.client(fa)
            db = firestore.client()
        blob = storage.bucket(storageBucket).blob(url)
        my_logger.info('images name after tagging with message_id :{}.jpeg'.format(msg_id))
        blob.download_to_filename('assets/severity_{}.jpeg'.format(msg_id))

        try:
            def score_prediction(softmax, step_size):
                vals = np.arange(3) * step_size + (step_size / 2.)
                vals = np.expand_dims(vals, axis=0)
                return np.sum(softmax * vals, axis=-1)

            class MetaModel:
                def __init__(self, meta_file, ckpt_file):
                    self.meta_file = meta_file
                    self.ckpt_file = ckpt_file

                    self.graph = tf.Graph()
                    with self.graph.as_default():
                        self.saver = tf.compat.v1.train.import_meta_graph(self.meta_file)
                        self.input_tr = self.graph.get_tensor_by_name('input_1:0')
                        self.phase_tr = self.graph.get_tensor_by_name('keras_learning_phase:0')
                        self.output_tr = self.graph.get_tensor_by_name('MLP/dense_1/MatMul:0')

                def infer(self, image):
                    with tf.compat.v1.Session(graph=self.graph) as sess:
                        self.saver.restore(sess, self.ckpt_file)

                        outputs = defaultdict(list)
                        outs = sess.run(self.output_tr,
                                        feed_dict={
                                            self.input_tr: np.expand_dims(image, axis=0),
                                            self.phase_tr: False
                                        })
                        outputs['logits'].append(outs)

                        for k in outputs.keys():
                            outputs[k] = np.concatenate(outputs[k], axis=0)

                        outputs['softmax'] = np.exp(outputs['logits']) / np.sum(
                            np.exp(outputs['logits']), axis=-1, keepdims=True)
                        outputs['score'] = score_prediction(outputs['softmax'], 1 / 3.)

                    return outputs['score']

                    ops.reset_default_graph()
        except Exception as e:
            print(e)


        if __name__ == '__main__':
            try:
                parser = argparse.ArgumentParser(description='COVID-Net Lung Severity Scoring')
                parser.add_argument('--weightspath_geo', default='models/COVIDNet-SEV-GEO', type=str, help='Path to output folder')
                parser.add_argument('--weightspath_opc', default='models/COVIDNet-SEV-OPC', type=str, help='Path to output folder')
                parser.add_argument('--metaname', default='model.meta', type=str, help='Name of ckpt meta file')
                parser.add_argument('--ckptname', default='model', type=str, help='Name of model ckpts')
                parser.add_argument('--imagepath', default='assets/severity_{}.jpeg'.format(msg_id), type=str, help='Full path to image to perfom scoring on')
                parser.add_argument('--input_size', default=480, type=int, help='Size of input (ex: if 480x480, --input_size 480)')
                parser.add_argument('--top_percent', default=0.08, type=float, help='Percent top crop from top of image')

                args = parser.parse_args()

                x = process_image_file(args.imagepath, args.top_percent, args.input_size)
                x = x.astype('float32') / 255.0
                # check if models exists

                infer_geo = os.path.exists(os.path.join(args.weightspath_geo, args.metaname))
                infer_opc = os.path.exists(os.path.join(args.weightspath_opc, args.metaname))

                if infer_geo:
                    model_geo = MetaModel(os.path.join(args.weightspath_geo, args.metaname),
                            os.path.join(args.weightspath_geo, args.ckptname))
                    output_geo = model_geo.infer(x)

                    print('Geographic severity: {:.3f}'.format(output_geo[0]))
                    Geographic_severity='{:.3f}'.format(output_geo[0])
                    GS = float(Geographic_severity)
                    print('Geographic extent score for right + left lung (0 - 8): {:.3f}'.format(output_geo[0]*8))
                    Geographic_extent_score = '{:.3f}'.format(output_geo[0]*8)
                    Geographic_extent_score = float(Geographic_extent_score)
                    print('For each lung: 0 = no involvement; 1 = <25%; 2 = 25-50%; 3 = 50-75%; 4 = >75% involvement.')

                if infer_opc:
                    model_opc = MetaModel(os.path.join(args.weightspath_opc, args.metaname),
                            os.path.join(args.weightspath_opc, args.ckptname))
                    output_opc = model_opc.infer(x)

                    print('Opacity severity: {:.3f}'.format(output_opc[0]))
                    Opacity_severity = ('{:.3f}'.format(output_opc[0]))
                    OS = float(Opacity_severity)
                    print('Opacity extent score for right + left lung (0 - 6): {:.3f}'.format(output_opc[0]*6))
                    Opacity_extent_score= ('{:.3f}'.format(output_opc[0]*6))
                    Opacity_extent_score = float(Opacity_extent_score)

                print('For each lung: 0 = no opacity; 1 = ground glass opacity; 2 =consolidation; 3 = white-out.')
                print('**DISCLAIMER**')
                print('Do not use this prediction for self-diagnosis. You should check with your local authorities for the latest advice on seeking medical assistance.')

                detection={'Geograph_and_Opacity':
                    {'Geographic_severity':("%.2f"%(GS*100)),
                        'Geographic_extent_score':(round(Geographic_extent_score,2)),
                        'Opacity_severity':("%.2f"%(OS*100)),
                        'Opacity_extent_score':(round(Opacity_extent_score,2))
                        }
                    }
            except Exception as e:
                my_logger.error('ERROR : ',e)
                print(e)

            try:
                if (not len(firebase_admin._apps)):
                    cred = credentials.Certificate(credential_json_file)
                    fa=firebase_admin.initialize_app(cred, {"databaseURL": databaseURL,'storageBucket':storageBucket })
                    fc=firebase_admin.firestore.client(fa)
                    db = firestore.client()
                    doc_ref = db.collection(u'stripe_customers/{0}/results'.format(userId)).document(currentTime)
                    doc_ref.update(detection)
                    my_logger.info(detection)
                    my_logger.info('Geograph : Saved in firestore & message Acknowledge')
                    message.ack()
                else:
                    # print('alredy initialize')
                    db = firestore.client()
                    doc_ref = db.collection(u'stripe_customers/{0}/results'.format(userId)).document(currentTime)
                    doc_ref.update(detection)
                    my_logger.info(detection)
                    my_logger.info('Geograph : Saved in firestore & message Acknowledge')
                    message.ack()

            except Exception as e :
                my_logger.error('Detection : NOT Saved , message Not Acknowledge' )

            storage_client = store.Client.from_service_account_json(credential_json_file)
            log_bucket = storage_client.get_bucket('bucket-name')
            log_blob = log_bucket.blob('{0}/logs/{1}/severity-{2}'.format(userId,date,currentTime))
            log_blob.upload_from_filename('severity_{}.log'.format(msg_id))
            os.remove('./severity_{}.log'.format(msg_id))
            os.remove('./assets/severity_{}.jpeg'.format(msg_id))

            print("waiting for new message")

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
