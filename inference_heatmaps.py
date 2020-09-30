"""
The Original Inference.py file belongs to COVID-Net Repository under GNU Affero General Public License v3.0
and is available here: https://github.com/lindawangg/COVID-Net/blob/master/inference.py
Neura Health made modifications to inference.py and added heatmap feature too to run with neuracovid application.
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
import cv2
from tensorflow.python.framework import ops

from data import process_image_file

import pandas as pd


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
        blob.download_to_filename('assets/{}.jpeg'.format(msg_id)) # tagging image with unique message id

        dirr = 'assets/{}'.format(msg_id)
        os.mkdir(dirr)
        my_logger.info('Processing model now')

        try:
            parser = argparse.ArgumentParser(description='COVID-Net Inference')
            parser.add_argument('--model', default="COVID-Net-Model.json", help="Path to model specification")
            parser.add_argument('--weightspath', default='models/COVIDNet-CXR-Large', type=str, help='Path to output folder')
            parser.add_argument('--metaname', default='model.meta', type=str, help='Name of ckpt meta file')
            parser.add_argument('--ckptname', default='model-8485', type=str, help='Name of model ckpts')
            parser.add_argument('--imagepath', default='assets/{}.jpeg'.format(msg_id), type=str, help='Full path to image to be inferenced')
            parser.add_argument('--in_tensorname', default='input_1:0', type=str, help='Name of input tensor to graph')
            parser.add_argument('--out_tensorname', default='dense_3/Softmax:0', type=str, help='Name of output tensor from graph')
            parser.add_argument('--input_size', default=224, type=int, help='Size of input (ex: if 480x480, --input_size 480)')
            parser.add_argument('--top_percent', default=0.08, type=float, help='Percent top crop from top of image')
            parser.add_argument('--outdir', default='assets/{}'.format(msg_id) , help="Output directory")
            parser.add_argument('--image_output_size', default=650, type=int, help='output heatmap image Size 650x650')

            args = parser.parse_args()
            model_info = json.load(open(args.model))
            mapping = {'normal': 0, 'pneumonia': 1, 'COVID-19': 2}
            inv_mapping = {0: 'normal', 1: 'pneumonia', 2: 'COVID-19'}

            sess = tf.compat.v1.Session()
            tf.compat.v1.get_default_graph()
            saver = tf.compat.v1.train.import_meta_graph(os.path.join(args.weightspath, args.metaname))
            saver.restore(sess, os.path.join(args.weightspath, args.ckptname))

            graph = tf.compat.v1.get_default_graph()

            image_tensor = graph.get_tensor_by_name(args.in_tensorname)
            pred_tensor = graph.get_tensor_by_name(args.out_tensorname)

            x = process_image_file(args.imagepath, args.top_percent, args.input_size)
            x = x.astype('float32') / 255.0
            pred = sess.run(pred_tensor, feed_dict={image_tensor: np.expand_dims(x, axis=0)})

            print('Prediction: {}'.format(inv_mapping[pred.argmax(axis=1)[0]]))
            print('Confidence')
            print('Normal: {:.3f}, Pneumonia: {:.3f}, COVID-19: {:.3f}'.format(pred[0][0], pred[0][1], pred[0][2]))
            print('**DISCLAIMER**')
            print('Do not use this prediction for self-diagnosis. You should check with your local authorities for the latest advice on seeking medical assistance.')

            N = pred[0][0] # Normal
            P = pred[0][1] # Pneumonia
            C = pred[0][2] # COVID-19

            result = ('{}'.format(inv_mapping[pred.argmax(axis=1)[0]]))

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
            heatmap_path = path+"heatmap.png"
            heatmap_path = {'heatmap_path':heatmap_path}
            my_logger.info('finish successfully and message acknowledge')
            my_logger.info('Detection : {}'.format(detection))

        except BaseException as error:
            my_logger.error('{}'.format(error))
            my_logger.info('error occurred need to reprocess message NOT acknowledge')

        #heatmap Genegration
        try:
            classes = [0,1,2]
            targetLayer=model_info["final_conv_tensor"]
            outLayer=args.out_tensorname
            if targetLayer is None:
                tensor_names = [t.name for op in tf.get_default_graph().get_operations() for t in op.values() if
                           "save" not in str(t.name)]
                for tensor_name in reversed(tensor_names):
                    tensor = graph.get_tensor_by_name(tensor_name)
                    if len(tensor.shape) == 4:
                        target = tensor
            else:
                target = graph.get_tensor_by_name(targetLayer)
            results = {} # grads of classes with keys being classes and values being normalized gradients
            for classIdx in classes:
                one_hot = tf.sparse_to_dense(classIdx, [len(classes)], 1.0)
                signal = tf.multiply(graph.get_tensor_by_name(outLayer),one_hot)
                loss = tf.reduce_mean(signal)

                grads = tf.gradients(loss, target)[0]

                norm_grads = tf.div(grads, tf.sqrt(tf.reduce_mean(tf.square(grads)))+tf.constant(1e-5))

                results[classIdx] = norm_grads

            grads = results
            origin_im =process_image_file(args.imagepath, args.top_percent, args.image_output_size)

            size_upsample = (origin_im.shape[1],origin_im.shape[0]) # (w, h)
            output, grads_val = sess.run([target, grads[mapping[result]]], feed_dict={image_tensor: np.expand_dims(x, axis=0)})

            conv_layer_out = output[0]
            grads_val = grads_val[0]
            upsample_size = size_upsample

            weights = np.mean(grads_val, axis=(0,1))
            cam = np.zeros(conv_layer_out.shape[0:2], dtype=np.float32)
            for i, w in enumerate(weights):
                cam += w*conv_layer_out[:,:,i]
            cam = np.maximum(cam, 0)
            cam = cam/np.max(cam)
            cam = cv2.resize(cam, upsample_size)

            cam3 = np.expand_dims(cam, axis=2)
            cam3 = np.tile(cam3,[1,1,3])

            # Overlay cam on image
            cam3 = np.uint8(255*cam3)
            cam3 = cv2.applyColorMap(cam3, cv2.COLORMAP_JET)

            new_im = cam3*0.3 + origin_im*0.5

            im_name = args.imagepath.split("/")[-1]
            ext = im_name.split(".")[-1]

            # Save the GradCAM
            cv2.imwrite(os.path.join(args.outdir, 'heatmap.png'), new_im)
            print("GradCAM image is save in ", args.outdir)
            ops.reset_default_graph()
            my_logger.info('Heatmaps saved : {}'.format(args.outdir))

            message.ack()

        except Exception as e :
            my_logger.error('heatmaps : not saved',e)

        credential_json_file = ['file_name']
        databaseURL = ['databaseURL']
        storageBucket = ['storageBucket']
        try:
            if (not len(firebase_admin._apps)):
                cred = credentials.Certificate(credential_json_file)
                fa=firebase_admin.initialize_app(cred, {"databaseURL": databaseURL,'storageBucket':storageBucket})
                fc=firebase_admin.firestore.client(fa)
                db = firestore.client()
                blob = storage.bucket(storageBucket).blob('{}heatmap.png'.format(path))
                blob.upload_from_filename('assets/{0}/heatmap.png'.format(msg_id))
                my_logger.error('Heatmaps : Saved to Storage')
                doc_ref = db.collection(u'stripe_customers/{0}/results'.format(userId)).document(currentTime)
                doc_ref.update(detection)
                doc_ref.update(heatmap_path)
                my_logger.error('Detection : Saved to firestore')
            else:
                # print('alredy initialize')
                db = firestore.client()
                blob = storage.bucket(storageBucket).blob('{}heatmap.png'.format(path))
                blob.upload_from_filename('assets/{0}/heatmap.png'.format(msg_id))
                my_logger.error('Heatmaps : Saved to Storage')
                doc_ref = db.collection(u'stripe_customers/{0}/results'.format(userId)).document(currentTime)
                doc_ref.update(detection)
                doc_ref.update(heatmap_path)
                my_logger.error('Detection : Saved to firestore')

        except Exception as e :
            my_logger.error('Detection : NOT Saved',e)

        storage_client = store.Client.from_service_account_json(credential_json_file)
        log_bucket = storage_client.get_bucket('logs_bucket')
        log_blob = log_bucket.blob('{0}/logs/{1}/{2}'.format(userId,date,currentTime))
        log_blob.upload_from_filename('{}.log'.format(msg_id))
        #print(msg_id)
        os.remove('./{}.log'.format(msg_id))
        os.remove('./assets/{}.jpeg'.format(msg_id))
        shutil.rmtree('./assets/{}'.format(msg_id))
        print("end of pubsubcallback")

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
