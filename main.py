from PIL import Image
import sys
import cv2
import numpy as np 
from matplotlib import pyplot as plt
import data_loading
import argparse
import os
from keras.preprocessing.image import ImageDataGenerator
import model_
from tqdm import tqdm
from keras.applications import ResNet50
from keras.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from sklearn.metrics import cohen_kappa_score, accuracy_score
import tensorflow as tf
import gc
import pandas as pd
import pickle
import json
import click
import tensorboard
import datetime



@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    pass


args = argparse.Namespace(
        checkpoint_dir="weights/best_weight.hdf5",
        history_dir="weights/history.json",
        dataset_dir="data",
        benign_dir="data/benign",
        malig_dir="data/malignant",
        train_sub_dir="train",
        valid_sub_dir="valid",
        test_sub_dir="test",
        gpus="0",
        batch_size=16,
        num_epochs=150,
        learning_rate=1e-6,
       
    )

@cli.command()
def train():
  #with tf.device(tf.DeviceSpec(device_type="GPU", device_index=args.gpus)):  
    X_train,Y_train=data_loading.load(sub_dir=args.train_sub_dir,args=args,resize=224)
    x_val,y_val=data_loading.load(sub_dir=args.valid_sub_dir,args=args,resize=224)
    train_generator = ImageDataGenerator(
        zoom_range=2,  # set range for random zoom
        rotation_range = 90,
        horizontal_flip=True,  # randomly flip images
        vertical_flip=True,  # randomly flip images
    )
    gc.collect()
    resnet = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(224,224,3)
)
    
    Model=model_.build_model(resnet,lr=args.learning_rate)

    #save best checkpoint    
    check="weights"
    os.mkdir(check)                          
    checkpoint_file=args.checkpoint_dir
    checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_file, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    
    #tensorboard log file
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)    

    
    history = Model.fit(
    train_generator.flow(X_train, Y_train, batch_size=args.batch_size),
    steps_per_epoch=X_train.shape[0] / args.batch_size,
    epochs=args.num_epochs,
    validation_data=(x_val, y_val),
    callbacks=[checkpoint,tensorboard_callback]
)
    #save history in to json file
    
    with open(args.history_dir,mode='w') as f:
        json.dump(str(history.history),f)
        hdf = pd.DataFrame(history.history)
        hdf[['accuracy', 'val_accuracy']].plot()
        plt.savefig("acc.jpg")
        plt.show()
        
        hdf1= pd.DataFrame(history.history)
        hdf1[['loss', 'val_loss']].plot()
        plt.savefig("loss.jpg")
        plt.show()
        

@cli.command()
def test():
  #with tf.device(tf.DeviceSpec(device_type="GPU", device_index=args.gpus)):  
    resnet = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(224,224,3)
)
    
    Model=model_.build_model(resnet,lr=args.learning_rate)
    Model.load_weights(args.checkpoint_dir)

    
    x_test,y_test,images=data_loading.load(sub_dir=args.test_sub_dir,args=args,resize=224,test=True)
    y_pred=Model.predict(x_test,batch_size=args.batch_size,
                                        steps = len(x_test)/args.batch_size)

    for i in range(len(images)):
        print("Image: ",images[i],", original value: ","malignant" if np.argmax(y_test[i])==1 else "benign", ", predicted result: ","malignant" if np.argmax(y_pred[i])==1 else "benign")
        
    print(f"accuracy: {accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))}")

        

@cli.command()
def preprocess():
    data_loading.distribute_data()
    data_loading.split_train_test(args)

if __name__ == "__main__":

    #os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    cli()
   
        
        
    
    

