from PIL import Image
import sys
import cv2
import numpy as np 
import os
from matplotlib import pyplot as plt
import shutil
from tqdm import tqdm
from pathlib import Path
import random
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator

def scan_folder(parent,target):
    # iterate over all the files in directory 'parent'
    current_path=parent
    for file_name in os.listdir(parent):
        if file_name.endswith(".png"):
            shutil.move(os.path.join(current_path,file_name),target)
        else:
            current_path = "".join((parent, "/", file_name))
            if os.path.isdir(current_path):
                # if we're checking a sub-directory, recursively call this method
                scan_folder(current_path,target)

def distribute_data():
    path="BreaKHis_v1/histology_slides/breast"
    ben="benign/SOB"
    mal="malignant/SOB"
    data="data"
    os.mkdir(data)
    benign="data/benign"
    malignant="data/malignant"
    os.mkdir(benign)
    scan_folder(os.path.join(path,ben),benign)
    os.mkdir(malignant)
    scan_folder(os.path.join(path,mal),malignant)

RNG = np.random.RandomState(4321)

def split_train_test(args):
    os.mkdir(args.dataset_dir+"/"+args.train_sub_dir)
    os.mkdir(args.dataset_dir+"/"+args.valid_sub_dir)
    os.mkdir(args.dataset_dir+"/"+args.test_sub_dir)
    
    for i in random.sample(os.listdir(args.benign_dir),int(len(os.listdir(args.benign_dir))*.7)):
        shutil.move(os.path.join(args.benign_dir,i),os.path.join(args.dataset_dir,args.train_sub_dir))
    for i in random.sample(os.listdir(args.malig_dir),int(len(os.listdir(args.malig_dir))*.7)):
        shutil.move(os.path.join(args.malig_dir,i),os.path.join(args.dataset_dir,args.train_sub_dir))
    for i in random.sample(os.listdir(args.benign_dir),int(len(os.listdir(args.benign_dir))*(2/3))):
        shutil.move(os.path.join(args.benign_dir,i),os.path.join(args.dataset_dir,args.valid_sub_dir))    
    for i in random.sample(os.listdir(args.malig_dir),int(len(os.listdir(args.malig_dir))*(2/3))):
        shutil.move(os.path.join(args.malig_dir,i),os.path.join(args.dataset_dir,args.valid_sub_dir))
    for i in random.sample(os.listdir(args.benign_dir),int(len(os.listdir(args.benign_dir)))):
        shutil.move(os.path.join(args.benign_dir,i),os.path.join(args.dataset_dir,args.test_sub_dir))    
    for i in random.sample(os.listdir(args.malig_dir),int(len(os.listdir(args.malig_dir)))):
        shutil.move(os.path.join(args.malig_dir,i),os.path.join(args.dataset_dir,args.test_sub_dir))    


    
        
def load(sub_dir, args, rng=RNG, resize=None,test=False):
        
        IMG = []
        Y=[]
        images=[]
        read = lambda imname: np.asarray(Image.open(imname).convert("RGB"))
        for IMAGE_NAME in tqdm(os.listdir(args.dataset_dir+"/"+sub_dir)):
                PATH = os.path.join(args.dataset_dir+"/"+sub_dir,IMAGE_NAME)
                _, ftype = os.path.splitext(PATH)
                if ftype == ".png":
                    img = read(PATH)
                    img = cv2.resize(img, (resize,resize))
                    IMG.append(np.array(img))
                    e=IMAGE_NAME.split("_")[1]
                    if e=="B":
                        Y.append(0)
                    elif e=="M":
                        Y.append(1)
                    images.append(IMAGE_NAME) 
        Y = to_categorical(Y, num_classes= 2)
        if test==True:
            return np.array(IMG),np.array(Y),images
        else:
            return np.array(IMG),np.array(Y)   

     

