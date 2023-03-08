'''
Created on 06-Mar-2023

@author: EZIGO
'''
import tensorflow as tf
from CVAE import CVAE
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.utils import plot_model
import pandas as pd
from tensorflow.keras.models import save_model
import cv2
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
from Loss import vae_loss

def load_celeba(img_dir, attributes_path, selected_attrs,data_limit=12000):
    with open(attributes_path, 'r') as f:
        lines = f.readlines()
        all_attr_names = lines[1].split()
        all_attr_vals = np.array([line.split() for line in lines[2:]])
        all_attr_vals=[list( map(float,i) ) for i in all_attr_vals[:,1:]]

    # Find indices of selected attributes
    selected_attr_indices = [all_attr_names.index(attr) for attr in selected_attrs]
    # Load images and corresponding labels
    img_filenames = np.array([img_dir + '/' + line.split()[0] for line in lines[2:]])
    img_filenames_sampled = []
    labels = []
    ctr=0
    for i,j in zip(all_attr_vals,img_filenames):
        for _ in selected_attr_indices:
            if i[_]==1.0:
                ctr+=1
        if ctr == len(selected_attr_indices):
            labels.append(i)
            img_filenames_sampled.append(j)
        ctr=0
    labels=np.array(labels[:data_limit])
    img_filenames_sampled=np.array(img_filenames_sampled[:data_limit])
    # Load and preprocess images
    images = []
    for filename in img_filenames_sampled:
        img = cv2.imread(filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (64, 64))
        images.append(img)
    images = np.array(images, dtype='float32') / 255.0

    return images, labels

img_dir = 'Data/img_align_celeba'
attributes_path = 'Data/list_attr_celeba.txt'
selected_attrs = []  # Example of two selected attributes
images, labels = load_celeba(img_dir, attributes_path, selected_attrs)

target_images = []
for i in images:
    target_images.append(cv2.resize(i,(79,79),interpolation = cv2.INTER_AREA))
target_images=np.array(target_images)

INIT_LR =0.001
BS= 100
NUM_EPOCHS =25

# def polynomial_decay(epoch):
#     maxEpochs = NUM_EPOCHS
#     baseLR = INIT_LR
#     power = 1
#     alpha = baseLR * (1 - (epoch / float(maxEpochs))) ** power
#     return alpha
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1,patience=2, min_lr=0.00001)
mcp_save = ModelCheckpoint("CVAE_wts_.hdf5", save_best_only=True, monitor='loss', mode='min')
callbacks = [mcp_save,reduce_lr]

# img_iter = datagen.flow(X_train, y_train, batch_size=BS)
# datagen.fit(X_train)

model = CVAE.build((64,64,3),40)

model.summary()    
plot_model(model,to_file="CVAE.png",show_shapes=True, show_dtype=True, show_layer_names=True, show_layer_activations=True)
opt = Adam(lr=INIT_LR)
model.compile(loss =[tf.keras.losses.MeanSquaredError(),tf.keras.losses.KLDivergence()],loss_weights=[0.5,0.5], optimizer=opt, metrics=['accuracy'])  

H = model.fit([images,labels],
              target_images,
              epochs=NUM_EPOCHS,
              steps_per_epoch=len(images)/BS, # Run same number of steps we would if we were not using a generator.
              callbacks=callbacks)

df=pd.DataFrame()
df.from_dict(H.history).to_csv("Training.csv",index=False)
model_json = model.to_json(indent=3)
with open("CVAE_architecture.json", "w") as json_file:
    json_file.write(model_json)
save_model(model, "CVAE.hp5", save_format="h5")