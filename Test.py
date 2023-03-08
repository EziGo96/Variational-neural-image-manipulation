'''
Created on 07-Mar-2023

@author: EZIGO
'''
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import model_from_json
from tensorflow import keras
import numpy as np
from sklearn import metrics
import cv2

print("Loading CVAE...")
json_file = open("CVAE_architecture.json", 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("CVAE_wts_.hdf5")

BS=1

def normalize(x):
    """
    Normalize a list of sample image data in the range of 0 to 1
    : x: List of image data.
    : return: Numpy array of normalized data
    """
    return np.array((x - np.min(x)) / (np.max(x) - np.min(x)))

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
    for i,j in zip(all_attr_vals,img_filenames):
        for _ in selected_attr_indices:
            if i[_]==-1.0:
                labels.append(i)
                img_filenames_sampled.append(j)
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

    return images, labels,selected_attr_indices

img_dir = 'Data/img_align_celeba'
attributes_path = 'Data/list_attr_celeba.txt'
selected_attrs_l = [
                    ["Smiling"],
                    ["Eyeglasses"],
                    ["Mustache"]  
                    ]
for selected_attrs in selected_attrs_l:
    images, labels,selected_attr_indices= load_celeba(img_dir, attributes_path, selected_attrs)
    id=1311
    print(images.shape, labels.shape)
    test_image =images[id]
    test_label=labels[id]
    print(selected_attr_indices)
    
    
    true_label=test_label
    print(true_label)
    plt.figure()
    plt.title(str(true_label))
    plt.imshow(test_image)
    
    '''model testing with true_label'''
    x_true = model.predict([np.array([test_image]),np.array([true_label])],batch_size=BS)
    
    plt.figure()
    plt.title("true_reconstruction")
    x_true = normalize(x_true[0])
    plt.imshow(x_true)
    plt.imsave(selected_attrs[0]+"_true_reconstruction.png",x_true)
    
    '''manipulating labels'''
    test_label[selected_attr_indices[0]]=1.0
    manipulated_test_label=test_label
    print(test_image.shape)
    print(manipulated_test_label)
    
    x_manipulated = model.predict([np.array([test_image]),np.array([manipulated_test_label])],batch_size=BS)
    
    plt.figure()
    plt.title("manipulated_reconstruction")
    x_manipulated=normalize(x_manipulated[0])
    plt.imshow(x_manipulated)
    plt.imsave(selected_attrs[0]+"_manipulated_reconstruction.png",x_manipulated)
    
    '''generative translation'''
    for delta in np.linspace(0,1,4,endpoint=True):
        plt.figure()
        plt.title("delta: "+str(delta))
        x_del=x_true*(1-delta)+x_manipulated*delta
        plt.imshow(x_del)
        plt.imsave(selected_attrs[0]+"_delta_"+str(delta)+".png",x_del)

# plt.show()