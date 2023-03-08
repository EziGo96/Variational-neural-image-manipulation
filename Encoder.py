'''
Created on 06-Mar-2023

@author: EZIGO
'''
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l2
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from Reparametrization import reparametrization
class Encoder:
    @staticmethod
    def build(input_dim,attribute_dim, latent_dim=128):
        (width, height, channels)=input_dim
        inputShape = (width, height, channels)
        chanDim = -1
        if K.image_data_format() == "channels_first":
            inputShape = (channels, height, width)
            chanDim = 1
        '''Input layer'''           
        inputs=Input(shape=inputShape,name='inputs')
        attributes=Input(shape=attribute_dim,name='attributes')
        attributes=Reshape((64,64,1))(attributes)
        x=Concatenate(axis=chanDim)([inputs,attributes])
        '''layer1'''
        x=ZeroPadding2D(padding=(1, 1))(x)
        x=Conv2D(filters=32, kernel_size=(3,3), strides=(2,2), padding='valid')(x)
        x=LeakyReLU()(x)
        '''layer2'''
        x=ZeroPadding2D(padding=(1, 1))(x)
        x=Conv2D(filters=64, kernel_size=(3,3), strides=(2,2), padding='valid')(x)
        x=LeakyReLU()(x)
        '''layer3'''
        x=ZeroPadding2D(padding=(1, 1))(x)
        x=Conv2D(filters=128, kernel_size=(3,3), strides=(2,2), padding='valid')(x)
        x=LeakyReLU()(x)
        '''layer4'''
        x=ZeroPadding2D(padding=(1, 1))(x)
        x=Conv2D(filters=256, kernel_size=(3,3), strides=(2,2), padding='valid')(x)
        x=LeakyReLU()(x)
        '''layer5'''
        x=ZeroPadding2D(padding=(1, 1))(x)
        x=Conv2D(filters=512, kernel_size=(3,3), strides=(2,2), padding='valid')(x)
        x=LeakyReLU()(x)
        '''FC layers'''
        x_vector = Flatten()(x)
        mu=Dense(latent_dim, name='mu')(x_vector)
        sigma=Dense(latent_dim, name='sigma')(x_vector)
        
        z=reparametrization(mu,sigma)
        model = Model(inputs=[inputs,attributes], outputs=z, name="Encoder")
        return model
    
# model = Encoder.build((64, 64, 3),(64, 64))
#
# model.summary()    
# plot_model(model,to_file="Encoder.png",show_shapes=True, show_dtype=True, show_layer_names=True, show_layer_activations=True)      

