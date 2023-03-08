'''
Created on 06-Mar-2023

@author: EZIGO
'''
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Reshape
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model

class Decoder:
    @staticmethod
    def build(z_dim,c_dim):
        '''Input layer'''
        z=Input(z_dim,name='z')
        c=Input(c_dim,name='raw attributes')
        x=Concatenate()([z,c])
        '''FC layer'''
        x=Dense(512*2*2)(x)
        x=Reshape((2,2,512))(x)
        '''layer1'''
        # x=ZeroPadding2D(padding=(1, 1))(x)
        x=Conv2DTranspose(filters=256, kernel_size=(3,3), strides=(2,2), padding='valid')(x)
        x=LeakyReLU()(x)
        '''layer2'''
        x=ZeroPadding2D(padding=(1, 1))(x)
        x=Conv2DTranspose(filters=128, kernel_size=(3,3), strides=(2,2), padding='valid')(x)
        x=LeakyReLU()(x)
        '''layer3'''
        x=ZeroPadding2D(padding=(1, 1))(x)
        x=Conv2DTranspose(filters=64, kernel_size=(3,3), strides=(2,2), padding='valid')(x)
        x=LeakyReLU()(x)
        '''layer4'''
        x=ZeroPadding2D(padding=(1, 1))(x)
        x=Conv2DTranspose(filters=32, kernel_size=(3,3), strides=(2,2), padding='valid')(x)
        x=LeakyReLU()(x)
        '''layer5'''
        x=ZeroPadding2D(padding=(1, 1))(x)
        x=Conv2DTranspose(filters=32, kernel_size=(3,3), strides=(2,2), padding='valid')(x)
        x=LeakyReLU()(x)
        '''layer6'''
        x=ZeroPadding2D(padding=(1, 1))(x)
        x=Conv2D(filters=3, kernel_size=(3,3), padding='valid')(x)
        x=Activation("tanh")(x)
        x=tf.image.central_crop(x, 0.5)
        model = Model(inputs=[z,c], outputs=x, name="Decoder")
        return model
        
# model = Decoder.build(128,40)
#
# model.summary()    
# plot_model(model,to_file="Decoder.png",show_shapes=True, show_dtype=True, show_layer_names=True, show_layer_activations=True)     

