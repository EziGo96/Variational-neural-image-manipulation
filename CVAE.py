'''
Created on 06-Mar-2023

@author: EZIGO
'''
from Encoder import Encoder
from Decoder import Decoder
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model

class CVAE:
    @staticmethod
    def build(input_dim,c_dim):
        (width, height, channels)=input_dim
        inputShape = (width, height, channels)
        chanDim = -1
        if K.image_data_format() == "channels_first":
            inputShape = (channels, height, width)
            chanDim = 1
        inputs=Input(shape=inputShape,name='inputs')
        c=Input(c_dim,name='raw attributes')
        attributes=Dense(inputShape[0]*inputShape[1],trainable=True)(c)
        attributes=Reshape((inputShape[0],inputShape[1]))(attributes)
        z=Encoder.build(inputShape,attributes.shape[1:])([inputs,attributes])
        x=Decoder.build(z.shape[1:],c.shape[1:])([z,c])
        model = Model(inputs=[inputs,c], outputs=x, name="CVAE")
        return model
    
# model = CVAE.build((64,64,3),40)
#
# model.summary()    
# plot_model(model,to_file="CVAE.png",show_shapes=True, show_dtype=True, show_layer_names=True, show_layer_activations=True)     

        
        
        