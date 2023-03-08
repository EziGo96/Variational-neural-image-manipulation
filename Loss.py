'''
Created on 06-Mar-2023

@author: EZIGO
'''

import keras.backend as K
from keras.losses import mse, kullback_leibler_divergence

def vae_loss(y_true, y_pred):

    # KL Divergence Loss
    kl_loss = kullback_leibler_divergence(y_true, y_pred)
    # # Reconstruction loss
    mse_loss = mse(y_true, y_pred)
    # Combine the two losses
    loss =  (kl_loss + mse_loss) 
    return loss
