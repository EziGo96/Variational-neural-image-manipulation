'''
Created on 07-Mar-2023

@author: EZIGO
'''
import tensorflow as tf

def reparametrization(mu,sigma):
    std=tf.math.exp(0.5*sigma)
    # print(std.shape)
    eps=tf.random.normal(tf.shape(std))
    z=mu+eps*sigma
    return z