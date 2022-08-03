import numpy as np
import math
import os 
import random 
import torch 
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
#from prob_vehicle_zoo import *
#from prob_vehicle_rgf import *  #probp, fine_grained_binary_search, fine_grained_binary_search_local, probp_attack_untargeted, train_data_seperation
#import prob_vehicle_rgf as pr

import tensorflow as tf
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ReduceLROnPlateau as ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint as ModelCheckpoint
import keras.backend as K
from keras.layers import Input, Dense, LSTM, multiply, concatenate, Activation, Masking, Reshape
from keras.layers import Conv1D, BatchNormalization, GlobalAveragePooling1D, Permute, Dropout
from keras.models import Model

#from models.AE import create_rnn_ae
#from models.VAE import create_rnn_vae
#from models.model import create_classification_model

from utils.generic_utils import load_dataset_at
from utils.constants import MAX_NB_VARIABLES, NB_CLASSES_LIST, MAX_TIMESTEPS_LIST 
from utils.keras_utils import train_model, evaluate_model, set_trainable
from utils.layer_utils import AttentionLSTM

DATASET_INDEX = 48

MAX_NB_VARIABLES = MAX_NB_VARIABLES[DATASET_INDEX]
NB_CLASSES_LIST = NB_CLASSES_LIST[DATASET_INDEX]
MAX_TIMESTEPS_LIST = MAX_TIMESTEPS_LIST[DATASET_INDEX]

X_train, Y_train, X_test, Y_test, is_timeseries = load_dataset_at(DATASET_INDEX, fold_index = None, normalize_timeseries = True) 

#np.save('X_combine', np.vstack((X_train, X_test)))
#np.save('Y_combine', np.vstack((Y_train, Y_test)))

X_test = pad_sequences(X_test, maxlen = MAX_NB_VARIABLES, padding='post', truncating='post')

Y_train = to_categorical(Y_train, len(np.unique(Y_train)))
Y_test = to_categorical(Y_test, len(np.unique(Y_test)))


print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

init_decoder_input = np.zeros(shape=(X_train.shape[0], 1, X_train.shape[2])) #(batch, 1, length_of_sequence)

np.min(X_test), np.max(X_test)

def generate_MLSTM_model():
    ip = Input(shape=(MAX_NB_VARIABLES, MAX_TIMESTEPS_LIST))

    x = Masking()(ip)
    x = LSTM(8)(x)
    x = Dropout(0.8)(x)

    y = Permute((2, 1))(ip)
    y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = squeeze_excite_block(y)

    y = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = squeeze_excite_block(y)

    y = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = GlobalAveragePooling1D()(y)

    x = concatenate([x, y])

    out = Dense(NB_CLASSES_LIST, activation='softmax')(x)

    model = Model(ip, out)
    model.summary()

    # add load model code here to fine-tune

    return model

def generate_MLSTM_attention_model():
    ip = Input(shape=(MAX_NB_VARIABLES, MAX_TIMESTEPS_LIST))
    # stride = 10

    # x = Permute((2, 1))(ip)
    # x = Conv1D(MAX_NB_VARIABLES // stride, 8, strides=stride, padding='same', activation='relu', use_bias=False,
    #            kernel_initializer='he_uniform')(x)  # (None, variables / stride, timesteps)
    # x = Permute((2, 1))(x)

    #ip1 = K.reshape(ip,shape=(MAX_TIMESTEPS,MAX_NB_VARIABLES))
    #x = Permute((2, 1))(ip)
    x = Masking()(ip)
    x = AttentionLSTM(8)(x)
    x = Dropout(0.8)(x)

    y = Permute((2, 1))(ip)
    y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = squeeze_excite_block(y)

    y = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = squeeze_excite_block(y)

    y = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = GlobalAveragePooling1D()(y)

    x = concatenate([x, y])

    out = Dense(NB_CLASSES_LIST, activation='softmax')(x)

    model = Model(ip, out)
    model.summary()

    # add load model code here to fine-tune

    return model

def generate_FCN_model():
    ip = Input(shape=(MAX_NB_VARIABLES, MAX_TIMESTEPS_LIST))

    #x = Masking()(ip)
    #x = LSTM(8)(x)
    #x = Dropout(0.8)(x)

    y = Permute((2, 1))(ip)
    y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = GlobalAveragePooling1D()(y)

    #x = concatenate([x, y])

    out = Dense(NB_CLASSES_LIST, activation='softmax')(y)

    model = Model(ip, out)
    model.summary()

    # add load model code here to fine-tune

    return model
    
def squeeze_excite_block(input):
    ''' Create a squeeze-excite block
    Args:
        input: input tensor
        filters: number of output filters
        k: width factor

    Returns: a keras tensor
    '''
    filters = input._keras_shape[-1] # channel_axis = -1 for TF

    se = GlobalAveragePooling1D()(input)
    se = Reshape((1, filters))(se)
    se = Dense(filters // 16,  activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    se = multiply([input, se])
    return se

'''
#vae, encoder_model, decoder_model = create_rnn_vae(MAX_NB_VARIABLES, MAX_TIMESTEPS_LIST, None, 256, 128) 
base_model = generate_MLSTM_attention_model()

train_model(base_model, DATASET_INDEX, dataset_prefix='MLSTM_attention', epochs=200, batch_size=128, learning_rate = 1e-4, normalize_timeseries=True, monitor="val_accuracy", optimization_mode="max")

base_model.save_weights('./weights/generate_MLSTM_attention_model_combine.h5')
#base_model.save_weights('./weights/generate_MLSTM_attention_model.h5')

evaluate_model(base_model, DATASET_INDEX, dataset_prefix='MLSTM_attention', batch_size=128, normalize_timeseries=True)
'''

base_model_2 = generate_FCN_model()

train_model(base_model_2, DATASET_INDEX, dataset_prefix='FCN', epochs=80, batch_size=128, normalize_timeseries=True, monitor="val_accuracy", optimization_mode="max")

base_model_2.save_weights('./weights/generate_FCN_model.h5')

evaluate_model(base_model_2, DATASET_INDEX, dataset_prefix='FCN', batch_size=128, normalize_timeseries=True)

'''
model = generate_MLSTM_attention_model()

ckpt_path = './weights/generate_MLSTM_attention_model.h5'

model.load_weights(ckpt_path)
'''
'''
model_2 = generate_MLSTM_model()

ckpt_path_2 = './weights/generate_MLSTM_model.h5'

model_2.load_weights(ckpt_path_2)
'''



'''
train_dataset, signals, maneuvers = train_data_seperation(X_train, Y_train, prop = 0.25)
#perturbation_best = probp_attack_untargeted(model, train_dataset, signals, maneuvers, p = 0.1, alpha = 0.2, beta = 0.001, iterations = 200)
perturbation_best = probp_attack_untargeted(model, train_dataset, signals, maneuvers, p = 0.25, alpha = 0.2, beta = 0.001, iterations = 200)
np.save('./perturbations/perturbation_best_t1.npy', perturbation_best)
#np.save('./perturbations/perturbation_grad_11.npy', perturbation_grad)
'''
#print(np.linalg.norm(signals[0]))
#print(np.linalg.norm(signals[0][0]))
#print(signals[0].shape)
#print(signals[0][0].shape)
