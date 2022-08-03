import numpy as np
import math
import os
import random 

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

from models.AE import create_rnn_ae
from models.VAE import create_rnn_vae
from models.model import create_classification_model

from utils.generic_utils import load_dataset_at
from utils.constants import MAX_NB_VARIABLES, NB_CLASSES_LIST, MAX_TIMESTEPS_LIST 
from utils.keras_utils import train_model
from utils.layer_utils import AttentionLSTM

DATASET_INDEX = 48

MAX_NB_VARIABLES = MAX_NB_VARIABLES[DATASET_INDEX]
NB_CLASSES_LIST = NB_CLASSES_LIST[DATASET_INDEX]
MAX_TIMESTEPS_LIST = MAX_TIMESTEPS_LIST[DATASET_INDEX]

X_train, Y_train, X_test, Y_test, is_timeseries = load_dataset_at(DATASET_INDEX, fold_index = None, normalize_timeseries = True) 


X_test = pad_sequences(X_test, maxlen = MAX_NB_VARIABLES, padding='post', truncating='post')

Y_train = to_categorical(Y_train, len(np.unique(Y_train)))
Y_test = to_categorical(Y_test, len(np.unique(Y_test)))


print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

init_decoder_input = np.zeros(shape=(X_train.shape[0], 1, X_train.shape[2])) #(batch, 1, length_of_sequence)

np.min(X_test), np.max(X_test)

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

vae, encoder_model, decoder_model = create_rnn_vae(MAX_NB_VARIABLES, MAX_TIMESTEPS_LIST, None, 256, 128) 
base_model = generate_MLSTM_attention_model()

#train_model(base_model, DATASET_INDEX, dataset_prefix='MLSTM_attention', epochs=40, batch_size=16, normalize_timeseries=True, monitor="val_accuracy", 
#            optimization_mode="max")

#base_model.save_weights('./weights/generate_MLSTM_attention_model.h5')

#train autoencoder
'''
history1=vae.fit(x=[X_train], y=[X_train], batch_size=32, epochs=40, 
                        validation_data=[X_train , X_train], 
                        callbacks=[ReduceLROnPlateau(patience=5), 
                        ModelCheckpoint('vae.h5',save_best_only=True,save_weights_only=True)])

decoder_model.save_weights('./weights/vae_decoder.h5')
'''
#MAKE graph .transpose(0,2,1)
sess_autoZoom = tf.InteractiveSession()

'''
    Construct tf-Graph
''' 
#latent_vector_shape = (MAX_TIMESTEPS_LIST,)
latent_vector_shape = (128,) #(128,)
X_shape = X_train.shape[1:]
Y_shape = Y_train.shape[1:]

k = 0.00
CONST_LAMBDA = tf.placeholder(tf.float32, name='lambda')
x0 = tf.placeholder(tf.float32, (None,) + X_shape, name='x0') #Input data
t0 = tf.placeholder(tf.float32, (None,) + Y_shape, name='t0') #Output

latent_adv = tf.placeholder(tf.float32, (None,) + latent_vector_shape, name='adv') #avdersarial example
init_dec_in = tf.placeholder(tf.float32, (None, 1, X_shape[1]), name ='Dec')

# compute loss
adv = decoder_model(latent_adv)
#adv = np.zeros((1,) + latent_vector_shape)
print(adv.shape)
print(x0.shape)

x = adv + x0
t = base_model(x)

Dist = tf.reduce_sum(tf.square(x - x0), axis=[1,2])

real = tf.reduce_sum(t0 * t, axis=1)
other = tf.reduce_max((1 - t0) * t - t0*10000, axis=1)

#untargeted attack    
Loss = CONST_LAMBDA * tf.maximum(tf.log(real + 1e-30) - tf.log(other + 1e-30), -k)

f = Dist + Loss
# # initialize variables and load target model
sess_autoZoom.run(tf.global_variables_initializer())

#weights are reset
base_model.load_weights('./weights/generate_MLSTM_attention_model.h5')
decoder_model.load_weights('./weights/vae_decoder.h5')

success_count = 0
success_with_se = 0
summary = {'init_l0': {}, 'init_l2': {}, 'l0': {}, 'l2': {}, 'adv': {}, 'query': {}, 'epoch': {}}
fail_count, invalid_count = 0, 0
S = 100
init_lambda = 10000

grad = np.zeros((1, latent_vector_shape[0]), dtype = np.float32)

########
X_len = 225
position = 0
cases = 1
a1=[]
a2=[]
a3=[]
a4=[]
a5=[]
a6=[]
a7=[]
a8=[]
a9=[]
a10=[]
a_list = [a1,a2,a3,a4,a5,a6,a7,a8,a9,a10]

while position <= 1830:
    for cases in range(10):
        cases += 1
        X_len = 225

        for i in range(cases+2):
            X = X_train[i+position:i+position+1]
            X_index = max(225-len(np.where(X[0,1] == X[0,1,-1])[0])-1, 4)
            if X_index < X_len:
                X_len = X_index

        move_f = []
        move_b = []
        for i in range(1,cases+1):
            a = X_train[i+position-1:i+position]
            X = X_train[i+position:i+position+1]
            b = X_train[i+position+1:i+position+2]
            a_index = max(225-len(np.where(a[0,1] == a[0,1,-1])[0])-1, 4)
            X_index = max(225-len(np.where(X[0,1] == X[0,1,-1])[0])-1, 4)
            b_index = max(225-len(np.where(b[0,1] == b[0,1,-1])[0])-1, 4)

            c = np.copy(X)
            for i_f in range(X_len+1):
                c = np.append(a[:,:,a_index-i_f:a_index-i_f+1],c,axis=-1)
                c = np.append(c[:,:,:X_index+1],c[:,:,X_index+2:],axis=-1)
                Y_temp = base_model.predict(c)
                if sum((max(Y_temp[0]) == Y_temp[0]) * Y_train[position+i]) == 0:
                    break
            move_f.append(i_f)

            c = np.copy(X)
            for j_b in range(X_len+1):
                c = np.append(c[:,:,1:X_index+1], b[:,:,j_b:j_b+1], axis=-1)
                c = np.append(c, X[:,:,X_index+1:], axis=-1)
                Y_temp = base_model.predict(c)
                if sum((max(Y_temp[0]) == Y_temp[0]) * Y_train[position+i]) == 0:
                    break
            move_b.append(j_b)
            
        move_f = min(move_f)
        move_b = min(move_b)
        
        a_list[cases-1].append((move_f-move_b)/2)
    position += 10
        
np.save('a.npy',a_list)
