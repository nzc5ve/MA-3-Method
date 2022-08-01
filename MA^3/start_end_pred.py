import numpy as np
import math
import os
import random
import time 

import tensorflow as tf

#tf.disable_eager_execution()
#tf.disable_v2_behavior()

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

vae, encoder_model, decoder_model = create_rnn_vae(MAX_NB_VARIABLES, MAX_TIMESTEPS_LIST, None, 450, 225) 
base_model = generate_MLSTM_model()

#train_model(base_model, DATASET_INDEX, dataset_prefix='MLSTM_attention', epochs=40, batch_size=16, normalize_timeseries=True, monitor="val_accuracy", 
#            optimization_mode="max")

#base_model.save_weights('./weights/generate_MLSTM_attention_model.h5')

#train autoencoder
'''
history1=vae.fit(x=[X_train], y=[X_train], batch_size=32, epochs=40, 
                        validation_data=[X_train , X_train], 
                        callbacks=[ReduceLROnPlateau(patience=5), 
                        ModelCheckpoint('vae.h5',save_best_only=True,save_weights_only=True)])

decoder_model.save_weights('./weights/vae_decoder_450.h5')
###decoder_model.save_weights('./weights/vae_decoder.h5')
'''
#MAKE graph .transpose(0,2,1)
sess_autoZoom = tf.InteractiveSession() #tf.InteractiveSession()

'''
    Construct tf-Graph
''' 
#latent_vector_shape = (MAX_TIMESTEPS_LIST,)
latent_vector_shape = (225,)
X_shape = X_train.shape[1:]
Y_shape = Y_train.shape[1:]

k = 0.00
#tf.disable_eager_execution()
CONST_LAMBDA = tf.placeholder(tf.float32, name='lambda') #tf.placeholder(tf.float32, name='lambda')
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
base_model.load_weights('./weights/generate_MLSTM1_model.h5')
decoder_model.load_weights('./weights/vae_decoder_450.h5')
#decoder_model.load_weights('./weights/vae_decoder.h5')

success_count = 0
success_with_se = 0
summary = {'init_l0': {}, 'init_l2': {}, 'l0': {}, 'l2': {}, 'adv': {}, 'query': {}, 'epoch': {}}
fail_count, invalid_count = 0, 0
S = 100
init_lambda = 10000

grad = np.zeros((2, latent_vector_shape[0]), dtype = np.float32)

########
time_off_start = time.time()
X_len = 225
position = 10
cases = 10
for i in range(cases+2):
    X = X_train[i+position:i+position+1]
    X_index = max(225-len(np.where(X[0,1] == X[0,1,-1])[0])-1, 4)
    if X_index < X_len:
        X_len = X_index

print('X_len is', X_len)
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

print('find the front move',min(move_f))
print('find the back move',min(move_b))
move_f = min(move_f)
move_b = min(move_b)

time_off_end = time.time()
print("time offline in finding start point is", time_off_end-time_off_start)
########
time_total = 0
size = 0
#Iterate for each test example
for i in range(1,X_train.shape[0]-1): #(X_train.shape[0]):
    time_start = time.time()
    print("\n start attacking target", i, "...")
       
    mt = 0           # accumulator m_t in Adam
    vt = 0           # accumulator v_t in Adam

    beta1 = 0.9            # parameter beta_1 in Adam
    beta2 = 0.999          # parameter beta_2 in Adam
    learning_rate = 1e-1 #2e-3          # learning rate in Adam
    
    batch_size = 1                # batch size
    Max_Query_count = 2000        # maximum number of queries allowed

    best_l2 = np.math.inf

    #For the time being it has the same shape as X
    init_adv = np.zeros((1,) + latent_vector_shape)           # initial adversarial perturbationi

    X = np.expand_dims(X_train[i], 0)           # target sample X
    Y = np.expand_dims(Y_train[i], 0)           # target sample's lable Y
    
    # check if (X, Y) is a valid target, y checking if it is classified correctly
    Y_pred = base_model.predict(X)
    if sum((max(Y_pred[0]) == Y_pred[0]) * Y[0]) == 0:
        #print("not a valid target.")
        invalid_count += 1
        continue

########
    X_true = np.copy(X)
    Y_true = np.copy(Y)
    a = X_train[i-1:i]
    b = X_train[i+1:i+2]
    a_index = max(225-len(np.where(a[0,1] == a[0,1,-1])[0])-1, 4)
    X_index = max(225-len(np.where(X[0,1] == X[0,1,-1])[0])-1, 4)
    b_index = max(225-len(np.where(b[0,1] == b[0,1,-1])[0])-1, 4)
    if move_f >= move_b:
        for i_f in range(random.randint(int((move_f-move_b)/2+0.1), round((move_f-move_b)/2+0.1))):
            X = np.append(a[:,:,a_index-i_f:a_index-i_f+1],X,axis=-1)
            X = np.append(X[:,:,:X_index+1],X[:,:,X_index+2:],axis=-1)
    else:
        for j_b in range(random.randint(int((move_b-move_f)/2+0.1), round((move_b-move_f)/2+0.1))):
            X = np.append(X[:,:,1:X_index+1], b[:,:,j_b:j_b+1], axis=-1)
            X = np.append(X, X_true[:,:,X_index+1:], axis=-1)

    Y = base_model.predict(X)
    Y = np.array([(max(Y[0])==Y[0])*np.array([1]*len(Y[0]))])
    #print('is X consistent?',X==X_true)
    #print('is Y consistent?',Y==Y_true)
    var_true = np.zeros((1,225))
########
    
    var_size = init_adv.size
    beta = 1/(var_size)

    query, epoch = 0, 0
    q = 1 
    b = q
    # main loop for the optimization
    while(query < Max_Query_count):
        epoch += 1
        #if initial attack is found fine tune the adversarial example buy increasing the q
        if(not np.math.isinf(best_l2)):
            q = 3 
            b = q
            grad = np.zeros((q, var_size), dtype = np.float32)

        query += q #q queries will be made in this iteration
        
        #Using random vector gradient estimation 

        #random noise
        u = np.random.normal(loc=0, scale=1000, size = (q, var_size))
        u_mean = np.mean(u, axis=1, keepdims=True)
        u_std = np.std(u, axis=1, keepdims=True)
        u_norm = np.apply_along_axis(np.linalg.norm, 1, u, keepdims=True)
        u = u/u_norm

        #For estimation of F(x + beta*u) and F(x)
        var = np.concatenate((init_adv, init_adv + beta * u.reshape((q,)+ (latent_vector_shape))), axis=0)
        
        l0_loss, l2_loss, losses, scores = sess_autoZoom.run([Loss, Dist, f, t], feed_dict={latent_adv: var, x0: X, t0: Y, 
                                                                            CONST_LAMBDA: init_lambda}) 

        #Gradient estimation
        for j in range(q):
            if len(losses) > 1:
                grad[j] = (b * (losses[j + 1] - losses[0])* u[j]) / beta
            else:
                grad[j] = (b * (losses[0])* u[j]) / beta
            
        avg_grad = np.mean(grad, axis=0)

        # ADAM update
        mt = beta1 * mt + (1 - beta1) * avg_grad
        vt = beta2 * vt + (1 - beta2) * (avg_grad * avg_grad)
        corr = (np.sqrt(1 - np.power(beta2, epoch))) / (1 - np.power(beta1, epoch))

        m = init_adv.reshape(-1)
        m -= learning_rate * corr * mt / (np.sqrt(vt) + 1e-8)

        #update the adversarial example
        init_adv = m.reshape(init_adv.shape)
        
        l2_loss = l2_loss[0]
        l0_loss = l0_loss[0]
        
        if(epoch%S == 0 and not np.math.isinf(best_l2)):
            init_lambda /= 2

        if(sum((scores[0] == max(scores[0]))*Y[0])==0 and l2_loss < best_l2):
           
            if(np.math.isinf(best_l2)):
                #print("Initial attack found on query {query} and l2 loss of {l2_loss}")
                summary['query'][i] = query
                summary['epoch'][i] = epoch
                summary['init_l0'][i] = l0_loss
                summary['init_l2'][i] = l2_loss

            best_l2 = l2_loss
            summary['l0'][i] = l0_loss
            summary['l2'][i] = l2_loss
            summary['adv'][i] = init_adv

            var_true = np.copy(var)
            '''
########
            success_count += 1

            l0_loss, l2_loss, losses, scores = sess_autoZoom.run([Loss, Dist, f, t], feed_dict={latent_adv: var, x0: X_true, t0: Y_true,
                                                                            CONST_LAMBDA: init_lambda})
            if(sum((scores[0] == max(scores[0]))*Y_true[0])==0):
                success_with_se += 1
                print('success with se is', success_with_se)
                break 
########
            '''
        if(query >= Max_Query_count and not np.math.isinf(best_l2)):
            success_count += 1
            
            ########
            #var = np.append(var[:,:128-int((i_f+1)*128/X_index)], np.array([[0]*int((i_f+1)*128/X_index)]*var.shape[0]),axis=-1)
            l0_loss, l2_loss, losses, scores = sess_autoZoom.run([Loss, Dist, f, t], feed_dict={latent_adv: var_true, x0: X_true, t0: Y_true,
                                                                            CONST_LAMBDA: init_lambda})
            if(sum((scores[0] == max(scores[0]))*Y_true[0])==0):
                success_with_se += 1
            print('success with se is', success_with_se)
            time_end = time.time()
            time_total += time_off_end + time_end - time_off_start - time_start
            size += np.linalg.norm(var_true)/np.linalg.norm(X_true)
            ########
            
        elif (query >= Max_Query_count and np.math.isinf(best_l2)):
            #print("attack failed!")
            fail_count += 1
            
            ########
            #var = np.append(var[:,:128-int((i_f+1)*128/X_index)], np.array([[0]*int((i_f+1)*128/X_index)]*var.shape[0]),axis=-1)
            l0_loss, l2_loss, losses, scores = sess_autoZoom.run([Loss, Dist, f, t], feed_dict={latent_adv: var_true, x0: X_true, t0: Y_true,
                                                                            CONST_LAMBDA: init_lambda})
            if(sum((scores[0] == max(scores[0]))*Y_true[0])==0):
                success_with_se += 1
            print('success with se is', success_with_se)
            time_end = time.time()
            time_total += time_off_end + time_end - time_off_start - time_start
            size += np.linalg.norm(var_true)/np.linalg.norm(X_true)
            ########
            
            break
'''
print(invalid_count, fail_count)
print(len(summary['adv'].keys()) /(len(summary['adv'].keys())+ fail_count))
print(np.average(list(summary['query'].values())))
print(np.average(list(summary['epoch'].values())))
print(np.average(list(summary['init_l2'].values())))
print(np.average(list(summary['init_l0'].values())))
print(np.average(list(summary['l2'].values())))
print(np.average(list(summary['l0'].values())))
'''

print("Invalid Count", invalid_count, "Fail Count", fail_count)
print("Success Rate:", (1 - ((fail_count+invalid_count)/X_train.shape[0])))
print("Success Rate With Start-End Effect", (success_with_se/(X_train.shape[0]-2-invalid_count)))
print("Time required", time_total/success_with_se)
print("Perturbation amount", size/success_with_se)
print('find the front move',move_f)
print('find the back move',move_b)
print(len(summary['adv'].keys()) /(len(summary['adv'].keys())+ fail_count))
print("Query Average", np.average(list(summary['query'].values())))
print("Iter Average", np.average(list(summary['epoch'].values())))
print("L2 Average:", np.average(list(summary['init_l2'].values())))
print("L0 Average:", np.average(list(summary['init_l0'].values())))
print("L2 Ratio Average:", np.average(list(summary['l2'].values())))


