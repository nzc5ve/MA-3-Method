from keras.layers.recurrent import SimpleRNN
from keras.layers import Dense, Dropout, Lambda, Input
import keras.backend as K
from keras.optimizers import Adam
from keras.models import Model

def create_rnn_ae(input_shape, latent_dims):
    timesteps = input_shape[0]
    encoded_seq_len = input_shape[1]

    ################################# AUTO ENCODER ###############################################
    #Encoder
    encoder_input = Input(shape=(None, encoded_seq_len))
    encoder_recurrent = SimpleRNN(latent_dims, return_sequences = False, return_state = True, name='Enc_Recurrent_layer')
    
    Y1, h = encoder_recurrent(encoder_input)
    encoder_model = Model(encoder_input, [Y1,h])

    #Decoder
    decoder_input = Input(shape=(1, encoded_seq_len))
    decoder_h_input = Input(shape=(latent_dims, ))

    # pay attention of the return_sequence and return_state flags.
    decoder_recurrent = SimpleRNN(latent_dims, activation='relu', return_sequences = True, return_state = True, name='Dec_Recurrent_layer')
    decoder_dense1 = Dense(40, activation='relu')
    decoder_dropout = Dropout(0.2)
    decoder_dense2 = Dense(encoded_seq_len, activation='linear')
        
    decoder_outputs = []
    X = decoder_input
    for i in range(input_shape[0]):
        X, h = decoder_recurrent(X, initial_state = h)
        Y = decoder_dense1(X)
        Y = decoder_dropout(X)
        Y = decoder_dense2(X)
        decoder_outputs.append(Y)
        X = Y

    Y = Lambda(lambda x: K.concatenate(x, axis=1))(decoder_outputs)
    
    autoencoder = Model(inputs=[encoder_input, decoder_input], output=Y)

    ################################# DECODER ###############################################
    h = decoder_h_input
    X = decoder_input
    decoder_outputs = []
    for i in range(input_shape[0]):
        X, h = decoder_recurrent(X, initial_state = h)
        Y = decoder_dense1(X)
        Y = decoder_dropout(X)
        Y = decoder_dense2(X)
        decoder_outputs.append(Y)
        X = Y

    Y = Lambda(lambda x: K.concatenate(x, axis=1))(decoder_outputs)

    decoder_model = Model(inputs=[decoder_input, decoder_h_input], outputs = Y)

    autoencoder.compile(optimizer=Adam(lr=1e-3), loss='mse')

    return encoder_model, decoder_model, autoencoder