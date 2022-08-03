from keras.layers import Input, Dense,  multiply, concatenate, Activation, Masking, Reshape, Lambda, Concatenate
from keras.layers import Conv1D, BatchNormalization, GlobalAveragePooling1D, Permute, Dropout, Reshape
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam
from keras.models import Model
from keras import regularizers

def create_classification_model(input_shape, output_shape, weights = None):
    #Squeeze excitation block
    def squeeze_excite_block(inputs):
        filters = inputs._keras_shape[-1] # channel_axis = -1 for TF
        se = GlobalAveragePooling1D()(inputs)
        se = Reshape((1, filters))(se)
        se = Dense(filters // 16,  activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
        se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
        se = multiply([inputs, se])
        return se


    ip = Input(shape=input_shape)

    x = Masking()(ip)
    x = LSTM(8)(x)
    x = Dropout(0.2)(x)

    y = Permute((2, 1))(ip)
    y = Conv1D(64, 5, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = squeeze_excite_block(y)

    y = Conv1D(128, 4, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = squeeze_excite_block(y)

    y = Conv1D(64, 3, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = GlobalAveragePooling1D()(y)

    x = concatenate([x, y])

    out = Dense(output_shape, activation='softmax')(x)

    model = Model(inputs=ip, outputs=out)
    
    model.compile(optimizer=Adam(lr=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
    
    if(not weights == None):
        model.load_weights(weights)

    return model