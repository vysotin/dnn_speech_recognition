from keras import backend as K
from keras.models import Model
from keras.layers import (BatchNormalization, Conv1D, Conv2D, MaxPooling2D, Dense, Input, 
    TimeDistributed, Activation, Bidirectional, SimpleRNN, GRU, LSTM,
    Lambda, GlobalMaxPooling2D, Dropout)
import functools

def simple_rnn_model(input_dim, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(output_dim, return_sequences=True, 
                 implementation=2, name='rnn')(input_data)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(simp_rnn)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def rnn_model(input_dim, units, activation, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    gru_rnn = GRU(units, activation=activation,
        return_sequences=True, implementation=2, name='gru_rnn')(input_data)
    # TODO: Add batch normalization 
    bn_rnn = BatchNormalization(name='batch_norm')(gru_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model


def cnn_rnn_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, output_dim=29):
    """ Build a recurrent + convolutional network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add convolutional layer
    conv_1d = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    # Add batch normalization
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)
    # Add a recurrent layer
    simp_rnn = SimpleRNN(units, activation='relu',
        return_sequences=True, implementation=2, name='rnn')(bn_cnn)
    # TODO: Add batch normalization
    bn_rnn = BatchNormalization(name='bn_rnn')(simp_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model

def cnn_output_length(input_length, filter_size, border_mode, stride,
                       dilation=1):
    """ Compute the length of the output sequence after 1D convolution along
        time. Note that this function is in line with the function used in
        Convolution1D class from Keras.
    Params:
        input_length (int): Length of the input sequence.
        filter_size (int): Width of the convolution kernel.
        border_mode (str): Only support `same` or `valid`.
        stride (int): Stride size used in 1D convolution.
        dilation (int)
    """
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'valid':
        output_length = input_length - dilated_filter_size + 1
    return (output_length + stride - 1) // stride

def deep_rnn_model(input_dim, units, recur_layers, output_dim=29):
    """ Build a deep recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add recurrent layers, each with batch normalization
    rnn_layers = []
    bn_layers = []
    for n in range(recur_layers):
        current = input_data if n==0 else bn_layers[n-1]
        rnn_layers.append(GRU(units, activation='relu', return_sequences=True, implementation=2, name='rnn_'+str(n))(current))
        bn_layers.append(BatchNormalization(name='batch_norm_'+str(n))(rnn_layers[n]))
		
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_layers[recur_layers-1])
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def bidirectional_rnn_model(input_dim, units, output_dim=29):
    """ Build a bidirectional recurrent network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add bidirectional recurrent layer
    bidir_rnn = Bidirectional(GRU(units, activation='relu',
        return_sequences=True, implementation=2, name='rnn'))(input_data)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bidir_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def final_model():
    """ Build a deep network for speech 
    """
    
    input_dim = 161
    recur_layers = 2
    rnn_units = 120
    output_dim=29
    freq_kernel_size = 5
    time_kernel_size = 3
    time_stride = 1
    
    freq_pool_size = 5
    freq_pool_stride = 2
    time_dilations = [1,2,2]
    conv_filters=[20,40,80]
    
    add_dropout2 = False
    add_dropout1 = True
    add_dropout3 = True
    dropout1_rate = 0.2
    dropout2_rate = 0.2
    dropout3_rate = 0.2
    dense1_size = 120
    
    
    def compose(functions):
        return functools.reduce(lambda f, g: lambda x: f(g(x)), functions, lambda x: x)

    def cnn_output_length_spec(args):
        x, dilations, n = args
        return (cnn_output_length( x, time_kernel_size, border_mode='valid', stride=time_stride,
                       dilation=dilations[n]), dilations, n+1)
    def cnn_output_length_fin(x, time_dilations):
        funcs = [cnn_output_length_spec for n in time_dilations]
        return compose(funcs)((x,time_dilations,0))
    def expand_third_dim(input_data):
        return K.expand_dims(input_data,axis=3)
    def squeese_second_dim(input_data):
        return K.squeeze(input_data,axis=2)
    
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    
    reshape = Lambda(expand_third_dim, name='expand')(input_data)
    
    # Convolutional layers
    conv_layer = None
    pool_layer = reshape
    current = reshape
    num_convolutions = len(time_dilations)
    for n in range(num_convolutions):
        #current = reshape if n==0 else conv_layers[n-1]
        conv_layer =  Conv2D(conv_filters[n],(time_kernel_size,freq_kernel_size),strides=(time_stride,1),dilation_rate=(time_dilations[n], 1), 
                     activation='relu', data_format='channels_last',name='conv2d_'+str(n))(current)
        pool_layer = MaxPooling2D(pool_size=(1,freq_pool_size), strides=(1,freq_pool_stride), name='maxPool_'+str(n))(conv_layer)
        current = pool_layer
    
    #last_layer1 = pool_layer if num_convolutions > 0 else reshape
   
    last_layer_shape = K.int_shape(current)
    print(K.int_shape(current))
    frPool=MaxPooling2D(pool_size=(1,last_layer_shape[2]), strides=(1,1), name='frPool')(current)
    print(K.int_shape(frPool))
    reshape_back = Lambda(squeese_second_dim, name='squeeze')(frPool)
    after_cnn_layer = Dropout(dropout1_rate)(reshape_back) if add_dropout1 else reshape_back
    
    
    # Bidirectional GRU layers
    rnn_layer = None
    bn_layer = after_cnn_layer
    for n in range(recur_layers):
        #current = dropout1 if n==0 else bn_layers[n-1]
        rnn_layer = Bidirectional(GRU(rnn_units, activation='relu', return_sequences=True, implementation=2, name='brnn_'+str(n)))(bn_layer)
        bn_layer = BatchNormalization(name='batch_norm_'+str(n))(rnn_layer)
        
        
    after_rnn_layer = Dropout(dropout2_rate)(bn_layer) if  add_dropout2 else  bn_layer  
    print(K.int_shape(after_rnn_layer))
   
    time_dense1 = TimeDistributed(Dense(dense1_size))(after_rnn_layer)
    after_dense1 = Dropout(dropout3_rate)(time_dense1) if add_dropout3 else time_dense1
    time_dense2 = TimeDistributed(Dense(output_dim))(after_dense1)
    # TODO: Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense2)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    # TODO: Specify model.output_length
    model.output_length = lambda x: cnn_output_length_fin(x, time_dilations)[0]
    print(model.summary())
    return model
    
    
if __name__ == '__main__':
     model_end = final_model()