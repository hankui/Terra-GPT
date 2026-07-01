
# Dec 29, 2020
# model builder
# import transformer_encoder44

# refer to
# https://machinelearningmastery.com/tensorflow-tutorial-deep-learning-with-tf-keras/
# https://www.tensorflow.org/tutorials/customization/custom_training_walkthrough
# https://www.tensorflow.org/tutorials/quickstart/advanced

import math
import numpy as np
import logging
import pandas as pd

import tensorflow as tf
from keras import backend as K

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Concatenate
from tensorflow.keras.layers import BatchNormalization,LayerNormalization
from tensorflow.keras.layers import LSTM,GRU,Bidirectional

from tensorflow.keras.layers import Dropout,Softmax
from tensorflow.keras.layers import LayerNormalization, MaxPooling1D, AveragePooling1D,Conv1D
from tensorflow.keras.layers import Masking,Embedding
from tensorflow.keras.layers import SimpleRNN, Attention, AdditiveAttention, TimeDistributed, MultiHeadAttention
from tensorflow.keras import Input, Model
# import keras_layers.layers as customized 

# N_times = 14
# N_feature = 14
# N_outputs = 8


##************************************************************************************************************
## point_wise_feed_forward_network
def point_wise_feed_forward_network(d_model, dff, reg=None, name=None):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu', kernel_regularizer=reg),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model, kernel_regularizer=reg)  # (batch_size, seq_len, d_model)
    ], name=name)

##************************************************************************************************************
## check test_mask.py to see why this is like adding two new axises
def create_padding_mask(inputs, mask_value=0):
    seq = tf.cast(tf.math.not_equal(inputs[:, :, 0], mask_value), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

## check test_mask.py to see why this is like adding two new axises
def create_padding_mask_any(inputs0, mask_value=0):
    # seq = tf.cast(tf.math.not_equal(inputs[:, :, 0], mask_value), tf.float32)
    seq = tf.cast(tf.math.reduce_any(tf.math.not_equal(inputs0, mask_value),axis=2), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :],seq[:, :, tf.newaxis]  # (batch_size, 1, 1, seq_len)

## *******************
# lat/lon encoder 
def embedding_lon_lat(layer_n=1, units=64, drop=0.1, reg=None):
    all_layers = list()
    unit_small = units
    # unit_small = units//2 ## tested for v8_75 and v8_76 
    for i in range(layer_n):
        if i==layer_n-1:
            unit_small = units
        
        all_layers.append(Dense(unit_small, activation='relu', kernel_regularizer=reg))
        all_layers.append(LayerNormalization(epsilon=1e-6))
        if drop>0:
            all_layers.append(Dropout(drop))
    
    return tf.keras.Sequential(all_layers)

# BNAD_N = 7
class AddLearnableEmbedding(tf.keras.layers.Layer):
    def __init__(self, d_model, seq_length):
        super(AddLearnableEmbedding, self).__init__()
        self.d_model = d_model
        self.seq_length = seq_length
        # Initialize learnable relative position bias
        self.relative_position_bias = self.add_weight(
            "relative_position_bias",
            shape=[2 * seq_length - 1, self.d_model],
            initializer="random_normal",
            trainable=True
        )
    def call(self, inputs):
        return self.relative_position_bias

# get this from ChatGPT
def create_relative_position_bias(seq_length, d_model, is_learn=False):
    """
    Create learnable relative positional encodings.
    """
    # relative_positions = tf.range(-seq_length + 1, seq_length)
    
    if is_learn:
        print ("AddLearnableEmbedding")
        relative_positional_encoding = AddLearnableEmbedding(d_model,2 * seq_length - 1)(8)
    else:
        relative_positions = tf.range(0, 2 * seq_length - 1)
        embedding_layer = tf.keras.layers.Embedding(input_dim=2 * seq_length - 1, output_dim=d_model)
        relative_positional_encoding = embedding_layer(relative_positions)
    
    # relative_positional_encoding = tf.reshape(relative_positional_encoding, [2 * seq_length - 1, d_model])
    return relative_positional_encoding


# Example usage
# batch_size = 2
# seq_length = 10  # Use a smaller length for simplicity
# d_model = 16

# Create dummy queries, keys, values, and position array
# queries = tf.random.uniform((batch_size, seq_length, d_model))
# keys = tf.random.uniform((batch_size, seq_length, d_model))
# values = tf.random.uniform((batch_size, seq_length, d_model))
# position_array = tf.constant([[1, 2, 3, -9999, 5, 6, -9999, 8, 9, 10], [1, -9999, 3, 4, 5, 6, 7, 8, -9999, 10]], dtype=tf.int32)

# Create relative positional encoding
# relative_position_bias = create_relative_position_bias(seq_length, d_model)

# Compute self-attention with relative position bias
# output = self_attention_with_relative_position_bias(queries, keys, values, position_array, relative_position_bias)
# print(output)


import multi_head_from_ChatGPT
import importlib
importlib.reload(multi_head_from_ChatGPT)
## *******************
## transformer_block
# xL,xL,padding_maskL,units,n_head,drop=drop,is_batch=True,is_att_score=False;
# relative=is_day_input>5
# x_doy=x_doy[:,:MAX_LANDSAT,0]
# queryx = xL
# x = xL 
# padding_mask = padding_maskL
def transformer_block (queryx, x, padding_mask, units, n_head,reg=None, drop=0.1, is_batch=True, is_att_score=True, relative=False, x_doy=None, name=None):
    if relative:
        attention_layer = multi_head_from_ChatGPT.MultiHeadAttentionWithRelativePositionBias(units, n_head, 366)
        attn_output  = attention_layer(queryx, x, x, x_doy, padding_mask)
    else:
        if is_att_score:
            attn_output, attn4 = MultiHeadAttention(key_dim=units//n_head, num_heads=n_head, kernel_regularizer=reg, name=name)(query=queryx, value=x, key=x,
                return_attention_scores=is_att_score, attention_mask=padding_mask)
        else:
            attn_output        = MultiHeadAttention(key_dim=units//n_head, num_heads=n_head, kernel_regularizer=reg, name=name)(query=queryx, value=x, key=x,
                return_attention_scores=is_att_score, attention_mask=padding_mask)
    
    if drop > 0:
        attn_output = Dropout(drop)(attn_output)
    
    out1 = queryx + attn_output
    if is_batch == True:
        out1 = LayerNormalization(epsilon=1e-6, name=name+'_ln1')(out1)
    
    ffn_output = point_wise_feed_forward_network(units, units * 4, reg=reg, name=name+'_Seq')(out1)
    if drop > 0:
        ffn_output = Dropout(drop)(ffn_output)
    
    out2 = out1 + ffn_output
    if is_batch == True:
        out2 = LayerNormalization(epsilon=1e-6, name=name+'_ln2')(out2)
    
    x = out2
    if is_att_score:
        return x, attn4
    else:
        return x

# *****************************************************************************************************************************************************
# ****************************** input is HLS  ***************************************************************
# *****************************************************************************************************************************************************
# layern1=3; layern2=3; units=64; n_head=4; drop=0.1; L2=0; concat=2
# is_day_input=1; is_xy=False; is_reflectance=True; active="linear"; is_sensor=True
# inputs = X_validation_t[:2,:,:] # no filled data
# https://www.tensorflow.org/text/tutorials/transformer
mask_value=-9999.0
def get_transformer_reflectance(MAX_LANDSAT=14, MAX_SENTINEL2=28, L8_bands_n=2, S2_bands_n=4, n_out=11,
    layern1=3, layern2=3, units=64, n_head=4, drop=0.1, L2=0, 
    is_day_input=1, is_sensor=False, is_xy=False, is_dem=False, is_reflectance=True, active="linear", concat=1):
    YEARS_DATA=1
    """using AveragePooling1D with mask"""
    is_batch=True, 
    reg = None
    if L2>0:
        reg = tf.keras.regularizers.l2(l=L2)
    ## *******************
    # reflectance
    LENGTH = MAX_LANDSAT+MAX_SENTINEL2
    inputs = Input(shape=(LENGTH*YEARS_DATA, S2_bands_n+2*is_xy+is_dem*4),name="input")
    ## *******************
    # positional -> need to change positional to day of year
    x_doy = inputs[:,:,:1]
    mask_multi2 = tf.cast(tf.math.not_equal(x_doy,mask_value), tf.float32)
    if is_day_input==1: ## day of year as position 
        # DOY_ARRAY = np.array(range(366))
        x_doy = x_doy*mask_multi2
        # x_doy = (x_doy - DOY_ARRAY.mean() ) * mask_multi2/ DOY_ARRAY.std() 
        x_doy = Dense(units, use_bias=False, name='doy_proj')(x_doy)
    elif is_day_input==2:
        print ("2nd method doy encoder using positional encoder function sin and cos")
        pos_enc = positional_encoding(366, units)
        ## SHIT I did this using ChatGPT 4 on Jun 12, 2024
        ## I input "In tensorflow, I have a tensor x_doy with shape [a,b], and another tensor pos_enc with shape [366,d], 
        ## the x_doy can be any values between 1 to 366 or filled value -9999. write a piece of code to generate another array with shape [a,b,d], 
        ## where the values are set as if any value in [a,b] if not filled (e.g., n), take the nth d vector from pos_enc as the [a,b,:] value, 
        ## if filled, set [a,b,:] as filled "
        # Create a mask for filled values (-9999)
        mask = tf.not_equal(x_doy, -9999)
        # Use tf.gather to index pos_enc with x_doy
        # gathered = tf.gather(pos_enc, tf.clip_by_value(x_doy, 0, 365))
        gathered = tf.gather(pos_enc, tf.cast(tf.clip_by_value(x_doy[:,:,0]-1, clip_value_min=0, clip_value_max=365),tf.int16))
        # Create the filled tensor with the same shape as the gathered tensor
        # filled_tensor = tf.fill([a, b, d], -9999.0)
        filled_tensor = tf.fill([tf.shape(x_doy)[0],LENGTH*YEARS_DATA, units], 0.0)      # fix bug on 4/15/2025
        # Use the mask to conditionally combine the tensors
        # result = tf.where(tf.expand_dims(mask, axis=-1), gathered, filled_tensor)
        x_doy = tf.where(mask, gathered, filled_tensor)
        # print(result)
        # tf.cond(condition, true_fn, false_fn)
    elif is_day_input==3:
        print ("3rd method doy encoder with sin and cos")
        n_times = 366
        xp1 = tf.cos((x_doy+0.5)/n_times*np.pi) * mask_multi2
        xp2  = tf.sin((x_doy+0.5)/n_times*np.pi) * mask_multi2
        xpp = tf.concat ([xp1,xp2],axis=2)    
        x_doy = Dense(units, use_bias=False)(xpp)
    elif is_day_input==4:
        print ("4th method doy encoder using positional embedding function (shit, embedding is not learned) ")
        pos_enc = create_relative_position_bias(366,units)
        mask = tf.not_equal(x_doy, -9999)
        gathered = tf.gather(pos_enc, tf.cast(tf.clip_by_value(x_doy[:,:,0]-1, clip_value_min=0, clip_value_max=365),tf.int32))
        filled_tensor = tf.fill([tf.shape(x_doy)[0],LENGTH,units], 0.0)
        x_doy = tf.where(mask, gathered, filled_tensor)
    elif is_day_input==5:
        print ("5th method doy encoder using learnable positional embedding function ")
        # pos_enc = create_relative_position_bias(366,units,is_learn=True) # does not implement learnable 
        pos_enc = AddLearnableEmbedding(units,366)(x_doy)
        mask = tf.not_equal(x_doy, -9999)
        gathered = tf.gather(pos_enc, tf.cast(tf.clip_by_value(x_doy[:,:,0]-1, clip_value_min=0, clip_value_max=365),tf.int32))
        filled_tensor = tf.fill([tf.shape(x_doy)[0],LENGTH,units], 0.0)
        x_doy = tf.where(mask, gathered, filled_tensor)
        # print(result)
        # tf.cond(condition, true_fn, false_fn)    
    else:
        print ("6th method doy encoder using RELATIVE learnable positional embedding function ")
        
    '''Embedding(2, units) 定义了一个嵌入层，它将输入的整数索引（这里的索引范围是 0 和 1）映射为一个 units 维的连续向量空间。'''
    ## sensor encoder 
    sensor_embed = Embedding(2, units, name='sensor_embed')
    
    ## Landsat 
    xL = inputs[:,:MAX_LANDSAT,1:L8_bands_n]
    mask_multi0 = tf.cast(tf.math.not_equal(xL, mask_value), tf.float32)
    xL = xL * mask_multi0  # 屏蔽掉无效的数据,将mask_value变成0，其他位置不变
    xL = Dense(units, use_bias=False, name='landsat_proj')(xL)
    if is_day_input<=5:
        xL = xL+x_doy[:,:MAX_LANDSAT,]
    
    if is_sensor:
        print("sensor Embedding is used");
        '''0和1始终大于-1的，也就是说embedding输入的是0'''
        xL = xL+sensor_embed(mask_multi0[:,:,0]<-1)
    
    ## Sentinel-2  
    xS = inputs[:,MAX_LANDSAT:,1:S2_bands_n]
    mask_multi0 = tf.cast(tf.math.not_equal(xS,mask_value), tf.float32)
    xS = xS * mask_multi0
    xS = Dense(units, use_bias=False, name='s2_proj')(xS)
    if is_day_input<=5:
        xS = xS+x_doy[:,MAX_LANDSAT:,]
    
    if is_sensor:
        '''0和1始终大于-1的，也就是说embedding输入的是1'''
        xS = xS+sensor_embed(mask_multi0[:,:,0]>-1)
    
    ## *******************
    # if DEM 
    if is_dem:
        dem = inputs[:,MAX_LANDSAT:,S2_bands_n:(S2_bands_n+4)]
        mask_multi2 = tf.cast(tf.math.not_equal(dem, mask_value), tf.float32)
        dem = dem * mask_multi2
        dem = Dense(units, use_bias=False, name='dem_proj')(dem)
        xL = xL+dem
        xS = xS+dem   
    
    ## *******************
    # lat lon 
    if is_xy:
        xxy = inputs[:,:,S2_bands_n:(S2_bands_n+2)]
        mask_multi2 = tf.cast(tf.math.not_equal(xxy,mask_value), tf.float32)
        xxy = xxy * mask_multi2
        xxy = Dense(units, use_bias=False, name='xy_proj')(xxy)
        xL = xL+xxy[:,:MAX_LANDSAT,]
        xS = xS+xxy[:,MAX_LANDSAT:,]
    
    ## *******************
    ## start to encoder and decoder    
    padding_mask, padding_mask3d = create_padding_mask_any(inputs0=inputs[:,:,1:8], mask_value=mask_value)
    '''attention_mask主要是为了表示每个时间步的位置是否有效。在标准的自注意力机制中，attention_mask 一般只会标记 时间步 是否有效，而不涉及 特征维度 的掩蔽。这是因为自注意力机制关心的是时间步之间的关系，而不是每个时间步内各个特征的关系。'''
    '''自注意力机制 主要关注 时间步之间的关系，也就是序列中各个位置的依赖性。在这个过程中，我们关心的通常是哪些时间步是有效的，而不是时间步内部的特征内容。'''
    padding_maskA, _ = create_padding_mask_any(inputs0=inputs[:,:           ,1:8], mask_value=mask_value) # fix this bug on Feb 18 2023, mask only applied for 6 bands but not for all data
    padding_maskL, _ = create_padding_mask_any(inputs0=inputs[:,:MAX_LANDSAT,1:8], mask_value=mask_value) # fix this bug on Feb 18 2023, mask only applied for 6 bands but not for all data
    padding_maskS, _ = create_padding_mask_any(inputs0=inputs[:,MAX_LANDSAT:,1:8], mask_value=mask_value) # fix this bug on Feb 18 2023, mask only applied for 6 bands but not for all data
    # if is_mask:
        # print ("Masking is used")
    # else:
        # print ("Masking is *NOT* used")
    # encoder
    for i in range(layern1):
        xL = transformer_block (xL,xL,padding_maskL,units,n_head,drop=drop,is_batch=True,is_att_score=False,relative=is_day_input>5, x_doy=x_doy[:,:MAX_LANDSAT,0], name=f'l_branch{i}')
        xS = transformer_block (xS,xS,padding_maskS,units,n_head,drop=drop,is_batch=True,is_att_score=False,relative=is_day_input>5, x_doy=x_doy[:,MAX_LANDSAT:,0], name=f's_branch{i}')
    
    if concat==1:
        x = tf.concat([xL, xS],axis=1) 
    elif concat==2:
        x = tf.concat([xL, xS],axis=2) # 6.22
        x = Dense(units)(x) # 6.21, 6.2 is Dense only 
        # if drop > 0:
            # x = Dropout(drop)(x)
        
        # padding_maskA = padding_maskL+padding_maskS
        padding_maskA = tf.cast(tf.logical_or(tf.cast(padding_maskL, tf.bool), tf.cast(padding_maskS, tf.bool)), tf.float32)
    elif concat==3:
        x = xL + xS # 6.22
        padding_maskA = tf.cast(tf.logical_or(tf.cast(padding_maskL, tf.bool), tf.cast(padding_maskS, tf.bool)), tf.float32)
    elif concat==4:
        x = tf.concat([xL, xS],axis=2) # 6.22
        # x = Dense(units)(x) # 6.21, 6.2 is Dense only 
        # if drop > 0:
            # x = Dropout(drop)(x)
        # padding_maskA = padding_maskL+padding_maskS
        padding_maskA = tf.cast(tf.logical_or(tf.cast(padding_maskL, tf.bool), tf.cast(padding_maskS, tf.bool)), tf.float32)
    
    for i in range(layern2):
        if concat<=3:
            x = transformer_block (x, x, padding_maskA,units,n_head,drop=drop,is_batch=True,is_att_score=False,relative=is_day_input>5, x_doy=x_doy[:,:,0], name=f'fuse_branch{i}')
        else:
            x = transformer_block (x, x, padding_maskA,units*2,n_head,drop=drop,is_batch=True,is_att_score=False,relative=is_day_input>5, x_doy=x_doy[:,:,0], name=f'fuse_branch{i}')
    
    ## *******************
    ## start to output 
    if is_reflectance==True:
        print (f"This is a dense prediction model with activation {active} ")
        if concat==1:
            output1 = Dense(n_out, activation=active, kernel_regularizer=reg, name='landsat_out')(x[:,:MAX_LANDSAT,:])
            output2 = Dense(n_out, activation=active, kernel_regularizer=reg, name='s2_out')(x[:,MAX_LANDSAT:,:])
        else:
            output1 = Dense(n_out, activation=active, kernel_regularizer=reg, name='landsat_out')(x)
            output2 = Dense(n_out, activation=active, kernel_regularizer=reg, name='s2_out')(x)
        output = tf.concat([output1, output2],axis=1)

    elif is_reflectance==2: # is_SM true
        print (f"this is the SM model with {active} activation function and Gaussion")
        output1 = Dense(1, activation=active  , kernel_regularizer=reg)(x)
        output2 = Dense(1, activation="linear", kernel_regularizer=reg)(x)
        output = tf.concat([output1, output2], axis=2)
        # output1 = Dense(units, activation="relu", kernel_regularizer=reg)(x)
        # output2 = Dense(units, activation="relu", kernel_regularizer=reg)(x)
        # x = tf.concat([output1, output2], axis=2)
        # output = Dense(n_out, activation="sigmoid", kernel_regularizer=reg)(x)

    elif is_reflectance==3: # is_SM true
        print (f"this is the fuel moisture model with {active} activation function and Gaussion");
        output = Dense(n_out, activation=active, kernel_regularizer=reg, name='output')(x)

    elif is_reflectance==4: # crop damage
        print(f'this is the crop damage model')
        enc_output = x
        output = Dense(n_out, kernel_regularizer=reg, name='output')(enc_output)

    else: # sequence to one classification
        # x (None, 123, 512)
        print(f"this is the crop mapping model")
        padding_mask3d = padding_maskA[:, 0, 0, :][:, :, tf.newaxis] # p*123*1
        enc_output2 = tf.math.multiply(x, padding_mask3d) #p*123*512
        enc_output = tf.math.divide(K.sum(enc_output2, axis=1), K.sum(padding_mask3d, axis=1))  # p * 512
        output = Dense(n_out, kernel_regularizer=reg, name='output')(enc_output) # p * n_out

    model = Model(inputs, output)    
    return model


## copied from tensorflow tutorials https://www.tensorflow.org/text/tutorials/transformer
def positional_encoding(length, depth):
    depth = depth // 2
    
    positions = np.arange(length)[:, np.newaxis]  # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :] / depth  # (1, depth)
    
    angle_rates = 1 / (10000 ** depths)  # (1, depth)
    angle_rads = positions * angle_rates  # (pos, depth)
    
    pos_encoding = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis=-1)
    
    return tf.cast(pos_encoding, dtype=tf.float32)


