from keras.models import Model
from keras.layers import Input, Concatenate, Conv2D, MaxPooling2D, UpSampling2D, Reshape, core, Dropout, Add, AveragePooling2D, BatchNormalization
from keras.optimizers import Adam

# Inception residual - channels_last
def get_inception_res_cust(patch_height, patch_width, n_ch):
    inputs = Input(shape=(patch_height, patch_width, n_ch))
    print(inputs.shape)
    stem_block, stem_segment = stem(inputs)
    print(stem_block.shape)
    
    inception_block_1 = inception1(stem_block)
    print(inception_block_1.shape)
    reduction_block_1 = reduction1(inception_block_1)
    print(reduction_block_1.shape)
    
    inception_block_2 = inception2(reduction_block_1)
    print(inception_block_2.shape)
    
    up1 = UpSampling2D(size=(2, 2), data_format='channels_last')(inception_block_2)
    up1 = Concatenate(axis=3)([inception_block_1, up1])
    
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_last')(up1)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_last')(conv1)
    
    up2 = UpSampling2D(size=(2, 2), data_format='channels_last')(conv1)
    up2 = Concatenate(axis=3)([stem_segment, up2])
    
    conv2 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_last')(up2)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_last')(conv2)
    
    up3 = UpSampling2D(size=(2, 2), data_format='channels_last')(conv2)
    up3 = Concatenate(axis=3)([inputs, up3])
    
    conv3 = Conv2D(16, (3, 3), activation='relu', padding='same',data_format='channels_last')(up3)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(16, (3, 3), activation='relu', padding='same',data_format='channels_last')(conv3)
    
    conv4 = Conv2D(2, (1, 1), activation='relu',padding='same',data_format='channels_last')(conv3)
    final = core.Reshape((patch_height*patch_width, 2))(conv4)
    
    out = core.Activation('softmax')(final)

    model = Model(inputs=inputs, outputs=out)
    print(model.summary())

    adam = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=0.1, decay=0.00001)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def stem(inputs):
    conv1 = Conv2D(32, (3, 3), strides=2, activation='relu', padding='same', data_format='channels_last')(inputs)
    conv1 = BatchNormalization()(conv1)
    
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_last')(conv1)
    conv2 = BatchNormalization()(conv2)

    conv3b = Conv2D(64, (1, 1), activation='relu', padding='same', data_format='channels_last')(conv2)
    conv3b = BatchNormalization()(conv3b)
    conv3c = Conv2D(96, (3, 3), activation='relu', padding='same', data_format='channels_last')(conv3b)
    conv3c = BatchNormalization()(conv3c)
    
    conv4a = Conv2D(64, (1, 1), activation='relu', padding='same', data_format='channels_last')(conv2)
    conv4a = BatchNormalization()(conv4a)
    conv4b = Conv2D(64, (1, 7), activation='relu', padding='same', data_format='channels_last')(conv4a)
    conv4b = BatchNormalization()(conv4b)
    conv4c = Conv2D(64, (7, 1), activation='relu', padding='same', data_format='channels_last')(conv4b)
    conv4c = BatchNormalization()(conv4c)
    conv4d = Conv2D(96, (3, 3), activation='relu', padding='same', data_format='channels_last')(conv4c)
    conv4d = BatchNormalization()(conv4d)
    
    concat1 = Concatenate(axis=3)([conv3c, conv4d])
    
    pool1 = MaxPooling2D((3, 3), strides=2, padding='same', data_format='channels_last')(concat1)
    conv5 = Conv2D(192, (3, 3), strides=2, activation='relu', padding='same', data_format='channels_last')(concat1)
    conv5 = BatchNormalization()(conv5)
    
    concat2 = Concatenate(axis=3)([pool1, conv5])
    
    # Pooling block used for upsampling
    pool2 = MaxPooling2D((2, 2), strides=1, padding='same', data_format='channels_last')(conv2)
    
    return concat2, concat1

def inception1(inputs):
    conv1a = Conv2D(32, (1, 1), activation='relu', padding='same', data_format='channels_last')(inputs)
    conv1a = BatchNormalization()(conv1a)
    conv1b = Conv2D(48, (3, 3), activation='relu', padding='same', data_format='channels_last')(conv1a)
    conv1b = BatchNormalization()(conv1b)
    conv1c = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_last')(conv1b)
    conv1c = BatchNormalization()(conv1c)
    
    conv2a = Conv2D(32, (1, 1), activation='relu', padding='same', data_format='channels_last')(inputs)
    conv2a = BatchNormalization()(conv2a)
    conv2b = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_last')(conv2a)
    conv2b = BatchNormalization()(conv2b)
    
    conv3a = Conv2D(32, (1, 1), activation='relu', padding='same', data_format='channels_last')(inputs)
    conv3a = BatchNormalization()(conv3a)
    
    concat1 = Concatenate(axis=3)([conv1c, conv2b, conv3a])
    
    conv4 = Conv2D(384, (1, 1), activation='relu', padding='same', data_format='channels_last')(concat1)
    
    add1 = Add()([inputs, conv4])
    
    return add1
    
def reduction1(inputs):
    # k = 192, l = 224, m = 256, n = 384
    conv1a = Conv2D(192, (1, 1), activation='relu', padding='same', data_format='channels_last')(inputs)
    conv1a = BatchNormalization()(conv1a)
    conv1b = Conv2D(224, (3, 3), activation='relu', padding='same', data_format='channels_last')(conv1a)
    conv1b = BatchNormalization()(conv1b)
    conv1c = Conv2D(256, (3, 3), strides=2, activation='relu', padding='same', data_format='channels_last')(conv1b)
    conv1c = BatchNormalization()(conv1c)
    
    conv2 = Conv2D(384, (3, 3), strides=2, activation='relu', padding='same', data_format='channels_last')(inputs)
    conv2 = BatchNormalization()(conv2)
    
    conv3 = MaxPooling2D((3, 3), strides=2, padding='same', data_format='channels_last')(inputs)
    
    concat1 = Concatenate(axis=3)([conv1c, conv2, conv3])
    
    return concat1

def inception2(inputs):
    conv1a = Conv2D(128, (1, 1), activation='relu', padding='same', data_format='channels_last')(inputs)
    conv1a = BatchNormalization()(conv1a)
    conv1b = Conv2D(160, (1, 7), activation='relu', padding='same', data_format='channels_last')(conv1a)
    conv1b = BatchNormalization()(conv1b)
    conv1c = Conv2D(192, (7, 1), activation='relu', padding='same', data_format='channels_last')(conv1b)
    conv1c = BatchNormalization()(conv1c)
    
    conv2 = Conv2D(192, (1, 1), activation='relu', padding='same', data_format='channels_last')(inputs)
    conv2 = BatchNormalization()(conv2)
    
    concat1 = Concatenate(axis=3)([conv1c, conv2])
    
    conv3 = Conv2D(1024, (1, 1), activation='relu', padding='same', data_format='channels_last')(concat1)
    conv3 = BatchNormalization()(conv3)
    
    add1 = Add()([inputs, conv3])
    
    return add1
    
def reduction2(inputs):
    conv1a = Conv2D(256, (1, 1), activation='relu', padding='same', data_format='channels_last')(inputs)
    conv1a = BatchNormalization()(conv1a)
    conv1b = Conv2D(288, (3, 3), activation='relu', padding='same', data_format='channels_last')(conv1a)
    conv1b = BatchNormalization()(conv1b)
    conv1c = Conv2D(320, (3, 3), strides=2, activation='relu', padding='same', data_format='channels_last')(conv1b)
    conv1c = BatchNormalization()(conv1c)
    
    conv2a = Conv2D(256, (1, 1), activation='relu', padding='same', data_format='channels_last')(inputs)
    conv2a = BatchNormalization()(conv2a)
    conv2b = Conv2D(288, (3, 3), strides=2, activation='relu', padding='same', data_format='channels_last')(conv2a)
    conv2b = BatchNormalization()(conv2b)
    
    conv3a = Conv2D(256, (1, 1), activation='relu', padding='same', data_format='channels_last')(inputs)
    conv3a = BatchNormalization()(conv3a)
    conv3b = Conv2D(384, (3, 3), strides=2, activation='relu', padding='same', data_format='channels_last')(conv3a)
    conv3b = BatchNormalization()(conv3b)
    
    conv4 = MaxPooling2D((3, 3), strides=2, padding='same', data_format='channels_last')(inputs)
    
    concat1 = Concatenate(axis=3)([conv1c, conv2b, conv3b, conv4])
    
    return concat1
    
def inception3(inputs):
    conv1a = Conv2D(192, (1, 1), activation='relu', padding='same', data_format='channels_last')(inputs)
    conv1a = BatchNormalization()(conv1a)
    conv1b = Conv2D(224, (1, 3), activation='relu', padding='same', data_format='channels_last')(conv1a)
    conv1b = BatchNormalization()(conv1b)
    conv1c = Conv2D(256, (3, 1), activation='relu', padding='same', data_format='channels_last')(conv1b)
    conv1c = BatchNormalization()(conv1c)

    conv2 = Conv2D(192, (1, 1), activation='relu', padding='same', data_format='channels_last')(inputs)
    conv2 = BatchNormalization()(conv2)
    
    concat1 = Concatenate(axis=3)([conv1c, conv2])
    
    conv3 = Conv2D(2016, (1, 1), activation='relu', padding='same', data_format='channels_last')(concat1)
    conv3 = BatchNormalization()(conv3)
    
    add1 = Add()([inputs, conv3])
    
    return add1

def dil_block(inputs):
    conv1 = Conv2D(2, (1, 1), activation='relu', padding='same', dilation_rate=1, data_format='channels_last')(inputs)
    conv1 = BatchNormalization()(conv1)
    
    conv2 = Conv2D(2, (1, 1), activation='relu', padding='same', dilation_rate=2, data_format='channels_last')(inputs)
    conv2 = BatchNormalization()(conv2)
    
    #conv3 = Conv2D(2, (1, 1), activation='relu', padding='same', dilation_rate=5, data_format='channels_last')(inputs)
    #conv3 = BatchNormalization()(conv3)
    
    #conv4 = Conv2D(2, (1, 1), activation='relu', padding='same', dilation_rate=15, data_format='channels_last')(inputs)
    #conv4 = BatchNormalization()(conv4)
    
    conv5 = Add()([conv1, conv2])
    
    #conv5 = Add()([conv1, conv2, conv3, conv4])
    
    return conv5