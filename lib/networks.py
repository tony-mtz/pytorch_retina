from keras.models import Model
from keras.layers import Input, Concatenate, Conv2D, MaxPooling2D, UpSampling2D, Reshape, core, Dropout, Add
from keras.optimizers import Adam
from pixel_dcn import pixel_dcl

# Unet with optimizer set as ADAM instead of SGD
def get_unet(patch_height, patch_width, n_ch):
    inputs = Input(shape=(patch_height, patch_width, n_ch))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_last')(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_last')(conv1)
    pool1 = MaxPooling2D((2, 2), data_format='channels_last')(conv1)
    
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_last')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_last')(conv2)
    pool2 = MaxPooling2D((2, 2), data_format='channels_last')(conv2)
    
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same',data_format='channels_last')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same',data_format='channels_last')(conv3)

    up1 = UpSampling2D(size=(2, 2), data_format='channels_last')(conv3)
    up1 = Concatenate(axis=3)([conv2, up1])
    
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_last')(up1)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_last')(conv4)
    
    up2 = UpSampling2D(size=(2, 2), data_format='channels_last')(conv4)
    up2 = Concatenate(axis=3)([conv1, up2])
    
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_last')(up2)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_last')(conv5)
    
    conv6 = Conv2D(2, (1, 1), activation='relu',padding='same',data_format='channels_last')(conv5)
    conv6 = core.Reshape((patch_height*patch_width, 2))(conv6)
    #conv6 = core.Permute((2,1))(conv6)
    
    conv7 = core.Activation('softmax')(conv6)

    model = Model(inputs=inputs, outputs=conv7)
    print(model.summary())

    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.3, nesterov=False)
    # model.compile(optimizer='sgd', loss='categorical_crossentropy',metrics=['accuracy'])
    opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=0.1, decay=0.0)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Deeper baseline unet
def get_deeper_unet(patch_height, patch_width, n_ch):
    inputs = Input(shape=(patch_height, patch_width, n_ch))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_last')(inputs)
    conv1 = Dropout(0.2)(conv1)
    pool1 = MaxPooling2D((2, 2), data_format='channels_last')(conv1)
    
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_last')(pool1)
    conv2 = Dropout(0.2)(conv2)
    pool2 = MaxPooling2D((2, 2), data_format='channels_last')(conv2)
    
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same',data_format='channels_last')(pool2)
    conv3 = Dropout(0.2)(conv3)
    pool3 = MaxPooling2D((3, 3), data_format='channels_last')(conv3)
    
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same',data_format='channels_last')(pool3)
    conv4 = Dropout(0.2)(conv4)

    up1 = UpSampling2D(size=(3, 3), data_format='channels_last')(conv4)
    up1 = Concatenate(axis=3)([conv3, up1])
    
    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same',data_format='channels_last')(up1)
    conv5= Dropout(0.2)(conv5)
    
    up2 = UpSampling2D(size=(2, 2), data_format='channels_last')(conv5)
    up2 = Concatenate(axis=3)([conv2, up2])
    
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_last')(up2)
    conv6 = Dropout(0.2)(conv6)
    
    up3 = UpSampling2D(size=(2, 2), data_format='channels_last')(conv6)
    up3 = Concatenate(axis=3)([conv1, up3])
    
    conv7 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_last')(up3)
    conv7 = Dropout(0.2)(conv7)
    
    conv8 = Conv2D(2, (1, 1), activation='relu',padding='same',data_format='channels_last')(conv7)
    conv8 = core.Reshape((2,patch_height*patch_width))(conv8)
    conv8 = core.Permute((2,1))(conv8)
    
    conv9 = core.Activation('softmax')(conv8)

    model = Model(inputs=inputs, outputs=conv9)
    print(model.summary())

    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.3, nesterov=False)
    # model.compile(optimizer='sgd', loss='categorical_crossentropy',metrics=['accuracy'])
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=0.1, decay=0.0)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def get_deeper_unet_2(patch_height, patch_width, n_ch):
    inputs = Input(shape=(patch_height, patch_width, n_ch))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_last')(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1_strided = Conv2D(32, (3, 3), activation='relu', padding='same',strides=2, data_format='channels_last')(conv1)
    
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_last')(conv1_strided)
    conv2 = Dropout(0.2)(conv2)
    conv2_strided = Conv2D(64, (3, 3), activation='relu', padding='same',strides=2, data_format='channels_last')(conv2)
    
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same',data_format='channels_last')(conv2_strided)
    conv3 = Dropout(0.2)(conv3)
    conv3_strided = Conv2D(128, (3, 3), activation='relu', padding='same', strides=2, data_format='channels_last')(conv3)
    
    conv4 = Conv2D(192, (3, 3), activation='relu', padding='same',data_format='channels_last')(conv3_strided)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(192, (3, 3), activation='relu', padding='same',data_format='channels_last')(conv4)
    conv4 = Dropout(0.2)(conv4)
    conv4_strided = Conv2D(192, (3, 3), activation='relu', padding='same', strides=2, data_format='channels_last')(conv4)
    
    conv5 = Conv2D(256, (3, 3), activation='relu', padding='same',data_format='channels_last')(conv4_strided)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(256, (3, 3), activation='relu', padding='same',data_format='channels_last')(conv5)
    conv5 = Dropout(0.2)(conv5)
    conv5_strided = Conv2D(256, (3, 3), activation='relu', padding='same', strides=2, data_format='channels_last')(conv5)
    
    conv6 = Conv2D(320, (3, 3), activation='relu', padding='same',data_format='channels_last')(conv5_strided)
    conv6 = Dropout(0.2)(conv6)
    conv6 = Conv2D(320, (3, 3), activation='relu', padding='same',data_format='channels_last')(conv6)

    up1 = UpSampling2D(size=(2, 2), data_format='channels_last')(conv6)
    up1 = Concatenate(axis=3)([conv5, up1])
    
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same',data_format='channels_last')(up1)
    conv7 = Dropout(0.2)(conv7)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same',data_format='channels_last')(conv7)
    
    up2 = UpSampling2D(size=(2, 2), data_format='channels_last')(conv7)
    up2 = Concatenate(axis=3)([conv4, up2])
    
    conv8 = Conv2D(192, (3, 3), activation='relu', padding='same',data_format='channels_last')(up2)
    conv8 = Dropout(0.2)(conv8)
    conv8 = Conv2D(192, (3, 3), activation='relu', padding='same',data_format='channels_last')(conv8)
    
    up3 = UpSampling2D(size=(2, 2), data_format='channels_last')(conv8)
    up3 = Concatenate(axis=3)([conv3, up3])
    
    conv9 = Conv2D(128, (3, 3), activation='relu', padding='same',data_format='channels_last')(up3)
    conv9 = Dropout(0.2)(conv9)
    conv9 = Conv2D(128, (3, 3), activation='relu', padding='same',data_format='channels_last')(conv9)
    
    up4 = UpSampling2D(size=(2, 2), data_format='channels_last')(conv9)
    up4 = Concatenate(axis=3)([conv2, up4])
    
    conv10 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_last')(up4)
    conv10 = Dropout(0.2)(conv10)
    conv10 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_last')(conv10)
    
    up5 = UpSampling2D(size=(2, 2), data_format='channels_last')(conv10)
    up5 = Concatenate(axis=3)([conv1, up5])
    
    conv11 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_last')(up5)
    conv11 = Dropout(0.2)(conv11)
    conv11 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_last')(conv11)
    
    conv12 = Conv2D(2, (1, 1), activation='relu',padding='same',data_format='channels_last')(conv11)
    conv12 = core.Reshape((patch_height*patch_width, 2))(conv12)
    
    outputs = core.Activation('softmax')(conv12)

    model = Model(inputs=inputs, outputs=outputs)
    print(model.summary())

    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=0.1, decay=0.0)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Doesn't quite make sense - 98.01%
def get_deeper_unet_3(patch_height, patch_width, n_ch):
    inputs = Input(shape=(patch_height, patch_width, n_ch))
    conv1 = Conv2D(32, (5, 5), activation='relu', padding='same',data_format='channels_last')(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_last')(conv1)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_last')(conv1)
    conv1 = Dropout(0.2)(conv1)
    conv1_strided = Conv2D(32, (3, 3), activation='relu', padding='same', strides=2, data_format='channels_last')(conv1)
    
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_last')(conv1_strided)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_last')(conv2)
    conv2 = Dropout(0.2)(conv2)
    conv2_strided = Conv2D(64, (3, 3), activation='relu', padding='same', strides=2, data_format='channels_last')(conv2)
    
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same',data_format='channels_last')(conv2_strided)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same',data_format='channels_last')(conv3)
    conv3 = Dropout(0.2)(conv3)
    conv3_strided = Conv2D(128, (3, 3), activation='relu', padding='same', strides=2, data_format='channels_last')(conv3)
    
    conv4 = Conv2D(192, (3, 3), activation='relu', padding='same',data_format='channels_last')(conv3_strided)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(192, (3, 3), activation='relu', padding='same',data_format='channels_last')(conv4)

    up1 = UpSampling2D(size=(2, 2), data_format='channels_last')(conv4)
    up1 = Concatenate(axis=3)([conv3, up1])
    
    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same',data_format='channels_last')(up1)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same',data_format='channels_last')(conv5)
    
    up2 = UpSampling2D(size=(2, 2), data_format='channels_last')(conv5)
    up2 = Concatenate(axis=3)([conv2, up2])
    
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_last')(up2)
    conv6 = Dropout(0.2)(conv6)
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_last')(conv6)
    
    up3 = UpSampling2D(size=(2, 2), data_format='channels_last')(conv6)
    up3 = Concatenate(axis=3)([conv1, up3])
    
    conv7 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_last')(up3)
    conv7 = Dropout(0.2)(conv7)
    conv7 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_last')(conv7)
    
    conv8 = Conv2D(2, (1, 1), activation='relu',padding='same',data_format='channels_last')(conv7)
    conv8 = core.Reshape((patch_height*patch_width, 2))(conv8)
    
    outputs = core.Activation('softmax')(conv8)

    model = Model(inputs=inputs, outputs=outputs)
    print(model.summary())

    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=0.1, decay=0.0)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Deeper dilated baseline unet
def get_deeper_dil_unet(patch_height, patch_width, n_ch):
    inputs = Input(shape=(patch_height, patch_width, n_ch))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_last')(inputs)
    conv1 = Dropout(0.2)(conv1)
    pool1 = MaxPooling2D((2, 2), data_format='channels_last')(conv1)
    
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_last')(pool1)
    conv2 = Dropout(0.2)(conv2)
    pool2 = MaxPooling2D((2, 2), data_format='channels_last')(conv2)
    
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same',data_format='channels_last')(pool2)
    conv3 = Dropout(0.2)(conv3)
    pool3 = MaxPooling2D((3, 3), data_format='channels_last')(conv3)
    
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same',data_format='channels_last')(pool3)
    conv4 = Dropout(0.2)(conv4)

    up1 = UpSampling2D(size=(3, 3), data_format='channels_last')(conv4)
    up1 = Concatenate(axis=3)([conv3, up1])
    
    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same',data_format='channels_last')(up1)
    conv5= Dropout(0.2)(conv5)
    
    up2 = UpSampling2D(size=(2, 2), data_format='channels_last')(conv5)
    up2 = Concatenate(axis=3)([conv2, up2])
    
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_last')(up2)
    conv6 = Dropout(0.2)(conv6)
    
    up3 = UpSampling2D(size=(2, 2), data_format='channels_last')(conv6)
    up3 = Concatenate(axis=3)([conv1, up3])
    
    dil2_conv7 = Conv2D(32, (3, 3), activation='relu', padding='same',dilation_rate=2, data_format='channels_last')(up3)
    dil2_conv7 = Dropout(0.2)(dil2_conv7)
    
    conv8 = Conv2D(2, (1, 1), activation='relu',padding='same',data_format='channels_last')(dil2_conv7)
    conv8 = core.Reshape((2,patch_height*patch_width))(conv8)
    conv8 = core.Permute((2,1))(conv8)
    
    conv9 = core.Activation('softmax')(conv8)

    model = Model(inputs=inputs, outputs=conv9)
    print(model.summary())

    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.3, nesterov=False)
    # model.compile(optimizer='sgd', loss='categorical_crossentropy',metrics=['accuracy'])
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=0.1, decay=0.0)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Unet without pooling
def get_fully_conv_unet(patch_height, patch_width, n_ch):
    inputs = Input(shape=(patch_height, patch_width, n_ch))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_last')(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1_strided = Conv2D(32, (3, 3), activation='relu', padding='same',strides=2,data_format='channels_last')(conv1)
    
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_last')(conv1_strided)
    conv2 = Dropout(0.2)(conv2)
    conv2_strided = Conv2D(64, (3, 3), activation='relu', padding='same',strides=2,data_format='channels_last')(conv2)
    
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same',data_format='channels_last')(conv2_strided)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same',data_format='channels_last')(conv3)

    up1 = UpSampling2D(size=(2, 2), data_format='channels_last')(conv3)
    up1 = Concatenate(axis=3)([conv2, up1])
    
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_last')(up1)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_last')(conv4)
    
    up2 = UpSampling2D(size=(2, 2), data_format='channels_last')(conv4)
    up2 = Concatenate(axis=3)([conv1, up2])
    
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_last')(up2)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_last')(conv5)
    
    conv6 = Conv2D(2, (1, 1), activation='relu',padding='same',data_format='channels_last')(conv5)
    conv6 = core.Reshape((2,patch_height*patch_width))(conv6)
    conv6 = core.Permute((2,1))(conv6)
    
    conv7 = core.Activation('softmax')(conv6)

    model = Model(inputs=inputs, outputs=conv7)
    print(model.summary())

    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=0.1, decay=0.0)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Unet without pooling and with almost all convolutions dilated
def get_fully_dil_unet(patch_height, patch_width, n_ch):
    inputs = Input(shape=(patch_height, patch_width, n_ch))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', dilation_rate=2, data_format='channels_last')(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1_strided = Conv2D(32, (3, 3), activation='relu', padding='same',strides=2,data_format='channels_last')(conv1)
    
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', dilation_rate=2, data_format='channels_last')(conv1_strided)
    conv2 = Dropout(0.2)(conv2)
    conv2_strided = Conv2D(64, (3, 3), activation='relu', padding='same',strides=2,data_format='channels_last')(conv2)
    
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', dilation_rate=2, data_format='channels_last')(conv2_strided)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same',data_format='channels_last')(conv3)

    up1 = UpSampling2D(size=(2, 2), data_format='channels_last')(conv3)
    up1 = Concatenate(axis=3)([conv2, up1])
    
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same', dilation_rate=2, data_format='channels_last')(up1)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_last')(conv4)
    
    up2 = UpSampling2D(size=(2, 2), data_format='channels_last')(conv4)
    up2 = Concatenate(axis=3)([conv1, up2])
    
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same', dilation_rate=2, data_format='channels_last')(up2)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_last')(conv5)
    
    conv6 = Conv2D(2, (1, 1), activation='relu',padding='same',data_format='channels_last')(conv5)
    conv6 = core.Reshape((2,patch_height*patch_width))(conv6)
    conv6 = core.Permute((2,1))(conv6)
    
    conv7 = core.Activation('softmax')(conv6)

    model = Model(inputs=inputs, outputs=conv7)
    print(model.summary())

    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=0.1, decay=0.0)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Skip concatenation replaced with add (residual connections)
def get_res_unet(patch_height, patch_width, n_ch):
    inputs = Input(shape=(patch_height, patch_width, n_ch))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_last')(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_last')(conv1)
    pool1 = MaxPooling2D((2, 2), data_format='channels_last')(conv1)
    
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_last')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_last')(conv2)
    pool2 = MaxPooling2D((2, 2), data_format='channels_last')(conv2)
    
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same',data_format='channels_last')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same',data_format='channels_last')(conv3)

    up1 = UpSampling2D(size=(2, 2), data_format='channels_last')(conv3)
    convup1 = (Conv2D(64, (1, 1), activation='relu', padding='same',data_format='channels_last'))(up1)
    upres1 = Add()([convup1, conv2])
    
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_last')(upres1)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_last')(conv4)
    
    up2 = UpSampling2D(size=(2, 2), data_format='channels_last')(conv4)
    convup2 = (Conv2D(32, (1, 1), activation='relu', padding='same',data_format='channels_last'))(up2)
    upres2 = Add()([convup2, conv1])
    
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_last')(upres2)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_last')(conv5)
    
    conv6 = Conv2D(2, (1, 1), activation='relu',padding='same',data_format='channels_last')(conv5)
    conv6 = core.Reshape((2,patch_height*patch_width))(conv6)
    conv6 = core.Permute((2,1))(conv6)

    conv7 = core.Activation('softmax')(conv6)

    model = Model(inputs=inputs, outputs=conv7)
    print(model.summary())

    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=0.1, decay=0.0)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Res unet with a deeper architecture
def get_deeper_res_unet(patch_height, patch_width, n_ch):
    inputs = Input(shape=(patch_height, patch_width, n_ch))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_last')(inputs)
    conv1 = Dropout(0.2)(conv1)
    pool1 = MaxPooling2D((2, 2), data_format='channels_last')(conv1)
    
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_last')(pool1)
    conv2 = Dropout(0.2)(conv2)
    pool2 = MaxPooling2D((2, 2), data_format='channels_last')(conv2)
    
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same',data_format='channels_last')(pool2)
    conv3 = Dropout(0.2)(conv3)
    pool3 = MaxPooling2D((3, 3), data_format='channels_last')(conv3)
    
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same',data_format='channels_last')(pool3)
    conv4 = Dropout(0.2)(conv4)

    up1 = UpSampling2D(size=(3, 3), data_format='channels_last')(conv4)
    convup1 = (Conv2D(128, (1, 1), activation='relu', padding='same',data_format='channels_last'))(up1)
    upres1 = Add()([convup1, conv3])
    
    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same',data_format='channels_last')(upres1)
    conv5= Dropout(0.2)(conv5)
    
    up2 = UpSampling2D(size=(2, 2), data_format='channels_last')(conv5)
    convup2 = (Conv2D(64, (1, 1), activation='relu', padding='same',data_format='channels_last'))(up2)
    upres2 = Add()([convup2, conv2])
    
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_last')(upres2)
    conv6 = Dropout(0.2)(conv6)
    
    up3 = UpSampling2D(size=(2, 2), data_format='channels_last')(conv6)
    convup3 = (Conv2D(32, (1, 1), activation='relu', padding='same',data_format='channels_last'))(up3)
    upres3 = Add()([convup3, conv1])
    
    conv7 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_last')(upres3)
    conv7 = Dropout(0.2)(conv7)
    
    conv8 = Conv2D(2, (1, 1), activation='relu',padding='same',data_format='channels_last')(conv7)
    conv8 = core.Reshape((2,patch_height*patch_width))(conv8)
    conv8 = core.Permute((2,1))(conv8)
    
    conv9 = core.Activation('softmax')(conv8)

    model = Model(inputs=inputs, outputs=conv9)
    print(model.summary())

    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.3, nesterov=False)
    # model.compile(optimizer='sgd', loss='categorical_crossentropy',metrics=['accuracy'])
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=0.1, decay=0.0)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


# Combined fully convolutional and residual unet
def get_fully_conv_res_unet(patch_height, patch_width, n_ch):
    inputs = Input(shape=(patch_height, patch_width, n_ch))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_last')(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1_strided = Conv2D(32, (3, 3), activation='relu', padding='same',strides=2,data_format='channels_last')(conv1)
    
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_last')(conv1_strided)
    conv2 = Dropout(0.2)(conv2)
    conv2_strided = Conv2D(64, (3, 3), activation='relu', padding='same',strides=2,data_format='channels_last')(conv2)
    
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same',data_format='channels_last')(conv2_strided)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same',data_format='channels_last')(conv3)

    up1 = UpSampling2D(size=(2, 2), data_format='channels_last')(conv3)
    convup1 = (Conv2D(64, (1, 1), activation='relu', padding='same',data_format='channels_last'))(up1)
    upres1 = Add()([convup1, conv2])
    
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_last')(upres1)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_last')(conv4)
    
    up2 = UpSampling2D(size=(2, 2), data_format='channels_last')(conv4)
    convup2 = (Conv2D(32, (1, 1), activation='relu', padding='same',data_format='channels_last'))(up2)
    upres2 = Add()([convup2, conv1])
    
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_last')(upres2)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_last')(conv5)
    
    conv6 = Conv2D(2, (1, 1), activation='relu',padding='same',data_format='channels_last')(conv5)
    conv6 = core.Reshape((2,patch_height*patch_width))(conv6)
    conv6 = core.Permute((2,1))(conv6)
    
    conv7 = core.Activation('softmax')(conv6)

    model = Model(inputs=inputs, outputs=conv7)
    print(model.summary())

    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=0.1, decay=0.0)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# TODO: Replace convolutions during up sampling with pixelDCN
def get_pixel_unet(patch_height, patch_width, n_ch):
    inputs = Input(shape=(patch_height, patch_width, n_ch))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_last')(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_last')(conv1)
    pool1 = MaxPooling2D((2, 2), data_format='channels_last')(conv1)
    
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_last')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_last')(conv2)
    pool2 = MaxPooling2D((2, 2), data_format='channels_last')(conv2)
    
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same',data_format='channels_last')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same',data_format='channels_last')(conv3)

    up1 = UpSampling2D(size=(2, 2), data_format='channels_last')(conv3)
    up1 = Concatenate(axis=3)([conv2, up1])
    
    conv4 = pixel_dcl(up1, 64, (3, 3), scope='pdcl_1', d_format='NCHW')
    #conv4 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_last')(up1)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_last')(conv4)
    
    up2 = UpSampling2D(size=(2, 2), data_format='channels_last')(conv4)
    up2 = Concatenate(axis=3)([conv1, up2])
    
    conv5 = pixel_dcl(up2, 32, (3, 3), scope='pdcl_1', d_format='NCHW')
    #conv5 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_last')(up2)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_last')(conv5)
    
    conv6 = Conv2D(2, (1, 1), activation='relu',padding='same',data_format='channels_last')(conv5)
    conv6 = core.Reshape((patch_height*patch_width, 2))(conv6)
    conv6 = core.Permute((1, 2))(conv6)
    
    conv7 = core.Activation('softmax')(conv6)

    model = Model(inputs=inputs, outputs=conv7)
    print(model.summary())

    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.3, nesterov=False)
    # model.compile(optimizer='sgd', loss='categorical_crossentropy',metrics=['accuracy'])
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=0.1, decay=0.0)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# TODO: Replace convolutions during up sampling with dilated convolutions
def get_dilated_unet(patch_height, patch_width, n_ch):
    
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=0.1, decay=0.0)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# TODO: Residual dilated convolutions
def get_dilated_net(patch_height, patch_width, n_ch):
    
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=0.1, decay=0.0)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    return model