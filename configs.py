"""
USING KERAS TUNER
SOURCE: https://www.kaggle.com/fchollet/keras-kerastuner-best-practices/

# Define a tunable model
We define a function def make_model(hp): which builds a compiled Keras model, parameterized by hyperparameters obtained from the hp argument.

Our model includes a stage that does random image data augmentation, via the augment_images function. Our image augmentation is itself tunable: we'll find the best augmentation configuration during the hyperparameter search.

from tensorflow import keras
from tensorflow.keras import layers

def augment_images(x, hp):
    use_rotation = hp.Boolean('use_rotation')
    if use_rotation:
        x = layers.experimental.preprocessing.RandomRotation(
            hp.Float('rotation_factor', min_value=0.05, max_value=0.2)
        )(x)
    use_zoom = hp.Boolean('use_zoom')
    if use_zoom:
        x = layers.experimental.preprocessing.RandomZoom(
            hp.Float('use_zoom', min_value=0.05, max_value=0.2)
        )(x)
    return x

def make_model(hp):
    inputs = keras.Input(shape=(28, 28, 1))
    x = layers.experimental.preprocessing.Rescaling(1. / 255)(inputs)
    x = layers.experimental.preprocessing.Resizing(64, 64)(x)
    x = augment_images(x, hp)

     num_block = hp.Int('num_block', min_value=2, max_value=5, step=1)
    num_filters = hp.Int('num_filters', min_value=32, max_value=128, step=32)
    for i in range(num_block):
        x = layers.Conv2D(
            num_filters,
            kernel_size=3,
            activation='relu',
            padding='same'
        )(x)
        x = layers.Conv2D(
            num_filters,
            kernel_size=3,
            activation='relu',
            padding='same'
        )(x)
        x = layers.MaxPooling2D(2)(x)
    
    reduction_type = hp.Choice('reduction_type', ['flatten', 'avg'])
    if reduction_type == 'flatten':
        x = layers.Flatten()(x)
    else:
        x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(
        units=hp.Int('num_dense_units', min_value=32, max_value=512, step=32),
        activation='relu'
    )(x)
    x = layers.Dropout(
        hp.Float('dense_dropout', min_value=0., max_value=0.7)
    )(x)
    outputs = layers.Dense(10)(x)
    model = keras.Model(inputs, outputs)
    
    learning_rate = hp.Float('learning_rate', min_value=3e-4, max_value=3e-3)
    optimizer = keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),

    optimizer=optimizer,
                  metrics=[keras.metrics.SparseCategoricalAccuracy(name='acc')])
    model.summary()
    return model

# RUNNING HYPERPARAMETER SEARCH (explores 100 different model configurations)
Note that we configure the calls to model.fit() to use the EarlyStopping callbacks. Indeed, we train for 100 epochs, but the model is likely to start overfitting much earlier than that -- in general, always use a large number of epochs + the EarlyStopping callback.

Our search is guided by validation accuracy, which is computed on a fixed 20% hold-out set of the training data.

tuner = kt.tuners.RandomSearch(
    make_model,
    objective='val_acc',
    max_trials=100,
    overwrite=True)

callbacks=[keras.callbacks.EarlyStopping(monitor='val_acc', mode='max', patience=3, baseline=0.9)]
tuner.search(x_train, y_train, validation_split=0.2, callbacks=callbacks, verbose=1, epochs=100)

# FINDING BEST EPOCH VALUE
Now, we can retrieve the best hyperparameters, use them to build the best model, and train the model for 50 epochs to find at which epoch training should stop.

####
best_hp = tuner.get_best_hyperparameters()[0]
model = make_model(best_hp)
history = model.fit(x_train, y_train, validation_split=0.2, epochs=50)

####### TRAINING THE MODEL
we can train the best model configuration from scratch for the optimal number of epochs.

This time, we train on the entirety of the training data -- no validation split. Our model parameters are already validated.

val_acc_per_epoch = history.history['val_acc']
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
model = make_model(best_hp)
model.fit(x_train, y_train, epochs=best_epoch)

"""

"""
########################### TRAINING MODEL #############################

from keras.callbacks import ModelCheckpoint, EarlyStopping

checkpoint = ModelCheckpoint("vgg16_1.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=20, verbose=1, mode='auto')
hist = model.fit_generator(steps_per_epoch=100,generator=traindata, validation_data= testdata, validation_steps=10,epochs=100,callbacks=[checkpoint,early])
"""

"""
########################### TIME TAKEN ##################################
time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(# time_elapsed // 60, time_elapsed % 60))
print('Best val Acc: {:4f}'.format(best_acc))
"""

"""
############################### TESTING ###################################
# from keras.preprocessing import image
# img = image.load_img("image.jpeg",target_size=(224,224))
# img = np.asarray(img)
# plt.imshow(img)
# img = np.expand_dims(img, axis=0)
# from keras.models import load_model
# saved_model = load_model("vgg16_1.h5")
# output = saved_model.predict(img)
# if output[0][0] > output[0][1]:
#     print("cat")
# else:
#     print('dog')

# def prepare(filepath):
#     arr = cv.imread(filepath, cv.IMREAD_GRAYSCALE)
#     arr = cv.resize(arr, (224,224))
#     return arr.reshape(-1,224,224,1)

# classes = ['Dog', 'Cat']
# prediction = model.predict([prepare('../input/test-images/_111434467_gettyimages-1143489763.jpg')])
# print(classes[int(prediction)])

# def predict(model, IMG_PATH, IMG_SIZE):
#     # show image
#     img = cv.imread(IMG_PATH)
#     plt.imshow(img)

#     # resize image and turn into array
#     x = readToGray(img, IMG_SIZE)
#     x = np.array(x).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
#     x = x/255
#     result = model.predict(x).tolist()

#     # make a prediction with confidence
#     if result[0][0] >= 0.5:
#         prediction = 'DOG 🐶'
#         confidence = ("{:.2%}".format(result[0][0]))
#     else:
#         prediction = 'CAT 🐱'
#         confidence = ("{:.2%}".format(result[0][1]))
        
#     print("I am {0} confident that this is a {1}".format(confidence, prediction))
"""

