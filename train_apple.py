from __future__ import print_function

# for reproducible
import numpy as np
np.random.seed(100)
import tensorflow as tf
tf.set_random_seed(100)
sess = tf.Session()

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K


import os
ls1=os.listdir('data/color')
dic1={}
import scipy.misc as sm
import numpy as np
count=0
for idx,i in enumerate(ls1):
    dic1[i]=idx
    ls2=os.listdir('data/color/'+i)
    for j in ls2:
        #im1=np.asarray(sm.imread('color/'+i+'/'+j))
        #temp=np.zeros((len(im1),len(im1[0]),len(im1[0][0])   ))
        count=count+1


import os
ls1=os.listdir('data/color')
dic1={}
import scipy.misc as sm
import numpy as np
X=np.zeros((count,256,256,3))
Y=np.zeros((count,1))
vap=0

for idx,i in enumerate(ls1):
    dic1[i]=idx
    ls2=os.listdir('data/color/'+i)
    for j in ls2:
        im1=np.asarray(sm.imread('data/color/'+i+'/'+j))
        X[vap,:,:,:]=im1
        Y[vap,0]=idx
        #temp=np.zeros((len(im1),len(im1[0]),len(im1[0][0])   ))
        vap=vap+1

# In[4]:


print (Y.shape)
np.random.permutation(5)


print (dic1)


batch_size = 10
num_classes = len(dic1)
epochs = 25

# input image dimensions
img_rows, img_cols = 256, 256


X /= 255.0


ind=np.random.permutation(X.shape[0])
len_ind=ind.shape[0]
train_ind= ind[0: int(0.8*len_ind)]
val_ind= ind[ int(0.8*len_ind) : int(0.9*len_ind)]
test_ind= ind[ int(0.9*len_ind) : len_ind]
X=X[ind]
Y=Y[ind]


sm.imsave('name.png',X[0])


X_train=X[0:int(0.8*len_ind)]
X_val=X[int(0.8*len_ind):int(0.9*len_ind)]
X_test=X[int(0.8*len_ind) :len_ind]

def visuals(num):
    sm.imsave('name'+str(num)+'.png',X_train[num])
    for i in dic1:
        if(dic1[i]== int(Y[num,0] ) ):
            print (i)
visuals(0)
visuals(1)
visuals(2)
visuals(3)

Y_train=Y[0:int(0.8*len_ind)]
Y_val=Y[int(0.8*len_ind):int(0.9*len_ind)]
Y_test=Y[int(0.8*len_ind) :len_ind]


# convert class vectors to binary class matrices
from keras.utils import np_utils
 
#Y_train = np_utils.to_categorical(Y_train,num_classes)
#Y_test = np_utils.to_categorical(Y_test,num_classes)
#Y_val = np_utils.to_categorical(Y_val,num_classes)


Y_train = keras.utils.to_categorical(Y_train, num_classes)
Y_test = keras.utils.to_categorical(Y_test, num_classes)
Y_val = keras.utils.to_categorical(Y_val, num_classes)



#model = Sequential()
#model.add(Conv2D(32, 3, 3,
#                 activation='relu',
#                 input_shape=(256,256,3)))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Conv2D(64, 3, 3, activation='relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Conv2D(64, 3, 3, activation='relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Conv2D(64, 3, 3, activation='relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Conv2D(64, 3, 3, activation='relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))
#model.add(Flatten())
#model.add(Dense(128, activation='relu'))
#model.add(Dropout(0.5))
#model.add(Dense(num_classes, activation='softmax'))

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(256,256,3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(X_train, Y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_val, Y_val))



score = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json 
# serialize model to JSON

model_json = model.to_json()
with open("model/cnn2.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model/cnn2.h5")
print("Saved model to disk")

######################################################################
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(256,256,3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(X_train, Y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_val, Y_val))



score = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json 
# serialize model to JSON

model_json = model.to_json()
with open("model/cnn3.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model/cnn3.h5")
print("Saved model to disk")

######################################################################
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(256,256,3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(X_train, Y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_val, Y_val))



score = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json 
# serialize model to JSON

model_json = model.to_json()
with open("model/cnn4.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model/cnn4.h5")
print("Saved model to disk")

######################################################################
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(256,256,3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())from __future__ import print_function

# for reproducible
import numpy as np
np.random.seed(100)
import tensorflow as tf
tf.set_random_seed(100)
sess = tf.Session()

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K


import os
ls1=os.listdir('data/color')
dic1={}
import scipy.misc as sm
import numpy as np
count=0
for idx,i in enumerate(ls1):
    dic1[i]=idx
    ls2=os.listdir('data/color/'+i)
    for j in ls2:
        #im1=np.asarray(sm.imread('color/'+i+'/'+j))
        #temp=np.zeros((len(im1),len(im1[0]),len(im1[0][0])   ))
        count=count+1


import os
ls1=os.listdir('data/color')
dic1={}
import scipy.misc as sm
import numpy as np
X=np.zeros((count,256,256,3))
Y=np.zeros((count,1))
vap=0

for idx,i in enumerate(ls1):
    dic1[i]=idx
    ls2=os.listdir('data/color/'+i)
    for j in ls2:
        im1=np.asarray(sm.imread('data/color/'+i+'/'+j))
        X[vap,:,:,:]=im1
        Y[vap,0]=idx
        #temp=np.zeros((len(im1),len(im1[0]),len(im1[0][0])   ))
        vap=vap+1

# In[4]:


print (Y.shape)
np.random.permutation(5)


print (dic1)


batch_size = 10
num_classes = len(dic1)
epochs = 25

# input image dimensions
img_rows, img_cols = 256, 256


X /= 255.0


ind=np.random.permutation(X.shape[0])
len_ind=ind.shape[0]
train_ind= ind[0: int(0.8*len_ind)]
val_ind= ind[ int(0.8*len_ind) : int(0.9*len_ind)]
test_ind= ind[ int(0.9*len_ind) : len_ind]
X=X[ind]
Y=Y[ind]


sm.imsave('name.png',X[0])


X_train=X[0:int(0.8*len_ind)]
X_val=X[int(0.8*len_ind):int(0.9*len_ind)]
X_test=X[int(0.8*len_ind) :len_ind]

def visuals(num):
    sm.imsave('name'+str(num)+'.png',X_train[num])
    for i in dic1:
        if(dic1[i]== int(Y[num,0] ) ):
            print (i)
visuals(0)
visuals(1)
visuals(2)
visuals(3)

Y_train=Y[0:int(0.8*len_ind)]
Y_val=Y[int(0.8*len_ind):int(0.9*len_ind)]
Y_test=Y[int(0.8*len_ind) :len_ind]


# convert class vectors to binary class matrices
from keras.utils import np_utils
 
#Y_train = np_utils.to_categorical(Y_train,num_classes)
#Y_test = np_utils.to_categorical(Y_test,num_classes)
#Y_val = np_utils.to_categorical(Y_val,num_classes)


Y_train = keras.utils.to_categorical(Y_train, num_classes)
Y_test = keras.utils.to_categorical(Y_test, num_classes)
Y_val = keras.utils.to_categorical(Y_val, num_classes)



#model = Sequential()
#model.add(Conv2D(32, 3, 3,
#                 activation='relu',
#                 input_shape=(256,256,3)))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Conv2D(64, 3, 3, activation='relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Conv2D(64, 3, 3, activation='relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Conv2D(64, 3, 3, activation='relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Conv2D(64, 3, 3, activation='relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))
#model.add(Flatten())
#model.add(Dense(128, activation='relu'))
#model.add(Dropout(0.5))
#model.add(Dense(num_classes, activation='softmax'))

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(256,256,3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(X_train, Y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_val, Y_val))



score = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json 
# serialize model to JSON

model_json = model.to_json()
with open("model/cnn2.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model/cnn2.h5")
print("Saved model to disk")

######################################################################
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(256,256,3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(X_train, Y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_val, Y_val))



score = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json 
# serialize model to JSON

model_json = model.to_json()
with open("model/cnn3.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model/cnn3.h5")
print("Saved model to disk")

######################################################################
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(256,256,3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(X_train, Y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_val, Y_val))



score = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json 
# serialize model to JSON

model_json = model.to_json()
with open("model/cnn4.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model/cnn4.h5")
print("Saved model to disk")

######################################################################
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(256,256,3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(X_train, Y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_val, Y_val))



score = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json 
# serialize model to JSON

model_json = model.to_json()
with open("model/cnn5.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model/cnn5.h5")
print("Saved model to disk")
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(X_train, Y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_val, Y_val))



score = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json 
# serialize model to JSON

model_json = model.to_json()
with open("model/cnn5.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model/cnn5.h5")
print("Saved model to disk")
