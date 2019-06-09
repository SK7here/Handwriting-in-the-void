#Importing packages
import numpy as np
#To use EMNIST dataset
from mnist import MNIST
from sklearn.model_selection import train_test_split
#To perform one hot encoding
from keras.utils import np_utils
#To build keras sequential model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.callbacks import ModelCheckpoint


##########################################################################################################################################################

#PRE-PROCESSING DATA
#Loading dataset
    #Converting images in dataset as numpy array
emnist_data = MNIST(path='data\\' , return_type='numpy')
#Loading the EMNIST letters dataset -> merges balaned uppercase and lowercase letters into 26 classes
emnist_data.select_emnist('letters')
#Loading the training dataset
X, y = emnist_data.load_training()
#Shape of training images and corresponding labels
print(X.shape)
print(y.shape)

# Reshape the data
    #Reshaping 124800 data having 784 dimensions into 28X28 dimensions (standard for MNIST)
X = X.reshape(124800, 28, 28)
y = y.reshape(124800, 1)

# Make it 0 based indices -> 0 to 25 classes 
y = y-1



##########################################################################################################################################################


# Split dataset into training and validation set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=111)

# Rescale the Images by Dividing Every Pixel in Every Image by 255 -> 0 to 255 as 0 to 1
X_train = X_train.astype('float32')/255
X_test = X_test.astype('float32')/255

# Performing one hot encoding
y_train = np_utils.to_categorical(y_train, num_classes = 26)
y_test = np_utils.to_categorical(y_test, num_classes = 26)

#Creating Sequential Architecture (Linear fashioned)
model = Sequential()

#Building layers of the model
#Perform flattening to convert multidimensional input to linear, to pass into dense
    #Converts 28x28 into 784
model.add(Flatten(input_shape=X_train.shape[1:]))
    #Converts 784 into 512 
model.add(Dense(512, activation='relu'))
    #Drops out 20 percent of neurons randomly -> prevents overfitting
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
#Output layer is set with 26 neurons
model.add(Dense(26, activation='softmax'))

#summarize the model
model.summary()

#Compile the Model
    #Categorical_crossentropy -> multi-class classification
    #rmsprop -> prevents vertical oscillations
model.compile(loss='categorical_crossentropy', optimizer='rmsprop',metrics=['accuracy'])

# Calculate the Classification Accuracy on the Test Set (Before Training)
score = model.evaluate(X_test, y_test, verbose=0)
accuracy = 100*score[1]
print('Before Training - Test accuracy: %.4f%%' % accuracy)

# Train the model
    #checkpointer saves the model in a hdf5 format file
checkpointer = ModelCheckpoint(filepath='emnist.model.best.hdf5', verbose = 1, save_best_only=True)
hist = model.fit(X_train, y_train, batch_size=128, epochs=10,
          validation_split=0.2, callbacks=[checkpointer], shuffle=True)

# Load the Model with the Best Classification Accuracy on the Validation Set
model.load_weights('emnist.model.best.hdf5')

# Save the best model
model.save('emnist_mlp_model.h5')

# Evaluate test accuracy (After training)
score = model.evaluate(X_test, y_test, verbose=0)
accuracy = 100*score[1]
print('Test accuracy: %.4f%%' % accuracy)

