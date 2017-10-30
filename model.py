import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers import Convolution2D, Dropout
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# read the log file
lines = []
with open('run2/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

# generate the training data on the fly
train_samples, valid_samples = train_test_split(lines, test_size=0.2)
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while True:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:(offset+batch_size)]
            images = []
            measurements = []
            for line in batch_samples:
                # use multiple camera images
                for i in range(3):
                    source_path = line[i]
                    filename = source_path.split('/')[-1]
                    current_path = "run2/IMG/" + filename
                    image = cv2.imread(current_path)
                    images.append(image)
                    measurement = float(line[3])
                    correction = 0.2  # adjusted steering measurement
                    if i == 1:  # left image
                        measurement += correction
                    if i == 2:  # right image
                        measurement -= correction
                    measurements.append(measurement)
                    # flip the image to take the opposite sign of steering measurement
                    image_flip = np.fliplr(image)
                    images.append(image_flip)
                    measurement_flip = -measurement
                    measurements.append(measurement_flip)
            X_train = np.array(images)
            y_train = np.array(measurements)
            yield shuffle(X_train, y_train)

"""
# read the training data
images = []
measurements = []
for line in lines:
    # use multiple camera images
    for i in range(3):
        source_path = line[i]
        filename = source_path.split('/')[-1]
        current_path = "run1/IMG/" + filename
        image = cv2.imread(current_path)
        images.append(image)
        measurement = float(line[3])
        correction = 0.2    # adjusted steering measurement
        if i == 1:    # left image
            measurement += correction
        if i == 2:    # right image
            measurement -= correction
        measurements.append(measurement)
        # flip the image to take the opposite sign of steering measurement
        image_flip = np.fliplr(image)
        images.append(image_flip)
        measurement_flip = -measurement
        measurements.append(measurement_flip)
X_train = np.array(images)
y_train = np.array(measurements)
"""

# create neural network
model = Sequential()

# image data normalized
model.add(Lambda(lambda x: (x/127.5)-1, input_shape=(160,320,3)))

# image data cropping
model.add(Cropping2D(cropping=((70,25),(0,0))))

# construct network model based on nvidia architecture
model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(128, 3, 3, activation='relu'))
model.add(Flatten())
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dropout(0.8))
model.add(Dense(1))

# train the model
model.compile(optimizer='adam', loss='mse')
#hist_ob = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)
train_generator = generator(train_samples, batch_size=32)
valid_generator = generator(valid_samples, batch_size=32)
hist_ob = model.fit_generator(train_generator, samples_per_epoch=len(train_samples),
                              validation_data=valid_generator, nb_val_samples=len(valid_samples),
                              nb_epoch=8, verbose=1)

# save the model
model.save('model.h5')

# plot the training and validation loss for each epoch
plt.plot(hist_ob.history['loss'])
plt.plot(hist_ob.history['val_loss'])
plt.title('model mean squared error loss')
plt.xlabel('epoch')
plt.ylabel('mean squared error loss')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.interactive(False)
plt.show()