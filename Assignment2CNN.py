#!/usr/bin/env python
# coding: utf-8

# ## Problem Statement
# The MNIST dataset consists of handwritten digit images and it is divided into 60,000 examples for the
# training set and 10,000 examples for testing. The dataset also provides images in such a way that you
# can extract labels for each image which tells you which digit does the image represent.
# Your task involves clustering these images such that all the images representing the same digit should
# belong to one cluster. You can use any method to provide the input to the clustering algorithm. Since
# 1you already have the labels available, you need to also report the accuracy you have obtained while
# clustering the images. 

# Comment this if running locally on Jupyter

# In[1]:


get_ipython().system(" unzip 'drive/My Drive/1272_2280_bundle_archive.zip'")


# ### Importing the libraries

# In[2]:


import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator


# In[3]:


tf.__version__


# ## Data Preprocessing

# ### Preprocessing the Training set

# In[4]:


train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
training_set = train_datagen.flow_from_directory('trainingSet/trainingSet',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')


# ### Preprocessing the Test set

# In[5]:


test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory('testSet',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'categorical')


# ## Building the CNN

# ### Initialising the CNN

# In[6]:


cnn = tf.keras.models.Sequential()


# ### Convolution

# In[7]:


cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))


# ### Pooling

# In[8]:


cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))


# ### Adding a second convolutional layer

# In[9]:


cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))


# ### Flattening

# In[10]:


cnn.add(tf.keras.layers.Flatten())


# ### Full Connection

# In[11]:


cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))


# ### Output Layer

# In[12]:


cnn.add(tf.keras.layers.Dense(units=10, activation='softmax'))


# ## Training the CNN

# ### Compiling the CNN

# In[13]:


cnn.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])


# ### Training the CNN on the Training set and evaluating it on the Test set

# In[14]:


cnn.fit(x = training_set, validation_data = test_set, epochs = 20)


# ## Making a single prediction

# In[15]:


import numpy as np
from keras.preprocessing import image
test_image = image.load_img('testSet/testSet/img_1225.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = cnn.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 0
elif result[0][1] == 1:
    prediction = 1
elif result[0][2] == 1:
    prediction = 2
elif result[0][3] == 1:
    prediction = 3
elif result[0][4] == 1:
    prediction = 4
elif result[0][5] == 1:
    prediction = 5
elif result[0][6] == 1:
    prediction = 6
elif result[0][7] == 1:
    prediction = 7
elif result[0][8] == 1:
    prediction = 8
elif result[0][9] == 1:
    prediction = 9

print(prediction)


# In[28]:


import numpy as np
from keras.preprocessing import image
test_image = image.load_img('3.1.png', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = cnn.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 0
elif result[0][1] == 1:
    prediction = 1
elif result[0][2] == 1:
    prediction = 2
elif result[0][3] == 1:
    prediction = 3
elif result[0][4] == 1:
    prediction = 4
elif result[0][5] == 1:
    prediction = 5
elif result[0][6] == 1:
    prediction = 6
elif result[0][7] == 1:
    prediction = 7
elif result[0][8] == 1:
    prediction = 8
elif result[0][9] == 1:
    prediction = 9

print(prediction)


# In[24]:


import numpy as np
from keras.preprocessing import image
test_image = image.load_img('5.jpeg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = cnn.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 0
elif result[0][1] == 1:
    prediction = 1
elif result[0][2] == 1:
    prediction = 2
elif result[0][3] == 1:
    prediction = 3
elif result[0][4] == 1:
    prediction = 4
elif result[0][5] == 1:
    prediction = 5
elif result[0][6] == 1:
    prediction = 6
elif result[0][7] == 1:
    prediction = 7
elif result[0][8] == 1:
    prediction = 8
elif result[0][9] == 1:
    prediction = 9

print(prediction)


# In[25]:


import numpy as np
from keras.preprocessing import image
test_image = image.load_img('6.jpeg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = cnn.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 0
elif result[0][1] == 1:
    prediction = 1
elif result[0][2] == 1:
    prediction = 2
elif result[0][3] == 1:
    prediction = 3
elif result[0][4] == 1:
    prediction = 4
elif result[0][5] == 1:
    prediction = 5
elif result[0][6] == 1:
    prediction = 6
elif result[0][7] == 1:
    prediction = 7
elif result[0][8] == 1:
    prediction = 8
elif result[0][9] == 1:
    prediction = 9

print(prediction)


# In[19]:


lst_test = []
import os
for filename in os.listdir('testSet/testSet'):
    img_path = os.path.join('testSet/testSet', filename)
    
    
    test_image = image.load_img(img_path, target_size = (64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = cnn.predict(test_image)
    if result[0][0] == 1:
        prediction = 0
    elif result[0][1] == 1:
        prediction = 1
    elif result[0][2] == 1:
        prediction = 2
    elif result[0][3] == 1:
        prediction = 3
    elif result[0][4] == 1:
        prediction = 4
    elif result[0][5] == 1:
        prediction = 5
    elif result[0][6] == 1:
        prediction = 6
    elif result[0][7] == 1:
        prediction = 7
    elif result[0][8] == 1:
        prediction = 8
    elif result[0][9] == 1:
        prediction = 9
        
    lst_test.append(prediction)

