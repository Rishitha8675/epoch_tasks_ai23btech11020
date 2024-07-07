import tensorflow as tf
from tensorflow.data.experimental import CsvDataset
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import cv2
import os
import pandas as pd


def preprocess(*record):
    label = record[0]
    features = tf.stack(record[1:])
    features = tf.reshape(features, [28, 28, 1])
    features = features / 255.0  
    return features, label

alphabets_types = [tf.string] + [tf.float32] * 784
dataset = tf.data.experimental.CsvDataset("alphabets_data.csv", record_defaults=alphabets_types, header=True)

x, y = zip(*dataset.map(preprocess))


x = np.array(x)
y = np.array([label.numpy().decode('utf-8') for label in y])

# Shuffle the dataset to get good results
indices = np.random.permutation(len(x))
x, y = x[indices], y[indices]

# Encode the labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split the dataset into training and testing sets
train_size = int(0.85 * len(x))
test_size = len(x) - train_size
x_train, x_test = x[:train_size], x[train_size:]
y_train, y_test = y[:train_size], y[train_size:]


model = Sequential([
    Conv2D(16, kernel_size=(3, 3), padding='valid', activation='relu', input_shape=(28, 28, 1)),
    Conv2D(8, kernel_size=(3, 3), padding='valid', activation='relu'),
    Flatten(),
    Dense(50, activation='relu'),
    Dense(26, activation='softmax')
])

model.summary()

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=10, batch_size=100, validation_split=0.15)


y_prob = model.predict(x_test)
y_pred = y_prob.argmax(axis=1)
accuracy = accuracy_score(y_test, y_pred)

print(f'Accuracy: {accuracy}')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()


def process_and_predict(directory):
    img_list = []
    img_files = sorted(os.listdir(directory))
    img_dict = {}  
    
    for img_file in img_files:
        img_path = os.path.join(directory, img_file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            print(f"Failed to read {img_path}")
            continue
        
        img = img / 255.0  
        img = np.expand_dims(img, axis=-1)  
        
        if not np.all(img == 0): 
            img_list.append((img_file, img))
    
    if img_list:
        img_array = np.array([img for _, img in img_list])
        predictions = model.predict(img_array)
        predicted_labels = predictions.argmax(axis=1)
        decoded_predictions = label_encoder.inverse_transform(predicted_labels)
        
        for (img_file, _), pred in zip(img_list, decoded_predictions):
            img_dict[img_file] = pred
            
    return img_dict      
    
     
target_labels= pd.read_csv("target_labels.csv")
sentiment_analysis=pd.read_csv("sentiment_analysis_dataset.csv")

for s in range(1, 7):
    directory = f'text_sentence_{s}'
    sentence=''
    if os.path.exists(directory):
        print(f"Processing directory: {directory}")
        img_dict = process_and_predict(directory)
        
        max_index = len(sorted(os.listdir(directory)))
        for i in range(max_index):
            key = f'file_name{i}.png'
            if key in img_dict:
                sentence+=img_dict[key]
                print(img_dict[key], end='')
                
            else:
                sentence+=' '
                print(' ', end='')
        
    else:
        print(f"Directory {directory} does not exist.")
    

    row_to_be_added = pd.DataFrame([[sentence,target_labels.iloc[s-1,1]]], columns=sentiment_analysis.columns)
    sentiment_analysis = sentiment_analysis._append(row_to_be_added)

    print('\n')

sentiment_analysis.to_csv("sentimental_analysis_dataset.csv", index=False)


    
        
    



























