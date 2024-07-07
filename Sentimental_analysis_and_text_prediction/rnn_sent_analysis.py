import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import SimpleRNN, Dense, Embedding, Flatten
from sklearn.preprocessing import LabelEncoder
from keras.utils import pad_sequences


df = pd.read_csv("processed_data.csv")
X = df['Text']
y = df['Target'].values
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
word_tokenizer = Tokenizer()
word_tokenizer.fit_on_texts(X)
X= word_tokenizer.texts_to_sequences(X)


X_train,X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train = pad_sequences(X_train, padding='post' , maxlen=15)
X_test = pad_sequences(X_test, padding = 'post', maxlen=15)


#Model's acccuracy is not good due to insufficient data

model = Sequential([

      SimpleRNN(50, activation='relu',input_shape=(15,1),return_sequences=False),
      Flatten(),
      Dense(8,activation='relu'),
      Dense(3,activation='softmax')

])


model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

history = model.fit(X_train,y_train,epochs=10,validation_data=(X_test,y_test),batch_size=4)
