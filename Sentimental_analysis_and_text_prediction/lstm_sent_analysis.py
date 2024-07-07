import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import LSTM, Dense, Embedding
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences


#Accuracy may not be good due to insufficient data for model training


df = pd.read_csv("processed_data.csv")


X = df['Text']
y = df['Target'].values


label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)


word_tokenizer = Tokenizer()
word_tokenizer.fit_on_texts(X)
X = word_tokenizer.texts_to_sequences(X)


vocab_len = len(word_tokenizer.word_index) + 1


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


max_len = 15  
X_train = pad_sequences(X_train, padding='post', maxlen=max_len)
X_test = pad_sequences(X_test, padding='post', maxlen=max_len)


model = Sequential([
    Embedding(input_dim=vocab_len, output_dim=10, input_length=max_len),
    LSTM(10),
    Dense(3, activation='softmax')
])


model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Build the model (needed before printing summary)
model.build(input_shape=(None, max_len))


model.summary()


history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=4)



print(f"Validation and Testing accuracy: {history.history['val_accuracy'][-1]}")

