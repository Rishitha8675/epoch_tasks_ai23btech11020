import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import  MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


df = pd.read_csv("processed_data.csv")


cv = CountVectorizer()


X = df['Text']
y = df['Target'].values


X = cv.fit_transform(X).toarray()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = MultinomialNB()

clf.fit(X_train,y_train)


y_pred=clf.predict(X_test)

print(accuracy_score(y_test,y_pred))


