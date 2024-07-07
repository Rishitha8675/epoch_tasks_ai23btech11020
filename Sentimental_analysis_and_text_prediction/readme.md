### Data Processing and Analysis Workflow

*1. Data Cleaning and Preparation*
   - Cleaned data in "alphabets_28x28.csv" using pandas and saved it as "alphabets_data.csv". Code in file: "proprocessing_alphabets_file.py".
   - Used TensorFlow to load the data due to its size and converted it into a TensorFlow DataFrame.
   - Cropped images to 28x28 pixels, stored in directories named "text_sentence" followed by a prefix number representing the image line number. Code in file: "image_breaking.py".

*2. CNN Training and Text Prediction*
   - Trained a CNN using "alphabets_data.csv" to predict letters from the cropped images. Code in file: "training_cnn_and_text_predicting.py".

*3. Dataset Combination*
   - Created "sentimental_analysis_dataset.csv" combining predicted text data and data from "sentiment_analysis_dataset.csv". Code at the bottom of file: "training_cnn_and_text_predicting.py".

*4. Data Preprocessing*
   - Preprocessed data from "sentimental_analysis_dataset.csv" and saved it as "processed_data.csv" to avoid repeated preprocessing. Code in file: "cleaning_data.py".

*5. Sentiment Analysis*
   - Utilized "processed_data.csv" for sentiment analysis:
     - NAIVE BAYES algo: Code in file: "naive_bayes_sent_analysis.py".
     - RNN: Code in file: "rnn_sent_analysis.py" (may have limited performance due to data).
     - LSTM: Code in file: "lstm_sent_analysis.py" (may have limited performance due to data).

Due to insufficient data, the RNN and LSTM models may not perform as well as expected.