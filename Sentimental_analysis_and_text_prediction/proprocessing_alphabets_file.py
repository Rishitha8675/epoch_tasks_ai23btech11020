import pandas as pd

alphabets_data= pd.read_csv("alphabets_28x28.csv",na_values='')          
for col in alphabets_data.columns[1:]:
    alphabets_data[col] = pd.to_numeric(alphabets_data[col], errors='coerce')
alphabets_data = alphabets_data.astype(float, errors='ignore')
alphabets_data.dropna(inplace=True)
alphabets_data.to_csv("alphabets_data.csv",index=False)
