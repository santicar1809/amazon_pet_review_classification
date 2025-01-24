import re
import pandas as pd
import fasttext
from sklearn.model_selection import train_test_split
from src.preprocessing.load_dataset import load_dataset
from src.preprocessing.preprocessing import preprocessing

def fast_text(data):
    data['sentiment'] = '__label__' + data['sentiment'].astype(str)

    data['sentiment_description'] = data['sentiment'] + ' '+data['review_body']

    def preprocessing(text):
        text=re.sub(r'[^\w\s]',' ',text)
        text=re.sub(r' +',' ',text)

        return text.strip().lower()

    data['review_body_fast'] = data['sentiment_description'].map(preprocessing)

    train,test=train_test_split(data,test_size=0.3)

    train.to_csv("train",columns=['review_body_fast'],index=False,header=False)
    test.to_csv("test",columns=['review_body_fast'],index=False,header=False)

    model= fasttext.train_supervised(input="train")
    n,accuracy,recall=model.test("test")
    
    fast_text_df=pd.DataFrame({'model':['fasttext'],'accuracy':accuracy,'recall':recall})
    fast_text_df.to_csv("./files/output/fast_text_result.csv",index=False)
    
    return fast_text_df