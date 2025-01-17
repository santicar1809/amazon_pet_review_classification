import spacy
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import joblib
import scipy.sparse

def feature_engineer(df):
    if not os.path.exists('./files/figs/'):
        os.makedirs('./files/figs/')
    fig_path='./files/figs/'
    nlp=spacy.load('en_core_web_sm')
    def lemmatize_text(text):
        doc=nlp(text)
        return " ".join([token.lemma_ for token in doc if not token.is_punct]) 
    df['tokens']=df['review_body'].apply(lemmatize_text)
    df.dropna(inplace=True)
    df.to_csv('data_final.csv',index=False)
    
    #df=pd.read_csv('data_final.csv')
    
    train_df,test_df=train_test_split(df,test_size=0.4,random_state=42,stratify=df['sentiment'])
    features=train_df['tokens']
    target=train_df['sentiment']

    
    target=target.apply(lambda x: 0 if x =='Negative' else (1 if x=='Positive' else 2))
    
    fig1,ax1=plt.subplots()
    target.value_counts().plot(kind='bar', ax=ax1)
    fig1.savefig(fig_path+'clases.png')
    
    features_train,features_valid,target_train,target_valid=train_test_split(features,target,test_size=0.3,random_state=42,stratify=target)
    features_test=test_df['tokens']
    target_test=test_df['sentiment']
    if not os.path.exists("./files/output/"):
        os.makedirs("./files/output/")

    output_path="./files/output/"
    target_test=target_test.apply(lambda x: 0 if x =='Negative' else (1 if x=='Positive' else 2))
    vectorizer=TfidfVectorizer(max_features=5000,ngram_range=(1,2))
    features_train_tfidf = vectorizer.fit_transform(features_train)  # Transformamos el texto de entrenamiento
    features_valid_tfidf = vectorizer.transform(features_valid)
    features_test_tfidf = vectorizer.transform(features_test)

    scipy.sparse.save_npz(output_path+'features_test.npz', features_test_tfidf)
    target_test.to_csv(output_path+'target_test.csv',index=False)

    joblib.dump(vectorizer,'./models/vectorizer.joblib')

    fig2,ax2=plt.subplots()
    target_train.value_counts().plot(kind='bar', ax=ax2)
    fig2.savefig(fig_path+'clases_balanced.png')

    return features_train_tfidf,features_valid_tfidf,target_train,target_valid



