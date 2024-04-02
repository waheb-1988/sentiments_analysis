import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, log_loss, accuracy_score
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
import matplotlib.pyplot as plt 
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


class TFIDF:
    def __init__(self, file_name : str, product_name: str):
        self.file_name = file_name
        self.product_name= product_name
        
    def read_data(self):
        df = pd.read_excel(os.path.join(Path(__file__).parent.parent,"data",self.file_name+".xlsx"))
        return df
    
    def select_product(self):
        df = self.read_data()
        df_product = df[df['product']==self.product_name].head(1000)
        print("The total number of row in the product {nm} is {nb}".format(nb=df_product.shape[0],nm=self.product_name))
        return df_product
    
    @staticmethod
    def preprocess_text(review_text):
        # Convert text to lowercase
        text = review_text.lower()

        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))

        # Tokenize the text
        tokens = word_tokenize(text)

        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [token for token in tokens if token not in stop_words]

        # Stem the tokens
        stemmer = PorterStemmer()
        stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]

        # Return preprocessed text as a string
        return ' '.join(stemmed_tokens)
    
    def process_select_product(self):
        df = self.select_product()
        df['review_cleaning'] = df['review_text'].astype(str).apply(TFIDF.preprocess_text)
        return df
    
    def data_analysis_report(self):
        df = self.select_product()
        # Calculate count and percentage of each rating
        rating_counts = df['rating'].value_counts().sort_index()
        rating_percentages = (rating_counts / len(df)) * 100

        # Create a bar chart
        fig, ax1 = plt.subplots()

        # Plotting count of ratings
        ax1.bar(rating_counts.index.astype(str), rating_counts, color='b', alpha=0.7, label='Count')
        ax1.set_xlabel('Rating')
        ax1.set_ylabel('Count', color='b')

        # Creating a secondary y-axis for percentages
        ax2 = ax1.twinx()
        ax2.plot(rating_percentages.index.astype(str), rating_percentages, color='r', marker='o', label='Percentage')
        ax2.set_ylabel('Percentage (%)', color='r')

        # Title and legends
        plt.title("Distribution of Ratings of {pr}".format(pr=self.product_name))
        fig.tight_layout()
        plt.savefig(os.path.join(Path(__file__).parent.parent,"output",'ratings_distribution_'+self.product_name+'.png'))
        
        pass
        
    def create_sentiment_var(self):
        df = self.select_product()
        
        #### Removing the neutral reviews
        df_sentiment = df[df['rating'] != 3]
        df_sentiment['sentiment'] = df_sentiment['rating'].apply(lambda rating : +1 if rating > 3 else 0)
        
        # Calculate count and percentage of each rating
        sentiment_counts = df_sentiment['sentiment'].value_counts().sort_index()
        sentiment_percentages = (sentiment_counts / len(df_sentiment)) * 100

        # Create a bar chart
        fig, ax1 = plt.subplots()

        # Plotting count of ratings
        ax1.bar(sentiment_counts.index.astype(str), sentiment_counts, color='b', alpha=0.7, label='Count')
        ax1.set_xlabel('Sentiment')
        ax1.set_ylabel('Count', color='b')

        # Creating a secondary y-axis for percentages
        ax2 = ax1.twinx()
        ax2.plot(sentiment_percentages.index.astype(str), sentiment_percentages, color='r', marker='o', label='Percentage')
        ax2.set_ylabel('Percentage (%)', color='r')

        # Title and legends
        plt.title("Distribution of sentiment of {pr}".format(pr=self.product_name))
        fig.tight_layout()
        plt.savefig(os.path.join(Path(__file__).parent.parent,"output",'sentiment_distribution_'+self.product_name+'.png'))
        
        #Positive vs Negative reviews
        df_sentiment_droppedna = df_sentiment.dropna()  
        #Positive vs Negative reviews
        positive = df_sentiment_droppedna[df_sentiment_droppedna["sentiment"] == 1].dropna()
        negative = df_sentiment_droppedna[df_sentiment_droppedna["sentiment"] == 0].dropna()
        return df_sentiment_droppedna , positive, negative
    
    def split_input(self):
        df, _ , _ = self.create_sentiment_var()
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(df['review_text'], df['sentiment'], test_size=0.2, random_state=42)
        # Create a TF-IDF vectorizer
        vectorizer = TfidfVectorizer(max_features=5000)
        # Fit the vectorizer to the training data
        vectorizer.fit(X_train)
        # Transform the training and testing data into TF-IDF vectors
        X_train_tfidf = vectorizer.transform(X_train).toarray()
        X_test_tfidf = vectorizer.transform(X_test).toarray()
        return X_train_tfidf,X_test_tfidf , y_train ,y_test 
    
    def models(self):
        # Step 5− Training the Model
        models = {
        'LogisticRegression': LogisticRegression(random_state=42),
         'KNeighborsClassifier': KNeighborsClassifier(),
            'SVC': SVC(random_state=42),
        'DecisionTreeClassifier': DecisionTreeClassifier(max_depth=1, random_state=42)
        }
        return models
    
    @staticmethod
    def loss(y_true, y_pred, retu=False):
        pre = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        loss = log_loss(y_true, y_pred)
        acc = accuracy_score(y_true, y_pred)

        if retu:
            return pre, rec, f1, loss, acc
        else:
            print('  pre: %.3f\n  rec: %.3f\n  f1: %.3f\n  loss: %.3f\n  acc: %.3f' % (pre, rec, f1, loss, acc))
            
        
    def train_model(self):
        f=[]
        X_train_tfidf,X_test_tfidf , y_train ,y_test  = self.split_input()
        models= self.models()
        for name, model in models.items():
            model.fit(X_train_tfidf,y_train )
            loss=TFIDF.loss(y_test, model.predict(X_test_tfidf))
            f=[].append(loss)
            print('-------{h}-------'.format(h=name))
            
        # y_pred = model.predict(X_test_tfidf)
        # accuracy = accuracy_score(y_test, y_pred)
        # precision = precision_score(y_test, y_pred, average='weighted')
        # recall = recall_score(y_test, y_pred, average='weighted')
        # f1 = f1_score(y_test, y_pred, average='weighted')
        # print("######## {ms} #######".format(ms=model_name))
        # print(f"Accuracy: {accuracy:}")
        # print(f"Precision: {precision:}")
        # print(f"Recall: {recall:}")
        # print(f"F1 score: {f1:}")
        return f
    
####### Test Data
   
instance = TFIDF("df_contact","Womens Clothing E-Commerce Reviews")
#data = instance.read_data()
f = instance.train_model()
print(f)
