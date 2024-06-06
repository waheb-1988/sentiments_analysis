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
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from catboost import (CatBoostClassifier,CatBoostRegressor)
from sklearn.model_selection import GridSearchCV
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import AdaBoostClassifier
from imblearn.over_sampling import BorderlineSMOTE, SMOTE, ADASYN, SMOTENC, RandomOverSampler
from sklearn.ensemble import BaggingClassifier
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report, confusion_matrix
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTEENN
import matplotlib.cm as cm
from matplotlib import rcParams
from collections import Counter
from nltk.tokenize import RegexpTokenizer
import re
import string
import tensorflow as tf
import torch
from torch.nn import Module, Sequential
from torch.nn import Linear, Embedding, Flatten, Dropout, Conv1d, MaxPool1d, Module
from torch.nn import LSTM, ReLU, Linear, Dropout, Module
from torch.nn import Module
from torch.optim import RMSprop

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim 
import torch.nn as nn
import torch.optim as optim
import torchtext; torchtext.disable_torchtext_deprecation_warning()
import torch.nn.functional as F
import pickle
n_jobs = -1 # This parameter conrols the parallel processing. -1 means using all processors.
random_state = 42 # This parameter controls the randomness of the data. Using some int value to get same results everytime this code is run.
max_len = 500


class TFIDF:
    def __init__(self, file_name : str, product_name: str):
        self.file_name = file_name
        self.product_name= product_name
        
    def read_data(self):
        df = pd.read_excel(os.path.join(Path(__file__).parent.parent,"data",self.file_name+".xlsx"))
        return df
    
    def select_product(self):
        df = self.read_data()
        df_product = df[df['product']==self.product_name]# .head(2000)
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
        df = self.process_select_product()
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
        plt.savefig(os.path.join(Path(__file__).parent.parent,"output","rating",'ratings_distribution_'+self.product_name+'.png'))
        
        return df
        
    def create_sentiment_var(self):
        df = self.process_select_product() 
        
        # #### Removing the neutral reviews
        df_sentiment = df[df['rating']!= 3]
        df_sentiment['sentiment'] = df_sentiment['rating'].apply(lambda rating : +1 if rating > 3 else -1)
        
        #df_sentiment = df[df['rating']!= 0]
        #df_sentiment['sentiment'] = df_sentiment['rating']
        
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
        plt.savefig(os.path.join(Path(__file__).parent.parent,"output","sentiment",'sentiment_distribution_'+self.product_name+'.png'))
        
        #Positive vs Negative reviews
        df_sentiment_droppedna = df_sentiment.dropna()  
        #Positive vs Negative reviews
        positive = df_sentiment_droppedna[df_sentiment_droppedna["sentiment"] == 1].dropna()
        negative = df_sentiment_droppedna[df_sentiment_droppedna["sentiment"] == 0].dropna()
        return df_sentiment_droppedna, positive, negative
    
    def word_map(self):
        _, positive, negative = self.create_sentiment_var()
        stopwords_set = set(stopwords.words('english'))
        stopwords_set.update(["br", "href"])

        fig, axs = plt.subplots(1, 2, figsize=(12, 6))  # Adjust figsize as needed
        
        # Generate and save positive sentiment word cloud
        text_positive = " ".join(review for review in positive["review_cleaning"])
        wordcloud_positive = WordCloud(stopwords=stopwords_set).generate(text_positive)
        axs[0].imshow(wordcloud_positive, interpolation='bilinear')
        axs[0].axis("off")
        axs[0].set_title('Positive Sentiment')

        # Generate and save negative sentiment word cloud
        text_negative = " ".join(review for review in negative["review_cleaning"])
        wordcloud_negative = WordCloud(stopwords=stopwords_set).generate(text_negative)
        axs[1].imshow(wordcloud_negative, interpolation='bilinear')
        axs[1].axis("off")
        axs[1].set_title('Negative Sentiment')

        plt.tight_layout()
        plt.savefig(os.path.join(Path(__file__).parent.parent, "output","wordmap", f'{self.product_name}_combined_wordclouds.png'))
        return "Graphic saved in output folder"
    def split_input(self):
        df, _, _ = self.create_sentiment_var()
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(df['review_cleaning'], df['sentiment'], test_size=0.2, random_state=42)
        # Create a TF-IDF vectorizer
        vectorizer = TfidfVectorizer(stop_words='english')
        # Fit and transform the training data
        X_train_tfidf = vectorizer.fit_transform(X_train)
        # Transform the testing data
        X_test_tfidf = vectorizer.transform(X_test)
        return X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer
    
class CustomDataset:
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        review = self.df[idx].toarray()[0]  # Convert sparse matrix to dense array
        sentiment = ...  # Assuming you have a sentiment array or series

        # Convert review to tensor
        review = torch.tensor([ord(c) for c in review])
        
        if self.transform:
            review = self.transform(review)
        
        return review, torch.tensor(sentiment)
    
def create_datasets_and_loaders(X_train_tfidf, y_train, X_test_tfidf, y_test, batch_size=32, shuffle=True):
    transform = TfidfVectorizer()

    train_dataset = CustomDataset(X_train_tfidf, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

    val_dataset = CustomDataset(X_test_tfidf, y_test, transform=transform)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)

    return train_loader, val_loader
    
   
    
class CNNmodel(nn.Module):
        def __init__(self, num_classes, last_units=64, conv_kernel_size=5, dropout=0.3):
            super(CNNmodel, self).__init__()
            self.conv1 = nn.Conv1d(1, 8, kernel_size=conv_kernel_size, stride=1)
            self.conv2 = nn.Conv1d(8, 16, kernel_size=conv_kernel_size, stride=1)
            self.fc1 = nn.Linear(16 * 56, last_units)  
            self.dropout = nn.Dropout(p=dropout)
            self.fc2 = nn.Linear(last_units, num_classes)

        def forward(self, x):
            x = x.unsqueeze(1)  # Add a channel dimension
            x = F.relu(self.conv1(x))  
            x = F.max_pool1d(x, kernel_size=2, stride=2)
            x = F.relu(self.conv2(x)) 
            x = F.max_pool1d(x, kernel_size=2, stride=2)
            x = x.view(-1, 16 * 56)
            x = self.dropout(x)
            x = F.sigmoid(self.fc1(x)) 
            x = self.fc2(x)
            return x
        
def train_Model(model, train_loader, criterion, optimizer, num_epochs=5):
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.to(device)
            
            for epoch in range(num_epochs):
                model.train()
                running_loss = 0.0
                for review, sentiment in train_loader:
                    reviews = reviews.to(device)  
                    sentiments = sentiments.to(device)
                    optimizer.zero_grad()
                    outputs = model(reviews)
                    loss = criterion(outputs, sentiments)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item() * reviews.size(0)
                
                epoch_loss = running_loss / len(train_loader.dataset)
                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")    
        

def evaluate_model(self, X_test_tfidf, y_test, model):
        # Make predictions
        y_pred = model.predict(X_test_tfidf)
        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        return accuracy, precision, recall, f1

def main_function(instance):
    X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer = instance.split_input()
    train_loader, test_loader = create_datasets_and_loaders(X_train_tfidf, y_train, X_test_tfidf, y_test, batch_size=32, shuffle=True)
    model = CNNmodel(num_classes=2, last_units=64, conv_kernel_size=5, dropout=0.1)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    train_Model(model, train_loader, criterion, optimizer, num_epochs=10)
    evaluate_model(model, test_loader)

    with open(os.path.join(Path(__file__).parent.parent, "output", "model", f'{instance.product_name}_model.pkl'), 'wb') as f:
        pickle.dump(model, f)

    print("Model saved in output folder")

# Create an instance of the TFIDF class
instance = TFIDF("df_contact","Twitter_Data naive bayes")
main_function(instance)