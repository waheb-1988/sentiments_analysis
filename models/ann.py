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
from tensorflow.keras.models import Sequential, Model # Import Model from tensorflow.keras.models
from tensorflow.keras.layers import Dense, Embedding, Flatten, Dropout, Conv1D, GlobalMaxPooling1D, Input
from tensorflow.keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
# Basic packages
import pandas as pd 
import numpy as np
import re
import collections
import matplotlib.pyplot as plt

# Packages for data preparation
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

# Packages for modeling
from keras import models
from keras import layers
from keras import regularizers


n_jobs = -1 # This parameter conrols the parallel processing. -1 means using all processors.
random_state = 42 # This parameter controls the randomness of the data. Using some int value to get same results everytime this code is run.
max_len = 500
class ANN:
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
        df['review_cleaning'] = df['review_text'].astype(str).apply(ANN.preprocess_text)
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
        #print(os.path.join(Path(__file__).parent.parent,"output",'ratings_distribution_'+self.product_name+'.png'))
        plt.savefig(os.path.join(Path(__file__).parent.parent,"output","rating",'ratings_distribution_'+self.product_name+'.png'))

        return df

    def create_sentiment_var(self):
        df = self.process_select_product()

        # #### Removing the neutral reviews
        df_sentiment = df[df['rating'] != 3]
        df_sentiment['sentiment'] = df_sentiment['rating'].apply(lambda rating : +1 if rating > 3 else -1)

        #df_sentiment = df[df['rating'] != 0]
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
        return df_sentiment_droppedna , positive, negative

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




    def split_input(self): ############# Start of the new changing
        df, _ , _ = self.create_sentiment_var()
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(df['review_cleaning'], df['sentiment'], test_size=0.2, random_state=42)
        # Create a TF-IDF vectorizer
        vectorizer = TfidfVectorizer(max_features=5000)
        # Fit the vectorizer to the training data
        vectorizer.fit(X_train)
        # Transform the training and testing data into TF-IDF vectors
        X_train_tfidf = vectorizer.transform(X_train).toarray()
        X_test_tfidf = vectorizer.transform(X_test).toarray()
        return X_train_tfidf,X_test_tfidf , y_train ,y_test

    @staticmethod
    def loss(y_true, y_pred):
        pre = round(precision_score(y_true, y_pred),2)
        rec = round(recall_score(y_true, y_pred),2)
        f1 = round(f1_score(y_true, y_pred),2)
        loss = round(log_loss(y_true, y_pred),2)
        acc = round(accuracy_score(y_true, y_pred),4)

        return pre, rec, f1, loss, acc
    NB_WORDS = 10000  # Parameter indicating the number of words we'll put in the dictionary
    VAL_SIZE = 1000  # Size of the validation set
    NB_START_EPOCHS = 20  # Number of epochs we usually start to train with
    BATCH_SIZE = 512  # Size of the batches used in the mini-batch gradient descent
    @staticmethod    
    def one_hot_seq(seqs, nb_features = NB_WORDS):
        ohs = np.zeros((len(seqs), nb_features))
        for i, s in enumerate(seqs):
            ohs[i, s] = 1.
        return ohs
    
    def ann(self):
        
        df, _ , _ = self.create_sentiment_var()
        tokenize = Tokenizer(num_words=1000)
        X_train, X_test, y_train, y_test = train_test_split(df['review_cleaning'], df['sentiment'], test_size=0.2, random_state=42)
        tk = Tokenizer(num_words=10000,
               filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
               lower=True,
               split=" ")
        tk.fit_on_texts(X_train)
        X_train_seq = tk.texts_to_sequences(X_train)
        X_test_seq = tk.texts_to_sequences(X_test)
        X_train_oh = ANN.one_hot_seq(X_train_seq)
        X_test_oh = ANN.one_hot_seq(X_test_seq)
        le = LabelEncoder()
        y_train_le = le.fit_transform(y_train)
        y_test_le = le.transform(y_test)
        y_train_oh = to_categorical(y_train_le)
        y_test_oh = to_categorical(y_test_le)
        X_train_rest, X_valid, y_train_rest, y_valid = train_test_split(X_train_oh, y_train_oh, test_size=0.1, random_state=37)

        assert X_valid.shape[0] == y_valid.shape[0]
        assert X_train_rest.shape[0] == y_train_rest.shape[0]
        base_model = models.Sequential()
        base_model.add(layers.Dense(64, activation='relu', input_shape=(200,)))
        base_model.add(layers.Dense(64, activation='relu'))
        base_model.add(layers.Dense(1, activation='sigmoid'))
        base_model.summary()
        
        return base_model,X_train_rest,y_train_rest,X_valid, y_valid
    def deep_model(self):
        model,X_train_rest,y_train_rest,X_valid, y_valid = self.ann()
        model.compile(optimizer='adam'
                  , loss='binary_crossentropy'
                  , metrics=['accuracy'])
    
        history = model.fit(X_train_rest
                        , y_train_rest
                        , epochs=20
                        , batch_size=50
                        , validation_data=(X_valid, y_valid)
                        , verbose=0)
        
        return history
    
    def eval_metric(self, metric_name):
        history= self.deep_model()
        metric = history.history[metric_name]
        val_metric = history.history['val_' + metric_name]

        e = range(1, 15 + 1)

        plt.plot(e, metric, 'bo', label='Train ' + metric_name)
        plt.plot(e, val_metric, 'b', label='Validation ' + metric_name)
        plt.legend()
        plt.show()
    
instance = ANN("df_contact","Womens Clothing E-Commerce Reviews") # Twitter_Data naive bayes
# # #data = instance.read_data()
df = instance.deep_model()


# df1 = instance.eval_metric( 'loss')
# print(df1)