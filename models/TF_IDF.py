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

    def solve_unbalance(self):
        X_train_tfidf,X_test_tfidf , y_train ,y_test = self.split_input()
        sm = ADASYN(random_state=777)
        X_res, y_res = sm.fit_resample(X_train_tfidf, y_train)
        return X_res, X_test_tfidf, y_res ,y_test



    @staticmethod
    def models():
        # Step 5− Training the Model
        models = {'LogisticRegression': LogisticRegression(), 'KNeighborsClassifier':KNeighborsClassifier(),
              'DecisionTreeClassifier' : DecisionTreeClassifier(),'RandomForestClassifier':RandomForestClassifier(),
              'AdaBoostClassifier': AdaBoostClassifier(),
              'XGBClassifier': XGBClassifier(),
              'CatBoostClassifier': CatBoostClassifier()
        }
        return models

    @staticmethod
    def loss(y_true, y_pred):
        pre = round(precision_score(y_true, y_pred),2)
        rec = round(recall_score(y_true, y_pred),2)
        f1 = round(f1_score(y_true, y_pred),2)
        loss = round(log_loss(y_true, y_pred),2)
        acc = round(accuracy_score(y_true, y_pred),4)

        return pre, rec, f1, loss, acc



    def train_model(self):
        results = []
        results1 = []
        X_train_tfidf,X_test_tfidf , y_train ,y_test  = self.split_input()
        models= TFIDF.models()
        for name, model in models.items():
            m1=model.fit(X_train_tfidf,y_train )
            pre, rec, f1, loss, acc=TFIDF.loss(y_test, m1.predict(X_test_tfidf))
            print('-------{h}-------'.format(h=name))
            print(pre, rec, f1, loss, acc)
            results.append([name, pre, rec, f1, loss, acc])
        df = pd.DataFrame(results, columns=['NAME', 'pre', 'rec', 'f1', 'loss', 'acc'])
        df.set_index('NAME', inplace=True)
        df.sort_values(by=['acc'], ascending=False, inplace=True)
        return  df
    
    def train_model_unb(self):
        results1 = []
        X_train_tfidf1,X_test_tfidf1 , y_train1 ,y_test1  = self.solve_unbalance()
        models= TFIDF.models()
        for name, model in models.items():
            m2=model.fit(X_train_tfidf1,y_train1 )
            pre1, rec1, f11, loss1, acc1=TFIDF.loss(y_test1, m2.predict(X_test_tfidf1))
            print('-------{h}-------'.format(h=name))
            print(pre1, rec1, f11, loss1, acc1)
            results1.append([name, pre1, rec1, f11, loss1, acc1])
        df1 = pd.DataFrame(results1, columns=['NAME', 'pre', 'rec', 'f1', 'loss', 'acc'])
        df1.sort_values(by=['acc'], ascending=False, inplace=True)
        return  df1
    
    def all_train(self):
        df=self.train_model()
        df1= self.train_model_unb()
        return df,df1
        
    # TODO Improv with good output
    def hyper_tun(self):
        models= TFIDF.models()
        X_train_tfidf,X_test_tfidf , y_train ,y_test  = self.split_input()
        results = []
        for name, model in models.items():
            if name == "DecisionTreeClassifier": ### Change name
                parameters1 = {'max_features': ['log2', 'sqrt', 'auto'],
                              'criterion': ['entropy', 'gini'],
                              'max_depth': [2, 3, 5, 10, 50],
                              'min_samples_split': [2, 3, 50, 100],
                              'min_samples_leaf': [1, 5, 8, 10]} ### Change parametrers
                grid_obj = GridSearchCV(model, parameters1, cv=5, n_jobs=-1) ### Change parametrers
                grid_obj = grid_obj.fit(X_train_tfidf, y_train)
                clf = grid_obj.best_estimator_

                clf.fit(X_train_tfidf, y_train)
                pre, rec, f1, loss, acc = TFIDF.loss(y_test, clf.predict(X_test_tfidf))
                results.append({'Model': name, 'Precision': pre, 'Recall': rec, 'F1 Score': f1, 'Log Loss': loss, 'Accuracy': acc})
                print(results)
            elif name == "RandomForestClassifier":
                parameters2 = {'n_estimators': [25, 50, 100, 150],
                              'max_features': ['sqrt', 'log2', None],
                              'max_depth': [3, 6, 9],
                              'max_leaf_nodes': [3, 6, 9]}
                grid_obj = GridSearchCV(model, parameters2, cv=5, n_jobs=-1)
                grid_obj = grid_obj.fit(X_train_tfidf, y_train)
                clf = grid_obj.best_estimator_
                clf.fit(X_train_tfidf, y_train)
                pre, rec, f1, loss, acc = TFIDF.loss(y_test, clf.predict(X_test_tfidf))
                results.append({'Model': name, 'Precision': pre, 'Recall': rec, 'F1 Score': f1, 'Log Loss': loss, 'Accuracy': acc})
            elif name == "AdaBoostClassifier":
                parameters3 = {'n_estimators': [50, 100, 200],
                          'learning_rate': [0.1, 1, 10]}
                grid_obj = GridSearchCV(model, parameters3, cv=5, n_jobs=-1)
                grid_obj = grid_obj.fit(X_train_tfidf, y_train)
                clf = grid_obj.best_estimator_
                clf.fit(X_train_tfidf, y_train)
                pre, rec, f1, loss, acc = TFIDF.loss(y_test, clf.predict(X_test_tfidf))
                results.append({'Model': name, 'Precision': pre, 'Recall': rec, 'F1 Score': f1, 'Log Loss': loss, 'Accuracy': acc})

            elif name == "CatBoostClassifier":
                parameters4 = {'iterations': [50, 100, 200],
                          'learning_rate': [0.01, 0.1, 1]}
                grid_obj = GridSearchCV(model, parameters4, cv=5, n_jobs=-1)
                grid_obj = grid_obj.fit(X_train_tfidf, y_train)
                clf = grid_obj.best_estimator_

                clf.fit(X_train_tfidf, y_train)
                pre, rec, f1, loss, acc = TFIDF.loss(y_test, clf.predict(X_test_tfidf))
                results.append({'Model': name, 'Precision': pre, 'Recall': rec, 'F1 Score': f1, 'Log Loss': loss, 'Accuracy': acc})

            elif name == "XGBoostClassifier":
                parameters5 = {'n_estimators': [50, 100, 200],
                          'learning_rate': [0.01, 0.1, 1],
                          'max_depth': [3, 6, 9],
                          'objective': ['binary:logistic']}
                grid_obj = GridSearchCV(model, parameters5, cv=5, n_jobs=-1)
                grid_obj = grid_obj.fit(X_train_tfidf, y_train)
                clf = grid_obj.best_estimator_

                clf.fit(X_train_tfidf, y_train)
                pre, rec, f1, loss, acc = TFIDF.loss(y_test, clf.predict(X_test_tfidf))
                results.append({'Model': name, 'Precision': pre, 'Recall': rec, 'F1 Score': f1, 'Log Loss': loss, 'Accuracy': acc})

            elif name == "SVM":
                parameters6 = {'kernel': ['linear', 'rbf'],
                          'C': [0.1, 1, 10]}
                grid_obj = GridSearchCV(model, parameters6, cv=5, n_jobs=-1)
                grid_obj = grid_obj.fit(X_train_tfidf, y_train)
                clf = grid_obj.best_estimator_

                clf.fit(X_train_tfidf, y_train)
                pre, rec, f1, loss, acc = TFIDF.loss(y_test, clf.predict(X_test_tfidf))
                results.append({'Model': name, 'Precision': pre, 'Recall': rec, 'F1 Score': f1, 'Log Loss': loss, 'Accuracy': acc})

            elif name == "KNeighborsClassifier":
                parameters7 = {'n_neighbors': [2, 3, 5, 10],
                          'weights': ['uniform', 'distance'],
                          'algorithm': ['auto', 'ball_tree', 'kd_tree']}
                grid_obj = GridSearchCV(model, parameters7, cv=5, n_jobs=-1)
                grid_obj = grid_obj.fit(X_train_tfidf, y_train)
                clf = grid_obj.best_estimator_

                clf.fit(X_train_tfidf, y_train)
                pre, rec, f1, loss, acc = TFIDF.loss(y_test, clf.predict(X_test_tfidf))
                results.append({'Model': name, 'Precision': pre, 'Recall': rec, 'F1 Score': f1, 'Log Loss': loss, 'Accuracy': acc})

        return pd.DataFrame(results)


        # TODO Improv with good output
    # def bagging_predictions(self):
    #         result=[]
    #         X_train_tfidf,X_test_tfidf , y_train ,y_test  = self.split_input()
    #         models= TFIDF.models()
    #         for name, model in models.items():

    #             bag_model = BaggingClassifier(
    #                     base_estimator=model,
    #                     n_estimators=100,
    #                     max_samples=0.8,
    #                     oob_score=True,
    #                     random_state=0
    #                 )
    #             bag_model.fit(X_train_tfidf, y_train)
    #             bag_model.oob_score_
    #             score = bag_model.score(X_test_tfidf, y_test)
    #             result.append({'Model': name, 'score': score})
    #             df=pd.DataFrame(result)
    #             df.sort_values(by=['score'], ascending=False, inplace=True)
    #         return df



    # @staticmethod
    # def ann_model(max_len):

    #     # Define the input layer
    #     inputs = Input(name='inputs', shape=[max_len])

    #     # Add an embedding layer
    #     layer = Embedding(input_dim=2000, output_dim=50, input_length=max_len)(inputs)

    #     # Flatten the embedding output to feed it into dense layers
    #     layer = Flatten()(layer)

    #     # Add a dense layer with ReLU activation
    #     layer = Dense(256, activation='relu', name='FC1')(layer)

    #     # Apply dropout
    #     layer = Dropout(0.5)(layer)

    #     # Add another dense layer with ReLU activation
    #     layer = Dense(128, activation='relu', name='FC2')(layer)

    #     # Apply dropout again
    #     layer = Dropout(0.5)(layer)

    #     # Add the output layer with sigmoid activation
    #     layer = Dense(1, activation='sigmoid', name='output')(layer)

    #     # Define the model
    #     model = Model(inputs=inputs, outputs=layer)

    #     # Compile the model
    #     model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    #     return model

    # @staticmethod
    # def models_dl():
    #     # Step 5− Training the Model
    #     models = {'ANN': TFIDF.ann_model(max_len)
    #     }
    #     return models

    # def train_dl_model(self):
    #     results = []
    #     df, _ , _ = self.create_sentiment_var()
    #     X=df['review_cleaning'].copy()
    #     y=df['sentiment'].copy()
    #     print("here")
    #     print(y.unique())
    #     max_len = 100
    #     tok = Tokenizer(num_words=2000)
    #     tok.fit_on_texts(X)
    #     sequences = tok.texts_to_sequences(X)
    #     sequences_matrix = sequence.pad_sequences(sequences)
    #     X_train, X_test, Y_train, Y_test = train_test_split(sequences_matrix, y, test_size=0.3, random_state=2)
    #     #X_train_tfidf1,X_test_tfidf1 , y_train1 ,y_test1  = self.solve_unbalance()
    #     # Define the input layer
    #     # Define the input layer
    #     inputs = Input(name='inputs', shape=[max_len])

    # # Add an embedding layer
    #     layer = Embedding(input_dim=2000, output_dim=50)(inputs)

    #     # Flatten the embedding output to feed it into dense layers
    #     layer = Flatten()(layer)

    #     # Add a dense layer with ReLU activation
    #     layer = Dense(256, activation='relu', name='FC1')(layer)

    #     # Apply dropout
    #     layer = Dropout(0.5)(layer)

    #     # Add another dense layer with ReLU activation
    #     layer = Dense(128, activation='relu', name='FC2')(layer)

    #     # Apply dropout again
    #     layer = Dropout(0.5)(layer)

    #     # Add the output layer with sigmoid activation
    #     layer = Dense(1, activation='sigmoid', name='output')(layer)

    #     # Define the model
    #     model = Model(inputs=inputs, outputs=layer)

    #     # Compile the model
    #     model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    #     historyANN=model.fit(X_train,Y_train,batch_size=32,epochs=10, validation_split=0.12)
    #         #m2=model.fit(X_train_tfidf1,y_train1 )
    #     pre, rec, f1, loss, acc=TFIDF.loss(Y_test, model.predict(X_test))
    #         #pre1, rec1, f11, loss1, acc1=TFIDF.loss(y_test1, m2.predict(X_test_tfidf1))
    #         #print('-------{h}-------'.format(h=name))
    #         #print(pre, rec, f1, loss, acc)
    #     name ='ANN'
    #     results.append([name, pre, rec, f1, loss, acc])
    #         #results1.append([name, pre1, rec1, f11, loss1, acc1])
    #     df = pd.DataFrame(results, columns=['NAME', 'pre', 'rec', 'f1', 'loss', 'acc'])
    #     #df1 = pd.DataFrame(results1, columns=['NAME', 'pre', 'rec', 'f1', 'loss', 'acc'])
    #     #df.set_index('NAME', inplace=True)
    #     df.sort_values(by=['acc'], ascending=False, inplace=True)
    #     #df1.sort_values(by=['acc'], ascending=False, inplace=True)
    #     return df
####### Test Data

instance = TFIDF("df_contact","Womens Clothing E-Commerce Reviews") # Twitter_Data naive bayes
# # #data = instance.read_data()
# df = instance.data_analysis_report()
# df = instance.create_sentiment_var()
# df = instance.word_map()
# df = instance.create_sentiment_var()
df,df1=instance.all_train()
print(df)
# print("######################################")
print(df1)