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
from sklearn.feature_extraction.text import CountVectorizer
n_jobs = -1 # This parameter conrols the parallel processing. -1 means using all processors.
random_state = 42 # This parameter controls the randomness of the data. Using some int value to get same results everytime this code is run.
n = 1 # N-gram value
min_samples = 1000 # Minimum number of samples for training
class BAGOFW:
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
        df['review_cleaning'] = df['review_text'].astype(str).apply(BAGOFW.preprocess_text)
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
        plt.savefig(os.path.join(Path(__file__).parent.parent,"output","sentiment",'sentiment_distribution_'+f'{self.product_name}_.png'))
        
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
        X_train, X_test, y_train, y_test = train_test_split(df['review_text'], df['sentiment'], test_size=0.2, random_state=42)
        # Define the N-gram model
        vectorizer = CountVectorizer(ngram_range=(n, n))
    
        # Fit the vectorizer to the training data
        vectorizer.fit(X_train)
        # Transform the training and testing data into TF-IDF vectors
        X_train_tfidf = vectorizer.transform(X_train).toarray()
        X_test_tfidf = vectorizer.transform(X_test).toarray()
        return X_train_tfidf,X_test_tfidf , y_train ,y_test 
    @staticmethod
    def models():
        # Step 5− Training the Model
        models = {'LogisticRegression': LogisticRegression(), 'KNeighborsClassifier':KNeighborsClassifier(), 'SVC':SVC()
             , 'GaussianNB': GaussianNB(), 'Perceptron': Perceptron(), 'LinearSVC': LinearSVC(),
              'SGDClassifier':SGDClassifier(),  
              'DecisionTreeClassifier' : DecisionTreeClassifier(),'RandomForestClassifier':RandomForestClassifier(),"XGBClassifier":XGBClassifier(),"CatBoostClassifier":CatBoostClassifier()
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
        X_train_tfidf,X_test_tfidf , y_train ,y_test  = self.split_input()
        models= BAGOFW.models()
        for name, model in models.items():
            model.fit(X_train_tfidf,y_train )
            pre, rec, f1, loss, acc=BAGOFW.loss(y_test, model.predict(X_test_tfidf))
            #print('-------{h}-------'.format(h=name))
            #print(pre, rec, f1, loss, acc)
            results.append([name, pre, rec, f1, loss, acc])
        df = pd.DataFrame(results, columns=['NAME', 'pre', 'rec', 'f1', 'loss', 'acc'])
        #df.set_index('NAME', inplace=True)    
        df.sort_values(by=['acc'], ascending=False, inplace=True) 
        return df
    
    # TODO Improv with good output
    def hyper_tun(self):
        models= BAGOFW.models()
        X_train_tfidf,X_test_tfidf , y_train ,y_test  = self.split_input()
        for name,model in models.items():
            if name == "DecisionTreeClassifier":
                # Hyperparameter Optimization
                parameters = {'max_features': ['log2', 'sqrt','auto'], 
                            'criterion': ['entropy', 'gini'],
                            'max_depth': [2, 3, 5, 10, 50], 
                            'min_samples_split': [2, 3, 50, 100],
                            'min_samples_leaf': [1, 5, 8, 10]
                            }
                grid_obj = GridSearchCV(model, parameters)
                grid_obj = grid_obj.fit(X_train_tfidf, y_train)
                clf = grid_obj.best_estimator_
                # Train the model using the training sets 
                clf.fit(X_train_tfidf, y_train)
                pre, rec, f1, loss, acc=BAGOFW.loss(y_test, clf.predict(X_test_tfidf))
                print('-------{h}-------'.format(h=name))
                print(pre, rec, f1, loss, acc)
                return pre, rec, f1, loss, acc
            
        # TODO Improv with good output
    def bagging_predictions(self,estimator):
            X_train_tfidf,X_test_tfidf , y_train ,y_test  = self.split_input()
            """
            I/P
            estimator: The base estimator from which the ensemble is grown.
            O/P
            br_y_pred: Predictions on test data for the base estimator.
            
            """
            regr = BaggingRegressor(base_estimator=estimator,
                                    n_estimators=10,
                                    max_samples=1.0,
                                    bootstrap=True, # Samples are drawn with replacement
                                    n_jobs= n_jobs,
                                    random_state=random_state).fit(X_train_tfidf, y_train)

            br_y_pred = regr.predict(X_test_tfidf)

            rmse_val = mean_squared_error(y_test, br_y_pred, squared= False) # squared= False > returns Root Mean Square Error   

            print(f'RMSE for base estimator {regr.base_estimator_} = {rmse_val}\n')
            ### 
            instance = BAGOFW("df_contact","jumia_reviews_df_multi_page")
            # #data = instance.read_data()
            # X_train_tfidf,X_test_tfidf , y_train ,y_test  = instance.split_input()


            # predictions = np.column_stack((instance.bagging_predictions(DecisionTreeClassifier()),
            #                               instance.bagging_predictions(KNeighborsClassifier()),
            #                               instance.bagging_predictions(LogisticRegression()),
            #                               instance.bagging_predictions(RandomForestClassifier())))
            # print(f"Bagged predictions shape: {predictions.shape}")
            # y_pred = np.mean(predictions, axis=1)

            # print("Aggregated predictions (y_pred) shape", y_pred.shape)

            # rmse_val = mean_squared_error(y_test, y_pred, squared= False) # squared= False > returns Root Mean Square Error  
            # models_scores = [] 
            # models_scores.append(['Bagging', rmse_val])

            # print(f'\nBagging RMSE= {rmse_val}')
            return br_y_pred
            
                
                
            
    
####### Test Data
   
instance = BAGOFW("df_contact","jumia_reviews_df_multi_page")
# # #data = instance.read_data()
# df = instance.data_analysis_report()
# df = instance.create_sentiment_var()
# df = instance.word_map()
df = instance.train_model()
print(df)

