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
from TF_IDF import TFIDF


class CustomDataset:
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        review = self.df.loc[idx, 'review_cleaning']
        sentiment = self.df.loc[idx, 'sentiment']
    
       # Convert review to tensor
        review = torch.tensor([ord(c) for c in review])
        
        if self.transform:
            review = self.transform(review)
        
        return review, torch.tensor(sentiment)
  
def create_datasets_and_loaders(X_train, X_test, batch_size=32, shuffle=True):
        transform = transforms.Compose([transforms.ToTensor()])
        train_dataset = CustomDataset(X_train, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

        val_dataset = CustomDataset('val_annotations.csv', 'val_images', transform=transform)
        val_loader = DataLoader(X_test, batch_size=batch_size, shuffle=shuffle)

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

           

        # Train the model
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

                
        # Evaluate the model
def evaluate_model(model, val_loader):
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.to(device)
            model.eval()
            total_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for reviews, sentiments in val_loader:
                    reviews, sentiments = reviews.to(device), sentiments.to(device)
                    outputs = model(reviews)
                    loss = criterion(outputs, sentiments)
                    total_loss += loss.item() * reviews.size(0)
                    _, predicted = torch.max(outputs.data, 1)
                    total += sentiments.size(0)
                    correct += (predicted == sentiments).sum().item()
            val_loss = total_loss / len(val_loader.dataset)
            val_accuracy = correct / total
            print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

            
            
instance = TFIDF("df_contact","Womens Clothing E-Commerce Reviews") # Twitter_Data naive bayes
X_train_tfidf,X_test_tfidf , y_train ,y_test=instance.split_input()
train_loader, val_loader = create_datasets_and_loaders(X_train_tfidf, X_test_tfidf, batch_size=32, shuffle=True)
model = CNNmodel(num_classes=2, last_units=64, conv_kernel_size=5, dropout=0.1)   
# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
train_Model(model, train_loader, criterion, optimizer, num_epochs=10)
evaluate_model(model, val_loader)