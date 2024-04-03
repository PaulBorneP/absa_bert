import numpy as np
import pandas as pd
from keras.utils import to_categorical
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import torch
from typing import List

from word2vec import Word2Vec


class Classifier:
    """Simple approach: Word2Vec embedding and Logistic Regression"""

    def __init__(self):
        self.columns = ['Polarity', 'Aspect_Category', 'Specific_Target_Aspect_Term', 'Character_Offset', 'Sentence']
        self.polarity_encoder = LabelEncoder()
        self.category_encoder = LabelEncoder()
        self.w2v = Word2Vec('../resources/crawl-300d-200k.vec', 150000)
        self.logreg = LogisticRegression(C=1, solver='liblinear', multi_class='ovr') 


    def train(self, train_filename: str, dev_filename: str, device: torch.device):
        """
        Trains the classifier model on the training set stored in file train_filename
        """
        train_df = pd.read_csv(train_filename, sep='\t', names=self.columns)
        # Encode sentences
        X_train = self.w2v.encode_parse(train_df.Sentence, False)

        # Category info
        train_categories_integer = self.category_encoder.fit_transform(train_df.Aspect_Category)
        train_categories_dummy = to_categorical(train_categories_integer) # classes 
        X_train = np.hstack((X_train, train_categories_dummy))

        # Get training labels
        y_train = self.polarity_encoder.fit_transform(train_df.Polarity)

        # Fit 
        self.logreg.fit(X_train, y_train)

    def predict(self, data_filename: str, device: torch.device) -> List[str]:
        """Predicts class labels for the input instances in file data_filename
        Returns the list of predicted labels
        """
        test_df = pd.read_csv(data_filename, sep='\t', names=self.columns)

        # Encode test words
        X_test = self.w2v.encode_parse(test_df.Sentence, False)

        # Category info
        test_categories_integer = self.category_encoder.transform(test_df.Aspect_Category)
        test_categories_dummy = to_categorical(test_categories_integer)
        X_test = np.hstack((X_test, test_categories_dummy))

        return list(self.polarity_encoder.inverse_transform(self.logreg.predict(X_test)))