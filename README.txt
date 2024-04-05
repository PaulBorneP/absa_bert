NLP Assignment: Aspect-Term Polarity Classification in Sentiment Analysis
Paul Borne--Pons, Emma Gauillard, Quentin Gopée, Maya Janvier


# Methodology Overview
The implemented classifier utilizes the BERT (Bidirectional Encoder Representations
from Transformers) model for aspect-based sentiment analysis (ABSA). ABSA involves
analyzing text to identify sentiment polarity towards specific aspects or entities
mentioned within the text. The classifier aims to accurately predict sentiment labels
(positive, neutral, or negative) for each aspect mentioned in the input text.

# Classifier Implementation Details
1. Data Preprocessing
The dataset is preprocessed using the ABSADataset class defined in dataset.py. This
class handles the loading and preprocessing of the dataset, including tokenization,
encoding, and padding of input sequences. The preprocessing includes transforming The
aspect_target into questions to make them more natural for the BERT model.

2. Model Architecture
The sentiment classification model is implemented in the BertClassifier class defined
in model.py. This class utilizes the pre-trained BERT model from the Hugging Face
Transformers library. The BERT model is fine-tuned for sentiment classification by
adding a classification layer on top. The classifier predicts sentiment labels based
on the contextualized representations learned by the BERT model.

3. Training and Evaluation
The Classifier class defined in classifier.py encapsulates the training, evaluation,
and prediction methods. During training, the model is trained on the training set while
optimizing model parameters using a specified loss function (cross-entropy loss).
The model's performance is evaluated on a separate development set to monitor training
progress and prevent overfitting.

# Script Execution
The tester.py script serves as the main entry point for running the training and
evaluation loops over multiple runs. It utilizes command-line arguments (parsed using
argparse) to specify the number of runs and the GPU device ID for execution. The script
handles reproducibility settings and GPU device selection based on the provided
arguments.

# Requirements
torch==2.1
transformers==4.34.1
tokenizers==0.14.1
datasets==2.14.5
scikit-learn==1.2.1
numpy==1.26.0
pandas==2.1.1

# Final Results
Model performance is evaluated using accuracy metrics calculated based on the predicted
labels compared to the ground truth labels. The implementation provides methods for
calculating accuracy on both the development set and, if available, the test set.

------------------------------
Accuracy on the dev set: 30.32
------------------------------

---------------------------------------------------------------------------------------

# Second Implementation
We also implemented a second method using word2vec and a logistic linear regression.
However this implementation uses packages not listed in the requirements of the
assigment.

To run it:
- Create a folder `resources`
- Download the 2 million word vectors trained on Common Crawl using fastText using the
  following link: 
  https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip
- Move the unziped file to `resources`
- install the requirements (listed below)
- cd to src and run teste_logreg.py

# Requirements:
torch==2.1
transformers==4.34.1
tokenizers==0.14.1
datasets==2.14.5
scikit-learn==1.2.1
numpy==1.26.0
pandas==2.1.1
spacy==3.7.4
inflect==7.2.0
pyspellchecker==0.8.1
keras==3.1.1
tensorflow==2.16.1
en-core-web-sm==3.7.1

------------------------------
Accuracy on the dev set: 80.85
------------------------------