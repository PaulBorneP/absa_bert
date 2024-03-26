
import torch.nn as nn
from transformers import BertModel

class BertClassifier:
    from transformers import BertModel
    import torch.nn as nn

    class BertClassifier(nn.Module):
        def __init__(self, bert_name: str, hidden_size: int = 768, num_classes: int = 3, dr_rate: float = None):
            """
            Initialize the BertClassifier model.

            Args:
                bert_name (str): The name of the pre-trained BERT model.
                hidden_size (int): The size of the hidden layer in the classifier. Default is 768.
                num_classes (int): The number of classes for classification. Default is 3.
                dr_rate (float): The dropout rate for the dropout layer. Default is None.
            """
            super(BertClassifier, self).__init__()
            self.bert = BertModel(bert_name)
            self.dr_rate = dr_rate
            self.classifier = nn.Linear(hidden_size, num_classes)
            if dr_rate:
                self.dropout = nn.Dropout(p=dr_rate)

    def forward(self, x):
        x = self.bert(x)[1]
        if self.dr_rate:
            x = self.dropout(x)
        return self.classifier(x)
    