from typing import List

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer


from dataset import ABSADataset
from model import BertClassifier

from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Classifier:
    """
    The Classifier: complete the definition of this class template by providing a constructor (i.e. the
    __init__() function) and the 2 methods train() and predict() below. Please do not change the signature
    of these methods
     """

    # complete the classifier class below

    def __init__(self):
        """
        This should create and initilize the model. Does not take any arguments.
        """
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.bert = 'bert-base-uncased'

        self.tokenizer = BertTokenizer.from_pretrained(self.bert)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.epochs = 10

    def train(self, train_filename: str, dev_filename: str, device: torch.device):
        """
        Trains the classifier model on the training set stored in file trainfile
        PLEASE:
          - DO NOT CHANGE THE SIGNATURE OF THIS METHOD
          - PUT THE MODEL and DATA on the specified device! Do not use another device
          - DO NOT USE THE DEV DATA AS TRAINING EXAMPLES, YOU CAN USE THEM ONLY FOR THE OPTIMIZATION
         OF MODEL HYPERPARAMETERS
        """
        train_dataset = ABSADataset(
            train_filename, self.tokenizer)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        dev_dataset = ABSADataset(dev_filename, self.tokenizer)
        dev_loader = DataLoader(dev_dataset, batch_size=32, shuffle=True)
        progress_bar = tqdm(total=self.epochs, desc='Training Progress')
        
        num_classes = train_dataset.df['label'].unique().shape[0]
        self.model = BertClassifier(self.bert, dr_rate=0.3, num_classes=num_classes)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)
        

        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []
        best_val_acc = 0
        self.total_steps = len(train_loader) * self.epochs

        self.scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            total_iters=self.total_steps
        )

        self.model = self.model.to(device)
        for _ in range(self.epochs):
            train_loss, train_acc = self.train_for_one_epoch(
                train_loader, device)
            val_loss, val_acc = self.evaluate(dev_loader, device)
            progress_bar.set_postfix(
                {'train_loss': train_loss, 'train_acc': train_acc, 'val_loss': val_loss, 'val_acc': val_acc})
            progress_bar.update(1)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            # Save the best  model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), 'model.pth')
                
        progress_bar.close()

        

    def train_for_one_epoch(self, train_loader: DataLoader, device: torch.device):
        loss_epoch = 0
        correct = 0
        total = 0
        self.model.train()
        for i, data in enumerate(train_loader):
            self.optimizer.zero_grad()
            
            ids = data['input_ids'].to(device)
            mask = data['attention_mask'].to(device)
            targets = data['label'].to(device)
        
            outputs = self.model(ids, mask)
            loss = self.criterion(outputs, targets)
            loss.backward()
            
            
            self.optimizer.step()
            self.scheduler.step()
            
            loss_epoch+=loss.item()
            total += targets.size(0)
            
            _, predicted = torch.max(outputs, 1)
            correct += torch.sum(predicted == targets).item()
            

        return loss_epoch/total, correct/total

    def evaluate(self, dev_loader: DataLoader, device: torch.device):
        correct = 0
        total = 0
        loss_epoch = 0

        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(dev_loader):
                
                ids = data['input_ids'].to(device)
                mask = data['attention_mask'].to(device)
                targets = data['label'].to(device)
                outputs = self.model(ids, mask)
                loss = self.criterion(outputs, targets)
                loss_epoch += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                
                total += targets.size(0)
                correct += torch.sum(predicted == targets).item()
        return loss_epoch/total, correct/total

    def predict(self, data_filename: str, device: torch.device) -> List[str]:
        """Predicts class labels for the input instances in file 'datafile'
        Returns the list of predicted labels
        PLEASE:
          - DO NOT CHANGE THE SIGNATURE OF THIS METHOD
          - PUT THE MODEL and DATA on the specified device! Do not use another device
        """
        label_dict = {0: 'negative', 1: 'neutral', 2: 'positive'}
        test_dataset = ABSADataset(data_filename, self.tokenizer)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        self.model.load_state_dict(torch.load('model.pth', map_location=device))
        self.model = self.model.to(device)
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                ids = data['input_ids'].to(device)
                mask = data['attention_mask'].to(device)
                outputs = self.model(ids, mask)
                
                logits = F.softmax(outputs, dim=1)
                logits = logits.detach().cpu().numpy()
                predicted = np.argmax(logits, axis=1)
                predictions.extend([label_dict[p.item()] for p in predicted])
        return np.array(predictions)
