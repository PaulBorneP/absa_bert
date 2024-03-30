from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import PreTrainedTokenizer


class ABSADataset(Dataset):
    import pandas as pd
    from typing import Any
    from transformers import PreTrainedTokenizer

        
    @staticmethod
    def polarity_to_label(polarity: str) -> int:
        """
        Convert the polarity (str) to a label {0,1,2}.
        0: negative
        1: neutral
        2: positive
        """
        if polarity == 'positive':
            return 2
        elif polarity == 'neutral':
            return 1
        elif polarity == 'negative':
            return 0
        else:
            raise ValueError('Polarity not found')
            
        
################ CHANGE THE ASPECT QUESTIONS #####################

#### maybe ask gpt to generate different questions  and try the results ####
    @staticmethod
    def aspect_questions(aspect: str) -> str:
        """
        Return the question for the aspect.
        """
    
        if aspect == 'AMBIENCE#GENERAL':
            question = "What do you think of the ambience ? "

        elif aspect == 'FOOD#QUALITY':
            question = "What do you think of the quality of the food ? "

        elif aspect == 'SERVICE#GENERAL':
            question = "What do you think of the service ? "

        elif aspect == 'FOOD#STYLE_OPTIONS':
            question = "What do you think of the food choices ? "

        elif aspect == 'DRINKS#QUALITY':
            question = "What do you think of the drinks? "

        elif aspect == 'RESTAURANT#MISCELLANEOUS' or aspect == 'RESTAURANT#GENERAL':
            question = "What do you think of the restaurant ? "

        elif aspect == 'LOCATION#GENERAL':
            question = 'What do you think of the location ? '

        elif aspect == 'DRINKS#STYLE_OPTIONS':
            question = "What do you think of the drink choices ? "

        elif aspect == 'RESTAURANT#PRICES' or aspect =='DRINKS#PRICES' or aspect == 'FOOD#PRICES':
            question = 'What do you think of the price of it ? '

        else:
            raise ValueError('Aspect not found')

        return question

    ################ CHANGE THE ASPECT QUESTIONS #####################
    def __init__(self, data_file: str, tokenizer: PreTrainedTokenizer, max_len: int) -> None:
        """
        Initialize the Dataset class.

        Args:
            data_file (str): The path to the data file.
            tokenizer (PreTrainedTokenizer): The tokenizer used for tokenization.
            max_len (int): The maximum length of the input sequence.

        Returns:
            None
        """
        self.df = pd.read_csv(data_file, sep='\t', header=None, names=['polarity','aspect', 'target', 'what?', 'text'],index_col=False)
        self.df['label'] = self.df['polarity'].apply(self.polarity_to_label)
        self.df['question'] = self.df['aspect'].apply(self.aspect_questions)
        self.df['aspect_target'] = self.df['question'] + self.df['target']
        self.tokenizer = tokenizer
        self.max_len = max_len


    def __len__(self):
        return len(self.df)


    def __getitem__(self, idx):
        text = self.df.loc[idx, 'text']
        aspect_target = self.df.loc[idx, 'aspect_target']
        label = self.df.loc[idx, 'label']

        encoding = self.tokenizer.encode_plus(
                text,
                aspect_target,
                add_special_tokens=True,
                max_length=self.max_len,
                padding='max_length',
                return_token_type_ids=False,
                return_attention_mask=True,
                return_tensors='pt',
                truncation=True)

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': label
        }

# count the number of unique aspects in the dataset
def count_aspects(data_file: str) -> int:
    """
    Count the number of unique aspects in the dataset.

    Args:
        data_file (str): The path to the data file.

    Returns:
        int: The number of unique aspects.
    """
    df = pd.read_csv(data_file, sep='\t', header=None, names=['polarity','aspect', 'target', 'what?', 'text'],index_col=False)
    return df['aspect'].nunique()

# count the number of each polarity in the dataset

def count_polarity(data_file: str) -> pd.DataFrame:
    """
    Count the number of each polarity in the dataset.

    Args:
        data_file (str): The path to the data file.

    Returns:
        pd.DataFrame: The count of each polarity.
    """
    df = pd.read_csv(data_file, sep='\t', header=None, names=['polarity','aspect', 'target', 'what?', 'text'],index_col=False)
    return df['polarity'].value_counts()


if __name__ == "__main__":
    data_file = "../data/traindata.csv"
    print(f"Number of unique aspects in the dataset: {count_aspects(data_file)}")
    print(f"Number of each polarity in the dataset:\n{count_polarity(data_file)}")