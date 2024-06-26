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


    @staticmethod
    def aspect_questions(aspect: str) -> str:
        """
        Return the question for the aspect.
        """
        if aspect == 'AMBIENCE#GENERAL':
            question = "How would you describe the atmosphere?"

        elif aspect == 'FOOD#QUALITY':
            question = "How do you feel about the food quality?"

        elif aspect == 'SERVICE#GENERAL':
            question = "What do you think of the service ?"

        elif aspect == 'FOOD#STYLE_OPTIONS':
            question = "How do you feel about the variety of food options?"

        elif aspect == 'DRINKS#QUALITY':
            question = "How would you assess the quality of the drinks?"

        elif aspect == 'RESTAURANT#MISCELLANEOUS' or aspect == 'RESTAURANT#GENERAL':
            question = "What are your overall thoughts on the restaurant?"

        elif aspect == 'LOCATION#GENERAL':
            question = "How do you feel about the restaurant's location?"

        elif aspect == 'DRINKS#STYLE_OPTIONS':
            question = "What do you think of the drink choices ? "

        elif aspect == 'RESTAURANT#PRICES' or aspect == 'DRINKS#PRICES' or aspect == 'FOOD#PRICES':
            question = "How do you feel about the pricing?"

        else:
            raise ValueError('Aspect not found')

        return question

    def __init__(self, data_file: str, tokenizer: PreTrainedTokenizer) -> None:
        """
        Initialize the Dataset class.

        Args:
            data_file (str): The path to the data file.
            tokenizer (PreTrainedTokenizer): The tokenizer used for tokenization.
            max_len (int): The maximum length of the input sequence.

        Returns:
            None
        """
        self.df = pd.read_csv(data_file, sep='\t', header=None, names=[
                              'polarity', 'aspect', 'target', 'what?', 'text'], index_col=False)
        self.df['label'] = self.df['polarity'].apply(self.polarity_to_label)
        self.df['question'] = self.df['aspect'].apply(self.aspect_questions)
        self.df['aspect_target'] = self.df['question'] +  ' ' + self.df['target']
        self.tokenizer = tokenizer
        self.max_len = sequence_length(self.df, tokenizer)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df.loc[idx, 'text']
        aspect_target = self.df.loc[idx, 'aspect_target']
        label = self.df.loc[idx, 'label']

        encoding = self.tokenizer.encode_plus(
            text,
            aspect_target,
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=self.max_len,
            return_token_type_ids=False,
            padding= "max_length",
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True)
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': label
        }


def sequence_length(df, tokenizer):
    token_lengths = []
    for sentence in df['text']:
        tokens = tokenizer.encode(sentence, max_length=1000)
        token_lengths.append(len(tokens))
    # Add 30 to the max in case there are longer sequences in the dev or test files
    return max(token_lengths) + 30

# count the number of unique aspects in the dataset


def count_aspects(data_file: str) -> int:
    """
    Count the number of unique aspects in the dataset.

    Args:
        data_file (str): The path to the data file.

    Returns:
        int: The number of unique aspects.
    """
    df = pd.read_csv(data_file, sep='\t', header=None, names=[
                     'polarity', 'aspect', 'target', 'what?', 'text'], index_col=False)
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
    df = pd.read_csv(data_file, sep='\t', header=None, names=[
                     'polarity', 'aspect', 'target', 'what?', 'text'], index_col=False)

    return df['polarity'].value_counts()


if __name__ == "__main__":
    from transformers import BertTokenizer
    data_file = "data/traindata.csv"
    # print(f"Number of unique aspects in the dataset: {count_aspects(data_file)}")
    # print(f"Number of each polarity in the dataset:\n{count_polarity(data_file)}")
    # show the first output of the dataset
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset = ABSADataset(data_file, tokenizer)
    for i in range(10):
        sample = dataset[i]
