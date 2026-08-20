import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        """Build vocabulary from a list of texts.

        Add special tokens first, then sorted unique words.
        """
        # YOUR CODE HERE
        special_tokens = [
            self.pad_token,
            self.unk_token,
            self.bos_token,
            self.eos_token,
        ]

        unique_words = set()
        for text in texts:
            words = text.lower().strip().split()
            for word in words:
                unique_words.add(word)

        sorted_words = sorted(list(unique_words))

        full_vocab = special_tokens + sorted_words

        self.word_to_id = {word: idx for idx, word in enumerate(full_vocab)}
        self.id_to_word = {idx: word for idx, word in enumerate(full_vocab)}
        self.vocab_size = len(full_vocab)
    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        # YOUR CODE HERE
        input_id = []
        words = text.lower().strip().split()
        for word in words:
            if word in self.word_to_id:
                input_id.append(self.word_to_id[word])
            else: input_id.append(self.word_to_id[self.unk_token])
        return input_id
    
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        # YOUR CODE HERE
        words = [self.id_to_word.get(token_id, self.unk_token) for token_id in ids]
        return " ".join(words)
