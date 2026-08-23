from typing import List, Dict

class WordPieceTokenizer:
    """
    WordPiece tokenizer for BERT.
    """
    
    def __init__(self, vocab: Dict[str, int], unk_token: str = "[UNK]", max_word_len: int = 100):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_word_len = max_word_len
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into WordPiece tokens.
        """
        tokens = []
        for word in text.lower().split():
            word_tokens = self._tokenize_word(word)
            tokens.extend(word_tokens)
        return tokens
    
    def _tokenize_word(self, word: str) -> List[str]:
        """
        Tokenize a single word into subwords.
        """
        # YOUR CODE HERE
        if len(word) > self.max_word_len:
            return [self.unk_token]

        is_bad = False
        start = 0
        subwords = []
        word_len = len(word)

        while start < word_len:
            end = word_len
            cur_substr = None
            
            # Tìm substring dài nhất từ vị trí 'start' có trong vocab
            while start < end:
                substr = word[start:end]
                if start > 0:
                    substr = f"##{substr}"
                
                if substr in self.vocab:
                    cur_substr = substr
                    break
                end -= 1

            # Nếu không tìm thấy bất kỳ subword hợp lệ nào
            if cur_substr is None:
                is_bad = True
                break

            subwords.append(cur_substr)
            start = end

        if is_bad:
            return [self.unk_token]
        return subwords
