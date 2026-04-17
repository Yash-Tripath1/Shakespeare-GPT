class CharTokenizer:
    def __init__(self, text):
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)        
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, text):
        """Converts a string of text into a list of integers."""
        return [self.char_to_idx[ch] for ch in text]

    def decode(self, indices):
        """Converts a list of integers back into a string of text."""

        return ''.join([self.idx_to_char[i] for i in indices])

    def save(self, path):
        """Saves the tokenizer's vocabulary to a file."""
        import json
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'chars': self.chars}, f)

    @classmethod
    def load(cls, path):
        """Loads a tokenizer from a saved vocabulary file."""
        import json
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    
        tokenizer_instance = cls("a")  # "a" is just a placeholder
        tokenizer_instance.chars = data['chars']
        tokenizer_instance.vocab_size = len(tokenizer_instance.chars)
        tokenizer_instance.char_to_idx = {ch: i for i, ch in enumerate(tokenizer_instance.chars)}
        tokenizer_instance.idx_to_char = {i: ch for i, ch in enumerate(tokenizer_instance.chars)}
        return tokenizer_instance