
class Vocabulary:
    """ 
    Vocabulary represents mapping between tokens and indices. 
    """
    
    def __init__(self):
        self.specials = []
        self.itos = []
        self.stoi = None
        self.DEFAULT_UNK_ID = None

    def add_tokens(self, tokens):
        """
        Add list of tokens to vocab
        
        Args:
            list of tokens to be added
        """
        for t in tokens:
            index = len(self.itos)
            if t not in self.itos: # check if not exist in itos
                self.itos.append(t)
                self.stoi[t] = index

    def _from_list(self, tokens):
        """
        Make vocabulary from list of tokens.
        Tokens are assumed to be unique and pre-selected.
        Special symbols are added if not in list.
        
        Args:
            tokens: list of tokens
        """
        self.add_tokens(self.specials + tokens)
        assert len(self.stoi) == len(self.itos), \
            'The length of stoi and itos is not same.'

    def _from_file(self, file):
        """
        Make vocabulary from contents of file.

        Args:
            file: path to file where the vocabulary is loaded from
        """
        raise NotImplementedError

    def is_unk(self, token):
        '''
        Check the given token is covered by the vocab we have.

        Args:
            token: a string
            return: True if covered, otherwise False
        '''
        return self.stoi[token] == self.DEFAULT_UNK_ID()

    def __len__(self):
        return len(self.itos)