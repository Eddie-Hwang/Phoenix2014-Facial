from dataset._vocab import Vocabulary
from utils import *
from opts import *

class TextVocabulary(Vocabulary):

    def __init__(self, tokens=None, file=None):
        """
        Create vocabulary from list of tokens or file.
        Special tokens are added if not already in file or list.
        
        Args:
            tokens: list of tokens
            file: file to load vocabulary from
        """

        super().__init__()
        self.specials = [UNK_TOKEN, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN]
        self.stoi = dict()

        # Make text vocab dictionary
        self._from_list(tokens)


class GlossVocabulary(Vocabulary):
    
    def __init__(self, tokens=None, file=None):
        """
        Create vocabulary from list of tokens or file.
        Special tokens are added if not already in file or list.

        Args:
            tokens: list of tokens
            file: file to load vocabulary from
        """
        super().__init__()
        self.specials = [SIL_TOKEN, UNK_TOKEN, PAD_TOKEN]
        self.stoi = dict()

        # Make gloss dictionary
        self._from_list(tokens)


def build_vocab(_file):
    '''
    Build text or gloss vocab dictionary

    Args:
        _file: a given json formatted data
    '''
    txt_tokens = list()
    gls_tokens = list()
    
    for d_name in _file.keys():
        txt_tokens += _file[d_name]['text'].split(' ')
        gls_tokens += _file[d_name]['gloss'].split(' ')
    
    
    txt_vocab = TextVocabulary(tokens=txt_tokens)
    gls_vocab = GlossVocabulary(tokens=gls_tokens)

    return txt_vocab, gls_vocab

    

        