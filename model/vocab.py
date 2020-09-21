from utils import *
from opts import *
from collections import Counter, defaultdict

'''
Token to be used
'''
SIL_TOKEN = "<si>"
UNK_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"
BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"

LANDMARK_SIZE = 10
BODY_PAD_FEATURE_SIZE = 50
PREPROCESSING_SIZE = 64
PREPROCESSING_SCALE = 1
SAMPLING_RATE = 3


class Vocabulary:
    
    def __init__(self):
        self.specials = []
        self.itos = []
        self.stoi = None
        self.DEFAULT_UNK_ID = None

    def add_tokens(self, tokens):
        
        for t in tokens:
            index = len(self.itos)
            if t not in self.itos: # check if not exist in itos
                self.itos.append(t)
                self.stoi[t] = index

    def _from_list(self, tokens):
        
        self.add_tokens(self.specials + tokens)
        assert len(self.stoi) == len(self.itos), \
            'The length of stoi and itos is not same.'

    def _from_file(self, file):
        
        raise NotImplementedError

    def is_unk(self, token):
        
        return self.stoi[token] == self.DEFAULT_UNK_ID()

    def __len__(self):
        return len(self.itos)


class TextVocabulary(Vocabulary):

    def __init__(self, tokens=None, file=None):

        super().__init__()
        self.specials = [UNK_TOKEN, PAD_TOKEN]
        # self.specials = [SIL_TOKEN, UNK_TOKEN, PAD_TOKEN]
        self.DEFAULT_UNK_ID = lambda: 0
        self.stoi = defaultdict(self.DEFAULT_UNK_ID)
        
        # Make text vocab dictionary
        if tokens is not None:
            self._from_list(tokens)
        else:
            raise NotImplementedError


class GlossVocabulary(Vocabulary):
    
    def __init__(self, tokens=None, _file=None):

        super().__init__()
        # self.specials = [SIL_TOKEN, UNK_TOKEN, PAD_TOKEN]
        self.specials = [UNK_TOKEN, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN]
        self.DEFAULT_UNK_ID = lambda: 0
        self.stoi = defaultdict(self.DEFAULT_UNK_ID)
        
        # Make gloss dictionary
        if tokens is not None:
            self._from_list(tokens)
        elif _file is not None:
            raise NotImplementedError
    
    def array_to_sentence(self, array, cut_at_eos=True):
        gls_seq = list()
        for idx in array:
            g = self.itos[idx]
            if cut_at_eos and g == EOS_TOKEN:
                break
            gls_seq.append(g)
        
        return gls_seq

    def arrays_to_sentences(self, arrays, cut_at_eos=True):
        sentences = list()
        for array in arrays:
            sentences.append(self.array_to_sentence(
                array=array,
                cut_at_eos=cut_at_eos,
            ))
        
        return sentences


def filter_min(counter, min_freq):
    filtered_counter = Counter({t: c for t, c in counter.items() if c >= min_freq})
    
    return filtered_counter

def sort_and_cut(counter, limit):
    # Sort by alphabetically
    tokens_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
    # Sort by numerically
    tokens_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)
    vocab_tokens = [i[0] for i in tokens_and_frequencies[:limit]]
    
    return vocab_tokens

def build_vocab(field, dataset, min_freq, max_size, vocab_file=None):
    '''
    Build text or gloss vocab dictionary

    Args:
        dataset: a given json formatted data
    '''
    if vocab_file is not None:
        if field == 'gls':
            vocab = GlossVocabulary(file=vocab_file)
        elif field == 'txt':
            vocab = TextVocabulary(file=vocab_file)
        else:
            raise ValueError('Unknown vocabulary type.')
    else:
        tokens = list()
        for d_name in dataset.keys():
            if d_name != 'estimator':
                if field == 'gls':
                    tokens += dataset[d_name]['gloss'].split()
                elif field == 'txt':
                    tokens += dataset[d_name]['text'].split()

        counter = Counter(tokens)
        if min_freq > -1:
            counter = filter_min(counter, min_freq)
        vocab_tokens = sort_and_cut(counter, max_size)
        assert len(vocab_tokens) <= max_size

        if field == 'gls':
            vocab = GlossVocabulary(tokens=vocab_tokens)
        elif field == 'txt':
            vocab = TextVocabulary(tokens=vocab_tokens)
        else:
            raise ValueError('Unkown vocabulary type.')

        assert len(vocab) <= max_size + len(vocab.specials)
        assert vocab.itos[vocab.DEFAULT_UNK_ID()] == UNK_TOKEN
    
    for i, s in enumerate(vocab.specials):
        if i != vocab.DEFAULT_UNK_ID():
            assert not vocab.is_unk(s)

    return vocab

    

        