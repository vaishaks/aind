import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def get_model_score(self, n):
        X, l = combine_sequences(range(len(self.sequences)), self.sequences)
        model = self.base_model(n)
        return -2 * model.score(X, l) + (n**2 + 2*n*model.n_features - 1) * math.log(len(self.sequences))

    def select(self):
        """ select based on BIC

        :return: GaussianHMM object
        """
        try:
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            best_score_so_far = float("Inf")
            for n_components in range(self.min_n_components, self.max_n_components + 1):
                model_score = self.get_model_score(n_components)
                if model_score < best_score_so_far:
                    self.X, self.lengths = combine_sequences(range(len(self.sequences)), self.sequences)
                    model = self.base_model(n_components)
                    best_score_so_far = model_score
            return model
        except Exception:
            return self.base_model(self.n_constant)    


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def get_model_score(self, n):
        model = self.base_model(n)
        scores = []
        for word, (X, l) in self.hwords.items():
            if word != self.this_word:
                scores.append(model.score(X, l))
        return model.score(self.X, self.lengths) - np.mean(scores)

    def select(self):
        """ select based on DIC

        :return: GaussianHMM object
        """        
        try:
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            best_score_so_far = float("-Inf")
            for n_components in range(self.min_n_components, self.max_n_components + 1):
                model_score = self.get_model_score(n_components)
                if model_score > best_score_so_far:
                    self.X, self.lengths = combine_sequences(range(len(self.sequences)), self.sequences)
                    model = self.base_model(n_components)
                    best_score_so_far = model_score
            return model
        except Exception:
            return self.base_model(self.n_constant)        


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def get_model_score(self, n):
        split_method = KFold(n_splits=2)
        scores = []
        for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
            self.X, self.lengths = combine_sequences(cv_train_idx, self.sequences)
            model = self.base_model(n)
            X, l = combine_sequences(cv_test_idx, self.sequences)
            logL = model.score(X, l)
            scores.append(logL)
        return np.mean(scores)

    def select(self):
        """ select based on CV

        :return: GaussianHMM object
        """        
        try:
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            best_score_so_far = float("Inf")
            for n_components in range(self.min_n_components, self.max_n_components + 1):
                model_score = self.get_model_score(n_components)
                if model_score < best_score_so_far:
                    self.X, self.lengths = combine_sequences(range(len(self.sequences)), self.sequences)
                    model = self.base_model(n_components)
                    best_score_so_far = model_score
            return model
        except Exception:
            return self.base_model(self.n_constant)    