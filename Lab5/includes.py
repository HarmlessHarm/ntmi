import numpy as np
from collections import defaultdict

class HMMLM:
    """
    This is our HMM language model class.
    
    It will be responsible for estimating parameters by MLE
    as well as computing probabilities using the HMM.
    
    We will use Laplace smoothing by default (because we do not want to assign 0 probabilities).
    
    GUIDELINES:
        - by convention we will use the string '-UNK-' for an unknown POS tag
            - and '<unk>' for an unknown word
        - don't forget that with Laplace smoothing the unknown symbols have to be in the support of distributions
        - now you will have 2 types of distributions, so you should deal with unknown symbols for both of them
        - we also need padding for sentences and tag sequences, by convention we will use 
            - '-BOS-' and '-EOS-' for padding tag sequences
            - '<s>' and '</s>' for padding sentences
        - do recall that '-BOS-' is **not** a valid tag
            in other words we never *generate* '-BOS-' tags, we only pretend they occur at
            the 0th position of the tag sequence in order to provide conditioning context
            for the first actual tag
        - similarly, '<s>' is not a valid word
            in other words, we never *generate* '<s>' as a word
            in fact '<s>' is optional as no emission event is based on it
        - on the other hand, '-EOS-' is a valid tag
            you should model it as the last event of a tag sequence
        - similarly, '</s>' is a valid word
            you should consider it as the last event of a sentence
            
    You can use whatever data structures you like for cpds
        - we suggest python dict or collections.defaultdict
            but you are free to experiment with list and/or np.array if you like
    """
    
    def __init__(self, transition_alpha=1.0, emission_alpha=1.0):
        self._vocab = set()
        self._tagset = set()
        self._emission_cpds = dict()
        self._transition_cpds = dict()
        self._transition_alpha = transition_alpha
        self._emission_alpha = emission_alpha
        
    def tagset(self):
        """
        Return the tagset: a set of all tags seen by the model (including '-UNK-').
        
        You can modify this if you judge necessary (for example, because you decided  to 
            use different datastructures, but do note that we provide you an implementation
            of the Viterbi algorithm that expects this functionality).        
        """        
        # the -BOS- tag is just something for internal representation
        #  in case you have added it to the tagset, we are removing it here
        #  as keeping it would be bad for algorithms such as Viterbi
        # the -UNK- tag must be in the support (due to Laplace smoothing)
        #  thus in case you forgot it, we are adding it now
        return self._tagset - {'-BOS-'} | {'-UNK-'}
    
    def addTag(self, tag):
        """
        Adds a tag to the tagset variable
        """
        self._tagset.add(tag)
    
    def vocab(self):
        """
        Return the vocabulary of words: all words seen by the model (including '<unk>').
        
        You can modify this if you judge necessary (for example, because you decided  to 
            use different datastructures, but do note that we provide you an implementation
            of the Viterbi algorithm that expects this functionality).        
        """        
        # the <s> token is just something for internal representation
        #  in case you have added it to the vocabulary, we are removing it here
        # the <unk> word must be in the support (due to Laplace smoothing)
        #  thus in case you forgot it, we are adding it now
        return self._vocab - {'<s>'} | {'<unk>'}
    
    def addWord(self, word):
        """
        Adds a word to the vocab variable
        """
        
        self._vocab.add(word)
        
    def preprocess_sentence(self, sentence, bos=True, eos=True):
        """
        Preprocess a sentence by lowercasing its words and possibly padding it.
        
        :param sentence: a list of tokens (each a string)
        :param bos: if True you will get <s> at the beginning 
        :param eos: if True you will get </s> at the end
        :returns: a list of tokens (lowercased strings)
        """
        # lowercase
        sentence = [x.lower() for x in sentence]
        # optional padding
        if bos: 
            sentence = ['<s>'] + sentence
        if eos:
            sentence = sentence + ['</s>']
        return sentence
        
    def preprocess_tag_sequence(self, tag_sequence, bos=True, eos=True):
        """
        Preprocess a tag sequence with optional padding.
        
        :param tag_sequence: a list of tags (each a string)
        :param bos: if True you will get -BOS- at the beginning 
        :param eos: if True you will get -EOS- at the end
        :returns: a list of tokens 
        """
        # optional padding
        if bos:
            tag_sequence = ['-BOS-'] + tag_sequence
        if eos:
            tag_sequence = tag_sequence + ['-EOS-']
        return tag_sequence
        
    def estimate_model(self, treebank):
        """
        :param treebank: a sequence of observations as provided by nltk
            each observation is a list of pairs (x_i, c_i)    
            and they have not yet been pre-processed 
        
        Estimate the model parameters.
        
        This method does not have to return anything, it simply computes the necessary cpds.        
        """
        
        # Create count table for emission and transition(defaultdict(int))
        emis_count_table = {'-UNK-' : {'<unk>': 0}}
        tran_count_table = {'-UNK-' : {'-UNK-': 0}}
             
        print("Start counting")
        for i, tag_sent in enumerate(treebank):
            # Preprocess the sentence and tag_sequence
            sentence, tags = map(list, zip(*tag_sent))
            sentence = self.preprocess_sentence(sentence)
            tags = self.preprocess_tag_sequence(tags)
            
            # Fill count tables
            for i, word in enumerate(sentence[1:], 1):
                tag = tags[i]
                tag_prev = tags[i-1]
                
                # Add word to tag emission_count: P(word|tag)
                if tag not in emis_count_table:
                    emis_count_table[tag] = defaultdict(int)
                emis_count_table[tag][word] += 1
                if '<unk>' not in emis_count_table[tag]:
                    emis_count_table[tag]['<unk>'] = 0
                
                # Add tag to prevtag in transition_count: P(tag|tag_prev)
                if tag_prev not in tran_count_table:
                    tran_count_table[tag_prev] = defaultdict(int)
                tran_count_table[tag_prev][tag] += 1
                if '-UNK-' not in tran_count_table[tag_prev]:
                    tran_count_table[tag_prev]['-UNK-'] = 0
                
                # Add tag and word to tagset andd vocab
                self.addTag(tag)
                self.addWord(word)
        
        print("Start calculating cpd's")
        # Parse count tables and convert them to CPD's
        for i, (tag, word_count) in enumerate(emis_count_table.items()):
            print('.', end='')
            self._emission_cpds[tag] = defaultdict(float)
            total_count = sum(word_count.values())
            for word, count in word_count.items():
                prob = (float(count) + self._emission_alpha) / \
                       (total_count + self._emission_alpha * len(self.vocab()))
                self._emission_cpds[tag][word] = prob

        for tag_prev, tag_count in tran_count_table.items():
            self._transition_cpds[tag_prev] = defaultdict(float)
            total_count = sum(tag_count.values())
            
            for tag, count in tag_count.items():
                prob = (float(count) + self._transition_alpha) / \
                       (total_count + self._transition_alpha * len(self.tagset()))
                self._transition_cpds[tag_prev][tag] = prob
                
        print("\nFinished cpd's")
    
    def transition_parameter(self, previous_tag, current_tag):
        """
        This method returns the transition probability for tag given the previous tag.
        
        Tips: do not forget that we have a smoothed model, thus 
            - if the either tag was never seen, you should pretend it to be '-UNK-'
        
        :param previous_tag: the previous tag (str)
        :param current_tag: the current tag (str)
        :return: transition parameter
        """
        if previous_tag not in self._transition_cpds:
            previous_tag = '-UNK-'
        if current_tag not in self._transition_cpds[previous_tag]:
            current_tag = '-UNK-'
        return self._transition_cpds[previous_tag][current_tag]        
    
    def emission_parameter(self, tag, word):
        """
        This method returns the emission probability for a word given a tag.
        Tips: do not forget that we have a smoothed model, thus 
            - if the tag was never seen, you should pretend it to be '-UNK-'
            - similarly, if the word was never seen, you shoud pretend it to be '<unk>'
        
        :param tag: the current tag (str)
        :param word: the current word (str)
        :return: the emission probability
        """
        if tag not in self._emission_cpds:
            tag = '-UNK-'
        if word not in self._emission_cpds[tag]:
            word = '<unk>'
        return self._emission_cpds[tag][word]
        
    def joint_parameter(self, previous_tag, current_tag, word):
        """
        This method returns the joint probability of (current tag, word) given the previous tag
            according to Equation (3)
            
        :param previous_tag: the previous tag (str)
        :param current_tag: the current tag (str)
        :param word: the current word (str)
        :returns: P(word, current_tag|previous_tag)
        """
        pcp = self.transition_parameter(previous_tag, current_tag)
        pxc = self.emission_parameter(current_tag, word)
        return pcp * pxc
    
    def marginal_x_given_cprev(self, previous_tag, word):
        """
        Return P(x|prev) as defined in Equation (4) by marginalising current tag.
        
        :param previous_tag: the previous tag (str)
        :param word: the current word (str)
        """
        return np.sum([self.joint_parameter(previous_tag, c, word) for c in self._tagset])
    
    def log_joint(self, sentence, tag_sequence):
        """
        Implement the logarithm of the joint probability over a sentence and tag sequence as in Equation (8)
        
        :param sentence: a sequence of words (each a string) not yet preprocessed
        :param tag_sequence: a sequence of tags (eac a string) not yet preprocessed
        :returns: log P(x_1^n, c_1^n|n) as defined in Equation (8)
        """ 
        sentence = self.preprocess_sentence(sentence)
        tag_sequence = self.preprocess_tag_sequence(tag_sequence)
        
        joint_prob = 0
        for i, word in enumerate(sentence[1:], 1):
            tag = tag_sequence[i]
            prev_tag = tag_sequence[i - 1]
            joint_prob += np.log(self.joint_parameter(prev_tag, tag, word))
        return joint_prob
    
    def log_marginal(self, sentence):
        """
        Implement the logarithm of the marginal probability of a sentence as in Equation (9)
            by marginalisation of all possible tag sequences. 
            
        :param sentence: a sequence of words (each a string) not yet preprocessed
        :returns: log P(x_1^m|n) as defined in Equation (9)
        """
        sentence = self.preprocess_sentence(sentence)
        
        return sum([np.log(sum([self.marginal_x_given_cprev(p,w) for p in self.tagset()])) for w in sentence])

def log_perplexity(sentences, hmm):
    """
    For a dataset of sentences (each sentence is a list of words)
        and an instance of the HMMLM class
        return the log perplexity as defined in Equation (12)
    """
    
    t = sum([len(s) + 2 for s in sentences])
    return -1.0 / t * sum([hmm.log_marginal(s) for s in sentences])

def extract_sentences(treebank_corpus):
    sentences = []
    for observations in treebank_corpus:
        sentences.append([x for x, c in observations])
    return sentences

def accuracy(gold_sequences, pred_sequences):
    """
    Return percentage of instances in the test data that our tagger labeled correctly.
    
    :param gold_sequences: a list of tag sequences that can be assumed to be correct
    :param pred_sequences: a list of tag sequences predicted by Viterbi    
    """
    count_correct, count_total = 0, 0
    for i, combined in enumerate(zip(pred_sequences, gold_sequences)):
        for p, g in list(zip(*combined)):
            if p == g:
                count_correct += 1
            count_total += 1
    if count_total:
        return count_correct / count_total
    return None

def predict_corpus(test_set, hmm):
    """
    Returns viterbi predictions for all sentences in a given corpus
    
    :param test_set: A corpus of tagged sentences
    :param hmm     : A language model
    """
    gold_sequences, pred_sequences = list(), list()
    print('Making predictions', end='')
    for i, sequence in enumerate(test_set):
        if i % round(len(test_set) / 10) == 0:
            print('.', end='')
        sentence , tags = map(list, zip(*sequence))
        viterbi_tags, _ = viterbi_recursion(sentence, hmm)
        gold_sequences.append(tags)
        pred_sequences.append(viterbi_tags)
    return gold_sequences, pred_sequences

def viterbi_recursion(sentence, hmm):
    """
    Computes the best possible tag sequence for a given input
    and also returns it log probability.
    
    This implementation uses recursion.
    
    :returns: tag sequence, log probability
    """
    # here we pad the sentence with </s> only
    sentence = hmm.preprocess_sentence(sentence, bos=False, eos=True)
    # this is the length (but recall that padding added 1 token) 
    n = len(sentence)
    # this is the complete tagset, which for convenience we will turn into a list
    tagset = list(hmm.tagset())
    t = len(tagset)
    # We need a table to store log alpha(i, j) values
    # - where i is an integer from 0 to n-1 which refers to a position in the list `sentence`
    #   i.e. sentence[i]
    # - and j is an integer from 0 to t-1 that refers to a tag in the list `tagset` 
    #   i.e. tagset[j] 
    # - together (i, j) means that we are setting `C_i = tagset[j]`    
    # - we will be exploring the space of possible tags per position
    #   thus our table has as many as n * t cells
    # - Recall that the value \log \alpha(i, j)
    #   corresponds to the log probability value of the best
    #   path (C_1, ..., C_i) such that C_i = j
    #   in other words the log probability of the best sequence up to the ith token where C_i = j
    # At the beginning path probabilities have not been computed, we use a probability of 0 to indicate that
    #  as we will be computing log probabilities, we use -inf instead
    #  numpy arrays are very handy and we can actually use the quantity -inf
    log_alpha_table = np.full([n, t], -float('inf'))
    # In a best path algorithm we are interested in two things
    #  the best score (or best log probability)
    #  as well as the path that corresponds to the best score
    # We compute the best score by moving i forward from 0 to n-1 computing the maximum value 
    #  and we traverse the table backwards following the path that led to the maximum
    #  thus we create a table of "back pointers"
    #  this is an integer for each cell (i, j) that tells us which tag `p` for position `i - 1`
    #   leads to the score stored in `log_alpha_table[i, j]`
    back_pointer_table = np.full([n, t], -1, dtype=int)

    # Here we define the log alpha recursion
    def log_alpha(i, j):
        """
        This function returns
                max_{c_1, ..., c_i=j} log P(c_1, ..., c_i=j) 
            where i is a (0-based) position in `sentence`
            and j is a (0-based) position in `tagset`
        """
        if i == 0:  # we do not need to tag the 0th position and it should not affect the probability
            return 0.  # np.log(1)
        # When we implement dynamic programs, we like to re-use computations already made
        # thus first of all we test if we have already computed a value for this cell
        # if so, it will not have a zero probability (-inf in log space)
        if log_alpha_table[i, j] != -float('inf'):  
            # then we can simply return it
            return log_alpha_table[i, j]
        # At this point we know we have not yet computed a score for this path
        #  thus we proceed to compute it
        # We will have to figure out the log prob of the best prefix
        #  and which tag best continues from it
        # There are exactly t classes that may tag this position
        #  thus we just go over the tagset trying one at a time
        #  and memorise the score we would have if we would select them
        path_max_log_prob = np.full(t, -float('inf'))
        for p in range(t):
            # this is the essential part of the recursion
            # we ask for the best score associated with the previous position 
            #  had it been tagged with p
            #  and we incorporate the probability of C_i = tagset[j] given that C_{i-1} = tagset[p]
            #   as well as the probability of X_i = sentence[i] given that C_i = tagset[j]
            path_max_log_prob[p] = log_alpha(i - 1, p) + np.log(hmm.joint_parameter(tagset[p], tagset[j], sentence[i]))
        # From all possibilities, we are only interested in the best
        log_alpha_table[i, j] = np.max(path_max_log_prob)
        # and we also want to store a pointer to the best
        back_pointer_table[i, j] = np.argmax(path_max_log_prob)
        return log_alpha_table[i, j]
    
    # Let's get the index associated with -EOS-
    #  which is the tag for the </s> symbol in sentence[-1]
    eos_index = tagset.index('-EOS-')
    # We want the last word in the sentence (</s>) to have the tag -EOS-
    #  thus we ask "what's the probability of the best path that ends in -EOS-?"
    max_log_prob = log_alpha(n - 1, eos_index)
    
    # Here we retrieve the backpointers for the best analysis
    #  the best analisys has n tags
    bwd_argmax = [None] * n
    #  the last tag is the -EOS- symbol
    bwd_argmax[-1] = eos_index
    # Here we maintain the "current tag" c_i
    c_i = eos_index
    for i in range(n - 1, 0, -1):  # we go backwards from c_{n-1} to c_1
        # and set the value of c_{i-1} for the current c_i
        bwd_argmax[i - 1] = back_pointer_table[i, c_i]
        # we need, of course, to update c_i
        c_i = bwd_argmax[i - 1]
    
    # Here we translate from ids back to actual tags (strings)
    #  we leave the -EOS- symbol out, since it was just a convenience 
    #  and return both the tag sequence and the total log probability
    return [tagset[c] for c in bwd_argmax[:-1]], max_log_prob