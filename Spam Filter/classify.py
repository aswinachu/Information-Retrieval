"""
Implementation of Multinomial Naive Bayes classifier for spam filtering.

Implemented 3 methods below:

train: compute the word probabilities and class priors given a list of documents labeled as spam or ham.
classify: compute the predicted class label for a list of documents
evaluate: compute the accuracy of the predicted class labels.

"""

from collections import defaultdict
from collections import Counter
import glob
import math
import os



class Document(object):
    """ A Document. 
    The instance variables are:

    filename....The path of the file for this document.
    label.......The true class label ('spam' or 'ham'), determined by whether the filename contains the string 'spmsg'
    tokens......A list of token strings.
    """

    def __init__(self, filename=None, label=None, tokens=None):
        """ Initialize a document either from a file, in which case the label
        comes from the file name, or from specified label and tokens, but not
        both.
        """
        if label: # specify from label/tokens, for testing.
            self.label = label
            self.tokens = tokens
        else: # specify from file.
            self.filename = filename
            self.label = 'spam' if 'spmsg' in filename else 'ham'
            self.tokenize()

    def tokenize(self):
        self.tokens = ' '.join(open(self.filename).readlines()).split()


class NaiveBayes(object):
    prior = defaultdict(float)
    condProb = defaultdict(defaultdict)
    uniqueTerms = set()

    def get_word_probability(self, label, term):
        """
        Return Pr(term|label). This is only valid after .train has been called.

        Params:
          label: class label.
          term: the term
        Returns:
          A float representing the probability of this term for the specified class.

        >>> docs = [Document(label='spam', tokens=['a', 'b']), Document(label='spam', tokens=['b', 'c']), Document(label='ham', tokens=['c', 'd'])]
        >>> nb = NaiveBayes()
        >>> nb.train(docs)
        >>> nb.get_word_probability('spam', 'a')
        0.25
        >>> nb.get_word_probability('spam', 'b')
        0.375
        """
        if label in self.condProb.keys():
            if term in self.condProb[label].keys() :
                prob = self.condProb[label][term]
        return prob
        pass

    def get_top_words(self, label, n):
        """ Return the top n words for the specified class, using the odds ratio.
        The score for term t in class c is: p(t|c) / p(t|c'), where c'!=c.

        Params:
          labels...Class label.
          n........Number of values to return.
        Returns:
          A list of (float, string) tuples, where each float is the odds ratio
          defined above, and the string is the corresponding term.  This list
          should be sorted in descending order of odds ratio.

        >>> docs = [Document(label='spam', tokens=['a', 'b']), Document(label='spam', tokens=['b', 'c']), Document(label='ham', tokens=['c', 'd'])]
        >>> nb = NaiveBayes()
        >>> nb.train(docs)
        >>> nb.get_top_words('spam', 2)
        [(2.25, 'b'), (1.5, 'a')]
        """
        result =[]
        other_label = 'spam'
        if label is 'spam':
            other_label ='ham'
        for term in self.condProb[label].keys():
            current_label_prob = float(self.condProb[label][term])
            other_label_term_prob = float(self.condProb[other_label][term])
            res = 0
            if other_label_term_prob != float(0):
                res = (current_label_prob)/(other_label_term_prob)
            result.append((res,term))
        return sorted(result, key=lambda item: item[0],reverse=True)[0:n]
        pass

    def train(self, documents):
        """
        Given a list of labeled Document objects, compute the class priors and
        word conditional probabilities, following Figure 13.2 of your
        book. Store these as instance variables, to be used by the classify
        method subsequently.
        Params:
          documents...A list of training Documents.
        Returns:
          Nothing.
        """
        #1 V <- ExtractVocabulary(D)
        #docID for each document while reading
        for docID in range(0, len(documents)):
            for term in documents[docID].tokens:
                self.uniqueTerms.add(term)
        self.uniqueTerms = list(self.uniqueTerms)

        #2 N <- CountDocs(D)
        N = len(documents)

        #3 for each c belongs to C
        for classOfDoc in ['ham','spam']:
            listOfClassTerms = []
            Tct = Counter()
            condprob = defaultdict(float)
            total = 0

            #4 do Nc <- CountDocsInClass(D,c)
            noOfClassDocs = len([1 for docID in range(0, N) if documents[docID].label is classOfDoc])

            #5 prior[c] <- Nc/N
            self.prior[classOfDoc] = float(noOfClassDocs) / float(N)

            #6 textc <- ConcatenateTextOfAllDocsInClass(D,c)
            for docID in range(0, N):
                if documents[docID].label is classOfDoc:
                    for term in documents[docID].tokens:
                        listOfClassTerms.append(term);

            #7 for each t belongs to V
            for term in listOfClassTerms:
                #8 do Tct <- CountTokensOfTerm(textc,t)
                Tct[term] += 1
            for term in self.uniqueTerms:
                total += (Tct[term] + 1)

            #9 for each t belongs to V
            for term in self.uniqueTerms:
                #10 do condprob[t][c] <-(Tct+1)/total
                condprob[term] = float(Tct[term] + 1) / float(total)

            self.condProb[classOfDoc] = condprob


    def classify(self, documents):
        """ Return a list of strings, either 'spam' or 'ham', for each document.
        Params:
          documents....A list of Document objects to be classified.
        Returns:
          A list of label strings corresponding to the predictions for each document.
        """

        final=[]
        testDocScore = defaultdict(defaultdict)
        N =len(documents)
        for c in ['ham','spam']:
            score = defaultdict(float)
            for docID in range(0, N):
                score[docID] = math.log10(self.prior[c])
                for term in documents[docID].tokens:
                    if self.condProb[c][term] > 0:
                        score[docID] += math.log10(self.condProb[c][term])
            testDocScore[c] = score

        for docID in range(0, N):
            temp_List = []
            for classOfDoc in ['ham','spam']:
                temp_List.append(testDocScore[classOfDoc][docID])
            final.append(['ham','spam'][temp_List.index(max(temp_List))])
        return final
        pass

def evaluate(predictions, documents):
    """ Evaluate the accuracy of a set of predictions.
    Return a tuple of three values (X, Y, Z) where
    X = percent of documents classified correctly
    Y = number of ham documents incorrectly classified as spam
    X = number of spam documents incorrectly classified as ham

    Params:
      predictions....list of document labels predicted by a classifier.
      documents......list of Document objects, with known labels.
    Returns:
      Tuple of three floats, defined above.
    """
    false_spam = 0
    false_ham = 0
    correct = 0
    N=len(documents)
    for i in range(0,N):
        if documents[i].label == predictions[i] :
            correct += 1
        else:
            if predictions[i] == 'spam':
                false_spam += 1
            else:
                false_ham += 1
    acc_value = (correct/float(N))
    results = [acc_value,false_spam,false_ham]
    return results


def main():
    """ Main function """
    if not os.path.exists('train'):  # download data
       from urllib.request import urlretrieve
       import tarfile
       urlretrieve('http://cs.iit.edu/~culotta/cs429/lingspam.tgz', 'lingspam.tgz')
       tar = tarfile.open('lingspam.tgz')
       tar.extractall()
       tar.close()
    train_docs = [Document(filename=f) for f in glob.glob("train/*.txt")]
    print('read', len(train_docs), 'training documents.')
    nb = NaiveBayes()
    nb.train(train_docs)
    test_docs = [Document(filename=f) for f in glob.glob("test/*.txt")]
    print('read', len(test_docs), 'testing documents.')
    predictions = nb.classify(test_docs)
    results = evaluate(predictions, test_docs)
    print('accuracy=%.3f, %d false spam, %d missed spam' % (results[0], results[1], results[2]))
    print('top ham terms: %s' % ' '.join('%.2f/%s' % (v,t) for v, t in nb.get_top_words('ham', 10)))
    print('top spam terms: %s' % ' '.join('%.2f/%s' % (v,t) for v, t in nb.get_top_words('spam', 10)))

if __name__ == '__main__':
    main()
