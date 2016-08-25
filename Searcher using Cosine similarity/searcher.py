""" 

Implementation of search engine based on cosine similarity.

The documents are read from documents.txt.gz.

The index will store tf-idf values using the formulae from class.

The search method will sort documents by the cosine similarity between the
query and the document (normalized only by the document length, not the query
length, as in the examples in class).

The search method also supports a use_champion parameter, which will use a
champion list (with threshold 10) to perform the search.

"""
from collections import defaultdict
import codecs
import gzip
import math
import re


class Index(object):

    def __init__(self, filename=None, champion_threshold=10):
        """ 
        Creating a new index by parsing the given file containing documents,
        one per line. """
        if filename:  
            self.documents = self.read_lines(filename)
            toked_docs = [self.tokenize(d) for d in self.documents]
            self.doc_freqs = self.count_doc_frequencies(toked_docs)
            self.index = self.create_tfidf_index(toked_docs, self.doc_freqs)
            self.doc_lengths = self.compute_doc_lengths(self.index)
            self.champion_index = self.create_champion_index(self.index, champion_threshold)

    def compute_doc_lengths(self, index):
        """
        Returning a dict mapping doc_id to length, computed as sqrt(sum(w_i**2)),
        where w_i is the tf-idf weight for each term in the document.

        E.g., in the sample index below, document 0 has two terms 'a' (with
        tf-idf weight 3) and 'b' (with tf-idf weight 4). It's length is
        therefore 5 = sqrt(9 + 16).

        >>> lengths = Index().compute_doc_lengths({'a': [[0, 3]], 'b': [[0, 4]]})
        >>> lengths[0]
        5.0
        """
        resultDict = defaultdict(lambda: 0)
        for val in index:
            for list in index[val]:
                resultDict[list[0]] += math.pow(list[1],2)
        for doc_id, value in resultDict.items():
            resultDict[doc_id] = math.sqrt(resultDict[doc_id])
        return resultDict
        pass

    def create_champion_index(self, index, threshold=10):
        """
        Creating an index mapping each term to its champion list, defined as the
        documents with the K highest tf-idf values for that term (the
        threshold parameter determines K).

        In the example below, the champion list for term 'a' contains
        documents 1 and 2; the champion list for term 'b' contains documents 0
        and 1.

        >>> champs = Index().create_champion_index({'a': [[0, 10], [1, 20], [2,15]], 'b': [[0, 20], [1, 15], [2, 10]]}, 2)
        >>> champs['a']
        [[1, 20], [2, 15]]
        """
        resultDict = {}
        for key in index:
            k = 0
            l = []
            l = sorted(index[key], key=lambda doc: doc[1], reverse = True)[:threshold]
            resultDict[key] = l
        return resultDict
        pass

    def create_tfidf_index(self, docs, doc_freqs):
        """
        Creating an index in which each postings list contains a list of
        [doc_id, tf-idf weight] pairs. For example:

        {'a': [[0, .5], [10, 0.2]],
         'b': [[5, .1]]}

        This entry means that the term 'a' appears in document 0 (with tf-idf
        weight .5) and in document 10 (with tf-idf weight 0.2). The term 'b'
        appears in document 5 (with tf-idf weight .1).

        Parameters:
        docs........list of lists, where each sublist contains the tokens for one document.
        doc_freqs...dict from term to document frequency (see count_doc_frequencies).

        Use math.log10 (log base 10).

        >>> index = Index().create_tfidf_index([['a', 'b', 'a'], ['a']], {'a': 2., 'b': 1., 'c': 1.})
        >>> sorted(index.keys())
        ['a', 'b']
        >>> index['a']
        [[0, 0.0], [1, 0.0]]
        >>> index['b']  # doctest:+ELLIPSIS
        [[0, 0.301...]]
        """
        tempDict = defaultdict(list)
        for id, doc in enumerate(docs):
            temp_doc = defaultdict(int)
            for val in doc:
                temp_doc[val] = temp_doc[val]+ 1
            for term in temp_doc:
                calc = (1 + math.log10(temp_doc[term]))*(math.log10(float(len(docs)) / float(doc_freqs[term])))
                tempDict[term].append([id, calc])
        return tempDict
        pass

    def count_doc_frequencies(self, docs):
        """ Returning a dict mapping terms to document frequency.
        >>> res = Index().count_doc_frequencies([['a', 'b', 'a'], ['a', 'b', 'c'], ['a']])
        >>> res['a']
        3
        >>> res['b']
        2
        >>> res['c']
        1
        """
        dict = {}
        list_result = []
        for doc in docs:
            for word in doc:
                if word not in dict:
                    dict[word] = 1
                    list_result.append(word)
                elif word not in list_result:
                    dict[word]+=1
                    list_result.append(word)
            list_result =[]
        return dict
        pass

    def query_to_vector(self, query_terms):
        """ Converting a list of query terms into a dict mapping term to inverse document frequency (IDF).
        Computing IDF of term T as log10(N / (document frequency of T)), where N is the total number of documents.
        You may need to use the instance variables of the Index object to compute this. Do not modify the method signature.

        If a query term is not in the index, simply omit it from the result.

        Parameters:
          query_terms....list of terms

        Returns:
          A dict from query term to IDF.
        """
        tempDict = defaultdict(int)
        N = len(self.documents)
        for word in query_terms:
            if word not in tempDict.keys():
                temp = math.log10(N/self.doc_freqs[word])
                tempDict[word] = temp
        return dict(tempDict)
        pass

    def search_by_cosine(self, query_vector, index, doc_lengths):
        """
        Returning a sorted list of doc_id, score pairs, where the score is the
        cosine similarity between the query_vector and the document. The
        document length should be used in the denominator, but not the query
        length (as discussed in class). You can use the built-in sorted method
        (rather than a priority queue) to sort the results.

        The parameters are:

        query_vector.....dict from term to weight from the query
        index............dict from term to list of doc_id, weight pairs
        doc_lengths......dict from doc_id to length (output of compute_doc_lengths)

        In the example below, the query is the term 'a' with weight
        1. Document 1 has cosine similarity of 2, while document 0 has
        similarity of 1.

        >>> Index().search_by_cosine({'a': 1}, {'a': [[0, 1], [1, 2]]}, {0: 1, 1: 1})
        [(1, 2.0), (0, 1.0)]
        """
        tempDict = defaultdict(int)
        for word in query_vector.keys():
            for list in index[word]:
                tempDict[list[0]] += query_vector[word] * list[1]

        for doc_id in tempDict.keys():
            tempDict[doc_id] /=  float(doc_lengths[doc_id])

        tempDict = dict(tempDict)
        tempDict = sorted(tempDict.items(), key = lambda key: key[1], reverse = True)
        return tempDict
        pass

    def search(self, query, use_champions=False):
        """ Returning the document ids for documents matching the query. Assume that
        query is a single string, possible containing multiple words. Assume
        queries with multiple words are AND queries. The steps are to:

        1. Tokenize the query (calling self.tokenize)
        2. Convert the query into an idf vector (calling self.query_to_vector)
        3. Compute cosine similarity between query vector and each document (calling search_by_cosine).

        Parameters:

        query...........raw query string, possibly containing multiple terms (though boolean operators do not need to be supported)
        use_champions...If True, Step 4 above will use only the champion index to perform the search.
        """
        tokens = self.tokenize(query)
        vectors = self.query_to_vector(tokens)
        if use_champions == False:
            result = self.search_by_cosine(vectors, self.index, self.doc_lengths)
        else:
            result = self.search_by_cosine(vectors, self.champion_index, self.doc_lengths)
        return result
        pass

    def read_lines(self, filename):
        """ 
        Read a gzipped file to a list of strings.
        """
        return [l.strip() for l in gzip.open(filename, 'rt', encoding="utf8").readlines()]

    def tokenize(self, document):
        """ 
        Converting a string representing one document into a list of
        words. Retain hyphens and apostrophes inside words. Remove all other
        punctuation and convert to lowercase.

        >>> Index().tokenize("Hi there. What's going on? first-class")
        ['hi', 'there', "what's", 'going', 'on', 'first-class']
        """
        return [t.lower() for t in re.findall(r"\w+(?:[-']\w+)*", document)]


def main():
    """ 
    Main method. Constructs an Index object and runs a sample query. """
    indexer = Index('documents.txt.gz')
    for query in ['pop love song', 'chinese american', 'city']:
        print('\n\nQUERY=%s' % query)
        print('\n'.join(['%d\t%e' % (doc_id, score) for doc_id, score in indexer.search(query)[:10]]))
        print('\n\nQUERY=%s Using Champion List' % query)
        print('\n'.join(['%d\t%e' % (doc_id, score) for doc_id, score in indexer.search(query, True)[:10]]))

if __name__ == '__main__':
    main()
