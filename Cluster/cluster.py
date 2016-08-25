"""
K-Means. S
"""

from collections import Counter
from collections import defaultdict
import gzip
import math

import numpy as np


class KMeans(object):

    def __init__(self, k=2):
        """ Initialize a k-means clusterer. Should not have to change this."""
        self.k = k

    def cluster(self, documents, iters=10):
        """
        Cluster a list of unlabeled documents, using iters iterations of k-means.
        Initialize the k mean vectors to be the first k documents provided.
        After each iteration, print:
        - the number of documents in each cluster
        - the error rate (the total Euclidean distance between each document and its assigned mean vector), rounded to 2 decimal places.
        See Log.txt for expected output.
        The order of operations is:
        1) initialize means
        2) Loop
          2a) compute_clusters
          2b) compute_means
          2c) print sizes and error
        """
        self.doc_norm=defaultdict(lambda:0)
        self.docs=documents
        self.mean_vectors=[]
        count=0

        #1) initialize means
        for i in range(self.k):
            self.mean_vectors.append(documents[i])
        self.mean_norms=[]
        for doc in documents:
            n_count=0
            for term in doc:
                n_count = n_count+self.sqnorm(doc[term])
            self.doc_norm[count]=n_count
            count=count+1
        for doc in self.mean_vectors:
            norm_val=0
            for term in doc:
                val=math.pow(doc[term],2)
                norm_val = norm_val+val
            self.mean_norms.append(norm_val)
        #2 Loop
        for j in range(iters):
            #2a) compute_clusters
            self.compute_clusters(documents)
            #2b) compute_means
            self.compute_means()
            #2c) print sizes and error
            no_of_docs=[]
            for i in self.cluster_doc:
                no_of_docs.append(len(self.cluster_doc[i]))
            print(no_of_docs)
            print(self.error(documents))

    def compute_means(self):
        """ Compute the mean vectors for each cluster (results stored in an
        instance variable of your choosing)."""
        del self.mean_vectors[:]
        for i in range(self.k):
            count=Counter()
            len=0
            for doc_id in self.cluster_doc[i]:
                count.update(self.docs[doc_id])
                len=len+1
            if (len>0):
                for doc in count:
                    count[doc]=count[doc]/len
            self.mean_vectors.append(count)
        self.mean_norms=[]

        for doc in self.mean_vectors:
            norm=0
            for term in doc:
                norm = norm + self.sqnorm(doc[term])
            self.mean_norms.append(norm)
        pass

    def compute_clusters(self, documents):
        """ Assign each document to a cluster. (Results stored in an instance
        variable of your choosing). """
        self.cluster_doc=defaultdict(list)
        doc_id=0
        for doc in documents:
            for i in range(self.k):
                mean_norm = self.mean_norms[i]+self.doc_norm[doc_id]
                dist=self.distance(doc,self.mean_vectors[i],mean_norm)
                if (i==0):
                   min=i
                   min_dist=dist
                else:
                    if (dist<min_dist):
                        min=i
                        min_dist=dist
            self.cluster_doc[min].append(doc_id)
            doc_id+=1
        pass

    def sqnorm(self, d):
        """ Return the vector length of a dictionary d, defined as the sum of
        the squared values in this dict. """
        val=math.pow(d,2)
        return val
        pass

    def distance(self, doc, mean, mean_norm):
        """ Return the Euclidean distance between a document and a mean vector.
        See here for a more efficient way to compute:
        http://en.wikipedia.org/wiki/Cosine_similarity#Properties"""
        dist=mean_norm
        for term in doc:
            val=-2.0*doc[term]*mean[term]
            dist = dist +val
        return math.sqrt(dist)
        pass

    def error(self, documents):
        """ Return the error of the current clustering, defined as the total
        Euclidean distance between each document and its assigned mean vector."""
        self.cluster_doc_dist=defaultdict(list)
        err_val=0

        for cluster in self.cluster_doc:
            for doc_id in self.cluster_doc[cluster]:
                doc=documents[doc_id]
                mean_norm =self.mean_norms[cluster]+self.doc_norm[doc_id]
                di=self.distance(doc,self.mean_vectors[cluster],mean_norm)
                err_val=err_val+di
                self.cluster_doc_dist[cluster].append((doc,di))
        return err_val
        #return ("%0.2f"%err_val)
        #pass

    def print_top_docs(self, n=10):
        """ Print the top n documents from each cluster. These are the
        documents that are the closest to the mean vector of each cluster.
        Since we store each document as a Counter object, just print the keys
        for each Counter (sorted alphabetically).
        Note: To make the output more interesting, only print documents with more than 3 distinct terms.
        See Log.txt for an example."""
        for i in self.cluster_doc:
            print ("CLUSTER %d" %i)
            top=sorted(self.cluster_doc_dist[i],key=lambda x:x[1])

            k=0
            j=0
            while (j<n and k<len(top)):
                if(len(top[k][0])>3):
                    print(' '.join(str(v) for v in sorted(top[k][0].keys())))
                    j+=1
                k+=1


def prune_terms(docs, min_df=3):
    """ Remove terms that don't occur in at least min_df different
    documents. Return a list of Counters. Omit documents that are empty after
    pruning words.
    >>> prune_terms([{'a': 1, 'b': 10}, {'a': 1}, {'c': 1}], min_df=2)
    [Counter({'a': 1}), Counter({'a': 1})]
    """
    doc_count = Counter()
    result = []
    for n in docs:
        for term in n:
            doc_count[term] += 1

    for n in docs:
        result_doc = Counter()
        for term in n:
            if doc_count[term] >= min_df:
                result_doc[term] = n[term]
        if len(result_doc) > 0:
            result.append(result_doc)
    return result
    pass

def read_profiles(filename):
    """ Read profiles into a list of Counter objects.
    """
    profiles = []
    with gzip.open(filename, mode='rt', encoding='utf8') as infile:
        for line in infile:
            profiles.append(Counter(line.split()))
    return profiles


def main():
    profiles = read_profiles('profiles.txt.gz')
    print('read', len(profiles), 'profiles.')
    profiles = prune_terms(profiles, min_df=2)
    km = KMeans(k=10)
    km.cluster(profiles, iters=20)
    km.print_top_docs()

if __name__ == '__main__':
    main()
