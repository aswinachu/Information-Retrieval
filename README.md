# Information-Retrieval

1)Implementing a simple in-memory boolean search engine over the jokes
from http://web.hawkesnest.net/~jthens/laffytaffy/.

The documents are read from documents.txt.
The queries to be processed are read from queries.txt.

search engine will only support AND queries. A multi-word query
is assumed to be an AND of the words. E.g., the query "why because" should be
processed as "why AND because."



2)Implementation of search engine based on cosine similarity.

The documents are read from documents.txt.gz.

The index will store tf-idf values using the formulae.

The search method will sort documents by the cosine similarity between the
query and the document (normalized only by the document length, not the query
length, as in the examples in class).

The search method also supports a use_champion parameter, which will use a
champion list (with threshold 10) to perform the search.



3)Implementation of Multinomial Naive Bayes classifier for spam filtering.

Implemented 3 methods below:

train: compute the word probabilities and class priors given a list of documents labeled as spam or ham.
classify: compute the predicted class label for a list of documents
evaluate: compute the accuracy of the predicted class labels.

4)Implementing K-Means cluster

5)Implementing PageRank
