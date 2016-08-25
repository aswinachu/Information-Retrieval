""" 
You will implement a simple in-memory boolean search engine over the jokes
from http://web.hawkesnest.net/~jthens/laffytaffy/.

The documents are read from documents.txt.
The queries to be processed are read from queries.txt.

search engine will only support AND queries. A multi-word query
is assumed to be an AND of the words. E.g., the query "why because" should be
processed as "why AND because."
"""
from collections import defaultdict
import re


def tokenize(document):

    document = document.lower()
    return re.findall("[A-Za-z'-]+", document)

def create_index(tokens):

    index = {}
    for i, sentence in enumerate(tokens):
        for word in sentence:
            docnum = []
            if word in index.keys():
                docnum = index[word]
                if i not in docnum:
                    docnum.append(i)
                    index[word] = docnum
            else:
                docnum.append(i)
                index[word] = docnum
                index[word] = docnum
    return index

def intersect(list1, list2):
    intersect_val = []
    val1 = 0
    val2 = 0
    while(val1<len(list1) and val2<len(list2)):
        if(list1[val1] == list2[val2]):
            intersect_val.append(list1[val1])
            val1 += 1
            val2 += 1
        elif(list1[val1] < list2[val2]):
            val1 +=1
        else:
            val2 +=1
    return intersect_val


def sort_by_num_postings(words, index):
    return (sorted(words, key=lambda i : len(index[i])))


def search(index, query):
    token_value = tokenize(query)
    sorted_list = sort_by_num_postings(token_value,index)
    intersection_list = []
    if(len(sorted_list) > 0):
        intersection_list = index[sorted_list[0]]
    i = 1
    while(i < len(sorted_list)):
        intersection_list = intersect(intersection_list,index[sorted_list[i]])
        i+=1
    return intersection_list


def main():
    
    documents = open('documents.txt').readlines()
    tokens = [tokenize(d) for d in documents]
    index = create_index(tokens)
    queries = open('queries.txt').readlines()
    for query in queries:
        results = search(index, query)
        print('\n\nQUERY:%s\nRESULTS:\n%s' % (query, '\n'.join(documents[r] for r in results)))


if __name__ == '__main__':
    main()
