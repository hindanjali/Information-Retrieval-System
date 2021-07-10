import numpy as np
import bisect
import time
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

class ESA():
    def __init__(self):
        self.indexArticles = None
        self.indexConcepts = None
        self.noOfDocuments = None
        self.articleWords = None
        self.conceptWords = None

    def buildIndexArticle(self, docs):
        index = []
        for i in docs:
            index.append(i)

        detokenized_doc = []
        for i in range(len(index)):
            t = ' '.join(index[i])
            detokenized_doc.append(t)

        df = pd.DataFrame(detokenized_doc, columns=['cleaned_docs'])

        vectorizer = TfidfVectorizer(stop_words='english', smooth_idf=True)
        X = vectorizer.fit_transform(df['cleaned_docs'])
        dictionary = vectorizer.get_feature_names()

        X = list(X.toarray())
        X.insert(0, dictionary)
        self.indexArticles = X


    def buildIndexConcept(self, docs, docIDs, queries):

        self.noOfDocuments = len(docs)

        index = []
        for i in docs:
            index.append(i)
        for i in queries:
            index.append(i)

        detokenized_doc = []
        for i in range(len(index)):
            t = ' '.join(index[i])
            detokenized_doc.append(t)

        df = pd.DataFrame(detokenized_doc, columns = ['cleaned_docs'])

        vectorizer = TfidfVectorizer(stop_words='english', smooth_idf=True)
        X = vectorizer.fit_transform(df['cleaned_docs'])

        dictionary = vectorizer.get_feature_names()
        X = list(X.toarray())
        X.insert(0, dictionary)
        self.indexConcepts = X

    def rank(self, concepts, queries):

        A = np.array(self.indexArticles[1:])
        C = np.array(self.indexConcepts[1:])

        articleWords = self.indexArticles[0]
        conceptWords = self.indexConcepts[0]


        Number_of_documents = self.noOfDocuments

        representationOfConcepts = []
        representationOfConcepts.append(['{}'.format(i+1) for i in range(C.shape[0])])
        for i in range(len(concepts)):
            arr = np.zeros(len(A))
            for j in range(len(concepts[i])):
                indexArticle = bisect.bisect_left(articleWords, concepts[i][j])
                indexConcept = bisect.bisect_left(conceptWords, concepts[i][j])
                if indexArticle < len(articleWords) and articleWords[indexArticle] == concepts[i][j]:
                    arr += np.array(A[:, indexArticle]) * C[i][indexConcept]
            representationOfConcepts.append(arr)

        for i in range(len(queries)):
            arr = np.zeros(len(A))
            for j in range(len(queries[i])):
                indexArticle = bisect.bisect_left(articleWords, queries[i][j])
                indexConcept = bisect.bisect_left(conceptWords, queries[i][j])
                if indexArticle < len(articleWords) and articleWords[indexArticle] == queries[i][j]:
                    arr += np.array(A[:, indexArticle]) * C[Number_of_documents + i][indexConcept]
            representationOfConcepts.append(arr)


        doc_IDs_ordered = []

        length_of_index = len(representationOfConcepts)

        for i in range(1 + Number_of_documents, length_of_index):
            Q = representationOfConcepts[i]
            doc = []
            for j in range(1, 1 + Number_of_documents):
                D = representationOfConcepts[j]
                if np.linalg.norm(D) == 0:
                    continue
                doc.append([Q.dot(D) / (np.linalg.norm(Q) * np.linalg.norm(D)), j])
            doc = sorted(doc, reverse=True)
            arr = np.array(doc, dtype=np.int)
            doc_IDs_ordered.append(arr[:, 1])

        return doc_IDs_ordered