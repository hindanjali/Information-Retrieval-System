import  numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.extmath import randomized_svd

class LSA():
    def __init__(self):
        self.dictionary = None

    def buildIndex(self, docs,queries):
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
        # print(df['cleaned_docs'])

        vectorizer = TfidfVectorizer(stop_words='english', smooth_idf=True)
        X = vectorizer.fit_transform(df['cleaned_docs'])
        # print(X)
        # print('************************************************************')

        dictionary = vectorizer.get_feature_names()
        self.dictionary = dictionary
        return X

    def rank(self, X, len_docs, len_queries, rank):
        # X = X.toarray()
        # # print(len(X), len(X[:len_docs]))
        # U, Sigma, VT = randomized_svd(X[:len_docs], n_components=rank, n_iter=100, random_state=122)
        # print(U.shape)
        # print(Sigma.shape)
        # print(VT.shape)
        # Sigma = np.diag(Sigma)
        # a = np.dot(U, Sigma)
        # rank_k_approximation = list(np.dot(a, VT))
        # rank_k_approximation.insert(0, self.dictionary)
        # # print("Rank of best - k  = ", np.linalg.matrix_rank(np.array(rank_k_approximation[1:])))
        # # print("Size of SVD matrix  = ", len(rank_k_approximation[1:]), len(rank_k_approximation[1]))
        # # print('************************************************************')
        # # print("rank_k_approximation  = ", rank_k_approximation)


        X = X.toarray()
        U, Sigma, VT = randomized_svd(np.array(X[:len_docs]).T, n_components=rank, n_iter=100, random_state=122)
        # print(U.shape)
        # print(Sigma.shape)
        # print(VT.shape)
        Sigma = np.diag(Sigma)
        rank_k_approximation = list(np.dot(Sigma, VT).T)

        # print("Rank of best - k  = ", np.linalg.matrix_rank(np.array(rank_k_approximation[1:])))
        # print("Size of SVD matrix  = ", len(rank_k_approximation), len(rank_k_approximation[1]))
        # print('************************************************************')
        # print("rank_k_approximation  = ", rank_k_approximation)

        queries = np.array(X[len_docs:])
        # print("lnvlsnklsnklbndfslvnslbnls = ",queries.shape)
        a = np.dot(queries, U)
        transformedQueries = np.dot(a, np.linalg.inv(Sigma))
        # print(transformedQueries.shape)

        for query in transformedQueries:
            rank_k_approximation.append(query)

        rank_k_approximation.insert(0, self.dictionary[rank])

        # print("Size of Final matrix  = ", len(rank_k_approximation[1:]), len(rank_k_approximation[1]))

        doc_IDs_ordered = []
        for i in range(1 + len_docs, 1 + len_docs + len_queries):
            Q = rank_k_approximation[i]
            doc = []
            for j in range(1, 1 + len_docs):
                D = rank_k_approximation[j]
                if np.linalg.norm(D) == 0:
                    continue
                doc.append([Q.dot(D) / (np.linalg.norm(Q) * np.linalg.norm(D)), j])
            doc = sorted(doc, reverse=True)
            arr = np.array(doc, dtype=np.int)
            doc_IDs_ordered.append(arr[:, 1])


        return doc_IDs_ordered