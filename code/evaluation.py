
import  numpy as np
from math import log2
# Add your import statements here




class Evaluation():

	def queryPrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of precision of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The precision value as a number between 0 and 1
		"""

		precision = -1

		#Fill in code here
		count_relevant = 0
		for i in range(k):
			if str(query_doc_IDs_ordered[i]) in true_doc_IDs:
				count_relevant += 1

		precision = count_relevant / k

		return precision


	def meanPrecision(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of precision of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean precision value as a number between 0 and 1
		"""

		meanPrecision = -1
		#Fill in code here

		val = 1
		checkpoints = [0]
		for i in range(len(qrels)):
			if int(qrels[i]["query_num"]) != val:
				checkpoints.append(i)
				val+=1
		checkpoints.append(len(qrels))


		Number_of_queries = len(query_ids)
		collect_all_precisions = np.zeros(Number_of_queries)
		for q in query_ids:
			true_ids = []
			for i in qrels[checkpoints[q - 1]:checkpoints[q]]:
				true_ids.append(i["id"])
			collect_all_precisions[q-1] = self.queryPrecision(doc_IDs_ordered[q-1], q, true_ids, k)

		meanPrecision = collect_all_precisions.mean()

		return meanPrecision

	
	def queryRecall(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of recall of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The recall value as a number between 0 and 1
		"""

		recall = -1

		#Fill in code here
		number_of_true_relevant = len(true_doc_IDs)
		count_relevant = 0
		for i in range(k):
			if str(query_doc_IDs_ordered[i]) in true_doc_IDs:
				count_relevant += 1

		recall = count_relevant / number_of_true_relevant

		return recall


	def meanRecall(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of recall of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean recall value as a number between 0 and 1
		"""

		meanRecall = -1
		#Fill in code here

		val = 1
		checkpoints = [0]
		for i in range(len(qrels)):
			if int(qrels[i]["query_num"]) != val:
				checkpoints.append(i)
				val+=1
		checkpoints.append(len(qrels))


		Number_of_queries = len(query_ids)
		collect_all_recalls = np.zeros(Number_of_queries)
		for q in query_ids:
			true_ids = []
			for i in qrels[checkpoints[q - 1]:checkpoints[q]]:
				true_ids.append(i["id"])
			collect_all_recalls[q-1] = self.queryRecall(doc_IDs_ordered[q-1], q, true_ids, k)

		meanRecall = collect_all_recalls.mean()


		return meanRecall


	def queryFscore(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of fscore of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The fscore value as a number between 0 and 1
		"""

		fscore = -1

		#Fill in code here
		P = self.queryPrecision(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
		R = self.queryRecall(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
		if P+R == 0:
			fscore = 0
		else:
			fscore = (2*P*R)/(P+R)

		return fscore


	def meanFscore(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of fscore of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value
		
		Returns
		-------
		float
			The mean fscore value as a number between 0 and 1
		"""

		meanFscore = -1

		#Fill in code here

		val = 1
		checkpoints = [0]
		for i in range(len(qrels)):
			if int(qrels[i]["query_num"]) != val:
				checkpoints.append(i)
				val+=1
		checkpoints.append(len(qrels))


		Number_of_queries = len(query_ids)
		collect_all_fscores = np.zeros(Number_of_queries)
		for q in query_ids:
			true_ids = []
			for i in qrels[checkpoints[q - 1]:checkpoints[q]]:
				true_ids.append(i["id"])
			collect_all_fscores[q-1] = self.queryFscore(doc_IDs_ordered[q-1], q, true_ids, k)

		meanFscore = collect_all_fscores.mean()

		return meanFscore
	

	def queryNDCG(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of nDCG of the Information Retrieval System
		at given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The nDCG value as a number between 0 and 1
		"""

		nDCG = -1

		#Fill in code here

		relavance_array = np.zeros(k, dtype=np.int)
		found = -1
		idx = 0
		for i in range(k):
			for j in true_doc_IDs:
				if str(query_doc_IDs_ordered[i]) == j["id"]:
					relavance_array[idx] = 5 - j["position"]
			idx +=1
		DCG = 0
		for i in range(k):
			DCG += relavance_array[i]/log2(i+2)


		relavance_array = sorted(relavance_array, reverse=True)

		IDCG = 0
		for i in range(k):
			IDCG += relavance_array[i]/log2(i+2)

		if IDCG == 0:
			nDCG = 0
		else:
			nDCG = DCG / IDCG

		return nDCG


	def meanNDCG(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of nDCG of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean nDCG value as a number between 0 and 1
		"""

		meanNDCG = -1

		#Fill in code here

		val = 1
		checkpoints = [0]
		for i in range(len(qrels)):
			if int(qrels[i]["query_num"]) != val:
				checkpoints.append(i)
				val+=1
		checkpoints.append(len(qrels))


		Number_of_queries = len(query_ids)
		collect_all_nDCGs = np.zeros(Number_of_queries)
		for q in query_ids:
			collect_all_nDCGs[q-1] = self.queryNDCG(doc_IDs_ordered[q-1], q,
													qrels[checkpoints[q - 1]:checkpoints[q]], k)

		meanNDCG = collect_all_nDCGs.mean()

		return meanNDCG


	def queryAveragePrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of average precision of the Information Retrieval System
		at a given value of k for a single query (the average of precision@i
		values for i such that the ith document is truly relevant)

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The average precision value as a number between 0 and 1
		"""

		avgPrecision = -1

		#Fill in code here

		store_precisions = np.zeros(k)
		for i in range(k):
			store_precisions[i] = self.queryPrecision(query_doc_IDs_ordered, query_id, true_doc_IDs, i+1)
		avgPrecision = store_precisions.mean()

		return avgPrecision


	def meanAveragePrecision(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of MAP of the Information Retrieval System
		at given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The MAP value as a number between 0 and 1
		"""

		meanAveragePrecision = -1

		#Fill in code here

		val = 1
		checkpoints = [0]
		for i in range(len(qrels)):
			if int(qrels[i]["query_num"]) != val:
				checkpoints.append(i)
				val+=1
		checkpoints.append(len(qrels))

		Number_of_queries = len(query_ids)
		collect_all_avgprecisions = np.zeros(Number_of_queries)
		for q in query_ids:
			true_ids = []
			for i in qrels[checkpoints[q - 1]:checkpoints[q]]:
				true_ids.append(i["id"])
			collect_all_avgprecisions[q-1] = self.queryAveragePrecision(doc_IDs_ordered[q-1], q, true_ids, k)

		meanAveragePrecision = collect_all_avgprecisions.mean()

		return meanAveragePrecision