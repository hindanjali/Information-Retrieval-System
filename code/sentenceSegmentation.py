
# Add your import statements here
import nltk
from nltk.tokenize import PunktSentenceTokenizer



class SentenceSegmentation():

	def naive(self, text):
		"""
		Sentence Segmentation using a Naive Approach

		Parameters
		----------
		arg1 : str
			A string (a bunch of sentences)

		Returns
		-------
		list
			A list of strings where each string is a single sentence
		"""

		segmentedText = []
		temp1 = []
		question_marks_tokenizer = text.split('?')
		for i in question_marks_tokenizer:
			temp1.append(i)
		temp2 = []
		for i in temp1:
			exclamation_mark_tokenizer = i.split('!')
			for j in exclamation_mark_tokenizer:
				temp2.append(j)
		for i in temp2:
			period_tokenizer = i.split('.')
			for j in period_tokenizer:
				segmentedText.append(j)

		return segmentedText[:-1]





	def punkt(self, text):
		"""
		Sentence Segmentation using the Punkt Tokenizer

		Parameters
		----------
		arg1 : str
			A string (a bunch of sentences)

		Returns
		-------
		list
			A list of strings where each string is a single sentence
		"""

		tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
		segmentedText = tokenizer.tokenize(text)
		
		return segmentedText