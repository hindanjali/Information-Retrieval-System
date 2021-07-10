
# Add your import statements here
from nltk.tokenize import TreebankWordTokenizer

class Tokenization():

	def naive(self, text):
		"""
		Tokenization using a Naive Approach

		Parameters
		----------
		arg1 : list
			A list of strings where each string is a single sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
		"""
		tokenizedText = []
		for i in text:
			tokenizedText.append(i.split(' '))

		return tokenizedText



	def pennTreeBank(self, text):
		"""
		Tokenization using the Penn Tree Bank Tokenizer

		Parameters
		----------
		arg1 : list
			A list of strings where each string is a single sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
		"""

		#Fill in code here
		tokenizer = TreebankWordTokenizer()
		tokenizedText = []
		for i in text:
			tokenizedText.append(tokenizer.tokenize(i))

		for i in tokenizedText:
			if i[-1] == '.':
				i.pop()

		return tokenizedText