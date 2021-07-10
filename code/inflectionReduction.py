
# Add your import statements here

import nltk

nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer


class InflectionReduction:

	def reduce(self, text):
		"""
		Stemming/Lemmatization

		Parameters
		----------
		arg1 : list
			A list of lists where each sub-list a sequence of tokens
			representing a sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of
			stemmed/lemmatized tokens representing a sentence
		"""

		reducedText = []

		lemmatizer = WordNetLemmatizer()
		for i in text:
			single_sentence = []
			for j in i:
				single_sentence.append(lemmatizer.lemmatize(j,pos='v'))
			reducedText.append(single_sentence)

		return reducedText
