# from util import *

from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')


class StopwordRemoval():

    def fromList(self, text, document_type):
        """
        Sentence Segmentation using the Punkt Tokenizer

        Parameters
        ----------
        arg1 : list
            A list of lists where each sub-list is a sequence of tokens
            representing a sentence

        Returns
        -------
        list
            A list of lists where each sub-list is a sequence of tokens
            representing a sentence with stopwords removed
        """


    # Fill in code here

        stop_words = set(stopwords.words("english"))

        import time

        time1 = time.time()
        stopwordRemovedText = []
        remove_from_stopwords = [',', '(', ')', '0', '1', '2', '3', '4', '5', '6', '7',
                                 '8', '9', '=', '.', ';', "'s", "'"]

        for i in text:
            for word in i:
                word = word.lower()
                if word not in stop_words:
                    word = word.replace('/', ' ')
                    word = word.replace('-', ' ')
                    word = word.strip()
                    flag = 0
                    for char in remove_from_stopwords:
                        if char in word or len(word) <= 2:
                            flag = 1
                            break
                    if flag == 0:
                        stopwordRemovedText.append(word)

        return stopwordRemovedText