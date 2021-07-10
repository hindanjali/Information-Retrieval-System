from sentenceSegmentation import SentenceSegmentation
from tokenization import Tokenization
from inflectionReduction import InflectionReduction
from stopwordRemoval import StopwordRemoval
from informationRetrieval import LSA
from evaluation import Evaluation
import random
import time
from ESA import ESA
# import wikipedia
import pandas as pd
from sys import version_info
import argparse
import json
import matplotlib.pyplot as plt

# Input compatibility for Python 2 and Python 3
if version_info.major == 3:
    pass
elif version_info.major == 2:
    try:
        input = raw_input
    except NameError:
        pass
else:
    print("Unknown python version - input function not safe")


class SearchEngine:

    def __init__(self, args):
        self.args = args

        self.tokenizer = Tokenization()
        self.sentenceSegmenter = SentenceSegmentation()
        self.inflectionReducer = InflectionReduction()
        self.stopwordRemover = StopwordRemoval()

        self.evaluator = Evaluation()

        self.latentSemanticAnalysis = LSA()
        self.explicitSemanticAnalysis = ESA()

    def segmentSentences(self, text):
        """
        Call the required sentence segmenter
        """
        if self.args.segmenter == "naive":
            return self.sentenceSegmenter.naive(text)
        elif self.args.segmenter == "punkt":
            return self.sentenceSegmenter.punkt(text)

    def tokenize(self, text):
        """
        Call the required tokenizer
        """
        if self.args.tokenizer == "naive":
            return self.tokenizer.naive(text)
        elif self.args.tokenizer == "ptb":
            return self.tokenizer.pennTreeBank(text)

    def reduceInflection(self, text):
        """
        Call the required stemmer/lemmatizer
        """
        return self.inflectionReducer.reduce(text)

    def removeStopwords(self, text, document_type):
        """
        Call the required stopword remover
        """
        return self.stopwordRemover.fromList(text, document_type)

    def preprocessQueries(self, queries, document_type):
        """
        Preprocess the queries - segment, tokenize, stem/lemmatize and remove stopwords
        """

        # Segment queries
        segmentedQueries = []
        for query in queries:
            segmentedQuery = self.segmentSentences(query)
            segmentedQueries.append(segmentedQuery)
        json.dump(segmentedQueries, open(self.args.out_folder + "segmented_queries.txt", 'w'))
        # Tokenize queries
        tokenizedQueries = []
        for query in segmentedQueries:
            tokenizedQuery = self.tokenize(query)
            tokenizedQueries.append(tokenizedQuery)
        json.dump(tokenizedQueries, open(self.args.out_folder + "tokenized_queries.txt", 'w'))
        # Stem/Lemmatize queries
        reducedQueries = []
        for query in tokenizedQueries:
            reducedQuery = self.reduceInflection(query)
            reducedQueries.append(reducedQuery)
        json.dump(reducedQueries, open(self.args.out_folder + "reduced_queries.txt", 'w'))
        # Remove stopwords from queries
        stopwordRemovedQueries = []
        for query in reducedQueries:
            stopwordRemovedQuery = self.removeStopwords(query, document_type)
            stopwordRemovedQueries.append(stopwordRemovedQuery)
        json.dump(stopwordRemovedQueries, open(self.args.out_folder + "stopword_removed_queries.txt", 'w'))

        preprocessedQueries = stopwordRemovedQueries
        return preprocessedQueries

    def preprocessDocs(self, docs, document_type):
        """
        Preprocess the documents
        """

        # Segment docs
        segmentedDocs = []
        for doc in docs:
            segmentedDoc = self.segmentSentences(doc)
            segmentedDocs.append(segmentedDoc)
        json.dump(segmentedDocs, open(self.args.out_folder + "segmented_docs.txt", 'w'))
        # Tokenize docs
        tokenizedDocs = []
        for doc in segmentedDocs:
            tokenizedDoc = self.tokenize(doc)
            tokenizedDocs.append(tokenizedDoc)
        json.dump(tokenizedDocs, open(self.args.out_folder + "tokenized_docs.txt", 'w'))
        # Stem/Lemmatize docs
        reducedDocs = []
        for doc in tokenizedDocs:
            reducedDoc = self.reduceInflection(doc)
            reducedDocs.append(reducedDoc)
        json.dump(reducedDocs, open(self.args.out_folder + "reduced_docs.txt", 'w'))
        # Remove stopwords from docs
        stopwordRemovedDocs = []
        for doc in reducedDocs:
            stopwordRemovedDoc = self.removeStopwords(doc, document_type)
            stopwordRemovedDocs.append(stopwordRemovedDoc)
        json.dump(stopwordRemovedDocs, open(self.args.out_folder + "stopword_removed_docs.txt", 'w'))

        preprocessedDocs = stopwordRemovedDocs
        return preprocessedDocs

    def handleDatasetLSA(self):
        """
        - preprocesses the queries and documents, stores in output folder
        - invokes the IR system
        - evaluates precision, recall, fscore, nDCG and MAP
          for all queries in the Cranfield dataset
        - produces graphs of the evaluation metrics in the output folder
        """

        # Read queries
        queries_json = json.load(open(args.dataset + "cran_queries.json", 'r'))[:]
        query_ids, queries = [item["query number"] for item in queries_json], \
                             [item["query"] for item in queries_json]
        # Process queries
        processedQueries = self.preprocessQueries(queries, 'docs')

        # Read documents
        docs_json = json.load(open(args.dataset + "cran_docs.json", 'r'))[:]
        doc_ids, docs = [item["id"] for item in docs_json], \
                        [item["body"] for item in docs_json]
        # Process documents
        processedDocs = self.preprocessDocs(docs, 'docs')

        # Build document index
        X = self.latentSemanticAnalysis.buildIndex(processedDocs, processedQueries)
        rank = 250
        doc_IDs_ordered = self.latentSemanticAnalysis.rank(X,
                                                           len(processedDocs), len(processedQueries), rank)

        # Read relevance judements
        qrels = json.load(open(args.dataset + "cran_qrels.json", 'r'))[:]

        # Calculate precision, recall, f-score, MAP and nDCG for k = 1 to 10
        precisions, recalls, fscores, MAPs, nDCGs = [], [], [], [], []
        for k in range(1, 11):
            precision = self.evaluator.meanPrecision(
                doc_IDs_ordered, query_ids, qrels, k)
            precisions.append(precision)

            recall = self.evaluator.meanRecall(
                doc_IDs_ordered, query_ids, qrels, k)
            recalls.append(recall)

            fscore = self.evaluator.meanFscore(
                doc_IDs_ordered, query_ids, qrels, k)
            fscores.append(fscore)

            print("Precision, Recall and F-score @ " +
                  str(k) + " : " + str(precision) + ", " + str(recall) +
                  ", " + str(fscore))
            MAP = self.evaluator.meanAveragePrecision(
                doc_IDs_ordered, query_ids, qrels, k)
            MAPs.append(MAP)
            nDCG = self.evaluator.meanNDCG(
                doc_IDs_ordered, query_ids, qrels, k)
            nDCGs.append(nDCG)
            print("MAP, nDCG @ " +
                  str(k) + " : " + str(MAP) + ", " + str(nDCG))

        # Plot the metrics and save plot
        plt.plot(range(1, 11), precisions, label="Precision")
        plt.plot(range(1, 11), recalls, label="Recall")
        plt.plot(range(1, 11), fscores, label="F-Score")
        plt.plot(range(1, 11), MAPs, label="MAP")
        plt.plot(range(1, 11), nDCGs, label="nDCG")
        plt.legend()
        plt.title("LSA - Cranfield Dataset")
        plt.xlabel("k")
        plt.savefig(args.out_folder + "eval_plot_LSA.png")

        '''
        # Code for finding the best rank for SVD matrix.


        time1 = time.time()
        l = []

        qrels = json.load(open(args.dataset + "cran_qrels.json", 'r'))[:]
        for rank in range(100, 1390, 10):

            doc_IDs_ordered = self.latentSemanticAnalysis.rank(X,
                                                           len(processedDocs), len(processedQueries), rank)

            # Read relevance judements

            # Calculate precision, recall, f-score, MAP and nDCG for k = 1 to 10
            precisions, recalls, fscores, MAPs, nDCGs = [], [], [], [], []
            for k in range(1, 11):
                nDCG = self.evaluator.meanNDCG(
                    doc_IDs_ordered, query_ids, qrels, k)
                nDCGs.append(nDCG)
            maxNDCG = max(nDCGs)

            print('Maximum nDCG = {} at {}'.format(maxNDCG, rank))
            l.append([maxNDCG, rank])
            mx = max(l)
            print("Max Score till now = {} at rank = {}".format(mx[0], mx[1]))
            time2 = time.time()
            print('Time = ', (time2 - time1) // 60, "mins", (time2 - time1) % 60, "secs")

        '''

    def handleDatasetESA(self):
        """
                - preprocesses the queries and documents, stores in output folder
                - invokes the IR system
                - evaluates precision, recall, fscore, nDCG and MAP
                  for all queries in the Cranfield dataset
                - produces graphs of the evaluation metrics in the output folder
                """

        # Read queries

        '''
        This is code to extract all 21981 wikipedia articles based on domain
        and properly clean the data and store it in a json file to save compute time.
        Kept here only for examining purpose.


        Read documents
        docs_json = json.load(open(args.dataset + "cran_docs.json", 'r'))[:]
        title_ids, titles = [item["id"] for item in docs_json], \
                        [item["title"] for item in docs_json]

        # Process documents
        vocabTitles = set()
        processedTitles = self.preprocessDocs(titles, 'docs')

        for docs in processedTitles:
            for word in docs:
                vocabTitles.add(word)
        print(len(vocabTitles))



        vocabBody = set()
        for docs in processedTitles:
            for word in docs:
                vocabBody.add(word)
        print(len(vocabBody))


        wordsToSearch = sorted(list(vocabBody.difference(vocabTitles)))


        print(len(wordsToSearch))
        print(wordsToSearch)

        wikipediaArticles = []
        wikipedia.set_lang("en")
        count = 0
        checkCount = 0
        for word in wordsToSearch:
            print("check count = ", checkCount)
            print('word = ', word)
            searched_pages = wikipedia.search(word, results=4)
            for page in searched_pages:
                try:
                    obj = wikipedia.summary(page)
                    wikipediaArticles.append({
                        'id': count+1,
                        'title': page,
                        'body': obj
                    })

                except:
                    pass
                count+=1
            checkCount += 1
        print(len(wikipediaArticles))

        with open('wikipediaArticlesNew.json', 'w') as outfile:
            json.dump(wikipediaArticles, outfile)

        time1 = time.time()

        docs_json = json.load(open("wikipedia1.json", 'r'))[:]
        article1 = [item["body"] for item in docs_json]
        processedArticles1 = self.preprocessDocs(article1, 'articles')


        docs_json = json.load(open("wikipedia2.json", 'r'))[:]
        article2 = [item["body"] for item in docs_json]
        processedArticles2 = self.preprocessDocs(article2, 'articles')


        docs_json = json.load(open("wikipedia3.json", 'r'))[:]
        article3 = [item["body"] for item in docs_json]
        processedArticles3 = self.preprocessDocs(article3, 'articles')


        docs_json = json.load(open("wikipedia4.json", 'r'))[:]
        article4 = [item["body"] for item in docs_json]
        processedArticles4 = self.preprocessDocs(article4, 'articles')


        docs_json = json.load(open("wikipedia5.json", 'r'))[:]
        article5 = [item["body"] for item in docs_json]
        processedArticles5 = self.preprocessDocs(article5, 'articles')


        docs_json = json.load(open("wikipedia6.json", 'r'))[:]
        article6 = [item["body"] for item in docs_json]
        processedArticles6 = self.preprocessDocs(article6, 'articles')

        time2 = time.time()

        allArticles = processedArticles1 + processedArticles2 + processedArticles3 + \
                        processedArticles4 + processedArticles5 + processedArticles6

        time3 = time.time()
        print("Time to proccess all articles = ",time2 - time1)
        print("Time to merge all proccessed docs = ", time3 - time2)

        print(len(allArticles))

        s = set()
        numArtices = len(allArticles)
        for i in range(numArtices):
            for j in allArticles[i]:
                s.add(j)


        print("Vocab of all articles = ", len(s))

        alphabetsToRemove = {'禁', 'Ī', '峨', '习', '̩', '|', 'ἔ', 'Η', 'B', 'ἰ', '合', '視', 'ἷ',
                             'す', '′', 'ٌ', '界', 'в', '大', 'ܝ', '（', 'ἠ', 'ɢ', '磁', 'ì', '小',
                             'ն', 'բ', 'ἥ', '書', '𐤮', 'a', 'p', '‐', '嗎', '可', '‒', '偉', '拾',
                             'ŋ', 'λ', '偵', '–', 'ک', '순', 'ל', 'ʉ', '典', '도', 'ộ', '二', 'h',
                             'ť', '戒', '往', '鷄', '健', 'ܡ', 'В', 'ট', 's', 'κ', '耻', 'č', 'ゾ',
                             'ザ', '舞', '綜', 'ם', 'ё', 'э', '人', 'ែ', 'ا', 'ლ', 'ے', 'š', 'ș',
                             '恥', '棋', 'ַ', '皮', '˨', '歌', '戰', '気', 'л', 'ò', 'R', '시', '無',
                             '廷', '始', 'ベ', '業', '誠', 'ɔ', 'ɵ', 'տ', 'ܓ', 'ს', '海', '^', '昊',
                             '女', '虽', 'き', 'あ', '島', 'Ｔ', '家', 'ɾ', '物', 'Ž', '宅', 'ീ',
                             '战', '̹', '世', '밖', 'พ', 'း', '…', 'θ', '鍾', '方', '֔', 'ç', '様',
                             'ˠ', 'Č', '玉', 'Վ', '争', '陳', 'ï', '油', 'ט', 'à', '‡', '愛', '义',
                             'ट', '双', 'Ŭ', '街', 'а', 'サ', 'め', 'ʋ', '色', 'כ', 'ὀ', '目', 'Ρ',
                             'ɥ', '奶', 'к', 'Π', 'х', 'б', 'ぐ', 'ד', '故', 'ᾶ', 'ˤ', '義', 'ּ',
                             '火', 'þ', 'ǫ', 'ה', '体', 'ن', '采', 'ュ', '≠', 'H', 'Á', '厲', '山',
                             'ܟ', '⋅', 'ட', 'پ', 'Υ', '״', '城', '있', ' ', '份', 'ŭ', 'պ', 'Ф',
                             '맨', 'ム', 'ጽ', '긍', 'ḏ', 'ĕ', 'ἀ', '남', '尽', '이', '之', '아',
                             'ा', '♯', 'ܬ', '明', 'ש', '～', '់', 'ɡ', '壓', '手', 'ṇ', '프', '極',
                             'ב', '四', '⟨', '决', 'Ṣ', '森', '诺', '𐤣', 'आ', 'ʻ', '盾', '序', 'ս',
                             'ق', 'ת', '바', '许', 'स', 'ִ', 'ă', 'ô', 'р', 'ς', '윤', 'ᓂ', 'r', 'О',
                             '湯', '영', '脳', '니', 'å', '皇', '絶', 'Ａ', 'η', '好', 'ナ', 'н', 'み',
                             '·', '社', 'ܫ', 'ὶ', '撃', '術', '種', 'ガ', 'ĭ', 'ソ', '荣', 'ὐ', 'ʔ',
                             'ت', 'コ', 'Ἀ', '소', 'ഹ', 'ɭ', '美', '̠', 'Ἱ', '爭', 'ḍ', '日', 'ǧ',
                             'ɨ', 'メ', '℘', '한', '改', '₂', 'ﻟ', 'נ', 'ṃ', 'ോ', 'ᑲ', 'Ἰ', '台',
                             'マ', 'g', 'е', 'ভ', 'ä', '）', 'の', 'ὑ', 'ή', 'π', 'Ｎ', '지', '舰',
                             '瞳', 'เ', 'י', '狐', '个', '트', 'ð', 'ὺ', 'ḗ', '李', '露', 'く', 'ө',
                             'ⲓ', '仔', 'ก', '・', '오', 'は', 'س', 'ズ', 'ိ', '浪', 'т', 'ी', 'ம',
                             'Ἅ', 'ٹ', '全', '东', 'ं', '天', '古', '士', 'צ', '마', 'õ', 'お', '분',
                             'ⲥ', '记', 'მ', '活', 'መ', 'x', '汶', 'प', 'n', '空', 'џ', '採', '̥',
                             '調', 'Я', '絞', '凱', 'ẽ', 'ֵ', '祖', 'न', '衛', 'ш', '법', 'ֹ', 'ร',
                             '雙', 'ة', 'た', '国', '岛', '藩', 'か', 'ೆ', 'ﺓ', '清', '“', 'ܨ',
                             '\u200c', '帝', 'կ', 'ξ', '坂', '彭', '계', 'ደ', 'Ō', '戦', '航', 'Д',
                             '祝', '듀', 'ワ', '浩', '께', 'ч', '်', '래', 'ē', '矛', 'ว', 'ខ', '기',
                             '飛', 'k', 'ǵ', 'ค', '𐤱', '軽', 'ឱ', 'セ', 'ب', 'ן', '르', 'ಗ', '″',
                             'ꦛ', '⁽', 'έ', '+', 'ջ', 'じ', 'و', '廸', '師', 'А', '⦵', 'w', '機',
                             'أ', '\u200b', 'リ', '❤', '답', '슈', '頭', 'ष', 'ṯ', 'に', '想', 'ɛ',
                             '和', 'Q', 'シ', 'ׁ', 'ⲱ', 'ɴ', 'Đ', 'ە', '行', '֫', '思', '振', '전',
                             'ἱ', 'ド', 'Š', 'Æ', 'ń', 'ḥ', 'ך', '斗', 'ō', '反', '法', 'Ş', '少',
                             '香', '憲', '理', 'ܥ', '管', 'ø', '姫', '勇', 'ĩ', '基', '클', '林',
                             'æ', 'ā', '라', 'ツ', '鬥', '钟', '♥', 'j', 'ペ', '遥', '放', '常', 'ա',
                             '準', 'ᠸ', 'ⲑ', 'Β', '̞', '无', 'Ü', '級', 'ク', 'ー', 'ꦕ', 'ğ', '🤖',
                             'ʈ', '•', '射', 'Ḍ', 'A', 'द', '化', '녀', 'ོ', 'ர', '話', 'й', '式',
                             'ἦ', '久', 'ǔ', '株', 'п', '다', 'ȝ', '准', '\u2060', 'Φ', 'ር', 'ι',
                             'E', '티', '限', '巨', 'ë', 'え', 'ὕ', '͡', 'ش', 'ᠤ', '碧', '스', '青',
                             '̃', 'ẓ', '卓', '—', '〇', 'ї', 'ζ', 'ト', 'ி', '批', 'Ц', 'ú', 'Մ',
                             'བ', 'ُ', '伝', 'ত', '國', '鳳', 'ŏ', 'ס', 'β', '庁', '今', 'ん', 'Z',
                             '隱', 'ˌ', 'չ', 'ћ', 'ض', 'ʒ', 'Ѻ', 'α', 'ﺍ', '국', 'ờ', '前', 'آ',
                             '鶯', '倾', '你', '康', 'ã', '틱', 'ン', '们', 'ห', 'Y', 'ɽ', 'ə', '̝',
                             '˧', 'і', 'រ', 'ܪ', 'ظ', '爐', 'i', '̀', 'ᖅ', 'ز', 'ப', '鎮', '副', '−',
                             'ை', '과', '精', '分', '𐤶', 'ָ', 'ί', '‑', '自', 'ɣ', 'ź', '湘', 'പ',
                             '隊', 'テ', '安', 'ំ', 'ⲁ', 'ṣ', '動', 'ﺪ', 'ɬ', '不', 'ま', 'ი', 'ւ',
                             '꧀', 'ɕ', 'Ε', 'ֽ', '없', '《', '”', '運', 'ו', '慎', 'ܢ', '่', 'ノ',
                             '會', '奈', 'ओ', 'ę', 'ص', '八', 'ю', 'ञ', 'מ', 'Ƿ', '៉', '魔', '\u202c',
                             'ܠ', '闘', 'ὖ', 'ᠯ', '示', 'ʾ', 'ா', '会', 'ⲙ', '指', '情', 'ら', '資',
                             '就', '응', '細', '王', '騎', 'ፈ', '袁', 'ר', 'ো', '月', '艦', 'ม',
                             '역', '寛', 'ῶ', '宗', '모', '郭', 'ά', '高', 'ス', '關', '虎', 'ê',
                             '興', '뜻', 'ʱ', 'น', '杜', '育', 'せ', '𐤳', 'Չ', 'ł', 'Ż', 'う', 'ˀ',
                             '澤', '汉', '，', 'Г', '勞', '๊', 'ր', 'â', '_', 'ふ', 'ሄ', '֚', 'ư',
                             '주', 'ꦩ', '嵋', 'ジ', 'ु', '間', 'é', 'ꦥ', 'ī', 'ň', '査', 'Ч', '刺',
                             'մ', '村', 'パ', 'Р', 'ʂ', 'ג', '~', 'ῷ', 'よ', 'ı', 'φ', 'า', 'ɟ', 'א',
                             'ɳ', 'ְ', 'ك', '如', 'ⲗ', 'ក', 'ग', 'ļ', '̌', '급', '毒', 'ʐ', 'ׂ', 'م',
                             '快', 'ⲏ', '俺', '失', '儂', 'Λ', '密', 'Î', '̑', '문', 'ﺠ', '》', '樑',
                             '由', 'ङ', '三', '〜', 'Ǝ', 'ь', '総', 'ф', 'ὸ', 'đ', 'Ш', '게', 'ʷ',
                             'Ս', 'ず', 'と', '省', '简', 'ء', 'გ', 'び', '遺', 'മ', 'Ј', 'V', 'ꦒ',
                             '主', 'O', 'ʁ', 'ّ', '院', '对', 'ピ', 'Х', '÷', 'օ', 'ᖏ', '太', '’',
                             'し', 'ミ', '字', 'ल', 'க', 'њ', 'キ', '使', '퍼', '嚴', '️', 'ν', 'î',
                             'µ', '贫', 'ṭ', '만', 'ἕ', '権', '일', 'ブ', '蛋', '里', '峠', 'ř', 'ハ',
                             'd', 'δ', 'ꦴ', 'ォ', 'í', 'b', 'ů', 'ž', 'Ġ', 'င', 'ო', 'ज', '⁾', '井',
                             '해', '∆', 'Σ', '車', '𐤯', 'ɷ', 'َ', '压', 'ц', 'غ', '一', '桜', '千',
                             'က', 'Ç', '討', '革', 'श', 'Ἄ', 'អ', '選', 'ロ', 'ِ', '𐤠', '말', 'ć',
                             '絵', '족', 'U', 'ῥ', 'ല', 'и', 'ษ', '足', 'ɫ', 'ἴ', 'τ', 'ヴ', '翼',
                             '입', 'ര', 'も', 'ボ', 'ゲ', 't', 'ശ', 'ק', '團', 'ダ', '学', 'σ', 'І',
                             'ж', 'ъ', '料', '壌', 'ö', '港', '電', '×', 'ꦃ', '吳', 'ÿ', 'Ж', '的',
                             'f', '动', 'ě', 'ὅ', '格', '£', 'ף', '식', 'D', '寿', 'И', 'Ó', 'ြ',
                             'च', 'ş', 'ז', 'ą', '방', 'у', 'Ø', 'ད', '⁄', 'Ｕ', '²', '̂', 'დ', 'گ',
                             '钱', '반', 'イ', 'ไ', '年', '∗', 'ɯ', '֥', 'ɦ', 'Ω', '„', '族', 'Т',
                             '去', 'ち', '時', 'ի', 'भ', 'û', 'プ', '文', 'F', 'ی', 'ϕ', 'д', '編',
                             'け', 'Κ', 'с', 'з', 'ח', '探', '米', 'だ', 'á', 'ó', 'ý', 'ำ', '∂',
                             '死', 'о', 'จ', '際', 'ാ', 'T', '語', '͜', '角', 'ئ', 'る', '运', '央',
                             '닝', 'ǒ', '噴', 'ニ', 'ꦶ', 'ẖ', 'ы', 'ո', 'ꦁ', 'ɑ', 'य', 'Ά', 'ح', '略',
                             '解', 'G', '中', 'ʝ', 'ネ', '德', '끌', 'W', '座', 'ѻ', 'ტ', '은', '集',
                             'ץ', 'С', '很', 'ビ', '½', '旅', '짓', 'É', 'ñ', '대', 'ポ', 'ې', '级',
                             'ύ', 'У', '→', 'ῆ', '恵', '韓', 'Լ', '録', '하', 'Ä', '所', '貧', 'ő',
                             '밀', 'ヨ', '養', '榮', 'ာ', '剿', '་', 'क', '疾', '위', '梯', 'သ',
                             '埴', '司', 'ख', '戸', '런', 'ភ', '崎', 'ა', '向', '吗', '信', 'ֲ', 'Θ',
                             'Ν', '神', '川', 'ܐ', '徳', '∠', 'ե', 'ܚ', 'Ɛ', 'ȟ', '告', 'タ', '☠',
                             '區', '超', '炉', '성', 'や', 'e', 'Ś', 'േ', 'ˑ', 'ǐ', 'ड', 'Е', '道',
                             '敏', 'ˈ', '자', 'Ṭ', '֖', '̪', 'ε', '境', '進', 'ψ', '雄', '奪', '九',
                             '潘', '龍', 'M', 'व', '로', 'ấ', '刻', '彼', 'Δ', 'ア', '็', 'ر', 'ầ',
                             'J', '心', '♭', '֣', 'ゼ', 'レ', 'ѣ', 'カ', '랑', '名', 'К', '恋', '괴',
                             'ێ', 'ぎ', 'ァ', '甲', 'ج', 'N', 'ത', '३', 'Α', 'د', '거', 'ע', 'ω',
                             '‿', '田', 'Э', 'Н', 'ย', 'ე', '零', 'ᠶ', '午', '怀', 'ܘ', 'ś', 'ց',
                             'ক', 'ន', 'ط', '事', 'へ', '雞', '子', 'ದ', 'զ', '瑶', 'ᠡ', 'ū', 'အ',
                             'ὁ', 'ű', 'ꦤ', '白', 'ह', 'ј', '武', 'ṅ', 'ù', '代', '사', '𐤭', '电',
                             'ع', '盤', 'Ա', '保', 'ግ', 'ः', 'я', 'ο', 'ả', '我', '在', 'ṽ', 'म',
                             '岭', 'Ö', 'È', '综', '郎', 'z', '有', '쟁', 'l', 'X', '繁', '船', 'ッ',
                             '科', 'ស', 'K', 'ֻ', 'ョ', '百', 'で', '்', 'ῖ', 'М', 'ბ', 'ि', '政',
                             'u', 'ʌ', 'ʲ', 'ʎ', 'ท', 'Կ', '旺', 'オ', 'y', '正', '永', 'Μ', '청',
                             'ា', '況', 'ل', 'ى', 'ᖠ', '`', 'フ', '研', '仙', '탕', '張', 'ċ', '通',
                             'œ', 'Ł', '립', '下', 'ꦭ', 'Ｏ', 'ġ', 'ክ', 'ǎ', '표', 'ٱ', 'ᠭ', 'ہ',
                             '辱', 'শ', '象', '究', 'ܣ', 'थ', 'v', 'ᕿ', 'ܒ', '奋', '龙', '起', 'q',
                             '江', 'μ', '発', '์', '重', 'Ю', '观', '조', '향', '風', '股', '挑', '्',
                             '觀', '∞', 'さ', '우', '飞', '־', '生', '志', '本', 'ု', 'ᴜ', 'त', 'ʼ',
                             'χ', 'ᠰ', '삼', '论', '爛', 'ャ', 'ʃ', 'ي', 'ძ', 'м', '⟩', '博', '蘇',
                             'ώ', '七', '務', 'バ', 'C', '\xad', '고', '풀', 'ィ', '°', 'っ', 'ល',
                             '𐤦', '特', 'ⲭ', 'ː', '望', 'ἡ', 'ば', 'こ', 'ﺴ', 'ố', '镇', '്', 'ゥ',
                             '蒼', 'บ', '蔘', 'ܛ', 'Б', '当', 'エ', '孔', '֑', '정', 'ሐ', '\u200d',
                             '平', 'ἄ', 'ვ', '쩌', 'լ', 'Γ', 'ֿ', 'Τ', '의', '伟', 'ኖ', 'ܙ', 'ė', '曾',
                             'ό', '˩', 'ล', '證', '洪', 'ラ', 'რ', 'ꦺ', '砲', 'ْ', 'ɪ', 'г', 'Ñ',
                             'ρ', '̯', 'Ḗ', 'خ', '輪', 'த', 'ᠠ', '담', '単', '実', '沖', 'פ', 'إ',
                             'γ', 'ә', '‘', '区', '穗', 'P', '\\', '호', '系', '\u200e', 'ṓ', '異',
                             '와', '逸', '論', '氏', 'S', '現', '̽', 'յ', 'ʰ', '圍', 'ช', 'ण', 'デ',
                             'İ', 'ƕ', 'П', '立', '함', 'Ψ', 'o', 'グ', '木', '¿', 'c', '者', 'チ',
                             'ħ', '宿', 'ắ', 'ध', '上', '여', '兴', '病', '星', 'L', 'ɒ', 'Ι', '記',
                             'ῦ', 'Ā', '淸', 'ß', 'モ', '駅', 'な', '―', 'ᠨ', 'ܕ', 'ܩ', 'Ḥ', '於',
                             'が', 'è', 'ʊ', '底', 'ɐ', '*', 'े', '́', 'ʀ', '무', '张', '≡', 'ֶ', '地',
                             '어', '신', 'ʏ', '造', 'ꜜ', 'ü', '能', 'ف', 'り', '影', '命', '報', '희',
                             'Ｒ', '알', 'ម', 'З', 'ɲ', '时', '׃', '鄭', 'र', '公', '體', '市', '鴬',
                             'ʿ', 'ἶ', 'ဦ', 'ル', '˗', '្', 'ه', 'm', 'ἐ', '駆', '球', '純', 'ք',
                             '扶', '裏', 'Л', '麻', 'I', '型', 'ง', '巻', 'ホ', '暮', 'い', '彰',
                             '才', '가', 'Å', 'ʕ', 'अ', 'ဂ', 'ो', 'چ', '̄', 'თ', 'υ'}

        Keep = {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
                'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
                '-', "'"}

        alphabetsToRemove = list(alphabetsToRemove.difference(Keep))

        for i in range(numArtices):
            newList = []
            for word in allArticles[i]:
                flag = 0
                for char in alphabetsToRemove:
                    if char in word:
                        flag = 1
                        break
                if flag == 0:
                    newList.append(word)
            allArticles[i] = newList


        s = set()
        numArtices = len(allArticles)
        for i in range(numArtices):
            for j in allArticles[i]:
                s.add(j)

        print("Vocab of all articles after preproccessing = ", len(s))

        with open('wikipediaArticlesProcessed.json', 'w') as outfile:
            json.dump(allArticles, outfile)

        '''

        # Load Queries
        queries_json = json.load(open(args.dataset + "cran_queries.json", 'r'))[:]

        query_ids, queries = [item["query number"] for item in queries_json], \
                             [item["query"] for item in queries_json]

        # Process queries
        processedQueries = self.preprocessQueries(queries, 'docs')

        # Load Cranfield Data
        docs_json = json.load(open(args.dataset + "cran_docs.json", 'r'))[:]
        title_ids, titles = [item["id"] for item in docs_json], \
                            [item["title"] for item in docs_json]

        # Process documents
        processedTitles = self.preprocessDocs(titles, 'docs')

        # Load all Preprocessed wikipedia articles
        allArticles = json.load(open("wikipediaArticlesProcessed.json", 'r'))[:5298]

        self.explicitSemanticAnalysis.buildIndexArticle(allArticles)

        self.explicitSemanticAnalysis.buildIndexConcept(processedTitles, title_ids, processedQueries)

        doc_IDs_ordered = self.explicitSemanticAnalysis.rank(processedTitles, processedQueries)

        # Read relevance judements
        qrels = json.load(open(args.dataset + "cran_qrels.json", 'r'))[:]

        # Calculate precision, recall, f-score, MAP and nDCG for k = 1 to 10
        precisions, recalls, fscores, MAPs, nDCGs = [], [], [], [], []
        for k in range(1, 11):
            precision = self.evaluator.meanPrecision(
                doc_IDs_ordered, query_ids, qrels, k)
            precisions.append(precision)

            recall = self.evaluator.meanRecall(
                doc_IDs_ordered, query_ids, qrels, k)
            recalls.append(recall)

            fscore = self.evaluator.meanFscore(
                doc_IDs_ordered, query_ids, qrels, k)
            fscores.append(fscore)

            print("Precision, Recall and F-score @ " +
                  str(k) + " : " + str(precision) + ", " + str(recall) +
                  ", " + str(fscore))
            MAP = self.evaluator.meanAveragePrecision(
                doc_IDs_ordered, query_ids, qrels, k)
            MAPs.append(MAP)
            nDCG = self.evaluator.meanNDCG(
                doc_IDs_ordered, query_ids, qrels, k)
            nDCGs.append(nDCG)
            print("MAP, nDCG @ " +
                  str(k) + " : " + str(MAP) + ", " + str(nDCG))

        # Plot the metrics and save plot
        plt.plot(range(1, 11), precisions, label="Precision")
        plt.plot(range(1, 11), recalls, label="Recall")
        plt.plot(range(1, 11), fscores, label="F-Score")
        plt.plot(range(1, 11), MAPs, label="MAP")
        plt.plot(range(1, 11), nDCGs, label="nDCG")
        plt.legend()
        plt.title("ESA - Cranfield Dataset")
        plt.xlabel("k")
        plt.savefig(args.out_folder + "eval_plot_ESA.png")


if __name__ == "__main__":

    # Create an argument parser
    parser = argparse.ArgumentParser(description='main.py')

    # Tunable parameters as external arguments
    parser.add_argument('-dataset', default="cranfield/",
                        help="Path to the dataset folder")
    parser.add_argument('-out_folder', default="output/",
                        help="Path to output folder")
    parser.add_argument('-segmenter', default="punkt",
                        help="Sentence Segmenter Type [naive|punkt]")
    parser.add_argument('-tokenizer', default="ptb",
                        help="Tokenizer Type [naive|ptb]")
    parser.add_argument('-custom', action="store_true",
                        help="Take custom query as input")

    # Parse the input arguments
    args = parser.parse_args()

    # Create an instance of the Search Engine
    searchEngine = SearchEngine(args)

    print("For running ESA type 0")
    print("For running LSA type 1")

    n = int(input())

    if n == 0:
        searchEngine.handleDatasetESA()
    else:
        searchEngine.handleDatasetLSA()