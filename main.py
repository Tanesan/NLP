# 自然言語処理(NLP)の一分野である感情分析を使用
# 極性(書き手の意見)に基づいて文書を分類する
# 今回はIMDbの50,000件のレビューで構成されたデータセットを肯定、否定で分類する

import pyprind
import os
import pandas as pd
import numpy as np

# やってることとしてはデータをすべてdata frame にしてるだけ
# basepath = 'aclImdb'
# labels = {'pos': 1, 'neg': 0}
# pbar = pyprind.ProgBar(5000)
# df = pd.DataFrame()
# for s in ('test', 'train'):  # testとtrainのディレクトリ全て行う
#     for i in ('pos', 'neg'):  # posとnegのディレクトリ全て行う
#         path = os.path.join(basepath, s, i)
#         for file in sorted(os.listdir(path)):
#             with open(os.path.join(path, file), 'r', encoding='utf8') as infile:
#                 txt = infile.read()
#             df = df.append([[txt, labels[i]]], ignore_index=True)
#             pbar.update()
# df.columns = ['review', 'sentiment']
#
# np.random.seed(0)
# df = df.reindex(np.random.permutation(df.index))
# df.to_csv('movie_data.csv', index=False, encoding='utf-8')
#
df = pd.read_csv('movie_data.csv', encoding='utf-8')
# print(df.head(3))
# print(df.shape)

# BoW
# 文章の集合全体から、例えば、単語という一意なトークンからなる語彙を作成する
# 各文書での各単語の出現回数を含んだ特徴量ベクトルを構築する
from sklearn.feature_extraction.text import CountVectorizer
count = CountVectorizer()
docs = np.array([
    'The sun is shining',
    'The weather is sweet',
    'The sun is shining, the weather is sweet, and one and one is two'
])
bag = count.fit_transform(docs)
# 生の出現頻度
print(count.vocabulary_)
# 特徴量ベクトル(出現頻度)
print(bag.toarray())
# TF-IDF を使って単語の関連性を評価する
# TF(単語の出現頻度) と　IDF(逆文書頻度) の積にして定義する
# tf-idf(t, d) = tf(t, d) * idf(t, d)

#                     n(d)
# idf(t, d) = logーーーーーーーーー
#                 1 + df(d, t)

# n = 文書の総数, df(d, t)は単語tを含んでいる文書dの個数を表す
# 1を足しているのは訓練データに出現するすべての単語に０以外の値を割り当てること、ゼロ割を回避するためである
# logを使用しているのは、頻度の低い文書に過剰な重みが与えられないようにするためである
# sklearnでは、TfidfTransformerクラスという変換器を使用.

from sklearn.feature_extraction.text import TfidfTransforme
tfidf = TfidfTransformer(use_idf=True, norm='l2', smooth_idf=True)
np.set_printoptions(precision=2)
print(tfidf.fit_transform(count.fit_transform(docs)).toarray())

# sklearnでの計算式
#                   1+  n(d)
# idf(t, d) = logーーーーーーーーー
#                 1 + df(d, t)
# tf-idf(t, d) = tf(t, d) * (idf(t, d) + 1 )

# +１はすべての重みを0にするのに役立つ(log(1)=0)
#　計算する前にl2正則化を行う

# テキストクレンジング
df.loc[0, 'review'][-50:]
# このように非英字文字が含まれていることがわかる。
# 感情分析に役立つ顔文字だけ残し、それ以外は消す
import re
def preprocessor(text):
    text = re.sub('<[^>]*>', ' ', text)  #HTMLマークアップ
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = (re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', ''))
    return text

preprocessor(df.loc[0, 'review'][-50 :])
print(preprocessor('</a>This :) is :( a test :-)!'))

def tokenizer(text):
    return text.split()

print(tokenizer('runners like running and thus they run'))

# ここでワードステミングという方法を行う。 単語をすべて原形に変換する (NLTKライブラリで実装)
from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()
def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]
# これで全部が原形になった。
tokenizer_porter('runners like running and thus they run')

# ストップワードの除去
# is and ahs likeのようにあまり関係ないものを削除する
from nltk.corpus import stopwords
stop = stopwords.words('english')
print([w for w in tokenizer_porter('a runner likes running and runs a lot')[-10:] if w not in stop])


# ここで、Bowモデルに基づいて、映画レビューを肯定的、否定的に分類 ロジスティック回帰モデルで求める
# Dataframe Objectを25,000個の練習用文章と25,000個のテスト用に分類
X_train = df.loc[:25000, 'review'].values
y_train = df.loc[:25000, 'sentiment'].values
X_test = df.loc[:25000, 'review'].values
y_test = df.loc[:25000, 'sentiment'].values
# gridsearchSVを使用し、ロジスティック回帰モデルの最適なパラメータ集合を求める
# ここでは5分割交差検証を使用する
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer


tfidf = TfidfVectorizer(strip_accents=None,
                        lowercase=False,
                        preprocessor=None)

param_grid = [{'vect__ngram_range': [(1, 1)],
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [tokenizer, tokenizer_porter],
               'clf__penalty': ['l1', 'l2'],
               'clf__C': [1.0, 10.0, 100.0]},
              {'vect__ngram_range': [(1, 1)],
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [tokenizer, tokenizer_porter],
               'vect__use_idf':[False],
               'vect__norm':[None],
               'clf__penalty': ['l1', 'l2'],
               'clf__C': [1.0, 10.0, 100.0]},
              ]

lr_tfidf = Pipeline([('vect', tfidf),
                     ('clf', LogisticRegression(random_state=0, solver='liblinear'))])

gs_lr_tfidf = GridSearchCV(lr_tfidf, param_grid,
                           scoring='accuracy',
                           cv=5,
                           verbose=2,
                           n_jobs=-1)

gs_lr_tfidf.fit(X_train, y_train)

# 終了後には性能指標がもっとも高くなるパラメータセットを出力できる　
print('Best parameter set: %s ' % gs_lr_tfidf.best_params_)
print('CV Accuracy: %.3f' % gs_lr_tfidf.best_score_)

