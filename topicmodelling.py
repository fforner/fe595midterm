from wiki_request import *
import nltk
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
from nltk import FreqDist
nltk.download('stopwords') # run this one time
import pandas as pd
pd.set_option("display.max_colwidth", 200)
import numpy as np
import re
import spacy
import gensim
from gensim import corpora
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
stop_words = stopwords.words('english')


wiki_link = "https://en.wikipedia.org/wiki/Donald_Trump"
# function to remove stopwords
def remove_stopwords(rev):
    rev_new = " ".join([i for i in rev if i not in stop_words])
    return rev_new

def topic_modelling(wiki_link = wiki_link):
    wiki_content = scrape_wiki(wiki_link)
    #content = ''
    #for i in range(len(wiki_content)):
    #    content = content + wiki_content[i][1] + ' '
    # remove unwanted characters, numbers and symbols
    wiki_content['text'] = wiki_content['text'].str.replace("[^a-zA-Z#]", " ")

    # remove short words (length < 3)
    wiki_content['text'] = wiki_content['text'].apply(lambda x: ' '.join([w for w in x.split() if len(w) > 2]))
    # remove stopwords from the text
    content = [remove_stopwords(r.split()) for r in wiki_content['text']]
    content = [r.lower() for r in content]
    freq_words(content)
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

    def lemmatization(texts, tags=['NOUN', 'ADJ']):  # filter noun and adjective
        output = []
        for sent in texts:
            doc = nlp(" ".join(sent))
            output.append([token.lemma_ for token in doc if token.pos_ in tags])
        return output
    tokenized_content = pd.Series(content).apply(lambda x: x.split())
    content_2 = lemmatization(tokenized_content)
    content_3 = []
    for i in range(len(content_2)):
        content_3.append(' '.join(content_2[i]))

    wiki_content['text'] =content_3
    freq_words(wiki_content['text'], 35)
    dictionary = corpora.Dictionary(content_2)
    doc_term_matrix = [dictionary.doc2bow(rev) for rev in content_2]
    # Creating the object for LDA model using gensim library
    LDA = gensim.models.ldamodel.LdaModel

    # Build LDA model
    lda_model = LDA(corpus=doc_term_matrix, id2word=dictionary, num_topics=4, random_state=100,
                    chunksize=1000, passes=50)
    print(1)


def freq_words(x, terms = 30):
    all_words = ' '.join([text for text in x])
    all_words = all_words.split()

    fdist = FreqDist(all_words)
    words_df = pd.DataFrame({'word':list(fdist.keys()), 'count':list(fdist.values())})

    # selecting top 20 most frequent words
    d = words_df.nlargest(columns="count", n = terms)
    plt.figure(figsize=(20,5))
    ax = sns.barplot(data=d, x= "word", y = "count")
    ax.set(ylabel = 'Count')
    plt.show()

topic_modelling()
