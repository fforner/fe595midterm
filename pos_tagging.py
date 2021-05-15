import spacy
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

#reference link: https://towardsdatascience.com/named-entity-recognition-with-nltk-and-spacy-8c4a7d88e7da

def pos_tagging(sent):
    sent = nltk.word_tokenize(sent)
    sent = nltk.pos_tag(sent)
    return sent