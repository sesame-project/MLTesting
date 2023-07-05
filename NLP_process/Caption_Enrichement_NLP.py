import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import wordnet
# from bs4 import BeautifulSoup
import requests
import re
taxonomy=['smile', 'wave','talk', 'sleep', 'sit','laught','jump', 'wear a mask']
import spacy
def synonym_antonym_extractor(phrase):
    synonyms = []
    antonyms = []

    for syn in wordnet.synsets(phrase):
        for l in syn.lemmas():
            print(l)
            synonyms.append(l.name())
            print(syn.definition())
            if l.antonyms():
                antonyms.append(l.antonyms()[0].name())
            token3 = l.name

    print(set(synonyms))
    print(set(antonyms))
    return set(synonyms), set(antonyms), token3
def dismilar(nlp,tax, word1):
    sim= {}
    word = nlp(word1)
    token=''
    for w in word:
        token = w
    for t in tax:
        t = nlp(t)
        sim_=token.similarity(t)
        print("Similarity:", sim_)
        sim[t]=sim_
    sorted_ = sorted(sim.items(), key=lambda x: x[1])

    print(sorted_)
    dis=min(sim, key=sim.get)
    print(dis)

    return dis
def Augment_caption(nlp,taxonomy,sentence):


    doc = nlp(sentence)
    wordtoreplace = {}
    for token in doc:
        if token.pos_ == 'VERB':
            verbtoreplace = token.text
            wordtoreplace["verb"] = verbtoreplace


    for word in wordtoreplace:
        print(wordtoreplace[word])

        rep = dismilar(nlp,taxonomy, wordtoreplace[word])
        ini_=wordtoreplace[word]
        rep=rep.text
        sent = sentence.replace(ini_, rep)

    return sent
def Change_caption(nlp, caption, cat,subcat, rep):
    sentence = str(caption)
    sentence = sentence.lower()
    cat=cat.lower()
    doc = nlp(sentence)

    wordtoreplace = {}
    new=''
    for token in doc:
        if token.text == cat:
            toreplace = token.text
            wordtoreplace[cat] = toreplace
        elif token.text == subcat:
            toreplace = token.text
            wordtoreplace[cat] = toreplace

    for word in wordtoreplace:
        # print(wordtoreplace[word])
        ini_ = wordtoreplace[word]
        rep = str(rep)
        new = sentence.replace(ini_, rep)
    if new=='':
        new='high-fidelity image of '+ sentence
    else:
        new='high-fidelity image of '+new
    print(new)
    return new
def Caption_category(caption, cat, rep):
    sentence = str(caption)
    sentence = sentence.lower()
    cat=cat.lower()

    if sentence != cat:
        new = cat+ rep
    if new=='':
        new='high-fidelity image of '+ cat
    else:
        new='high-fidelity image of '+new

    return new
if __name__ == "__main__":
    nlp = spacy.load("en_core_web_sm")
    taxonomy = ['smiling', 'waving', 'talking', 'sleeping', 'siting', 'laughting', 'jumping', 'wearing a mask']
    sent = Augment_caption(nlp, taxonomy,"the nurse is eating")
    print(sent)