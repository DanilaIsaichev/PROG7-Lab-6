"""
import string
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from wordcloud import WordCloud
import matplotlib.pyplot as plt
"""
from natasha import (
    Segmenter,
    MorphVocab,
    
    NewsEmbedding,
    NewsMorphTagger,
    NewsSyntaxParser,
    NewsNERTagger,
    
    PER,
    NamesExtractor,

    Doc
)

segmenter = Segmenter()
morph_vocab = MorphVocab()

emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
syntax_parser = NewsSyntaxParser(emb)
ner_tagger = NewsNERTagger(emb)

names_extractor = NamesExtractor(morph_vocab)


text = """Клара у Карла украла каралы, а Карл у Клары украл кларнет"""

doc = Doc(text)

doc.segment(segmenter)
doc.tag_morph(morph_tagger)
for token in doc.tokens:
    token.lemmatize(morph_vocab)
    print(token.lemma)
doc.tag_ner(ner_tagger)
doc.ner.print()
"""
spec_chars = string.punctuation + '\n\xa0«»\t—…' 

text = "".join([ch for ch in text if ch not in spec_chars])

text_tokens = word_tokenize(text)

russian_stopwords = stopwords.words("russian")

filtered_text = [token for token in text_tokens if not token.lower() in russian_stopwords]

text = nltk.Text(filtered_text)

fdist = FreqDist(text)

fdist.most_common(5)

text_raw = " ".join(text)

wordcloud = WordCloud().generate(text_raw)

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
"""