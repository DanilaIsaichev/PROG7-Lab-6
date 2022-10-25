# -*- coding: utf-8 -*-

from wordcloud import WordCloud
import matplotlib.pyplot as plt

from natasha import (
    Segmenter,
    MorphVocab,
    
    NewsEmbedding,
    NewsMorphTagger,
    NewsSyntaxParser,
    NewsNERTagger,
    
    PER,
    ORG,
    LOC,
    NamesExtractor,

    Doc
)
from typing import Dict
from collections import Counter


def extract_ners(spans: list) -> Dict[str, list]:
    """Функция, принимающая на вход список span'ов и возращающая словарь с ключами actors, locations, organizations"""
    actors = []
    locations = []
    organizations = []
    
    for span in spans:
        if span.type == PER:
            # Нормализируем
            span.normalize(morph_vocab)

            # Извлекаем имя
            span.extract_fact(names_extractor)

            # Парсим имя
            actor_dict = span.fact.as_dict

            if "first" in actor_dict:
                if "middle" in actor_dict:
                    if "last" in actor_dict:
                        actors.append(actor_dict["last"] + " " + actor_dict["first"] + " " + actor_dict["middle"])
                    else:
                        actors.append(actor_dict["last"] + " " + actor_dict["middle"])

                elif "last" in actor_dict:
                    actors.append(actor_dict["last"] + " " + actor_dict["first"])

                else:
                    actors.append(actor_dict["first"])

            elif "middle" in actor_dict:
                if "last" in actor_dict:
                    actors.append(actor_dict["last"] + " " + actor_dict["first"])

            else:
                actors.append(actor_dict["last"])

        elif span.type == ORG:
            span.normalize(morph_vocab)
            
            # Проверка не является ли название организации аббревиатурой
            if span.tokens[0].text.isupper() == True:
                # Добавление ненормализированной строки
                # РГПУ им. А.И. Герцена - ненормализированная строка
                # РГПУ им. А.И. Герцен - нормализированная строка
                organizations.append(span.text)
            else:
                # Добавление нормализированной строки
                organizations.append(span.normal)

        elif span.type == LOC:
            span.normalize(morph_vocab)
            locations.append(span.normal)

        else: print(span.text)

    return {"actors": actors, "locations": locations, "organizations": organizations}

segmenter = Segmenter()
morph_vocab = MorphVocab()

emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
syntax_parser = NewsSyntaxParser(emb)
ner_tagger = NewsNERTagger(emb)

names_extractor = NamesExtractor(morph_vocab)


text = """Уже три дня РГПУ им. А. И. Герцена совместно со школами Санкт-Петербурга реализует Тематическую смену для обучающихся классов психолого-педагогической направленности.
Первый день прошел на базе РГПУ им. А. И. Герцена, второй день- на базе ГБОУ СОШ №27 с углубленным изучением литературы, истории и иностранных языков Василеостровского района им. И.А.Бунина, третий, завершающий день 21 октября, прошёл в ГБОУ СОШ № 47 им. Д.С. Лихачева Петроградского района.
Ученики психолого-педагогических классов и студенты-тьюторы совместно дорабатывали проекты, тренировались защищать работы в виде стендовых докладов. Из учеников, тьюторов и мастеров-преподавателей были сформированы экспертные группы, которые, посетив все мастерские и изучив проекты, задав уточняющие вопросы ребятам, выставили оценки и выбрали несколько лучших проектов.
В рамках подведения итогов директор ГБОУ СОШ № 47 им. Д.С. Лихачева Петроградского района Обухова М.Ю. выступила с напутственным словом, в котором призвала учеников «чтить прошлое, творить настоящее, верить в будущее!», научный руководитель работы с классами психолого-педагогической направленности РГПУ им. А. И. Герцена Ирина Кондракова пожелала ребятам найти свое место в жизни и выбрать правильный путь, после чего обучающиеся презентовали лучшие проекты.
Поздравляем с завершением Тематической смены и желаем ребятам успешной практики!"""

doc = Doc(text)

doc.segment(segmenter)
doc.tag_morph(morph_tagger)
doc.tag_ner(ner_tagger)

ner_elements_lists = extract_ners(doc.spans)

# Счётчик действующих лиц
actors_amount = Counter(ner_elements_lists["actors"])

# Счётчик мест
locations_amount = Counter(ner_elements_lists["locations"])

# Счётчик организаций
organizations_amount = Counter(ner_elements_lists["organizations"])

# Список слов, из которых состоят actors, locations и organizations
ners_words = []

for keys in ner_elements_lists:
    for ner_element in ner_elements_lists[keys]:
        for word in ner_element.split(" "):
            # Удаляем служебные символы
            ners_words.append(word.lower().strip("""!()-[]{};?@#$%:'"\,./^&amp;*_"""))
print(ners_words)

key_words = []
key_words_tokens= []

for token in doc.tokens:
    # Проверка токенов 
    if token.pos not in ["PROPN", "PUNCT", "CCONJ", "ADP", "PART", "PRON", "NUM", "SYM"] and token.rel != 'iobj':
        token.lemmatize(morph_vocab)

        # Удаляем служебные символы
        token.text = token.text.strip("""!()-[]{};?@#$%:'"\,./^&amp;*_""")
        token.lemma = token.lemma.strip("""!()-[]{};?@#$%:'"\,./^&amp;*_""")

        # Проверяем, не встречается ли слово в actors, organizations и locations
        if token.text not in ners_words and token.lemma not in ners_words:
            key_words_tokens.append(token)
            key_words.append(token.lemma)

# Счётчик ключевых слов
key_words_amount = Counter(key_words)

# ТОП 10
if len(actors_amount) < 10:
    print(list(actors_amount.items()))
else:
    print(actors_amount.most_common(10))


if len(locations_amount) < 10:
    print(list(locations_amount.items()))
else:
    print(locations_amount.most_common(10))


if len(organizations_amount) < 10:
    print(list(organizations_amount.items()))
else:
    print(organizations_amount.most_common(10))


if len(key_words_amount) < 10:
    print(list(key_words_amount.items()))
else:
    print(key_words_amount.most_common(10))


text_raw = " ".join(key_word for key_word in key_words_amount.elements())
print(text_raw)
wordcloud = WordCloud().generate(text_raw)

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
