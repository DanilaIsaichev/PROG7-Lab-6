from urllib.request import urlopen
from bs4 import BeautifulSoup
from wordcloud import WordCloud
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

import matplotlib.pyplot as plt
import json


def get_news_text(news_tags: list) -> str:
  """Функция для извлечения содержимого из тэгов страницы с новостью в текстовом виде"""

  news_text = ""

  for news_tag in news_tags:
    tag_text = " ".join(news_tag.text.split())

    if news_tag != "\n" and tag_text != "":
      news_text += tag_text + " "

  return news_text


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

      # Записываем ФИО в зависимости от наличия его частей
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
      
      # Проверка не содержит ли название организации "им"
      if "им " in span.text:
        #Получение и добавление строки с сохранённой формой ФИО после "им "/"им."/"имени"
        organization_name = span.normal[:span.normal.find("имя")] + span.text[span.text.find("им "):]
        organizations.append(organization_name)

      elif "им." in span.text:
        #Получение и добавление строки с сохранённой формой ФИО после "им "/"им."/"имени"
        organization_name = span.normal[:span.normal.find("имя")] + span.text[span.text.find("им."):]
        organizations.append(organization_name)

      elif "имени" in span.text:
        #Получение и добавление строки с сохранённой формой ФИО после "им "/"им."/"имени"
        organization_name = span.normal[:span.normal.find("имя")] + span.text[span.text.find("имени"):]
        organizations.append(organization_name)

      else:
        # Добавление нормализированной строки
        organizations.append(span.normal)

    elif span.type == LOC:
      span.normalize(morph_vocab)
      locations.append(span.normal)

  return {"actors": actors, "locations": locations, "organizations": organizations}

def get_news(file_name = 'news', silent = False):
  """Функция, извлекающая новости и сторящая облака слов"""

  base_url = 'https://www.herzen.spb.ru'

  news = []

  url = base_url + '/main/news/'
  html = urlopen(url)
  bs = BeautifulSoup(BeautifulSoup(html, 'html.parser').prettify(), 'html.parser')

  tags = bs.find_all('a', {'class': 'news_header_link'})

  for tag in tags:

    # Получаем url страницы с новостями
    news_url = base_url + tag['href'].strip()

    news_html = urlopen(news_url)

    news_bs = BeautifulSoup(BeautifulSoup(news_html, 'html.parser').prettify(), 'html.parser')

    news_tags = news_bs.find('div', {'style': 'padding:0 5 5 5px'})

    news_text = get_news_text(news_tags.contents)

    doc = Doc(news_text)

    # Сегментация
    doc.segment(segmenter)

    # Проставление морфологических меток
    doc.tag_morph(morph_tagger)

    # Проставление меток имё собственных
    doc.tag_ner(ner_tagger)

    ner_elements_lists = extract_ners(doc.spans)

    # Счётчик действующих лиц
    news_actors = list(Counter(ner_elements_lists["actors"]))

    # Счётчик мест
    news_locations = list(Counter(ner_elements_lists["locations"]))

    # Счётчик организаций
    news_organizations = list(Counter(ner_elements_lists["organizations"]))

    # Список слов, из которых состоят actors, locations и organizations
    ners_words = []

    for keys in ner_elements_lists:
      for ner_element in ner_elements_lists[keys]:
        for word in ner_element.split(" "):
          # Удаляем служебные символы
          ners_words.append(word.lower().strip("""!()-[]{};?@#$%:'"\,./^&amp;*_"""))

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

    # Вывод в зависимости от наполненности полей
    if len(news_actors) != 0:
      if len(news_locations) != 0:
        if len(news_organizations) != 0:
          news.append(
            {
              "header": tag.get_text().strip(),
              "link": base_url + tag['href'].strip(),
              "text": news_text,
              "actors": news_actors,
              "locations": news_locations,
              "organiztions": news_organizations,
              "key_words": key_words
            })
        else:
          news.append(
            {
              "header": tag.get_text().strip(),
              "link": base_url + tag['href'].strip(),
              "text": news_text,
              "actors": news_actors,
              "locations": news_locations,
              "key_words": key_words
            })
      elif len(news_organizations) != 0:
        news.append(
          {
            "header": tag.get_text().strip(),
            "link": base_url + tag['href'].strip(),
            "text": news_text,
            "actors": news_actors,
            "organiztions": news_organizations,
            "key_words": key_words
          }) 
      else:
        news.append(
            {
              "header": tag.get_text().strip(),
              "link": base_url + tag['href'].strip(),
              "text": news_text,
              "actors": news_actors,
              "key_words": key_words
            })
    elif len(news_locations) != 0:
      if len(news_organizations) != 0:
        news.append(
          {
            "header": tag.get_text().strip(),
            "link": base_url + tag['href'].strip(),
            "text": news_text,
            "locations": news_locations,
            "organiztions": news_organizations,
            "key_words": key_words
          })
      else:
        news.append(
          {
            "header": tag.get_text().strip(),
            "link": base_url + tag['href'].strip(),
            "text": news_text,
            "locations": news_locations,
            "key_words": key_words
          })
    else:
      news.append(
        {
          "header": tag.get_text().strip(),
          "link": base_url + tag['href'].strip(),
          "text": news_text,
          "organiztions": news_organizations,
          "key_words": key_words
        })

    print(news)

    if not silent:
      # Получаем сырую строку с колючевыми словами для облака слов
      text_raw = r" ".join(key_word for key_word in key_words)

      # Генерируем облако слов
      wordcloud = WordCloud().generate(text_raw)

      # Отображаем облако слов
      plt.imshow(wordcloud, interpolation='bilinear')
      plt.axis("off")
      plt.show()

  with open(file_name + '.json', 'w', encoding='utf-8') as file:
    json.dump(news, file, ensure_ascii=False, indent=4)


if __name__ == "__main__":
  segmenter = Segmenter()
  morph_vocab = MorphVocab()

  emb = NewsEmbedding()
  morph_tagger = NewsMorphTagger(emb)
  syntax_parser = NewsSyntaxParser(emb)
  ner_tagger = NewsNERTagger(emb)

  names_extractor = NamesExtractor(morph_vocab)
  
  get_news()
