# Import telebot dari pyTelegramBotAPI
from aiogram import Bot, Dispatcher, executor, types
from scipy.fft import set_backend
from telebot import*

# Import word_tokenize dari NLTK
from nltk.tokenize import word_tokenize
from nltk.tag import CRFTagger

# Import wordnet dari nltk.corpus
from nltk.corpus import wordnet as wn

# Library
import string
import re
import matplotlib.pyplot as plt

# Import pandas
import pandas as pd

# Import Ontology/RDF
from rdflib import Graph

from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import seaborn as sn
import numpy as np

API_KEY = '5334960092:AAFNbId6NjUhliSz63L1UUSEldz5yAzoc8w'
bot = Bot(token=API_KEY)
sb = Dispatcher(bot)


exit_list = ['exit', 'see you later', 'bye', 'quit',
             'break', 'stop', 'finish', 'selesai', 'berhenti']


def huruf_kecil(message):
    return message.lower()


def hapus_angka(message):
    h_angka = re.sub(r'\d+', '', message)
    return h_angka


def hapus_tanda_baca(message):
    translator = str.maketrans('', '', string.punctuation)
    return message.translate(translator)


def hapus_spasi(message):
    hapus_spasi_putih = " ".join(message.split())
    return hapus_spasi_putih


def postag(message):
    c_tagger = CRFTagger()
    c_tagger.set_model_file('all_indo_man_tag_corpus_model.crf.tagger')
    return c_tagger.tag_sents([message])


def preprocessing(message):
    message = huruf_kecil(message)
    message = hapus_angka(message)
    message = hapus_tanda_baca(message)
    message = hapus_spasi(message)
    message = word_tokenize(message)
    message = postag(message)
    return message[0]


pertanyaan = pd.read_csv('dataset.csv')
pertanyaan = pertanyaan.fillna('0')

pertanyaan['preprocess pertanyaan'] = pertanyaan.text.apply(preprocessing)

sub = pertanyaan['preprocess pertanyaan'].iloc[0]


def get_feature(sub):
    try:
        QW = sub[0][0]
        QW_bigram = [item[0] for item in sub[:2]][0] + \
            ' ' + [item[0] for item in sub[:2]][1]
        QW_trigram = [item[0] for item in sub[:3]][0] + ' ' + [item[0]
                                                               for item in sub[:3]][1] + ' ' + [item[0] for item in sub[:3]][2]
        QW_fourgram = [item[0] for item in sub[:4]][0] + ' ' + [item[0] for item in sub[:4]
                                                                ][1] + ' ' + [item[0] for item in sub[:4]][2] + ' ' + [item[0] for item in sub[:4]][3]
        # QW_fivegram = [item[0] for item in sub[:5]][0] + ' ' + [item[0] for item in sub[:5]][1] + ' ' + [item[0]
        #                                                                                                  for item in sub[:5]][2] + ' ' + [item[0] for item in sub[:5]][3] + ' ' + [item[0] for item in sub[:5]][4]
        tag = sub[0][1]
        post_tag = sub[1][1]
        return [QW, QW_bigram, QW_trigram, QW_fourgram, tag, post_tag]
    except:
        print("Pertanyaan Kurang Jelas")


feature = pertanyaan['preprocess pertanyaan'].apply(get_feature)
feature = pd.DataFrame([item for item in feature.values], columns=[
                       'QW', 'QW-bigram', 'QW-trigram', 'QW-fourgram', 'tag', 'pos tag'])

matriks_feature = pd.get_dummies(feature, prefix='', prefix_sep='')

svc_subject = LinearSVC()


@sb.message_handler(commands='start')
async def start(message: types.Message):
    await message.answer(f"Selamat Datang {message.from_user.full_name} !!\n\nSaya Bot Sumedang Larang.\n\nJika ingin bertanyaan seputar kerajaan Sumedang Larang.\nGunakan bahasa indonesia yang baku")


@sb.message_handler(commands='mulai')
async def mulai(message: types.Message):
    await message.answer(f"Selamat Datang {message.from_user.full_name} !!\n\nSaya Bot Sumedang Larang.\n\nJika ingin bertanyaan seputar kerajaan Sumedang Larang.\nGunakan bahasa indonesia yang baku")


@sb.message_handler(commands='help')
async def help(message: types.Message):
    await message.answer(f"Halo {message.from_user.full_name} !!\n\nAda yang bisa dibantu?")


@sb.message_handler(content_types=exit_list)
async def help(message: types.Message):
    await message.answer(f"Terima kasih {message.from_user.full_name}, Semoga membantu ya")


@set_backend.message_handler(content_types='text')
async def test(message: types.Message):
    text = message.text

    while True:
        svc_subject = LinearSVC()
        svc_subject.fit(matriks_feature, pertanyaan['subject'])

        svc_object = LinearSVC()
        svc_object.fit(matriks_feature, pertanyaan['object'])

        svc_katakunci = LinearSVC()
        svc_katakunci.fit(matriks_feature, pertanyaan['katakunci'])

        def extract_feature(input):
            new_feature = feature.to_numpy().tolist().copy()
            new_feature.append(get_feature(preprocessing(text)))
            new_feature = pd.DataFrame([item for item in new_feature], columns=[
                                       'QW', 'QW-bigram', 'QW-trigram', 'QW-fourgram', 'tag', 'pos tag'])
            return pd.get_dummies(new_feature, prefix='', prefix_sep='')

        def prediksi(model_subject, model_object, model_katakunci, teks_user):
            try:
                ef = extract_feature(teks_user)
                try:
                    return model_subject.predict(ef)[-1], model_object.predict(ef)[-1], model_katakunci.predict(ef)[-1]
                except:
                    model_subject = LinearSVC()
                    model_subject.fit(ef[:-1], pertanyaan['subject'])

                    model_object = LinearSVC()
                    model_object.fit(ef[:-1], pertanyaan['object'])

                    model_katakunci = LinearSVC()
                    model_katakunci.fit(ef[:-1], pertanyaan['katakunci'])
                return model_subject.predict(ef)[-1], model_object.predict(ef)[-1], model_katakunci.predict(ef)[-1]
            except:
                print('Ekstraksi fitur gagal')

        predicted = prediksi(svc_subject, svc_object, svc_katakunci, text)

        subject = predicted[0]
        object = predicted[1]
        katakunci = predicted[2]

        if text in exit_list:
            await message.answer(f"Terima kasih {message.from_user.full_name}, Semoga sehat selalu ya :)\n\nSalam Hormat\nSulaBot :)")
            break
        else:
            g = Graph()
            g.parse('sumedang-larang.owl')

            q = """
                    PREFIX table: <http://www.semanticweb.org/asus/ontologies/2022/4/sumedanglarang#>
                SELECT *
                    WHERE { ?subject 	rdf:type	sumedanglarang:"""+subject+""" ;			
                            sumedanglarang:kata_kunci	?katakunci ;
                            sumedanglarang:"""+object+"""	?object			
                                FILTER(STR(?katakunci) = '"""+katakunci+"""')
                            }
                """

            for r in g.query(q):
                answer = r["object"]

            await message.answer(answer)
            break

if __name__ == "__main__":
    executor.start_polling(sb)


bot.polling()
