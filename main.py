import sys

import torch
import torchaudio
sys.setrecursionlimit(10000)
import librosa
import numpy as np

import editdistance
from difflib import SequenceMatcher
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from flask import Flask, request, render_template, jsonify, send_file
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Configura o backend para não-GUI
import re
import os
import tempfile
import wave
import pickle
import random
import pandas as pd
from gtts import gTTS
from WordMatching import get_best_mapped_words
import epitran
import noisereduce as nr
from concurrent.futures import ThreadPoolExecutor
import logging
app = Flask(__name__, template_folder="templates", static_folder="static")


# Configuração do logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Variáveis globais para modelos
model_asr, processor_asr, translation_model, tokenizer = None, None, None, None

# Executor para processamento assíncrono
executor = ThreadPoolExecutor(max_workers=1)

# Carregar frases categorizadas e arquivos --------------------------------------------------------------------------------------------------


# Carregar frases aleatórias
try:
    with open('data_de_en_fr.pickle', 'rb') as f:
        random_sentences_df = pickle.load(f)
    # Verificar se é um DataFrame e converter para lista de dicionários
    if isinstance(random_sentences_df, pd.DataFrame):
        random_sentences = random_sentences_df.to_dict(orient='records')
    else:
        random_sentences = random_sentences_df
except Exception as e:
    print(f"Erro ao carregar data_de_en_fr.pickle: {e}")
    random_sentences = []

try:
    with open('frases_categorias.pickle', 'rb') as f:
        categorized_sentences = pickle.load(f)
except Exception as e:
    print(f"Erro ao carregar frases_categorias.pickle: {e}")
    categorized_sentences = {}

# Carregar o Modelo ASR Wav2Vec2 para Francês
processor_asr = Wav2Vec2Processor.from_pretrained("jonatasgrosman/wav2vec2-xls-r-1b-french")
model_asr = Wav2Vec2ForCTC.from_pretrained("jonatasgrosman/wav2vec2-xls-r-1b-french")




'''
# Exemplo com um modelo alternativo
processor_asr = Wav2Vec2Processor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-french")
model_asr = Wav2Vec2ForCTC.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-french")

# Carregar o Modelo ASR Wav2Vec2 para Francês
processor_asr = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-xlsr-53-french")
model_asr = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-xlsr-53-french")

# Este modelo faz parte do projeto VoxPopuli, focado em dados de fala europeus.
processor_asr = Wav2Vec2Processor.from_pretrained("jonatasgrosman/wav2vec2-large-fr-voxpopuli-french")
model_asr = Wav2Vec2ForCTC.from_pretrained("jonatasgrosman/wav2vec2-large-fr-voxpopuli-french")

# Outra variante treinada para o francês, oferecendo diferentes características de desempenho.
processor_asr = Wav2Vec2Processor.from_pretrained("m3hrdadfi/wav2vec2-large-xlsr-53-french")
model_asr = Wav2Vec2ForCTC.from_pretrained("m3hrdadfi/wav2vec2-large-xlsr-53-french")

# Desenvolvido pelo grupo Helsinki-NLP, este modelo também é otimizado para o francês.
processor_asr = Wav2Vec2Processor.from_pretrained("Helsinki-NLP/wav2vec2-large-xlsr-53-french")
model_asr = Wav2Vec2ForCTC.from_pretrained("Helsinki-NLP/wav2vec2-large-xlsr-53-french")

#patrickvonplaten/wav2vec2-large-xlsr-53-french
Este modelo é uma variante do Wav2Vec2 treinada especificamente para o francês.
processor_asr = Wav2Vec2Processor.from_pretrained("patrickvonplaten/wav2vec2-large-xlsr-53-french")
model_asr = Wav2Vec2ForCTC.from_pretrained("patrickvonplaten/wav2vec2-large-xlsr-53-french")


'''



##-----------------------------------------------------------------------------------------------------------
# Iniciar o Epitran e funções de tradução

# Inicializar Epitran para Francês
epi = epitran.Epitran('fra-Latn')

source_language = 'fr'  # Francês
target_language = 'pt'  # Português


# Mapeamento atualizado com regras contextuais
french_to_portuguese_phonemes = {
    # Vogais orais
    'i': {'default': 'i'},
    'e': {'default': 'e'},
    'ɛ': {
        'default': 'é',
        'before_nasal': 'ê',
    },
    'a': {'default': 'a'},
    'ɑ': {'default': 'a'},
    'ɔ': {'default': 'ó'},
    'o': {'default': 'ô'},
    'u': {'default': 'u'},
    'y': {'default': 'u'},
    'ø': {'default': 'eu'},
    'œ': {'default': 'é'},
    'ə': {'default': 'e'},

    # Vogais nasais
    'ɛ̃': {'default': 'ẽ'},
    'ɑ̃': {'default': 'ã'},
    'ɔ̃': {'default': 'õ'},
    'œ̃': {'default': 'ũ'},

    # Semivogais
    'j': {'default': 'i'},
    'w': {'default': 'u'},
    'ɥ': {'default': 'u'},

    # Consoantes
    'b': {'default': 'b'},
    'd': {
    'default': 'd',
    'before_i': 'dj',
},
    'f': {'default': 'f'},
    'g': {'default': 'g'},
    'ʒ': {
        'default': 'j',
        'word_initial': 'j',
        'after_nasal': 'j',
    },
    'k': {
        'default': 'k',
        'before_front_vowel': 'qu',
    },
    'l': {'default': 'l'},
    'm': {'default': 'm'},
    'n': {'default': 'n'},
    'p': {'default': 'p'},
    'ʁ': {
        'default': 'r',
        'word_initial': 'h',
        'after_vowel': 'rr',
        'after_consonant': 'r',
    },
    's': {
    'default': 's',
    'between_vowels': 'z',
},
    't': {
    'default': 't',
    'before_i': 'tch',
},

    'v': {'default': 'v'},
    'z': {'default': 'z'},
    'ʃ': {'default': 'ch'},
    'dʒ': {'default': 'dj'},
    'tʃ': {'default': 'tch'},
    'ɲ': {'default': 'nh'},
    'ŋ': {'default': 'ng'},
    'ç': {'default': 's'},

    # Fonemas compostos
    'sj': {'default': 'ch'},  # Exemplo: "attention"
    'ks': {'default': 'x'},   # Exemplo: "exact"
    # Adicione outros conforme necessário
}

# Características fonéticas
phonetic_features = {
    # Vogais
    'a':  {'type': 'vowel', 'height': 'open', 'backness': 'front',  'rounded': False, 'nasal': False},
    'e':  {'type': 'vowel', 'height': 'close-mid', 'backness': 'front',  'rounded': False, 'nasal': False},
    'i':  {'type': 'vowel', 'height': 'close', 'backness': 'front',  'rounded': False, 'nasal': False},
    'o':  {'type': 'vowel', 'height': 'close-mid', 'backness': 'back',   'rounded': True,  'nasal': False},
    'u':  {'type': 'vowel', 'height': 'close', 'backness': 'back',   'rounded': True,  'nasal': False},
    'ɛ':  {'type': 'vowel', 'height': 'open-mid', 'backness': 'front',  'rounded': False, 'nasal': False},
    'ɔ':  {'type': 'vowel', 'height': 'open-mid', 'backness': 'back',   'rounded': True,  'nasal': False},
    'ɑ':  {'type': 'vowel', 'height': 'open', 'backness': 'back',   'rounded': False, 'nasal': False},
    'ø':  {'type': 'vowel', 'height': 'close-mid', 'backness': 'front',  'rounded': True,  'nasal': False},
    'œ':  {'type': 'vowel', 'height': 'open-mid', 'backness': 'front',  'rounded': True,  'nasal': False},
    'ə':  {'type': 'vowel', 'height': 'mid', 'backness': 'central', 'rounded': False, 'nasal': False},
    'y':  {'type': 'vowel', 'height': 'close', 'backness': 'front',  'rounded': True,  'nasal': False},
    # Vogais nasais
    'ɛ̃': {'type': 'vowel', 'height': 'open-mid', 'backness': 'front',  'rounded': False, 'nasal': True},
    'ɑ̃': {'type': 'vowel', 'height': 'open', 'backness': 'back',   'rounded': False, 'nasal': True},
    'ɔ̃': {'type': 'vowel', 'height': 'open-mid', 'backness': 'back',   'rounded': True,  'nasal': True},
    'œ̃': {'type': 'vowel', 'height': 'open-mid', 'backness': 'front',  'rounded': True,  'nasal': True},
    # Semivogais
    'j':  {'type': 'approximant', 'place': 'palatal', 'voiced': True,  'nasal': False},
    'w':  {'type': 'approximant', 'place': 'labio-velar', 'voiced': True,  'nasal': False},
    'ɥ':  {'type': 'approximant', 'place': 'labio-palatal', 'voiced': True,  'nasal': False},
    # Consoantes
    'p':  {'type': 'consonant', 'place': 'bilabial',     'manner': 'plosive',           'voiced': False},
    'b':  {'type': 'consonant', 'place': 'bilabial',     'manner': 'plosive',           'voiced': True},
    't':  {'type': 'consonant', 'place': 'alveolar',     'manner': 'plosive',           'voiced': False},
    'd':  {'type': 'consonant', 'place': 'alveolar',     'manner': 'plosive',           'voiced': True},
    'k':  {'type': 'consonant', 'place': 'velar',        'manner': 'plosive',           'voiced': False},
    'g':  {'type': 'consonant', 'place': 'velar',        'manner': 'plosive',           'voiced': True},
    'f':  {'type': 'consonant', 'place': 'labiodental',  'manner': 'fricative',         'voiced': False},
    'v':  {'type': 'consonant', 'place': 'labiodental',  'manner': 'fricative',         'voiced': True},
    's':  {'type': 'consonant', 'place': 'alveolar',     'manner': 'fricative',         'voiced': False},
    'z':  {'type': 'consonant', 'place': 'alveolar',     'manner': 'fricative',         'voiced': True},
    'ʃ':  {'type': 'consonant', 'place': 'postalveolar', 'manner': 'fricative',         'voiced': False},
    'ʒ':  {'type': 'consonant', 'place': 'postalveolar', 'manner': 'fricative',         'voiced': True},
    'ʁ':  {'type': 'consonant', 'place': 'uvular',       'manner': 'fricative',         'voiced': True},
    'h':  {'type': 'consonant', 'place': 'glottal',      'manner': 'fricative',         'voiced': False},
    'm':  {'type': 'consonant', 'place': 'bilabial',     'manner': 'nasal',             'voiced': True},
    'n':  {'type': 'consonant', 'place': 'alveolar',     'manner': 'nasal',             'voiced': True},
    'ɲ':  {'type': 'consonant', 'place': 'palatal',      'manner': 'nasal',             'voiced': True},
    'ŋ':  {'type': 'consonant', 'place': 'velar',        'manner': 'nasal',             'voiced': True},
    'l':  {'type': 'consonant', 'place': 'alveolar',     'manner': 'lateral_approximant', 'voiced': True},
    'tʃ': {'type': 'consonant', 'place': 'postalveolar', 'manner': 'affricate',         'voiced': False},
    'dʒ': {'type': 'consonant', 'place': 'postalveolar', 'manner': 'affricate',         'voiced': True},
    'ç':  {'type': 'consonant', 'place': 'palatal',      'manner': 'fricative',         'voiced': False},
    'ɥ':  {'type': 'consonant', 'place': 'labial-palatal', 'manner': 'approximant',     'voiced': True},
    # Outros fonemas
    'x':  {'type': 'consonant', 'place': 'velar',        'manner': 'fricative',         'voiced': False},
    'ʎ':  {'type': 'consonant', 'place': 'palatal',      'manner': 'lateral_approximant', 'voiced': True},
    # Fonemas adicionais conforme necessário
    'ʔ':  {'type': 'consonant', 'place': 'glottal',      'manner': 'plosive',           'voiced': False},
    'θ':  {'type': 'consonant', 'place': 'dental',       'manner': 'fricative',         'voiced': False},
    'ð':  {'type': 'consonant', 'place': 'dental',       'manner': 'fricative',         'voiced': True},
    'ʒ':  {'type': 'consonant', 'place': 'postalveolar', 'manner': 'fricative',         'voiced': True},
    'w':  {'type': 'approximant', 'place': 'labio-velar', 'voiced': True},
    'ɾ':  {'type': 'consonant', 'place': 'alveolar',     'manner': 'tap',               'voiced': True},
    'ʕ':  {'type': 'consonant', 'place': 'pharyngeal',   'manner': 'fricative',         'voiced': True},
}




# Lista de palavras com 'h' aspirado
h_aspirate_words = [
    "hache", "hagard", "haie", "haillon", "haine", "haïr", "hall", "halo", "halte", "hamac",
    "hamburger", "hameau", "hamster", "hanche", "handicap", "hangar", "hanter", "happer",
    "harceler", "hardi", "harem", "hareng", "harfang", "hargne", "haricot", "harnais", "harpe",
    "hasard", "hâte", "hausse", "haut", "havre", "hennir", "hérisser", "hernie", "héron",
    "héros", "hêtre", "heurter", "hibou", "hic", "hideur", "hiérarchie", "hiéroglyphe", "hippie",
    "hisser", "hocher", "hockey", "hollande", "homard", "honte", "hoquet", "horde", "hors",
    "hotte", "houblon", "houle", "housse", "huard", "hublot", "huche", "huer", "huit", "humer",
    "hurler", "huron", "husky", "hutte", "hyène"
]

# Funções de pronúncia e transcrição
def get_pronunciation(word):
    try:
        pronunciation = epi.transliterate(word)
        return pronunciation
    except Exception as e:
        logger.error(f"Erro ao transliterar '{word}': {e}")
        return word  # Retorna a palavra original como fallback

def remove_silent_endings(pronunciation, word):
    # Verificar se a palavra termina com 'ent' e a pronúncia termina com 't'
    if word.endswith('ent') and pronunciation.endswith('t'):
        pronunciation = pronunciation[:-1]
    # Adicionar outras regras conforme necessário
    return pronunciation

def transliterate_and_convert_sentence(sentence):
    words = sentence.split()
    # Tratar apóstrofos
    words = handle_apostrophes(words)
    pronunciations = [get_pronunciation(word) for word in words]
    # Aplicar liaisons
    pronunciations = apply_liaisons(words, pronunciations)
    # Remover terminações silenciosas para cada palavra e pronúncia
    pronunciations = [remove_silent_endings(pron, word) for pron, word in zip(pronunciations, words)]
    # Converter cada pronúncia para português
    pronunciations_pt = [
        convert_pronunciation_to_portuguese(pron)
        for pron in pronunciations
    ]
    return ' '.join(pronunciations_pt)

def split_into_phonemes(pronunciation):
    phonemes = []
    idx = 0
    while idx < len(pronunciation):
        matched = False
        for phoneme in sorted(french_to_portuguese_phonemes.keys(), key=len, reverse=True):
            length = len(phoneme)
            if pronunciation[idx:idx+length] == phoneme:
                phonemes.append(phoneme)
                idx += length
                matched = True
                break
        if not matched:
            phonemes.append(pronunciation[idx])
            idx += 1
    return phonemes


def convert_pronunciation_to_portuguese(pronunciation):
    phonemes = split_into_phonemes(pronunciation)
    result = []
    idx = 0
    length = len(phonemes)
    
    while idx < length:
        phoneme = phonemes[idx]
        mapping = french_to_portuguese_phonemes.get(phoneme, {'default': phoneme})
        context = 'default'
        
        next_phoneme = phonemes[idx + 1] if idx + 1 < length else ''
        
        # Definir listas de vogais
        vowels = ['a', 'e', 'i', 'o', 'u', 'ɛ', 'ɔ', 'ɑ', 'ø', 'œ', 'ə']
        next_is_i = next_phoneme in ['i', 'j', 'ɥ']
        
        # Definir o contexto
        if phoneme in ['t', 'd'] and next_is_i:
            context = 'before_i'
            idx += 1  # Pular o próximo fonema para evitar duplicação
        elif phoneme == 's':
            prev_phoneme = phonemes[idx - 1] if idx > 0 else ''
            prev_is_vowel = prev_phoneme in vowels
            next_is_vowel = next_phoneme in vowels
            if prev_is_vowel and next_is_vowel:
                context = 'between_vowels'
        
        # Obter o mapeamento
        mapped_phoneme = mapping.get(context, mapping['default'])
        result.append(mapped_phoneme)
        idx += 1
    
    return ''.join(result)




def handle_apostrophes(words_list):
    new_words = []
    skip_next = False
    for i, word in enumerate(words_list):
        if skip_next:
            skip_next = False
            continue
        if "'" in word:
            prefix, sep, suffix = word.partition("'")
            # Contrações comuns
            if prefix.lower() in ["l", "d", "j", "qu", "n", "m", "c"]: 
                if i + 1 < len(words_list):
                    combined_word = prefix + suffix + words_list[i + 1]
                    new_words.append(combined_word)
                    skip_next = True
                else:
                    new_words.append(word)
            else:
                new_words.append(word)
        else:
            new_words.append(word)
    return new_words


def apply_liaisons(words_list, pronunciations):
    new_pronunciations = []
    for i in range(len(words_list) - 1):
        current_word = words_list[i]
        next_word = words_list[i + 1]
        current_pron = pronunciations[i]

        # Verificar se a próxima palavra começa com "h" aspirado
        next_word_clean = re.sub(r"[^a-zA-Z']", '', next_word).lower()
        h_aspirate = next_word_clean in h_aspirate_words

        # Verificar se a próxima palavra começa com vogal ou 'h' mudo
        if re.match(r"^[aeiouyâêîôûéèëïüÿæœ]", next_word, re.IGNORECASE) and not h_aspirate:
            # Aplicar liaison
            if current_word.lower() == "les":
                current_pron = current_pron.rstrip('e') + 'z'
            elif current_word[-1] in ['s', 'x', 'z']:
                current_pron = current_pron + 'z'
            elif current_word[-1] == 'd':
                current_pron = current_pron + 't'
            elif current_word[-1] == 'g':
                current_pron = current_pron + 'k'
            elif current_word[-1] == 't':
                current_pron = current_pron + 't'
            elif current_word[-1] == 'n':
                current_pron = current_pron + 'n'
            elif current_word[-1] == 'p':
                current_pron = current_pron + 'p'
            elif current_word[-1] == 'r':
                current_pron = current_pron + 'r'
            # Liaison com 'd' (ex: d'une)
            elif current_word.lower() == "d'" and re.match(r"^[aeiouyâêîôûéèëïüÿæœ]", next_word, re.IGNORECASE):
                current_pron = current_pron.rstrip('e') + 'z'  # Adiciona /z/
        # Caso contrário, não aplica liaison
        new_pronunciations.append(current_pron)
    # Adicionar a última pronúncia
    new_pronunciations.append(pronunciations[-1])
    return new_pronunciations


def normalize_text(text):
    text = text.lower()
    text = text.replace("’", "'")
    # Manter apóstrofos dentro das palavras e remover outros caracteres especiais
    text = re.sub(r"[^\w\s']", '', text)
    # Garantir que não haja espaços desnecessários ao redor dos apóstrofos
    text = re.sub(r"\s+'", "'", text)
    text = re.sub(r"'\s+", "'", text)
    return text.strip()

def remove_punctuation_end(sentence):
    return sentence.rstrip('.')
#--------------------------------------------------------------------------
#funções para comparação fonetica
def compare_phonetics(phonetic1, phonetic2, threshold=0.90):
    alignment_score = needleman_wunsch(phonetic1, phonetic2)
    max_possible_score = max(len(phonetic1), len(phonetic2))
    normalized_score = alignment_score / max_possible_score
    return normalized_score >= threshold

def needleman_wunsch(seq1, seq2, gap_penalty=-1):
    n = len(seq1)
    m = len(seq2)
    score_matrix = [[0] * (m + 1) for _ in range(n + 1)]

    # Inicializar a primeira linha e coluna
    for i in range(n + 1):
        score_matrix[i][0] = gap_penalty * i
    for j in range(m + 1):
        score_matrix[0][j] = gap_penalty * j

    # Preencher a matriz de pontuação
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            match = score_matrix[i - 1][j - 1] + get_similarity(seq1[i - 1], seq2[j - 1])
            delete = score_matrix[i - 1][j] + gap_penalty
            insert = score_matrix[i][j - 1] + gap_penalty
            score_matrix[i][j] = max(match, delete, insert)

    return score_matrix[n][m]

def calculate_similarity(p1, p2):
    features1 = phonetic_features.get(p1)
    features2 = phonetic_features.get(p2)
    if not features1 or not features2:
        return -1  # Penalidade máxima para fonemas desconhecidos
    score = 0
    total = 0

    # Comparar características comuns
    for feature in ['type', 'place', 'manner', 'height', 'backness', 'rounded', 'voiced', 'nasal']:
        if feature in features1 and feature in features2:
            total += 1
            if features1[feature] == features2[feature]:
                score += 1

    # Retornar a proporção de características correspondentes
    return (score / total) * 2 - 1  # Normalizado entre -1 e 1


# Atualizar get_similarity para usar calculate_similarity
def get_similarity(p1, p2):
    if p1 == p2:
        return 1.0
    else:
        return calculate_similarity(p1, p2) - 1  # Subtraia 1 para manter a consistência com penalidades

##--------------------------------------------------------------------------------------------------------------------------------# Processamento de áudio:
# Processamento de áudio:
def reduce_noise(waveform, sample_rate):
    try:
        # Ajustar parâmetros para melhor redução de ruído
        reduced_waveform = nr.reduce_noise(y=waveform, sr=sample_rate, prop_decrease=1.0, stationary=False)
        return reduced_waveform
    except Exception as e:
        logger.error(f"Erro na redução de ruído: {e}")
        return waveform  # Retorna o waveform original em caso de erro


def normalize_waveform(waveform):
    rms = waveform.pow(2).mean().sqrt()
    if rms > 0:
        return waveform / rms
    return waveform

def resample_waveform(waveform, orig_sr, target_sr=16000):
    if orig_sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=target_sr)
        waveform = resampler(waveform)
    return waveform


def process_audio(file_path):
    try:
        # Carregar o áudio
        waveform, sample_rate = torchaudio.load(file_path)
        
        # Converter para mono se necessário
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Resamplear para 16000 Hz
        waveform = resample_waveform(waveform, sample_rate, target_sr=16000)
        sample_rate = 16000  # Atualizar o sample_rate após resamplear
        
        # Redução de ruído
        waveform_np = waveform.squeeze().numpy()
        waveform_np = reduce_noise(waveform_np, sample_rate)
        
        # Converter de volta para tensor
        waveform = torch.tensor(waveform_np, dtype=torch.float32).unsqueeze(0)
        
        # Normalização
        waveform = normalize_waveform(waveform)
        
        # Processamento pelo modelo ASR
        logger.info("Iniciando processamento de áudio pelo modelo ASR")
        inputs = processor_asr(waveform.squeeze(0), sampling_rate=sample_rate, return_tensors="pt")
        with torch.no_grad():
            logits = model_asr(inputs.input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        
        # Decodificação correta
        transcription = processor_asr.decode(predicted_ids[0], skip_special_tokens=True)
        logger.info(f"Transcrição obtida: {transcription}")
        return transcription
    except Exception as e:
        logger.error(f"Erro ao processar áudio: {e}")
        raise e
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


##--------------------------------------------------------------------------------------------------------------------------------
# Rotas de API
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/pronounce', methods=['POST'])
def pronounce():
    text = request.form['text']
    # Certifique-se de que os apóstrofos estão corretos
    text = text.replace("’", "'")
    # Transliterar e converter a frase inteira
    pronunciation = transliterate_and_convert_sentence(text)
    return jsonify({'pronunciations': pronunciation})

@app.route('/get_sentence', methods=['POST'])
def get_sentence():
    try:
        category = request.form.get('category', 'random')

        if category == 'random':
            if random_sentences:
                sentence = random.choice(random_sentences)
                sentence_text = remove_punctuation_end(sentence.get('fr_sentence', "Frase não encontrada."))
            else:
                return jsonify({"error": "Nenhuma frase disponível para seleção aleatória."}), 500
        else:
            if category in categorized_sentences:
                sentences_in_category = categorized_sentences[category]
                sentence_text = random.choice(sentences_in_category)
                sentence_text = remove_punctuation_end(sentence_text)
            else:
                return jsonify({"error": "Categoria não encontrada."}), 400

        return jsonify({'fr_sentence': sentence_text, 'category': category})

    except Exception as e:
        logger.error(f"Erro no endpoint /get_sentence: {e}")
        return jsonify({"error": "Erro interno no servidor."}), 500

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files.get('audio')
    if not file:
        return jsonify({"error": "Nenhum arquivo de áudio enviado."}), 400
    
    # Limitar tamanho do arquivo (por exemplo, 10 MB)
    max_size = 10 * 1024 * 1024  # 10 MB
    file.seek(0, os.SEEK_END)
    file_length = file.tell()
    if file_length > max_size:
        return jsonify({"error": "Arquivo de áudio muito grande. O limite é de 10 MB."}), 400
    file.seek(0)
    
    text = request.form.get('text')
    if not text:
        return jsonify({"error": "Texto de referência não fornecido."}), 400
    
    category = request.form.get('category', 'random')
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(file.read())
        tmp_file_path = tmp_file.name
    
    # Enviar para processamento assíncrono
    future = executor.submit(process_audio, tmp_file_path)
    try:
        transcription = future.result(timeout=120)  # Timeout de 12 segundos
    except Exception as e:
        logger.error(f"Erro ao processar áudio: {e}")  # Log do erro
        return jsonify({"error": "Erro ao processar o áudio."}), 500
    
    # Normalização e comparação de transcrições
    normalized_transcription = normalize_text(transcription)
    normalized_text = normalize_text(text)
    words_estimated = normalized_transcription.split()
    words_real = normalized_text.split()
    mapped_words, mapped_words_indices = get_best_mapped_words(words_estimated, words_real)

    # Inicialização de variáveis para feedback
    diff_html = []
    pronunciations = {}
    feedback = {}
    correct_count = 0
    incorrect_count = 0

    # Comparação palavra a palavra
    for idx, real_word in enumerate(words_real):
        if idx < len(mapped_words):
            mapped_word = mapped_words[idx]
            correct_pronunciation = transliterate_and_convert_sentence(real_word)
            user_pronunciation = transliterate_and_convert_sentence(mapped_word)
            if compare_phonetics(correct_pronunciation, user_pronunciation):
                diff_html.append(f'<span class="word correct" onclick="showPronunciation(\'{real_word}\')">{real_word}</span>')
                correct_count += 1
            else:
                diff_html.append(f'<span class="word incorrect" onclick="showPronunciation(\'{real_word}\')">{real_word}</span>')
                incorrect_count += 1
                feedback[real_word] = {
                    'correct': correct_pronunciation,
                    'user': user_pronunciation,
                    'suggestion': f"Tente pronunciar '{real_word}' como '{correct_pronunciation}'"
                }
            pronunciations[real_word] = {
                'correct': correct_pronunciation,
                'user': user_pronunciation
            }
        else:
            diff_html.append(f'<span class="word missing" onclick="showPronunciation(\'{real_word}\')">{real_word}</span>')
            incorrect_count += 1
            feedback[real_word] = {
                'correct': transliterate_and_convert_sentence(real_word),
                'user': '',
                'suggestion': f"Tente pronunciar '{real_word}' como '{transliterate_and_convert_sentence(real_word)}'"
            }
            pronunciations[real_word] = {
                'correct': transliterate_and_convert_sentence(real_word),
                'user': ''
            }

    diff_html = ' '.join(diff_html)
    total_words = correct_count + incorrect_count
    ratio = (correct_count / total_words) * 100 if total_words > 0 else 0
    completeness_score = (len(words_estimated) / len(words_real)) * 100

    return jsonify({
        'ratio': f"{ratio:.2f}",
        'diff_html': diff_html,
        'pronunciations': pronunciations,
        'feedback': feedback,
        'completeness_score': f"{completeness_score:.2f}"
    })


@app.route('/speak', methods=['POST'])
def speak():
    text = request.form['text']
    tts = gTTS(text=text, lang='fr')
    file_path = tempfile.mktemp(suffix=".mp3")
    tts.save(file_path)
    return send_file(file_path, as_attachment=True, mimetype='audio/mp3')



# Inicialização e execução do aplicativo
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=os.getenv("PORT", default=5000))


    '''
# Exemplo com um modelo alternativo
processor_asr = Wav2Vec2Processor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-french")
model_asr = Wav2Vec2ForCTC.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-french")

# Carregar o Modelo ASR Wav2Vec2 para Francês
processor_asr = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-xlsr-53-french")
model_asr = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-xlsr-53-french")

# Este modelo faz parte do projeto VoxPopuli, focado em dados de fala europeus.
processor_asr = Wav2Vec2Processor.from_pretrained("VoxPopuli/wav2vec2-large-xlsr-53-french")
model_asr = Wav2Vec2ForCTC.from_pretrained("VoxPopuli/wav2vec2-large-xlsr-53-french")

# Outra variante treinada para o francês, oferecendo diferentes características de desempenho.
processor_asr = Wav2Vec2Processor.from_pretrained("m3hrdadfi/wav2vec2-large-xlsr-53-french")
model_asr = Wav2Vec2ForCTC.from_pretrained("m3hrdadfi/wav2vec2-large-xlsr-53-french")

# Desenvolvido pelo grupo Helsinki-NLP, este modelo também é otimizado para o francês.
processor_asr = Wav2Vec2Processor.from_pretrained("Helsinki-NLP/wav2vec2-large-xlsr-53-french")
model_asr = Wav2Vec2ForCTC.from_pretrained("Helsinki-NLP/wav2vec2-large-xlsr-53-french")

#patrickvonplaten/wav2vec2-large-xlsr-53-french
Este modelo é uma variante do Wav2Vec2 treinada especificamente para o francês.
processor_asr = Wav2Vec2Processor.from_pretrained("patrickvonplaten/wav2vec2-large-xlsr-53-french")
model_asr = Wav2Vec2ForCTC.from_pretrained("patrickvonplaten/wav2vec2-large-xlsr-53-french")

Modelos Whisper da OpenAI

processor = WhisperProcessor.from_pretrained("openai/whisper-large")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large")

'''

