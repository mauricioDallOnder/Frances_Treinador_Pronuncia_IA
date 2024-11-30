import sys
import torch
import torchaudio
sys.setrecursionlimit(10000)
import unicodedata
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from flask import Flask, request, render_template, jsonify, send_file
import re
import os
import tempfile
import pickle
import random
import pandas as pd
from gtts import gTTS
import epitran
import noisereduce as nr
from concurrent.futures import ThreadPoolExecutor
import logging
import json
# Importar os módulos WordMatching e WordMetrics
import WordMatching
import WordMetrics

app = Flask(__name__, template_folder="templates", static_folder="static")

# Configuração do logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Variáveis globais para modelos
model_asr, processor_asr = None, None

# Executor para processamento assíncrono
executor = ThreadPoolExecutor(max_workers=1)

# Limite de tempo para mapeamento
TIME_THRESHOLD_MAPPING = 5.0

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
    logger.error(f"Erro ao carregar data_de_en_fr.pickle: {e}")
    random_sentences = []

try:
    with open('frases_categorias.pickle', 'rb') as f:
        categorized_sentences = pickle.load(f)
except Exception as e:
    logger.error(f"Erro ao carregar frases_categorias.pickle: {e}")
    categorized_sentences = {}

# Carregar o Modelo ASR Wav2Vec2 para Francês
processor_asr = Wav2Vec2Processor.from_pretrained("jonatasgrosman/wav2vec2-xls-r-1b-french")
model_asr = Wav2Vec2ForCTC.from_pretrained("jonatasgrosman/wav2vec2-xls-r-1b-french")

# Iniciar o Epitran e funções de tradução
# Inicializar Epitran para Francês
epi = epitran.Epitran('fra-Latn')

# Carregar o dic.json
with open('dic.json', 'r', encoding='utf-8') as f:
    ipa_dictionary = json.load(f)

# Mapeamento de fonemas francês para português com regras contextuais
french_to_portuguese_phonemes = {
    # Vogais orais
    'i': {'default': 'i'},
    'e': {'default': 'e'},
    'ɛ': {'default': 'é', 'before_nasal': 'ê'},
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
    'd': {'default': 'd', 'before_i': 'dj'},
    'f': {'default': 'f'},
    'g': {'default': 'g'},
    'ʒ': {'default': 'j', 'word_initial': 'j', 'after_nasal': 'j'},
    'k': {'default': 'k', 'before_front_vowel': 'qu'},
    'l': {'default': 'l'},
    'm': {'default': 'm'},
    'n': {'default': 'n'},
    'p': {'default': 'p'},
    'ʁ': {'default': 'r', 'word_initial': 'h', 'after_vowel': 'rr', 'after_consonant': 'r'},
    's': {'default': 's', 'between_vowels': 'z'},
    't': {'default': 't', 'before_i': 'tch'},
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

    # Outros fonemas
    'x': {'default': 'x'},
    'ʎ': {'default': 'lh'},
    'ʔ': {'default': ''},  # Fonema glotal stop geralmente não tem equivalente em português
    'θ': {'default': 't'},  # Adapte conforme necessário
    'ð': {'default': 'd'},  # Adapte conforme necessário
    'ɾ': {'default': 'r'},
    'ʕ': {'default': 'r'},  # Adapte conforme necessário
}

# Características fonéticas

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
    word_normalized = word.lower()
    try:
        # Tentar obter a pronúncia do dic.json
        pronunciation = ipa_dictionary.get(word_normalized)
        if pronunciation:
            return pronunciation
        else:
            # Se não encontrado, usar Epitran como fallback
            pronunciation = epi.transliterate(word)
            return pronunciation
    except Exception as e:
        logger.error(f"Erro ao obter pronúncia para '{word}': {e}")
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
        convert_pronunciation_to_portuguese(pron, idx, pronunciations)
        for idx, pron in enumerate(pronunciations)
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
            logger.warning(f"Fonema não mapeado: '{pronunciation[idx]}' na pronúncia '{pronunciation}'")
            idx += 1
    return phonemes

def convert_pronunciation_to_portuguese(pronunciation, word_idx, all_pronunciations):
    phonemes = split_into_phonemes(pronunciation)
    result = []
    idx = 0
    length = len(phonemes)
    word_start = idx == 0
    while idx < length:
        phoneme = phonemes[idx]
        mapping = french_to_portuguese_phonemes.get(phoneme, {'default': phoneme})
        context = 'default'

        next_phoneme = phonemes[idx + 1] if idx + 1 < length else ''
        prev_phoneme = phonemes[idx - 1] if idx > 0 else ''

        # Definir listas de vogais
        vowels = ['a', 'e', 'i', 'o', 'u', 'ɛ', 'ɔ', 'ɑ', 'ø', 'œ', 'ə']
        front_vowels = ['i', 'e', 'ɛ', 'ɛ̃', 'œ', 'ø', 'y']

        next_is_i = next_phoneme == 'i'
        prev_is_vowel = prev_phoneme in vowels
        next_is_vowel = next_phoneme in vowels
        next_is_front_vowel = next_phoneme in front_vowels

        # Definir o contexto
        if phoneme == 'd' and next_is_i:
            context = 'before_i'
        elif phoneme == 't' and next_is_i:
            context = 'before_i'
        elif phoneme == 'k' and next_is_front_vowel:
            context = 'before_front_vowel'
        elif phoneme == 'ʁ':
            if word_start:
                context = 'word_initial'
            elif prev_is_vowel:
                context = 'after_vowel'
            else:
                context = 'after_consonant'
        elif phoneme == 's' and prev_is_vowel and next_is_vowel:
            context = 'between_vowels'
        elif phoneme == 'ʒ' and phonemes[idx - 1] in ['ɛ̃', 'ɑ̃', 'ɔ̃', 'œ̃']:
            context = 'after_nasal'

        # Obter o mapeamento
        mapped_phoneme = mapping.get(context, mapping['default'])
        result.append(mapped_phoneme)
        idx += 1
        word_start = False  # Apenas a primeira iteração é o início da palavra

    return ''.join(result)

def handle_apostrophes(words_list):
    new_words = []
    for word in words_list:
        if "'" in word:
            prefix, sep, suffix = word.partition("'")
            # Contrações comuns
            if prefix.lower() in ["l", "d", "j", "qu", "n", "m", "c"]:
                combined_word = prefix + suffix
                new_words.append(combined_word)
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
            elif current_word.lower() == "d'" and re.match(r"^[aeiouyâêîôûéèëïüÿæœ]", next_word, re.IGNORECASE):
                current_pron = current_pron.rstrip('e') + 'z'  # Adiciona /z/
        # Caso contrário, não aplica liaison
        new_pronunciations.append(current_pron)
    # Adicionar a última pronúncia
    new_pronunciations.append(pronunciations[-1])
    return new_pronunciations

def remove_accents(text):
    return ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )

def normalize_text(text):
    text = text.lower()
    text = text.replace("’", "'")
    text = remove_accents(text)
    text = re.sub(r"[^\w\s']", '', text)
    text = re.sub(r"\s+'", "'", text)
    text = re.sub(r"'\s+", "'", text)
    return text.strip()


'''
def remove_punctuation_end(sentence):
    return sentence.rstrip('.')
'''

def add_punctuation_end(sentence):
    # Adiciona um ponto final se não houver um já
    if not sentence.endswith('.'):
        return sentence + '.'
    return sentence

# Funções para comparação fonética
# Funções para comparação fonética
def compare_phonetics(phonetic1, phonetic2, threshold=0.8):
    # Usar distância de edição normalizada para comparação
    distance = WordMetrics.edit_distance_python(phonetic1, phonetic2)
    max_len = max(len(phonetic1), len(phonetic2))
    similarity = 1 - (distance / max_len)
    return similarity >= threshold
# Processamento de áudio:
# Funções de processamento de áudio
def remove_silence(waveform, sample_rate, threshold=0.01):
    # Remove silêncios no início e no final do áudio
    waveform = waveform.squeeze(0)  # Remove a dimensão do canal se existir
    energy = waveform.abs()
    mask = energy > threshold
    if mask.any():
        indices = torch.nonzero(mask).squeeze()
        start = indices[0].item()
        end = indices[-1].item()
        trimmed_waveform = waveform[start:end+1]
        return trimmed_waveform.unsqueeze(0)  # Adiciona de volta a dimensão do canal
    else:
        return waveform.unsqueeze(0)  # Retorna o waveform original com dimensão do canal

def reduce_noise(waveform, sample_rate):
    try:
        reduced_waveform = nr.reduce_noise(y=waveform, sr=sample_rate, prop_decrease=0.5, stationary=False)
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
        # Remover silêncios
        waveform = remove_silence(waveform, sample_rate)
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
        inputs = processor_asr(waveform.squeeze(0), sampling_rate=sample_rate, return_tensors="pt")
        with torch.no_grad():
            logits = model_asr(inputs.input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        # Decodificação correta
        transcription = processor_asr.decode(predicted_ids[0], skip_special_tokens=True)
        return transcription
    except Exception as e:
        logger.exception(f"Erro ao processar áudio: {e}")
        raise e
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

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
                sentence_text = add_punctuation_end(sentence.get('fr_sentence', "Frase não encontrada"))
            else:
                return jsonify({"error": "Nenhuma frase disponível para seleção aleatória."}), 500
        else:
            if category in categorized_sentences:
                sentences_in_category = categorized_sentences[category]
                sentence_text = random.choice(sentences_in_category)
                sentence_text = add_punctuation_end(sentence_text)
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
        transcription = future.result(timeout=120)  # Timeout de 120 segundos
    except Exception as e:
        logger.exception(f"Erro ao processar áudio: {e}")  # Log do erro
        return jsonify({"error": "Erro ao processar o áudio."}), 500

    # Normalização e comparação de transcrições
    normalized_transcription = normalize_text(transcription)
    normalized_text = normalize_text(text)
    words_estimated = normalized_transcription.split()
    words_real = normalized_text.split()

    # Alinhar as palavras usando o alinhamento otimizado
    mapped_words, mapped_indices = WordMatching.get_best_mapped_words(words_estimated, words_real)

    # Calcular WER e acurácia
    wer = calculate_wer(words_real, mapped_words)
    accuracy = (1 - wer) * 100

    # Calcular acurácia de fonemas
    phoneme_accuracy = calculate_phoneme_accuracy(words_real, mapped_words)

    print(f"WER: {wer}, Acurácia: {accuracy}, Acurácia de Fonemas: {phoneme_accuracy}")

    # Inicialização de variáveis para feedback
    diff_html = []
    pronunciations = {}
    feedback = {}
    correct_count = 0
    incorrect_count = 0

    # Comparação palavra a palavra usando as palavras mapeadas
    for idx, real_word in enumerate(words_real):
        mapped_word = mapped_words[idx]
        if mapped_word != '-':
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
    completeness_score = (len(mapped_words) / len(words_real)) * 100 if len(words_real) > 0 else 0

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

# Funções adicionais necessárias
# Função para calcular WER
def calculate_wer(reference, hypothesis):
    # Usar edit distance para calcular WER
    distance = WordMetrics.edit_distance_python(reference, hypothesis)
    wer = distance / len(reference) if len(reference) > 0 else 0
    return wer

def calculate_phoneme_accuracy(reference_words, hypothesis_words):
    total_phonemes = 0
    total_distance = 0

    for ref_word, hyp_word in zip(reference_words, hypothesis_words):
        ref_pronunciation = transliterate_and_convert_sentence(ref_word)
        hyp_pronunciation = transliterate_and_convert_sentence(hyp_word)

        ref_phonemes = list(ref_pronunciation)
        hyp_phonemes = list(hyp_pronunciation)

        total_phonemes += len(ref_phonemes)
        distance = WordMetrics.edit_distance_python(ref_phonemes, hyp_phonemes)
        total_distance += distance

    if total_phonemes > 0:
        phoneme_accuracy = ((total_phonemes - total_distance) / total_phonemes) * 100
    else:
        phoneme_accuracy = 0

    return phoneme_accuracy

# Inicialização e execução do aplicativo
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=os.getenv("PORT", default=3000))