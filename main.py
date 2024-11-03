import sys
sys.setrecursionlimit(10000)
import numpy as np
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
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
from WordMetrics import edit_distance_python2
from WordMatching import get_best_mapped_words
from unidecode import unidecode
import epitran
from datetime import datetime
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

app = Flask(__name__, template_folder="./templates", static_folder="./static")

# Carregar o Modelo de SST Francês atualizado
model_name = "jonatasgrosman/wav2vec2-large-xlsr-53-french"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name)

# Modelo para tradução
translation_model_name = 'facebook/m2m100_418M'  # Ou 'facebook/m2m100_1.2B' para um modelo maior
tokenizer = M2M100Tokenizer.from_pretrained(translation_model_name)
translation_model = M2M100ForConditionalGeneration.from_pretrained(translation_model_name)

source_language = 'fr'  # Francês
target_language = 'pt'  # Português

# Função de Tradução
def translate_to_portuguese(text):
    try:
        tokenizer.src_lang = source_language
        encoded = tokenizer(text, return_tensors='pt')
        generated_tokens = translation_model.generate(
            **encoded,
            forced_bos_token_id=tokenizer.get_lang_id(target_language)
        )
        translated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        return translated_text
    except Exception as e:
        print(f"Erro na tradução: {e}")
        return "Tradução indisponível."

# Funções para carregar e salvar dados de desempenho
performance_file = 'performance_data.pkl'

def load_performance_data():
    if os.path.exists(performance_file):
        with open(performance_file, 'rb') as f:
            return pickle.load(f)
    else:
        return []

def save_performance_data(data):
    with open(performance_file, 'wb') as f:
        pickle.dump(data, f)

performance_data = load_performance_data()

# Arquivo para armazenar o progresso do usuário
user_progress_file = 'user_progress.pkl'

def load_user_progress():
    if os.path.exists(user_progress_file):
        with open(user_progress_file, 'rb') as f:
            return pickle.load(f)
    else:
        return {}

def save_user_progress(progress):
    with open(user_progress_file, 'wb') as f:
        pickle.dump(progress, f)

user_progress = load_user_progress()

# Inicializar Epitran para Francês
epi = epitran.Epitran('fra-Latn')

# Mapeamento atualizado de fonemas franceses para português
french_to_portuguese_phonemes = {
    # Vogais orais
    'i': 'i',     # como em "fil"
    'e': 'ê',     # como em "clé"
    'ɛ': 'é',     # como em "fait"
    'a': 'a',     # como em "abat"
    'ɑ': 'a',     # como em "pâte"
    'ɔ': 'ó',     # como em "dormir"
    'o': 'ô',     # como em "dôme"
    'u': 'u',     # como em "hibou"
    'y': 'u',     # aproximado ao "u" francês em "tu"
    'ø': 'e',     # como em "jeu"
    'œ': 'é',     # aproximado ao som em "immeuble"
    'ə': 'e',     # vogal neutra, como em "premier"

    # Vogais nasais
    'ɛ̃': 'ẽ',    # como em "sein"
    'ɑ̃': 'ã',    # como em "grand"
    'ɔ̃': 'õ',    # como em "sombre"
    'œ̃': 'ũ',    # aproximado ao som nasal em "brun"

    # Semivogais
    'j': 'i',     # como em "yeux"
    'w': 'u',     # como em "jouet"
    'ɥ': 'ui',    # como em "lui"

    # Consoantes
    'b': 'b',     # como em "blond"
    'd': 'd',     # como em "don"
    'f': 'f',     # como em "fait"
    'g': 'g',     # como em "gage"
    'ʒ': 'j',     # como em "gel"
    'k': 'k',     # como em "cou"
    'l': 'l',     # como em "lire"
    'm': 'm',     # como em "maire"
    'n': 'n',     # como em "nuire"
    'p': 'p',     # como em "présent"
    'ʁ': 'rr',    # como em "rat"
    'r': 'rr',    # representação alternativa
    's': 's',     # como em "sans"
    't': 't',     # como em "talent"
    'v': 'v',     # como em "vitesse"
    'z': 'z',     # como em "zone"
    'ʃ': 'ch',    # como em "chaleur"
    'ɲ': 'nh',    # como em "gagner"
    'ŋ': 'n',     # como em "camping" (aproximação)
}

def get_pronunciation(word):
    # Transcreve a palavra usando Epitran
    pronunciation = epi.transliterate(word)
    return pronunciation

def omit_schwa(pronunciation):
    # Remove o schwa ('ə') em finais de palavra ou onde for apropriado
    pronunciation = re.sub(r'ə\b', '', pronunciation)
    return pronunciation

def normalize_vowels(pronunciation):
    # Normaliza vogais para consistência, se necessário
    # Exemplo: substitui variantes de 'œ' por 'ø'
    pronunciation = pronunciation.replace('œ', 'ø')
    return pronunciation

def handle_special_cases(pronunciation):
    # Regras especiais para contextos específicos
    # Exemplo: ajustar pronúncias específicas de certas palavras
    return pronunciation

def convert_pronunciation_to_portuguese(pronunciation):
    pronunciation = omit_schwa(pronunciation)
    pronunciation = normalize_vowels(pronunciation)
    pronunciation = handle_special_cases(pronunciation)

    # Substituir símbolos fonéticos usando o mapeamento
    # Ordenar os fonemas por tamanho decrescente para evitar conflitos
    sorted_phonemes = sorted(french_to_portuguese_phonemes.keys(), key=len, reverse=True)
    for phoneme in sorted_phonemes:
        pronunciation = pronunciation.replace(phoneme, french_to_portuguese_phonemes[phoneme])

    return pronunciation

def transliterate_and_convert(word):
    pronunciation = get_pronunciation(word)
    pronunciation_pt = convert_pronunciation_to_portuguese(pronunciation)
    return pronunciation_pt

def compare_pronunciations(correct_pronunciation, user_pronunciation, similarity_threshold=0.9):
    distance = edit_distance_python2(correct_pronunciation, user_pronunciation)
    max_length = max(len(correct_pronunciation), len(user_pronunciation))
    if max_length == 0:
        return False  # Evita divisão por zero
    similarity = 1 - (distance / max_length)
    return similarity >= similarity_threshold

def normalize_text(text):
    text = text.lower()
    # Substituir apóstrofos especiais pelo apóstrofo ASCII
    text = text.replace("’", "'")
    # Remover pontuação, exceto apóstrofos
    text = re.sub(r"[^\w\s']", '', text)
    # Recombinar contrações separadas por espaços
    text = re.sub(r"\b(\w+)\s'\s(\w+)\b", r"\1'\2", text)
    text = text.strip()
    return text

def remove_punctuation_end(sentence):
    return sentence.rstrip('.')

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

# Carregar frases categorizadas
try:
    with open('frases_categorias.pickle', 'rb') as f:
        categorized_sentences = pickle.load(f)
except Exception as e:
    print(f"Erro ao carregar frases_categorias.pickle: {e}")
    categorized_sentences = {}

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['audio']
    text = request.form['text']
    category = request.form.get('category', 'random')

    # Salvar o arquivo enviado em um arquivo temporário
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(file.read())
        tmp_file_path = tmp_file.name

    try:
        # Ler o arquivo de áudio usando o módulo wave
        with wave.open(tmp_file_path, 'rb') as wav_file:
            sample_rate = wav_file.getframerate()
            num_frames = wav_file.getnframes()
            waveform = wav_file.readframes(num_frames)
            waveform = np.frombuffer(waveform, dtype=np.int16).astype(np.float32) / 32768.0
            waveform = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0)

        # Resample se necessário
        if sample_rate != 16000:
            waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)

        waveform = waveform.squeeze(0)

        # Normalizar o volume
        waveform = waveform / waveform.abs().max()

        # Ajustar os parâmetros do modelo
        inputs = processor(
            waveform,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )

        with torch.no_grad():
            logits = model(inputs.input_values).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)[0]
    finally:
        os.remove(tmp_file_path)

    # Após normalizar a transcrição
    normalized_transcription = normalize_text(transcription)
    normalized_text = normalize_text(text)

    # Estimativas e palavras reais
    words_estimated = normalized_transcription.split()
    words_real = normalized_text.split()

    # Utilizar get_best_mapped_words para obter o mapeamento
    mapped_words, mapped_words_indices = get_best_mapped_words(words_estimated, words_real)

    # Gerar HTML com palavras codificadas por cores e feedback
    diff_html = []
    pronunciations = {}
    feedback = {}
    correct_count = 0
    incorrect_count = 0

    for idx, real_word in enumerate(words_real):
        if idx < len(mapped_words):
            mapped_word = mapped_words[idx]
            correct_pronunciation = transliterate_and_convert(real_word)
            user_pronunciation = transliterate_and_convert(mapped_word)
            if compare_pronunciations(correct_pronunciation, user_pronunciation):
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
            # Palavra não reconhecida
            diff_html.append(f'<span class="word missing" onclick="showPronunciation(\'{real_word}\')">{real_word}</span>')
            incorrect_count += 1
            feedback[real_word] = {
                'correct': transliterate_and_convert(real_word),
                'user': '',
                'suggestion': f"Tente pronunciar '{real_word}' como '{transliterate_and_convert(real_word)}'"
            }
            pronunciations[real_word] = {
                'correct': transliterate_and_convert(real_word),
                'user': ''
            }

    diff_html = ' '.join(diff_html)

    # Calcula a taxa de acerto e completude
    total_words = correct_count + incorrect_count
    ratio = (correct_count / total_words) * 100 if total_words > 0 else 0
    completeness_score = (len(words_estimated) / len(words_real)) * 100

    # Armazena os resultados diários
    performance_data.append({
        'date': pd.Timestamp.now().strftime('%Y-%m-%d'),
        'correct': correct_count,
        'incorrect': incorrect_count,
        'ratio': ratio,
        'completeness_score': completeness_score,
        'sentence': text
    })
    save_performance_data(performance_data)

    # Atualizar o progresso do usuário
    if category != 'random':
        user_category_progress = user_progress.get(category, {'sentences_done': [], 'performance': []})

        if text not in user_category_progress['sentences_done']:
            user_category_progress['sentences_done'].append(text)

        user_category_progress['performance'].append({
            'sentence': text,
            'ratio': ratio,
            'completeness_score': completeness_score
        })

        user_progress[category] = user_category_progress
        save_user_progress(user_progress)

    # Logging para depuração
    print(f"Correct: {correct_count}, Incorrect: {incorrect_count}, Total: {total_words}, Ratio: {ratio}")
    formatted_ratio = "{:.2f}".format(ratio)
    formatted_completeness = "{:.2f}".format(completeness_score)

    return jsonify({
        'ratio': formatted_ratio,
        'diff_html': diff_html,
        'pronunciations': pronunciations,
        'feedback': feedback,
        'completeness_score': formatted_completeness
    })

@app.route('/translate', methods=['POST'])
def translate():
    text = request.form['text']
    translated_text = translate_to_portuguese(text)
    return jsonify({'translation': translated_text})

@app.route('/')
def index():
    return render_template('index.html')

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
                user_category_progress = user_progress.get(category, {'sentences_done': [], 'performance': []})
                sentences_done = user_category_progress.get('sentences_done', [])
                sentences_remaining = list(set(sentences_in_category) - set(sentences_done))

                if not sentences_remaining:
                    # Todas as frases foram praticadas
                    performance_list = user_category_progress.get('performance', [])
                    if performance_list:
                        avg_ratio = sum(p['ratio'] for p in performance_list) / len(performance_list)
                        avg_completeness = sum(p['completeness_score'] for p in performance_list) / len(performance_list)
                    else:
                        avg_ratio = 0
                        avg_completeness = 0

                    return jsonify({
                        "message": "Você completou todas as frases desta categoria.",
                        "average_ratio": f"{avg_ratio:.2f}",
                        "average_completeness": f"{avg_completeness:.2f}"
                    })
                else:
                    sentence_text = random.choice(sentences_remaining)
                    sentence_text = remove_punctuation_end(sentence_text)
            else:
                return jsonify({"error": "Categoria não encontrada."}), 400

        return jsonify({'fr_sentence': sentence_text, 'category': category})

    except Exception as e:
        print(f"Erro no endpoint /get_sentence: {e}")
        return jsonify({"error": "Erro interno no servidor."}), 500

@app.route('/pronounce', methods=['POST'])
def pronounce():
    text = request.form['text']
    # Certifique-se de que os apóstrofos estão corretos
    text = text.replace("’", "'")
    words = text.split()
    pronunciations = [transliterate_and_convert(word) for word in words]
    return jsonify({'pronunciations': ' '.join(pronunciations)})

@app.route('/speak', methods=['POST'])
def speak():
    text = request.form['text']
    tts = gTTS(text=text, lang='fr')
    file_path = tempfile.mktemp(suffix=".mp3")
    tts.save(file_path)
    return send_file(file_path, as_attachment=True, mimetype='audio/mp3')

@app.route('/performance', methods=['GET'])
def performance():
    if not performance_data:
        return "Nenhum dado de desempenho disponível.", 204
    df = pd.DataFrame(performance_data)
    if 'date' not in df.columns:
        return "Dados de desempenho inválidos.", 500
    grouped = df.groupby('date').agg({
        'ratio': 'mean',
        'completeness_score': 'mean'
    }).reset_index()

    dates = grouped['date']
    ratios = grouped['ratio']
    completeness_scores = grouped['completeness_score']

    x = np.arange(len(dates))  # the label locations

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(dates, ratios, marker='o', label='Taxa de Acerto na Pronúncia (%)')
    ax.plot(dates, completeness_scores, marker='x', label='Taxa de Completude (%)')

    ax.set_xlabel('Data')
    ax.set_ylabel('Percentagem')
    ax.set_title('Desempenho Diário')
    ax.set_xticks(x)
    ax.set_xticklabels(dates, rotation=45)
    ax.set_ylim(0, 100)
    ax.legend()

    fig.tight_layout()

    graph_path = 'static/performance_graph.png'
    plt.savefig(graph_path, bbox_inches='tight')
    plt.close()

    return send_file(graph_path, mimetype='image/png')

@app.route('/get_progress', methods=['GET'])
def get_progress():
    progress_data = {}
    for category in categorized_sentences.keys():
        total_sentences = len(categorized_sentences[category])
        user_category_progress = user_progress.get(category, {'sentences_done': []})
        sentences_done = len(user_category_progress.get('sentences_done', []))
        progress_data[category] = {
            'total_sentences': total_sentences,
            'sentences_done': sentences_done
        }
    return jsonify(progress_data)

if __name__ == '__main__':
    app.run(debug=True)
