import os
import re
import json
import pickle
import random
import tempfile
import logging
import unicodedata
import traceback
from flask import Flask, request, render_template, jsonify, send_file
from concurrent.futures import ThreadPoolExecutor
from SpecialRoules import *

import WordMatching
import WordMetrics
from SpecialRoules import (
    handle_est_ce_que,
    handle_est_pronunciation,
    handle_plus_pronunciation
)
from getPronunciation import get_pronunciation_hints
from functions import (
    epi,
    ipa_dictionary,
    transliterate_and_convert_sentence,
    normalize_text,
    remove_punctuation_end,
    get_pronunciation  # caso precise direto
)

# -----------------------------------------------------------------------------
# Configuração básica
# -----------------------------------------------------------------------------
app = Flask(__name__, template_folder="templates", static_folder="static")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Thread pool para processar tarefas em background
executor = ThreadPoolExecutor(max_workers=2)

# Carrega frases de exemplo
try:
    with open('data_de_en_fr.pickle', 'rb') as f:
        df = pickle.load(f)
        random_sentences = df.to_dict(orient='records') if hasattr(df, 'to_dict') else df
except Exception as e:
    logger.error(f"Erro ao carregar data_de_en_fr.pickle: {e}")
    random_sentences = []

try:
    with open('frases_categorias.pickle', 'rb') as f:
        categorized_sentences = pickle.load(f)
except Exception as e:
    logger.error(f"Erro ao carregar frases_categorias.pickle: {e}")
    categorized_sentences = {}

# -----------------------------------------------------------------------------
# Rotas
# -----------------------------------------------------------------------------
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/pronounce', methods=['POST'])
def pronounce():
    text = request.form.get('text', '')
    logger.info(f"[DEBUG] /pronounce recebido: {text!r}")
    try:
        pron = transliterate_and_convert_sentence(text)
        logger.info(f"[DEBUG] pron: {pron!r}")
        return jsonify({'pronunciations': pron})
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"Erro em /pronounce: {e}\n{tb}")
        return jsonify({
            'error': str(e),
            'traceback': tb.splitlines()[-3:]  # só as últimas linhas
        }), 500

@app.route('/hints', methods=['POST'])
def hints():
    text = request.form.get('text', '')
    logger.info(f"[DEBUG] /hints recebido: {text!r}")
    try:
        words = text.split()
        hints = [get_pronunciation_hints(w) for w in words]
        filtered = [h for h in hints if h['explanations']]
        logger.info(f"[DEBUG] hints filtrados: {filtered!r}")
        return jsonify({'hints': filtered})
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"Erro em /hints: {e}\n{tb}")
        return jsonify({
            'error': str(e),
            'traceback': tb.splitlines()[-3:]
        }), 500


@app.route('/get_sentence', methods=['POST'])
def get_sentence():
    cat = request.form.get('category', 'random')
    try:
        if cat == 'random':
            sent = random.choice(random_sentences) if random_sentences else {}
            txt = sent.get('fr_sentence','')
        else:
            txt = random.choice(categorized_sentences.get(cat, []))
        return jsonify({
            'fr_sentence': remove_punctuation_end(txt),
            'category': cat
        })
    except Exception as e:
        logger.exception("Erro em /get_sentence")
        return jsonify({'error': str(e)}), 500

"""
@app.route('/upload', methods=['POST'])
def upload():
    file = request.files.get('audio')
    text = request.form.get('text')
    if not file or not text:
        return jsonify({'error': 'Áudio ou texto faltando'}), 400

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    tmp.write(file.read())
    tmp.close()

    transcription = executor.submit(process_audio, tmp.name).result(timeout=120)

    user_norm = normalize_text(transcription).split()
    ref_norm  = normalize_text(text).split()

    mapped, _ = WordMatching.get_best_mapped_words(user_norm, ref_norm)

    diff_html = []
    feedback = {}
    pron = {}
    correct, incorrect = 0, 0

    for real, got in zip(ref_norm, mapped):
        corr = transliterate_and_convert_sentence(real)
        usr  = transliterate_and_convert_sentence(got) if got!='-' else ''
        ok   = WordMetrics.hybrid_similarity(corr, usr) >= 0.8
        cls  = 'correct' if ok else 'incorrect'
        if ok:
            correct += 1
        else:
            incorrect += 1
            feedback[real] = {'correct': corr, 'user': usr,
                              'suggestion': f"Tente '{corr}'"}
        pron[real] = {'correct': corr, 'user': usr}
        diff_html.append(f'<span class="word {cls}">{real}</span>')

    total = correct + incorrect or 1
    ratio = correct/total*100
    completeness = len([m for m in mapped if m!='-']) / len(ref_norm) * 100

    return jsonify({
        'ratio': f"{ratio:.2f}",
        'diff_html': ' '.join(diff_html),
        'pronunciations': pron,
        'feedback': feedback,
        'completeness_score': f"{completeness:.2f}"
    })
"""

# -----------------------------------------------------------------------------
# Start
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    # dev local
    app.run(host='0.0.0.0', port=3000, debug=True)
else:
    # em produção (quando importado pelo Railway), use Waitress:
    from waitress import serve
    serve(app, host='0.0.0.0', port=int(os.environ.get('PORT', 3000)))
