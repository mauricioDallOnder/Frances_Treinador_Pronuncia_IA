import epitran
import unicodedata
import re
import json
import logging
import os
from typing import List, Dict, Tuple, Optional, Set, Any

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicializar Epitran para Francês
try:
    epi = epitran.Epitran('fra-Latn')
except Exception as e:
    logger.warning(f"Erro ao inicializar Epitran: {e}")
    epi = None

# Carregar dicionário de pronúncias
script_dir = os.path.dirname(os.path.abspath(__file__))
dic_path = os.path.join(script_dir, 'dic.json')

try:
    with open(dic_path, 'r', encoding='utf-8') as f:
        ipa_dictionary = json.load(f)
except Exception as e:
    logger.warning(f"Não foi possível carregar o dicionário: {e}")
    ipa_dictionary = {}

# Caracteres a serem ignorados na transcrição fonética
IGNORED_CHARS = set([',', '.', ';', ':', '!', '?', ' ', '\t', '\n', '-', '_', '(', ')', '[', ']', '{', '}', '"', "'"])

# Mapeamento de fonemas IPA para representação simplificada
IPA_TO_SIMPLE = {
    'a': 'a', 'ɑ': 'a', 'ɑ̃': 'ã',
    'e': 'e', 'ɛ': 'é', 'ə': 'e', 'œ': 'ö', 'ø': 'ö',
    'i': 'i', 'ɪ': 'i',
    'o': 'o', 'ɔ': 'ó', 'ɔ̃': 'õ',
    'u': 'u', 'ʊ': 'u',
    'y': 'ü',
    'ɛ̃': 'ẽ', 'œ̃': 'ẽ',
    'j': 'j', 'w': 'w', 'ɥ': 'ɥ',
    'b': 'b', 'd': 'd', 'f': 'f', 'g': 'g', 'k': 'k',
    'l': 'l', 'm': 'm', 'n': 'n', 'ɲ': 'nh', 'ŋ': 'ng',
    'p': 'p', 'ʁ': 'r', 'r': 'r', 's': 's', 'ʃ': 'ch',
    't': 't', 'v': 'v', 'z': 'z', 'ʒ': 'j',
    'x': 'ks', 'ts': 'ts', 'dz': 'dz', 'tʃ': 'tch', 'dʒ': 'dj',
    'h': 'h'
}

# Mapeamento de fonemas franceses para português com regras contextuais
french_to_portuguese_phonemes = {
    # VOGAIS ORAIS
    'i': {'default': 'i'},
    'e': {'default': 'e'},
    'ɛ': {'default': 'é'},
    'a': {'default': 'a'},
    'ɑ': {'default': 'a'},
    'ɔ': {'default': 'ó'},
    'o': {'default': 'ô'},
    'u': {'default': 'u'},
    'y': {'default': 'u'},
    'ø': {'default': 'eu'},
    'œ': {'default': 'eu'},
    'ə': {'default': 'e'},

    # VOGAIS NASAIS
    'ɛ̃': {'default': 'ẽ'},
    'ɑ̃': {'default': 'ã'},
    'ɔ̃': {'default': 'õ'},
    'œ̃': {'default': 'ũ'},

    # SEMIVOGAIS
    'w': {'default': 'u'},
    'ɥ': {'default': 'u', 'after_vowel': 'w'},

    # CONSOANTES
    'b': {'default': 'b'},
    'd': {'default': 'd', 'before_i': 'dj'},
    'f': {'default': 'f'},
    'g': {'default': 'g', 'before_front_vowel': 'j'},
    'ʒ': {'default': 'j'},
    'k': {'default': 'k', 'before_front_vowel': 'qu'},
    'l': {'default': 'l'},
    'm': {'default': 'm'},
    'n': {'default': 'n'},
    'p': {'default': 'p'},
    'ʁ': {'default': 'r'},
    's': {'default': 's', 'between_vowels': 'z', 'word_final': 's'},
    't': {'default': 't', 'before_i': 'tch'},
    'v': {'default': 'v'},
    'z': {'default': 'z'},
    'ʃ': {'default': 'ch'},
    'ɲ': {'default': 'nh'},
    'ŋ': {'default': 'ng'},
    'ç': {'default': 's'},
    'ʎ': {'default': 'lh'},
    'ʔ': {'default': ''},
    'θ': {'default': 't'},
    'ɾ': {'default': 'r'},
    'ʕ': {'default': 'r'},

    # COMPOSTOS
    'sj': {'default': 'si'},
    'ks': {'default': 'x'},
    'gz': {'default': 'gz'},
    'ʃj': {'default': 'chi'},
    'ʒʁ': {'default': 'jr'},

    # FINAIS
    'k$': {'default': 'c'},
    'g$': {'default': 'g'},
    'p$': {'default': 'p'},
    't$': {'default': 't'}
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

# Casos especiais de pronúncia
SPECIAL_CASES = {
    "c'est": "sé",
    "c'était": "seté",
    "c'étaient": "setén",
    "j'aie": "je",
    "j'ai": "jé",
    "est-ce": "és",
    "est-ce que": "és k",
    "qu'est-ce": "kés",
    "qu'est-ce que": "kés k",
    # Novos casos para 'jusqu'au' e 'jusqu'aux'
    "jusqu'au": "jusko",
    "jusqu'aux": "jusko",
}

# Padrões de substituição para correções específicas
PATTERN_REPLACEMENTS = [
    (r'sjõ\b', 'siõn'),
    (r'ésts k', 'ésk'),
    (r'késts k', 'késk'),
]


def normalize_text(text: str) -> str:
    """Normaliza o texto removendo acentos e convertendo para minúsculas."""
    text = unicodedata.normalize('NFD', text)
    text = ''.join([c for c in text if not unicodedata.combining(c)])
    return text.lower()


def get_ipa_pronunciation(word: str) -> str:
    """Obtém a pronúncia IPA de uma palavra."""
    word = word.lower()
    if word in ipa_dictionary:
        return ipa_dictionary[word]
    if epi:
        try:
            return epi.transliterate(word)
        except Exception as e:
            logger.warning(f"Erro ao transliterar '{word}': {e}")
    return word


def split_into_phonemes(ipa: str) -> List[str]:
    """Divide uma string IPA em fonemas individuais, ignorando caracteres não fonéticos."""
    phonemes = []
    i = 0
    cleaned_ipa = ''.join([c for c in ipa if c not in IGNORED_CHARS])
    while i < len(cleaned_ipa):
        found = False
        for length in [3, 2]:
            if i + length <= len(cleaned_ipa):
                seq = cleaned_ipa[i:i+length]
                if seq in IPA_TO_SIMPLE or seq in IPA_TO_SIMPLE:
                    phonemes.append(seq)
                    i += length
                    found = True
                    break
        if not found:
            char = cleaned_ipa[i]
            if char in IPA_TO_SIMPLE:
                phonemes.append(char)
            elif char not in IGNORED_CHARS:
                logger.debug(f"Fonema não mapeado: '{char}' na pronúncia '{ipa}'")
            i += 1
    return phonemes


def map_phonemes_to_simple(phonemes: List[str]) -> List[str]:
    """Mapeia fonemas IPA para representação simplificada."""
    result = []
    for p in phonemes:
        if p in IPA_TO_SIMPLE:
            result.append(IPA_TO_SIMPLE[p])
        else:
            mapped = True
            for char in p:
                if char in IPA_TO_SIMPLE:
                    result.append(IPA_TO_SIMPLE[char])
                elif char not in IGNORED_CHARS:
                    mapped = False
                    logger.debug(f"Caractere não mapeado: '{char}' no fonema '{p}'")
            if not mapped and p not in IGNORED_CHARS:
                result.append(p)
    return result


def apply_french_rules(phonemes: List[str]) -> List[str]:
    """Aplica regras específicas do francês à sequência de fonemas."""
    if not phonemes:
        return phonemes

    # Regra 1: Nasalização antes de consoantes
    for i in range(len(phonemes) - 1):
        if phonemes[i] in ['a','e','i','o','u'] and phonemes[i+1] == 'n':
            if i+2 >= len(phonemes) or phonemes[i+2] in ['b','p','m']:
                nasal_map = {'a':'ã','e':'ẽ','i':'ẽ','o':'õ','u':'ẽ'}
                phonemes[i] = nasal_map.get(phonemes[i], phonemes[i])
                phonemes[i+1] = ''

    # Regra 2: Simplificação de grupos finais
    if len(phonemes) >= 2 and phonemes[-2:] in [['r','d'],['r','t'],['s','t']]:
        phonemes = phonemes[:-1]

    # Regra 3: Só remover schwa (ə) no final
    if phonemes[-1] == 'ə':
        phonemes = phonemes[:-1]

    # Remover vazios
    phonemes = [p for p in phonemes if p]
    return phonemes


def simplify_pronunciation(word: str) -> str:
    """Simplifica a pronúncia de uma palavra francesa."""
    wl = word.lower()
    if wl in SPECIAL_CASES:
        return SPECIAL_CASES[wl]
    ipa = get_ipa_pronunciation(word)
    phonemes = split_into_phonemes(ipa)
    simple = map_phonemes_to_simple(phonemes)
    simple = apply_french_rules(simple)
    return ''.join(simple)


def transcrever_fonetica(texto: str) -> str:
    """Transcreve um texto em francês para representação fonética simplificada."""
    # Casos especiais no texto completo
    for expr, pron in SPECIAL_CASES.items():
        if expr in texto.lower():
            texto = texto.lower().replace(expr, f" {pron} ")

    # Tokeniza preservando apóstrofos
    palavras = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ]+(?:'[A-Za-zÀ-ÖØ-öø-ÿ]+)?", texto.lower())
    resultado = []
    for palavra in palavras:
        pron = simplify_pronunciation(palavra)
        resultado.append(pron)
    return ' '.join(resultado)

# Função principal exposta

def get_pronunciation(text: str) -> str:
    try:
        return transcrever_fonetica(text)
    except Exception as e:
        logger.error(f"Erro ao processar texto '{text}': {e}")
        return text

# Compatibilidade

def remove_punctuation_end(text: str) -> str:
    return re.sub(r'[.,;:!?]+$', '', text)


def handle_apostrophes(words_list: List[str]) -> List[str]:
    new_words = []
    for word in words_list:
        if "'" in word:
            prefix, _, suffix = word.partition("'")
            if prefix.lower() in ["l","d","j","qu","n","m","c"]:
                new_words.append(prefix + suffix)
            else:
                new_words.append(prefix)
                new_words.append(suffix)
        else:
            new_words.append(word)
    return new_words


def apply_liaisons(words: List[str], pronunciations: List[str]) -> List[str]:
    return pronunciations


def convert_pronunciation_to_portuguese(pronunciation: str, word_idx: int, all_pronunciations: List[str]) -> str:
    return pronunciation


def transliterate_and_convert_sentence(sentence: str) -> str:
    return get_pronunciation(sentence)
