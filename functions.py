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
    'ɛ̃': 'ẽ', 'œ̃': 'ẽ', 'ɑ̃': 'ã', 'ɔ̃': 'õ',
    'j': 'j', 'w': 'w', 'ɥ': 'ɥ',
    'b': 'b', 'd': 'd', 'f': 'f', 'g': 'g', 'k': 'k',
    'l': 'l', 'm': 'm', 'n': 'n', 'ɲ': 'nh', 'ŋ': 'ng',
    'p': 'p', 'ʁ': 'r', 'r': 'r', 's': 's', 'ʃ': 'ch',
    't': 't', 'v': 'v', 'z': 'z', 'ʒ': 'j',
    'x': 'ks', 'ts': 'ts', 'dz': 'dz', 'tʃ': 'tch', 'dʒ': 'dj',
    'h': 'h'
}

# Mapeamento de fonemas francês para português com regras contextuais aprimoradas
# Cada entrada deve ser um dicionário com, no mínimo, a chave 'default'.
# Se houver contextos adicionais (ex.: 'before_front_vowel', 'word_initial', etc.),
# mantenha também o 'default' para evitar KeyError.

french_to_portuguese_phonemes = {
    # VOGAIS ORAIS
    'i': { 'default': 'i' },
    'e': { 'default': 'e' },
    'ɛ': { 'default': 'é' },
    'a': { 'default': 'a' },
    'ɑ': { 'default': 'a' },    # se não quiser "á" aberto
    'ɔ': { 'default': 'ó' },
    'o': { 'default': 'ô' },
    'u': { 'default': 'u' },
    'y': { 'default': 'u' },
    'ø': { 'default': 'eu' },   # ou 'ô', se preferir "vou" ~ "vô"
    'œ': { 'default': 'eu' },   # ou 'é'
    'ə': { 'default': 'e'  },   # TROCA IMPORTANTE: schwa -> "e"

    # VOGAIS NASAIS
    'ɛ̃': { 'default': 'ẽ' },
    'ɑ̃': { 'default': 'ã' },
    'ɔ̃': { 'default': 'õ' },
    'œ̃': { 'default': 'ũ' },
    'ð':  { 'default': 'd'  },

    # SEMIVOGAIS
    'w': { 'default': 'u' },
    'ɥ': { 'default': 'u', 'after_vowel': 'w' },

    # CONSOANTES
    'b':  { 'default': 'b' },
    'd':  { 'default': 'd', 'before_i': 'dj' },
    'f':  { 'default': 'f' },
    'g':  { 'default': 'g', 'before_front_vowel': 'j' },
    'ʒ':  { 'default': 'j' },
    'k':  { 'default': 'k', 'before_front_vowel': 'qu' },
    'l':  { 'default': 'l' },
    'm':  { 'default': 'm' },
    'n':  { 'default': 'n' },
    'p':  { 'default': 'p' },
    # REMOVE o "rr" e "h" aqui:
    'ʁ':  { 'default': 'r' }, 
    's':  { 
        'default': 's',
        'between_vowels': 'z',
        'word_final': 's'
    },
    't':  { 'default': 't', 'before_i': 'tch' },
    'v':  { 'default': 'v' },
    'z':  { 'default': 'z' },
    'ʃ':  { 'default': 'ch' },
    'dʒ': { 'default': 'dj' },
    'tʃ': { 'default': 'tch' },
    'ɲ':  { 'default': 'nh' },
    'ŋ':  { 'default': 'ng' },
    'ç':  { 'default': 's' },
    'ʎ':  { 'default': 'lh' },
    'ʔ':  { 'default': '' },
    'θ':  { 'default': 't' },
    'ɾ':  { 'default': 'r' },
    'ʕ':  { 'default': 'r' },

    # FONEMAS COMPOSTOS
    'sj': { 'default': 'si' },  
    'ks': { 'default': 'x' },
    'gz': { 'default': 'gz' },
    'x':  { 'default': 'x' },
    'ʃj': { 'default': 'chi' },
    'ʒʁ':{ 'default': 'jr' },

    # H aspirado ou mudo
    'h':  {
        'default': '',
        'aspirated': 'h',
        'mute': ''
    },

    # Consoantes duplas
    'kk': { 'default': 'c' },
    'tt': { 'default': 't' },
    'pp': { 'default': 'p' },
    'bb': { 'default': 'b' },
    'gg': { 'default': 'g' },

    # Finais
    'k$': { 'default': 'c' },
    'g$': { 'default': 'g' },
    'p$': { 'default': 'p' },
    't$': { 'default': 't' },

    # Outros
    'ɡə': { 'default': 'gue' },
    'ɡi': { 'default': 'gi' },
    'ʧ': { 'default': 'tch' },
    'ʤ': { 'default': 'dj' }
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
    "qu'est-ce que": "kés k"
}

# Padrões de substituição para correções específicas
PATTERN_REPLACEMENTS = [
    # Corrigir terminações -ion
    (r'sjõ\b', 'iõ'),
    # Corrigir est-ce que
    (r'ésts k', 'és k'),
    # Corrigir qu'est-ce que
    (r'késts k', 'kés k'),
]

def normalize_text(text: str) -> str:
    """Normaliza o texto removendo acentos e convertendo para minúsculas."""
    text = unicodedata.normalize('NFD', text)
    text = ''.join([c for c in text if not unicodedata.combining(c)])
    return text.lower()

def get_ipa_pronunciation(word: str) -> str:
    """Obtém a pronúncia IPA de uma palavra."""
    word = word.lower()
    
    # Verificar no dicionário primeiro
    if word in ipa_dictionary:
        return ipa_dictionary[word]
    
    # Usar epitran como fallback
    if epi:
        try:
            return epi.transliterate(word)
        except Exception as e:
            logger.warning(f"Erro ao transliterar '{word}': {e}")
    
    return word  # Retorna a palavra original se não conseguir obter a pronúncia

def split_into_phonemes(ipa: str) -> List[str]:
    """Divide uma string IPA em fonemas individuais, ignorando caracteres não fonéticos."""
    phonemes = []
    i = 0
    
    # Limpar a string IPA de caracteres não fonéticos
    cleaned_ipa = ''.join([c for c in ipa if c not in IGNORED_CHARS])
    
    while i < len(cleaned_ipa):
        # Verificar dígrafos e trígrafos primeiro
        found = False
        for length in [3, 2]:  # Verificar sequências de 3 e 2 caracteres
            if i + length <= len(cleaned_ipa):
                seq = cleaned_ipa[i:i+length]
                # Verificar se a sequência é um fonema válido
                if any(seq == k or seq in IPA_TO_SIMPLE for k in IPA_TO_SIMPLE):
                    phonemes.append(seq)
                    i += length
                    found = True
                    break
        
        if not found:
            # Verificar se o caractere atual é um fonema válido
            if cleaned_ipa[i] in IPA_TO_SIMPLE or any(cleaned_ipa[i] == k for k in IPA_TO_SIMPLE):
                phonemes.append(cleaned_ipa[i])
            # Se não for um fonema válido e não for um caractere ignorado, registrar warning
            elif cleaned_ipa[i] not in IGNORED_CHARS:
                logger.debug(f"Fonema não mapeado: '{cleaned_ipa[i]}' na pronúncia '{ipa}'")
            i += 1
    
    return phonemes

def map_phonemes_to_simple(phonemes: List[str]) -> List[str]:
    """Mapeia fonemas IPA para representação simplificada."""
    result = []
    for p in phonemes:
        if p in IPA_TO_SIMPLE:
            result.append(IPA_TO_SIMPLE[p])
        else:
            # Tentar mapear caractere por caractere
            mapped = True
            for char in p:
                if char in IPA_TO_SIMPLE:
                    result.append(IPA_TO_SIMPLE[char])
                elif char not in IGNORED_CHARS:
                    mapped = False
                    logger.debug(f"Caractere não mapeado: '{char}' no fonema '{p}'")
            
            if not mapped and p not in IGNORED_CHARS:
                # Se não conseguiu mapear e não é um caractere ignorado, adicionar o fonema original
                result.append(p)
    
    return result

def apply_french_rules(phonemes: List[str]) -> List[str]:
    """Aplica regras específicas do francês à sequência de fonemas."""
    if not phonemes:
        return phonemes
    
    # Regra 1: Nasalização antes de consoantes
    for i in range(len(phonemes) - 1):
        if phonemes[i] in ['a', 'e', 'i', 'o', 'u'] and phonemes[i+1] == 'n':
            if i + 2 >= len(phonemes) or phonemes[i+2] in ['b', 'p', 'm']:
                if phonemes[i] == 'a':
                    phonemes[i] = 'ã'
                elif phonemes[i] == 'e':
                    phonemes[i] = 'ẽ'
                elif phonemes[i] == 'i':
                    phonemes[i] = 'ẽ'
                elif phonemes[i] == 'o':
                    phonemes[i] = 'õ'
                elif phonemes[i] == 'u':
                    phonemes[i] = 'ẽ'  # 'un' soa como 'ẽ' em francês
                phonemes[i+1] = ''  # Remover o 'n' após nasalizar
    
    # Regra 2: Simplificação de grupos consonantais finais
    if len(phonemes) >= 2 and phonemes[-2:] in [['r', 'd'], ['r', 't'], ['s', 't']]:
        phonemes = phonemes[:-1]  # Remover a última consoante
    
    # Regra 3: Tratamento de 'e' mudo no final
    if phonemes[-1] == 'e' and len(phonemes) > 1:
        phonemes = phonemes[:-1]  # Remover 'e' mudo no final
    
    # Remover elementos vazios
    phonemes = [p for p in phonemes if p]
    
    return phonemes

def simplify_pronunciation(word: str) -> str:
    """Simplifica a pronúncia de uma palavra francesa para uma representação mais próxima do português."""
    # Verificar casos especiais primeiro
    word_lower = word.lower()
    if word_lower in SPECIAL_CASES:
        return SPECIAL_CASES[word_lower]
    
    # Obter pronúncia IPA
    ipa = get_ipa_pronunciation(word)
    
    # Dividir em fonemas
    phonemes = split_into_phonemes(ipa)
    
    # Mapear para representação simplificada
    simple_phonemes = map_phonemes_to_simple(phonemes)
    
    # Aplicar regras específicas do francês
    simple_phonemes = apply_french_rules(simple_phonemes)
    
    # Juntar fonemas em uma string
    return ''.join(simple_phonemes)

def process_sentence(sentence: str) -> str:
    """Processa uma frase completa, palavra por palavra."""
    # Importar SpecialRoules apenas quando necessário para evitar importação circular
    try:
        from SpecialRoules import apply_special_rules
    except ImportError:
        logger.warning("Módulo SpecialRoules não encontrado. Algumas regras especiais não serão aplicadas.")
        apply_special_rules = lambda words: words
    
    # Dividir a frase em palavras
    words = re.findall(r'\b\w+\b', sentence.lower())
    
    # Aplicar regras especiais (liaison, etc.)
    words = apply_special_rules(words)
    
    # Processar cada palavra
    pronunciations = []
    for word in words:
        pron = simplify_pronunciation(word)
        pronunciations.append(pron)
    
    # Juntar as pronúncias com espaços
    return ' '.join(pronunciations)

def remove_silent_endings(pronunciation, word):
    # Verificar se a palavra termina com 'ent' e a pronúncia termina com 't'
    if word.endswith('ent') and pronunciation.endswith('t'):
        pronunciation = pronunciation[:-1]
    # Verificar se a palavra termina com 'es' e a pronúncia termina com 's'
    if word.endswith('es') and pronunciation.endswith('s'):
        pronunciation = pronunciation[:-1]
    # Verificar se a palavra termina com 'e' mudo
    if word.endswith('e') and not word.endswith('le') and not word.endswith('re'):
        if pronunciation.endswith('ə'):
            pronunciation = pronunciation[:-1]
    # Adicionar outras regras conforme necessário
    return pronunciation


# Ajustar listas conforme sua necessidade
vogais_orais = ['a', 'e', 'i', 'o', 'u', 'é', 'ê', 'í', 'ó', 'ô', 'ú', 'ø', 'œ', 'ə', 'ü']
vogais_nasais = ['ã', 'ẽ', 'ĩ', 'õ', 'ũ']
semivogais = ['j', 'w', 'ɥ']
grupos_consonantais_especiais = ['tch', 'dj', 'sj', 'dʒ', 'ks', 'ts', 'ch', 'nh', 'lh']
consoantes_base = [
    'b','d','f','g','k','l','m','n','p','ʁ','r','s','t','v','z','ʃ','ʒ','ɲ','ŋ','ç'
]

excecoes_semivogais = {
    # Exemplos de exceções: padrão -> substituição
    # Caso queira ajustar manualmente certos clusters após a primeira passagem.
    # Por exemplo, se "sós.jó" sempre deveria ficar "sósjó"
    ("sós","jó"): ["sósjó"],
    ("próm","uv"): ["pró","muv"],
    ("a","prê"): ["a","prê"],
    ("di","né"): ["di","né"],
    ("si","lã"): ["si","lã"],
    ("sjeu"): ["siê"],
    ("ʀe","mé"): ["re","mé"],
    ("ʀe","la"): ["re","la"],
    ("tün"): ["ün"],
    ("smén"): ["semén"],
    # Novas exceções para corrigir problemas específicos
    ("relasiõ, relatiõ"): ["relasiõ"],  # Corrigir problema com vírgula
    ("él"): ["él"],  # Garantir que 'él' seja tratado corretamente
}

def e_vogal(c):
    return (c in vogais_orais) or (c in vogais_nasais)

def e_vogal_nasal(c):
    return c in vogais_nasais

def e_semivogal(c):
    return c in semivogais

def e_consoante(c):
    return c in consoantes_base

def e_grupo_consonantal(seq):
    return seq in grupos_consonantais_especiais

def tokenizar_palavra(palavra):
    i = 0
    tokens = []
    while i < len(palavra):
        matched = False
        for gc in grupos_consonantais_especiais:
            length = len(gc)
            if i + length <= len(palavra) and palavra[i:i+length] == gc:
                tokens.append(gc)
                i += length
                matched = True
                break
        if not matched:
            tokens.append(palavra[i])
            i += 1
    return tokens

def ajustar_semivogais(silabas):
    # Primeiro, mover semivogais do final da sílaba se a próxima inicia com vogal
    novas_silabas = []
    i = 0
    while i < len(silabas):
        s = silabas[i]
        if i < len(silabas)-1 and s:
            ultima_letra = s[-1] if s else ''
            proxima_silaba = silabas[i+1]
            if ultima_letra in semivogais and proxima_silaba and e_vogal(proxima_silaba[0]):
                # Move a semivogal para a próxima sílaba
                s = s[:-1]
                proxima_silaba = ultima_letra + proxima_silaba
                novas_silabas.append(s)
                silabas[i+1] = proxima_silaba
            else:
                novas_silabas.append(s)
        else:
            # Última sílaba, apenas adiciona
            novas_silabas.append(s)
        i += 1

    silabas = novas_silabas

    # Agora, tentar mesclar semivogais do início de uma sílaba anterior se a anterior terminou em vogal
    # Exemplo: se anterior terminar em vogal e a atual começar com semivogal + vogal, podemos unir.
    novas_silabas = []
    i = 0
    while i < len(silabas):
        if i > 0 and silabas[i] and len(silabas[i]) >= 2:
            primeira_letra = silabas[i][0]
            segunda_letra = silabas[i][1]
            silaba_anterior = novas_silabas[-1]
            if (e_semivogal(primeira_letra) and e_vogal(segunda_letra) and 
                silaba_anterior and e_vogal(silaba_anterior[-1])):
                # Mover a semivogal para a sílaba anterior
                silaba_anterior = silaba_anterior + primeira_letra
                s = silabas[i][1:]
                novas_silabas[i-1] = silaba_anterior
                if s:  # Se ainda sobrou algo na sílaba atual
                    novas_silabas.append(s)
            else:
                novas_silabas.append(silabas[i])
        else:
            novas_silabas.append(silabas[i])
        i += 1

    return novas_silabas

def silabificar(tokens):
    silabas = []
    silaba_atual = ""
    
    for i, token in enumerate(tokens):
        silaba_atual += token
        
        # Se o token atual é uma vogal e o próximo é uma consoante seguida de vogal,
        # ou se o token atual é uma consoante e o próximo é uma vogal,
        # terminamos a sílaba atual
        if i < len(tokens) - 2:
            if (e_vogal(token) and e_consoante(tokens[i+1]) and e_vogal(tokens[i+2])):
                silabas.append(silaba_atual)
                silaba_atual = ""
            elif (e_consoante(token) and e_vogal(tokens[i+1]) and 
                  i > 0 and e_consoante(tokens[i-1])):
                # Se temos um padrão CVC, a sílaba termina após o segundo C
                silabas.append(silaba_atual)
                silaba_atual = ""
        
        # Se chegamos ao final, adicionamos o que sobrou
        if i == len(tokens) - 1 and silaba_atual:
            silabas.append(silaba_atual)
            
    # Se não conseguimos dividir em sílabas, retornamos a palavra inteira como uma sílaba
    if not silabas and tokens:
        silabas = [''.join(tokens)]
        
    return silabas

def aplicar_excecoes(silabas):
    # Verificar se temos uma exceção para a sequência de sílabas
    for padrao, substituicao in excecoes_semivogais.items():
        if isinstance(padrao, tuple) and len(padrao) == len(silabas):
            match = True
            for i, p in enumerate(padrao):
                if silabas[i] != p:
                    match = False
                    break
            if match:
                return substituicao
        elif isinstance(padrao, str) and len(silabas) == 1 and silabas[0] == padrao:
            return substituicao
    return silabas

def aplicar_padroes_substituicao(pronuncia):
    """Aplica padrões de substituição para corrigir problemas específicos."""
    for pattern, replacement in PATTERN_REPLACEMENTS:
        pronuncia = re.sub(pattern, replacement, pronuncia)
    return pronuncia

def transcrever_fonetica(texto):
    """
    Transcreve um texto em francês para uma representação fonética simplificada.
    
    Args:
        texto: O texto em francês a ser transcrito
        
    Returns:
        Uma string com a transcrição fonética
    """
    # Verificar casos especiais de expressões completas
    for expr, pron in SPECIAL_CASES.items():
        if expr in texto.lower():
            texto = texto.lower().replace(expr, f" {pron} ")
    
    # Limpar o texto de caracteres não alfanuméricos, exceto espaços
    texto_limpo = re.sub(r'[^\w\s\'-]', ' ', texto)
    
    # Dividir em palavras
    palavras = texto_limpo.split()
    
    # Processar cada palavra
    resultado = []
    for palavra in palavras:
        # Ignorar palavras vazias
        if not palavra:
            continue
        
        # Verificar casos especiais primeiro
        palavra_lower = palavra.lower()
        if palavra_lower in SPECIAL_CASES:
            resultado.append(SPECIAL_CASES[palavra_lower])
            continue
            
        # Obter pronúncia IPA
        ipa = get_ipa_pronunciation(palavra)
        
        # Dividir em fonemas
        fonemas = split_into_phonemes(ipa)
        
        # Mapear para representação simplificada
        fonemas_simples = map_phonemes_to_simple(fonemas)
        
        # Aplicar regras específicas do francês
        fonemas_simples = apply_french_rules(fonemas_simples)
        
        # Juntar fonemas em uma string
        pronuncia = ''.join(fonemas_simples)
        
        # Remover terminações silenciosas
        pronuncia = remove_silent_endings(pronuncia, palavra)
        
        # Tokenizar a palavra para silabificação
        tokens = tokenizar_palavra(pronuncia)
        
        # Silabificar
        silabas = silabificar(tokens)
        
        # Ajustar semivogais
        silabas = ajustar_semivogais(silabas)
        
        # Aplicar exceções
        silabas = aplicar_excecoes(silabas)
        
        # Juntar sílabas
        pronuncia_final = ''.join(silabas)
        
        # Adicionar ao resultado
        resultado.append(pronuncia_final)
    
    # Juntar as palavras com espaços
    resultado_final = ' '.join(resultado)
    
    # Aplicar padrões de substituição para correções finais
    resultado_final = aplicar_padroes_substituicao(resultado_final)
    
    return resultado_final

# Função principal para uso externo
def get_pronunciation(text):
    """
    Função principal para obter a pronúncia de um texto em francês.
    
    Args:
        text: O texto em francês
        
    Returns:
        A pronúncia simplificada
    """
    try:
        return transcrever_fonetica(text)
    except Exception as e:
        logger.error(f"Erro ao processar texto '{text}': {e}")
        return text  # Retorna o texto original em caso de erro

# Funções para compatibilidade com o código original
def remove_punctuation_end(text):
    """Remove pontuação do final do texto."""
    return re.sub(r'[.,;:!?]+$', '', text)

def handle_apostrophes(words_list):
    """Trata apóstrofos em palavras."""
    new_words = []
    for word in words_list:
        if "'" in word:
            prefix, sep, suffix = word.partition("'")
            # Contrações comuns
            if prefix.lower() in ["l", "d", "j", "qu", "n", "m", "c"]:
                combined_word = prefix + suffix
                new_words.append(combined_word)
            else:
                new_words.append(prefix)
                new_words.append(suffix)
        else:
            new_words.append(word)
    return new_words

def apply_liaisons(words, pronunciations):
    """Aplica regras de liaison entre palavras."""
    # Implementação simplificada para compatibilidade
    return pronunciations

def convert_pronunciation_to_portuguese(pronunciation, word_idx, all_pronunciations):
    """Converte pronúncia IPA para representação em português."""
    # Usar a função simplify_pronunciation para compatibilidade
    return pronunciation

def transliterate_and_convert_sentence(sentence):
    """
    Função de compatibilidade para transcrever uma frase.
    
    Args:
        sentence: A frase em francês
        
    Returns:
        A pronúncia simplificada
    """
    return get_pronunciation(sentence)
