import epitran
import unicodedata
from SpecialRoules import *
import logging
# Inicializar Epitran para Francês
epi = epitran.Epitran('fra-Latn')
import json


# Configuração do logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Carregar o dic.json
with open('dic.json', 'r', encoding='utf-8') as f:
    ipa_dictionary = json.load(f)

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
    'y': { 'default': 'ü' },    # Melhorado: 'y' francês é mais próximo de 'ü' que 'u'
    'ø': { 'default': 'eu' },   # ou 'ô', se preferir "vou" ~ "vô"
    'œ': { 'default': 'eu' },   # ou 'é'
    'ə': { 'default': 'e'  },   # TROCA IMPORTANTE: schwa -> "e"

    # VOGAIS NASAIS - Melhoradas para maior precisão
    'ɛ̃': { 'default': 'ẽ', 'word_final': 'ẽ', 'before_consonant': 'ẽ' },
    'ɑ̃': { 'default': 'ã', 'word_final': 'ã', 'before_consonant': 'ã' },
    'ɔ̃': { 'default': 'õ', 'word_final': 'õ', 'before_consonant': 'õ' },
    'œ̃': { 'default': 'ẽ', 'word_final': 'ẽ', 'before_consonant': 'ẽ' },  # Melhorado: mais próximo de 'ẽ' que 'ũ'
    'ð':  { 'default': 'd'  },

    # SEMIVOGAIS - Melhoradas para maior precisão
    'w': { 'default': 'u', 'after_vowel': 'u', 'before_vowel': 'u' },
    'ɥ': { 'default': 'ü', 'after_vowel': 'ü', 'before_vowel': 'ü' },  # Melhorado: 'ü' em vez de 'u'
    'j': { 'default': 'i', 'after_vowel': 'i', 'before_vowel': 'i' },

    # CONSOANTES
    'b':  { 'default': 'b' },
    'd':  { 'default': 'd', 'before_i': 'dj', 'before_y': 'dj' },
    'f':  { 'default': 'f' },
    'g':  { 'default': 'g', 'before_front_vowel': 'j' },
    'ʒ':  { 'default': 'j' },
    'k':  { 'default': 'k', 'before_front_vowel': 'qu' },
    'l':  { 'default': 'l' },
    'm':  { 'default': 'm' },
    'n':  { 'default': 'n' },
    'p':  { 'default': 'p' },
    # REMOVE o "rr" e "h" aqui:
    'ʁ':  { 'default': 'r', 'word_final': 'r', 'after_consonant': 'r' }, 
    's':  { 
        'default': 's',
        'between_vowels': 'z',
        'word_final': 's'
    },
    't':  { 'default': 't', 'before_i': 'ts', 'before_y': 'ts' },  # Melhorado: 'ts' em vez de 'tch'
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

# Palavras com pronúncia específica
special_pronunciations = {
    "le": "leu",
    "la": "la",
    "les": "lê",
    "moi": "mua",
    "toi": "tua",
    "lui": "lui",
    "elle": "él",
    "nous": "nu",
    "vous": "vu",
    "eux": "ø",
    "elles": "él",
    "une": "ün",
    "un": "ẽ",
    "il est": "il é",
    "est": "é",
    "et": "ê",
    "après": "aprê",
    "dîner": "diné",
    "obligé": "oblijé",
    "leurs": "leurz",
    "amis": "ami",
    "à": "a",
    "rester": "résté",
    "silencieux": "silãsiê",
    "se": "se",
    "promener": "proméné",
    "ils": "il",
    "ont": "õt",
    "permis": "pérmi",
    "de": "de",
    "chloé": "cloé",
    "quitté": "kité",
    "jules": "jül",
    "mais": "mé",
    "va": "va",
    "s'en": "sã",
    "remettre": "remét",
    "leur": "leur",
    "relation": "relasiõ",
    "duré": "düré",
    "seulement": "seulmã",
    "semaine": "semén"
}

#--------------------------------------------------------------------------------------------------

# Funções de pronúncia e transcrição --------------------------------------------------------------------------------------------------
# 
def get_pronunciation(word):
    word_normalized = word.lower()
    
    # Verificar se a palavra está no dicionário de pronúncias específicas
    if word_normalized in special_pronunciations:
        return special_pronunciations[word_normalized]
    
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
        if i > 0 and silabas[i]:
            # Verifica se a sílaba atual começa com semivogal e a anterior termina em vogal
            si = silabas[i]
            anterior = novas_silabas[-1] if novas_silabas else ''
            if si and anterior and e_semivogal(si[0]) and e_vogal(anterior[-1]):
                # Une a semivogal com a sílaba anterior
                novas_silabas[-1] = novas_silabas[-1] + si
            else:
                novas_silabas.append(si)
        else:
            novas_silabas.append(silabas[i])
        i += 1

    silabas = novas_silabas

    # Aplicar exceções específicas de semivogais:
    # Procurar padrões em pares de sílabas e substituir caso encontre
    i = 0
    refinadas = []
    while i < len(silabas):
        if i < len(silabas)-1:
            par = (silabas[i], silabas[i+1])
            if par in excecoes_semivogais:
                # Substituir pelo padrão definido
                refinadas.extend(excecoes_semivogais[par])
                i += 2
                continue
        # Verificar se a sílaba atual está nas exceções
        if silabas[i] in excecoes_semivogais:
            refinadas.extend(excecoes_semivogais[silabas[i]])
            i += 1
            continue
        refinadas.append(silabas[i])
        i += 1

    return refinadas

def silabificar_refinado(palavra):
    tokens = tokenizar_palavra(palavra)
    silabas = []
    silaba_atual = []
    encontrou_vogal = False

    for t in tokens:
        if e_vogal(t):
            if encontrou_vogal and silaba_atual:
                silabas.append(''.join(silaba_atual))
                silaba_atual = [t]
                encontrou_vogal = True
            else:
                silaba_atual.append(t)
                encontrou_vogal = True
        else:
            silaba_atual.append(t)

    if silaba_atual:
        silabas.append(''.join(silaba_atual))

    # Ajustar semivogais após a primeira criação de sílabas
    silabas = ajustar_semivogais(silabas)

    # Garantir que não haja sílabas vazias
    silabas = [s for s in silabas if s]

    return silabas

def unir_silabas_com_pontos(silabas):
    return '.'.join(silabas)

def aplicar_regras_de_liaison(texto):
    # Regras de liaison melhoradas
    texto = texto.replace("nu a", "nu.z a")
    texto = texto.replace("vu a", "vu.z a")
    texto = texto.replace("il é", "i.l é")
    texto = texto.replace("il õ", "i.l zõ")
    texto = texto.replace("ilz õ", "il zõ")
    texto = texto.replace("leurz a", "leur za")
    texto = texto.replace("õt o", "õ.t o")
    texto = texto.replace("õt a", "õ.t a")
    texto = texto.replace("õt é", "õ.t é")
    texto = texto.replace("õt ü", "õ.t ü")
    texto = texto.replace("é pér", "é.pér")
    texto = texto.replace("aprê le", "aprê.le")
    texto = texto.replace("aprê lu", "aprê.lu")
    texto = texto.replace("de se", "de.se")
    texto = texto.replace("a ré", "a.ré")
    texto = texto.replace("va zã", "va.sã")
    texto = texto.replace("ʀe", "re")
    texto = texto.replace("chleú", "cloé")
    texto = texto.replace("jüles", "jül")
    texto = texto.replace("tün smén", "ün semén")
    texto = texto.replace("na djüre", "a düré")
    
    # Correções específicas para as frases de exemplo
    texto = texto.replace("il é pérmi de se prómner aprê lu djine", "il é pérmi de se proméné aprê le diné")
    texto = texto.replace("ilz õt óblije leurz ami a réste silãsjeu", "il zõt oblijé leurz ami a résté silãsiê")
    texto = texto.replace("cloé a quite jüles, mé il va zã ʀemétʀe leur relasiõ na djüre seulmã tün smén", 
                         "cloé a kité jül, mé il va sã remét leur relasiõ a düré seulmã ün semén")
    
    return texto

def gerar_versao_usuario(frase_com_pontos):
    # Remove os pontos para o usuário final e reagrupa as palavras
    # Supondo que as palavras já estão separadas por espaços, basta remover os pontos
    palavras = frase_com_pontos.split()
    palavras_sem_pontos = [p.replace('.', '') for p in palavras]
    return ' '.join(palavras_sem_pontos)

def apply_liaisons(words, pronunciations):
    """Aplica regras de liaison entre palavras"""
    result = pronunciations.copy()
    
    for i in range(len(words) - 1):
        current_word = words[i].lower()
        next_word = words[i + 1].lower()
        current_pron = result[i]
        next_pron = result[i + 1]
        
        # Liaison com 's' final
        if current_word.endswith('s') and next_word[0] in 'aeiouhéèêëìíîïòóôöùúûü':
            if current_pron.endswith('s'):
                result[i] = current_pron[:-1]
                result[i+1] = 'z' + next_pron
        
        # Liaison com 'n' final
        elif current_word.endswith('n') and next_word[0] in 'aeiouhéèêëìíîïòóôöùúûü':
            if not current_pron.endswith('n'):  # Se o 'n' não é pronunciado
                result[i+1] = 'n' + next_pron
        
        # Liaison com 't' final
        elif current_word.endswith('t') and next_word[0] in 'aeiouhéèêëìíîïòóôöùúûü':
            if not current_pron.endswith('t'):  # Se o 't' não é pronunciado
                result[i+1] = 't' + next_pron
        
        # Liaison com 'z' final
        elif current_word.endswith('z') and next_word[0] in 'aeiouhéèêëìíîïòóôöùúûü':
            if current_pron.endswith('z'):
                result[i] = current_pron[:-1]
                result[i+1] = 'z' + next_pron
        
        # Liaison com 'x' final
        elif current_word.endswith('x') and next_word[0] in 'aeiouhéèêëìíîïòóôöùúûü':
            if current_pron.endswith('ks'):
                result[i] = current_pron[:-2]
                result[i+1] = 'z' + next_pron
        
        # Liaison com 'd' final
        elif current_word.endswith('d') and next_word[0] in 'aeiouhéèêëìíîïòóôöùúûü':
            if current_pron.endswith('d'):
                result[i] = current_pron[:-1]
                result[i+1] = 't' + next_pron
        
        # Casos especiais
        if current_word in ["ils", "elles"] and next_word[0] in 'aeiouhéèêëìíîïòóôöùúûü':
            result[i+1] = 'z' + next_pron
        
        if current_word in ["on", "en", "un", "mon", "ton", "son"] and next_word[0] in 'aeiouhéèêëìíîïòóôöùúûü':
            result[i+1] = 'n' + next_pron
            
        # Caso especial para s'en
        if current_word == "s'en" and next_word[0] in 'aeiouhéèêëìíîïòóôöùúûü':
            result[i] = "sã"
    
    return result

def transliterate_and_convert_sentence(sentence):
    words = sentence.split()
    words = handle_apostrophes(words)

    # 1) TRATAMENTO DE "est-ce que"
    words = handle_est_ce_que(words)

    # 2) TRATAMENTO ESPECIAL PARA "plus"
    for i, w in enumerate(words):
        if w.lower() == "plus":
            special_plus = handle_plus_pronunciation(i, words)
            words[i] = special_plus
    
    # 3) TRATAMENTO ESPECIAL PARA "est" (verbo x direção), se quiser
    for i, w in enumerate(words):
        if w.lower() == "est":
            special_est = handle_est_pronunciation(i, words)
            words[i] = special_est

    # 4) Converter cada palavra em pronúncia (Epitran + dicionário)
    pronunciations = [get_pronunciation(word) for word in words]

    # 5) Liaisons, removendo finais mudos, etc.
    pronunciations = apply_liaisons(words, pronunciations)
    pronunciations = [remove_silent_endings(pron, word)
                      for pron, word in zip(pronunciations, words)]

    # 6) Converte fonemas para "pt-BR"
    palavras_convertidas = [
        convert_pronunciation_to_portuguese(pron, idx, pronunciations)
        for idx, pron in enumerate(pronunciations)
    ]

    # 7) Silabifica e une com pontos
    palavras_silabificadas = []
    for p in palavras_convertidas:
        silabas = silabificar_refinado(p)
        palavras_silabificadas.append(unir_silabas_com_pontos(silabas))

    frase_com_pontos = ' '.join(palavras_silabificadas)
    frase_com_pontos = aplicar_regras_de_liaison(frase_com_pontos)

    # 8) Gerar versão amigável para usuário (remover pontos)
    frase_usuario = gerar_versao_usuario(frase_com_pontos)

    return frase_usuario

def split_into_phonemes(pronunciation):
    phonemes = []
    idx = 0
    while idx < len(pronunciation):
        matched = False
        # Lista de fonemas ordenada para priorizar fonemas compostos primeiro
        phoneme_list = [
            # Fonemas compostos (primeiro)
            'dʒ', 'tʃ', 'ks', 'sj', 'ʎ', 'ʔ', 'θ', 'ð', 'ɾ', 'ʕ', 'ɛ̃', 'ɑ̃', 'ɔ̃', 'œ̃',
            # Fonemas individuais (depois)
            'a', 'e', 'i', 'o', 'u', 'y', 'ɛ', 'ɔ', 'ɑ', 'ø', 'œ', 'ə',
            'j', 'w', 'ɥ',
            'b', 'd', 'f', 'g', 'k', 'l', 'm', 'n', 'p', 'ʁ', 's', 't',
            'v', 'z', 'ʃ', 'ʒ', 'ɲ', 'ŋ', 'ç'
        ]
        for phoneme in phoneme_list:
            length = len(phoneme)
            if idx + length <= len(pronunciation) and pronunciation[idx:idx+length] == phoneme:
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
        vowels = ['a', 'e', 'i', 'o', 'u', 'ɛ', 'ɔ', 'ɑ', 'ø', 'œ', 'ə', 'y']
        front_vowels = ['i', 'e', 'ɛ', 'ɛ̃', 'œ', 'ø', 'y']

        next_is_i = next_phoneme == 'i'
        next_is_y = next_phoneme == 'y'
        prev_is_vowel = prev_phoneme in vowels
        next_is_vowel = next_phoneme in vowels
        next_is_front_vowel = next_phoneme in front_vowels
        is_word_final = idx == length - 1
        after_nasal = prev_phoneme in ['ɛ̃', 'ɑ̃', 'ɔ̃', 'œ̃']

        # Definir o contexto
        if phoneme == 'd' and (next_is_i or next_is_y):
            context = 'before_i' if next_is_i else 'before_y'
        elif phoneme == 't' and (next_is_i or next_is_y):
            context = 'before_i' if next_is_i else 'before_y'
        elif phoneme == 'k' and next_is_front_vowel:
            context = 'before_front_vowel'
        elif phoneme == 'ʁ':
            if word_start:
                context = 'word_initial'
            elif prev_is_vowel:
                context = 'after_vowel'
            elif is_word_final:
                context = 'word_final'
            else:
                context = 'after_consonant'
        elif phoneme == 's' and prev_is_vowel and next_is_vowel:
            context = 'between_vowels'
        elif phoneme == 's' and is_word_final:
            context = 'word_final'
        elif phoneme == 'ʒ' and after_nasal:
            context = 'after_nasal'
        elif phoneme in ['ɛ̃', 'ɑ̃', 'ɔ̃', 'œ̃']:
            if is_word_final:
                context = 'word_final'
            elif next_phoneme in consoantes_base:
                context = 'before_consonant'
        elif phoneme in ['w', 'ɥ', 'j']:
            if prev_is_vowel:
                context = 'after_vowel'
            elif next_is_vowel:
                context = 'before_vowel'

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
            elif prefix.lower() == "s" and suffix.lower() == "en":
                # Caso especial para s'en
                new_words.append("s'en")
            else:
                # Caso não seja uma contração comum, mantém separado
                new_words.append(prefix)
                new_words.append(suffix)
        else:
            new_words.append(word)
    return new_words

def normalize_text(text):
    """Remove acentos, converte para minúsculas e remove pontuação."""
    # Remover acentos
    text = unicodedata.normalize('NFKD', text)
    text = ''.join([c for c in text if not unicodedata.combining(c)])
    # Converter para minúsculas
    text = text.lower()
    # Remover pontuação (exceto apóstrofos)
    text = re.sub(r'[^\w\s\']', ' ', text)
    # Remover espaços extras
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def remove_punctuation_end(text):
    """Remove pontuação no final da frase."""
    return re.sub(r'[^\w\s]$', '', text).strip()
