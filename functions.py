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
IPA_TO_SIMPLE_PT = {
   'a': 'a', 'ɑ': 'a', 'ɑ̃': 'ã',
    'e': 'e', 'ɛ': 'é', 'ə': 'e', 
    'œ': 'é', 'ø': 'ê', # More Portuguese-like vowels ('eu' might be alternative for ø/œ)
    'i': 'i', 'ɪ': 'i',
    'o': 'o', 'ɔ': 'ó', 'ɔ̃': 'õ',
    'u': 'u', 'ʊ': 'u',
    'y': 'i', # Map French 'u' sound to Portuguese 'i'
    'ɛ̃': 'ẽ', 'œ̃': 'ẽ', # Keep nasal e (œ̃ could be 'ũ')
    'j': 'i', # Map French 'yod' to Portuguese 'i' semi-vowel
    'w': 'u', # Map French 'w' to Portuguese 'u' semi-vowel
    'ɥ': 'i', # Map French 'hu' sound to Portuguese 'i'
    'b': 'b', 'd': 'd', 'f': 'f', 'g': 'g', 'k': 'k',
    'l': 'l', 'm': 'm', 'n': 'n', 'ɲ': 'nh', 'ŋ': 'ng',
    'p': 'p', 'ʁ': 'r', 'r': 'r', 's': 's', 'ʃ': 'ch',
    't': 't', 'v': 'v', 'z': 'z', 'ʒ': 'j', # French 'j' sound maps to PT 'j'
    'x': 'ks', 'ts': 'ts', 'dz': 'dz', 'tʃ': 'tch', 'dʒ': 'dj',
    'h': '' # H is usually silent
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
    "j'aie": "je", # User wanted 'e' fechado, which is 'e' in our map
    "j'ai": "jé", # 'é' aberto
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
    """Obtém a pronúncia IPA de uma palavra, consultando dicionário e fallback para epitran."""
    word = word.lower()
    if word in ipa_dictionary:
        # Handle multiple pronunciations in dictionary
        pron = ipa_dictionary[word]
        if isinstance(pron, list):
            return pron[0] # Return the first pronunciation if multiple exist
        return pron
    if epi:
        try:
            return epi.transliterate(word)
        except Exception as e:
            logger.warning(f"Erro ao transliterar '{word}' com Epitran: {e}")
    return word # Return original word if IPA not found

def split_into_phonemes(ipa: str) -> List[str]:
    """Divide uma string IPA em fonemas individuais baseados no mapeamento PT."""
    phonemes = []
    i = 0
    # Limpar IPA de caracteres ignorados antes de processar
    cleaned_ipa = ''.join([c for c in ipa if c not in IGNORED_CHARS])
    valid_phonemes = set(IPA_TO_SIMPLE_PT.keys())

    while i < len(cleaned_ipa):
        matched = False
        # Verificar fonemas mais longos primeiro (ex: 3 caracteres como 'ɛ̃')
        for length in range(3, 0, -1): 
            if i + length <= len(cleaned_ipa):
                seq = cleaned_ipa[i:i+length]
                if seq in valid_phonemes:
                    phonemes.append(seq)
                    i += length
                    matched = True
                    break
        if not matched:
            # Se nenhum fonema conhecido for encontrado, registrar e avançar
            # Isso pode indicar um problema no IPA ou no mapeamento
            if cleaned_ipa[i] not in IGNORED_CHARS:
                 logger.debug(f"Caractere IPA não reconhecido/mapeado: '{cleaned_ipa[i]}' em IPA '{ipa}'")
            i += 1
            
    return phonemes
    
def map_phonemes_to_portuguese(phonemes: List[str]) -> List[str]:
    """Mapeia fonemas IPA para representação adaptada ao português usando IPA_TO_SIMPLE_PT."""
    result = []
    for p in phonemes:
        if p in IPA_TO_SIMPLE_PT:
            result.append(IPA_TO_SIMPLE_PT[p])
        else:
            # Logar aviso se um fonema IPA não estiver no mapa e pular
            logger.warning(f"Fonema IPA '{p}' não encontrado no mapeamento PT. Pulando.")
    return result

def apply_french_rules(phonemes_pt: List[str], word: str) -> List[str]:
    """Aplica regras de silenciamento francesas comuns sobre os fonemas já mapeados para PT."""
    if not phonemes_pt:
        return phonemes_pt

    processed = phonemes_pt[:] # Trabalhar em uma cópia

    # Regra: Remover 'e' final mudo (mapeado para 'e')
    # Condições: palavra termina em 'e' (não 'ée'), não é palavra curta (le, de, me, te, se, ce), 
    # e o último fonema PT é 'e'.
    if (word.endswith('e') and not word.endswith('ée') and len(word) > 2 and 
            processed and processed[-1] == 'e'):
        # Verificação adicional: não remover se for a única vogal ou sílaba tônica?
        # Por simplicidade, removemos por enquanto.
        processed = processed[:-1]

    # Regras para consoantes finais mudas (s, t, d, x, z, p, g) são complexas
    # e dependem muito de exceções e liaisons. É mais seguro confiar que
    # a fonte IPA (dicionário ou Epitran) já removeu essas consoantes mudas.
    # Ex: 'petit' -> /pəti/, 'petits' -> /pəti/, 'parlent' -> /paʁl/
    # Portanto, não aplicaremos regras genéricas de remoção de consoantes finais aqui.

    # Remover fonemas vazios resultantes do mapeamento (ex: 'h' -> '')
    processed = [p for p in processed if p]

    return processed

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
    """Pipeline completo para simplificar a pronúncia de uma palavra para PT."""
    word_lower = word.lower()
    logger.debug(f"--- Processando Palavra: '{word_lower}' ---")

    # 1. Casos Especiais (palavras/expressões inteiras)
    if word_lower in SPECIAL_CASES:
        logger.debug(f"Aplicando caso especial: {SPECIAL_CASES[word_lower]}")
        return SPECIAL_CASES[word_lower]

    # 2. Obter Pronúncia IPA
    ipa = get_ipa_pronunciation(word_lower)
    if not ipa or ipa == word_lower: # Se IPA não for encontrado ou falhar
        logger.warning(f"Não foi possível obter IPA para '{word_lower}'. Retornando palavra original.")
        return word_lower # Ou retornar vazio? Depende do comportamento desejado.
    
    # Lidar com múltiplas pronúncias (ex: 'relasjɔ̃, relasjɔ̃') - pegar a primeira
    if ',' in ipa:
        ipa = ipa.split(',')[0].strip()
    logger.debug(f"IPA Obtido: '{ipa}'")

    # 3. Dividir IPA em Fonemas
    phonemes_ipa = split_into_phonemes(ipa)
    logger.debug(f"Fonemas IPA: {phonemes_ipa}")

    # 4. Mapear Fonemas IPA para PT
    phonemes_pt = map_phonemes_to_portuguese(phonemes_ipa)
    logger.debug(f"Fonemas PT Mapeados: {phonemes_pt}")

    # 5. Aplicar Regras Francesas (silenciamento) sobre Fonemas PT
    phonemes_pt_rules = apply_french_rules(phonemes_pt, word_lower)
    logger.debug(f"Fonemas PT Pós-Regras Francesas: {phonemes_pt_rules}")

    # 6. Juntar Fonemas PT
    final_pronunciation = ''.join(phonemes_pt_rules)
    logger.debug(f"Pronúncia Unida Inicial: '{final_pronunciation}'")

    # 7. Aplicar Padrões de Substituição Globais (ajustes finos)
    for pattern, replacement in PATTERN_REPLACEMENTS:
        final_pronunciation = re.sub(pattern, replacement, final_pronunciation)
    logger.debug(f"Após Substituições Globais: '{final_pronunciation}'")

    # 8. FIX para Duplicação (Solução temporária e simples)
    # Verifica se a string é uma repetição exata de sua primeira metade
    if len(final_pronunciation) > 4 and len(final_pronunciation) % 2 == 0:
        half = len(final_pronunciation) // 2
        if final_pronunciation[:half] == final_pronunciation[half:]:
             logger.warning(f"Detectada duplicação exata em '{word}': '{final_pronunciation}'. Corrigindo para '{final_pronunciation[:half]}'")
             final_pronunciation = final_pronunciation[:half]

    logger.debug(f"Pronúncia Final para '{word}': '{final_pronunciation}'")
    return final_pronunciation

def process_sentence(sentence: str) -> str:
    """Processa uma frase completa, palavra por palavra, aplicando regras especiais."""
    logger.debug(f"--- Processando Frase: '{sentence}' ---")
    
    # Importar SpecialRoules dinamicamente para aplicar liaisons, etc.
    apply_special_rules = lambda words, sentence: words # Função dummy padrão
    try:
        # Assumindo que SpecialRoules.py tem apply_special_rules(words: List[str], sentence: str) -> List[str]
        from SpecialRoules import apply_special_rules 
        logger.info("Módulo SpecialRoules carregado com sucesso.")
    except ImportError:
        logger.warning("Módulo SpecialRoules não encontrado. Regras de liaison e outras não serão aplicadas.")
    except Exception as e:
        logger.error(f"Erro ao importar ou usar SpecialRoules: {e}")

    # Tokenizar preservando hífens e apóstrofos nas palavras
    # Usar regex mais simples para separar palavras e pontuações
    words_and_punctuation = re.findall(r"([\w'-]+)|([^\w'\s-]+)", sentence)
    # words_only = [match[0].lower() for match in words_and_punctuation if match[0]] # Extrai apenas as palavras
    
    # Preparar lista de palavras para SpecialRoules
    words_for_rules = [match[0].lower() for match in words_and_punctuation if match[0]]

    # Aplicar regras especiais (liaison, etc.) from SpecialRoules
    try:
        # Passar a lista de palavras e a frase original para contexto
        processed_word_list = apply_special_rules(words_for_rules, sentence) 
        logger.debug(f"Palavras após SpecialRoules: {processed_word_list}")
    except Exception as e:
        logger.error(f"Erro ao aplicar regras especiais de SpecialRoules: {e}")
        processed_word_list = words_for_rules # Fallback para lista original

    # Processar cada palavra (ou grupo unido por liaison) da lista retornada por SpecialRoules
    pronunciations = []
    word_index_rules = 0
    for word_part, punct_part in words_and_punctuation:
        if word_part: # Se for uma palavra
            if word_index_rules < len(processed_word_list):
                # Pega a palavra correspondente da lista processada por SpecialRoules
                # Isso assume que SpecialRoules mantém a ordem e número de palavras
                # ou retorna palavras modificadas/unidas na posição correta.
                # Se SpecialRoules alterar muito a estrutura, essa lógica pode falhar.
                current_word_to_pronounce = processed_word_list[word_index_rules]
                if current_word_to_pronounce: # Evitar processar strings vazias
                    pron = simplify_pronunciation(current_word_to_pronounce)
                    pronunciations.append(pron)
                word_index_rules += 1
            else:
                # Se SpecialRoules retornou menos palavras que o original (improvável?)
                logger.warning(f"Inconsistência entre palavras originais e processadas por SpecialRoules perto de '{word_part}'")
                # Tentar processar a palavra original como fallback
                pron = simplify_pronunciation(word_part.lower())
                pronunciations.append(pron)
        # elif punct_part: # Se for pontuação, adicionar à saída? Não, a função retorna só a pronúncia.
        #     # pronunciations.append(punct_part) # Manter pontuação?
        #     pass

    # Juntar as pronúncias com espaços
    final_sentence_pronunciation = ' '.join(filter(None, pronunciations)) # Filtra Nones/vazios
    logger.debug(f"Pronúncia Final da Frase: '{final_sentence_pronunciation}'")
    return final_sentence_pronunciation


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
