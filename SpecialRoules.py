import re
import logging

# Configuração do logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def handle_plus_pronunciation(index, words):
    """
    Decide como pronunciar "plus" de acordo com o contexto.
    Retorna a string de IPA aproximada (ex.: 'plys', 'ply', 'plyz').
    
    Regras aprimoradas:
    - Em construção negativa "ne ... plus" → "ply" (sem 's')
    - Em construção negativa + vogal → "plyz" (liaison)
    - Em comparativo "plus que/de" → "plys"
    - Em expressões como "en plus", "de plus" → "plys"
    - Em "plus ou moins" → "plys"
    - Em início de frase → "plys"
    """
    # A palavra atual é "plus"
    # 1) Verificar se há construção negativa "ne ... plus" (ou "n' ... plus")
    is_negative = False
    if index > 0:
        # Procurar "ne" ou "n'" antes de "plus", mesmo que não seja imediatamente antes
        for i in range(max(0, index-5), index):
            prev_word_lower = words[i].lower()
            if prev_word_lower in ("ne", "n'"):  
                is_negative = True
                break

    # 2) Verificar próxima palavra
    plus_pron = "plys"  # valor padrão = "plys" (caso signifique "mais" e seja seguido de substantivo)
    
    if index < len(words) - 1:
        next_word = words[index + 1]
        # Remove pontuação e pega só letras iniciais para ver se é vogal
        next_word_alpha = re.sub(r"[^a-zA-Zàâêîôûéèëïüÿæœ']", '', next_word.lower())

        if is_negative:
            # Construção negativa + plus
            # a) se a próxima palavra começa com vogal -> liaison => "plyz"
            if next_word_alpha and re.match(r'^[aeiouhâêîôûéèëïüÿæœ]', next_word_alpha):
                plus_pron = "plyz"
            else:
                # b) se não tem próxima vogal => "ply"
                plus_pron = "ply"
        else:
            # Se NÃO é negativo, verificar contextos específicos
            if next_word_alpha in ("de", "du", "des", "que", "qu'", "d'"):
                # Comparativo: "plus de/que..." => "plys"
                plus_pron = "plys"
            elif next_word_alpha in ("ou", "et"):
                # Expressões como "plus ou moins", "plus et plus" => "plys"
                plus_pron = "plys"
            elif next_word_alpha and re.match(r'^[aeiouhâêîôûéèëïüÿæœ]', next_word_alpha):
                # Seguido de vogal (possível liaison) => "plyz"
                if index > 0 and words[index-1].lower() in ("en", "de"):
                    # Expressões como "en plus", "de plus" => "plys"
                    plus_pron = "plys"
                else:
                    plus_pron = "plyz"
            else:
                # Default para outros casos => "plys"
                plus_pron = "plys"
    else:
        # "plus" é a última palavra da frase
        if is_negative:
            plus_pron = "ply"  # Negativo no final => "ply"
        elif index > 0 and words[index-1].lower() in ("en", "de"):
            # Expressões como "en plus", "de plus" => "plys"
            plus_pron = "plys"
        else:
            plus_pron = "plys"  # Default no final => "plys"

    logger.debug(f"'plus' em contexto: {' '.join(words[max(0, index-2):min(len(words), index+3)])} => {plus_pron}")
    return plus_pron


def handle_est_pronunciation(index, words):
    """
    Decide como pronunciar "est" de acordo com o contexto.
    Retorna 'é' quando for verbo (il est / elle est / c'est)
    Retorna 'ést' quando for outra acepção (ex.: leste/direção).
    
    Regras aprimoradas:
    - Verbo após pronomes pessoais (il, elle, ce, c', on) → "é"
    - Verbo após pronomes demonstrativos (cela, ceci) → "é"
    - Verbo em "qu'est-ce que" → "é"
    - Substantivo (ponto cardeal) → "ést"
    - Em expressões como "c'est-à-dire" → "é"
    """
    # Pegamos a palavra atual:
    current_word = words[index].lower()

    # Valor padrão
    est_pron = "ést"

    # Verificar se é parte de expressão fixa
    if index < len(words) - 1:
        next_word = words[index + 1].lower()
        if next_word.startswith("à-") or next_word == "à":
            # Expressão "est-à-dire" ou similar
            est_pron = "é"
            return est_pron

    # Verificar se é precedido por pronome ou determinante
    if index > 0:
        prev_word = words[index - 1].lower()
        # Remove pontuação
        prev_word_alpha = re.sub(r"[^a-zA-Zàâêîôûéèëïüÿæœ']", '', prev_word)

        # Pronomes pessoais e demonstrativos
        if prev_word_alpha in ("il", "elle", "on", "ce", "cela", "ceci", "ça") or prev_word_alpha.startswith("c'"):
            est_pron = "é"  # Ex.: "il est" => "il é"
        
        # Verificar se é parte de "qu'est-ce que"
        elif prev_word_alpha in ("qu'", "que") and index < len(words) - 1:
            if words[index + 1].lower() in ("ce", "c'", "ce-"):
                est_pron = "é"
    
    # Verificar se é parte de expressão "est-ce que"
    if index < len(words) - 2:
        if words[index + 1].lower() in ("ce", "c'", "ce-") and words[index + 2].lower() in ("que", "qu'"):
            est_pron = "é"

    logger.debug(f"'est' em contexto: {' '.join(words[max(0, index-2):min(len(words), index+3)])} => {est_pron}")
    return est_pron


def handle_est_ce_que(words):
    """
    Detecta e processa as variações de 'est-ce que' para substituir por tokens
    de pronúncia mais natural.
    
    Regras aprimoradas:
    - "est-ce que" → ["éss", "ke"]
    - "est-ce-que" → ["éss", "ke"]
    - "qu'est-ce que" → ["késs", "ke"]
    - "qu'est-ce qui" → ["késs", "ki"]
    - "est-ce qui" → ["éss", "ki"]
    """
    new_words = []
    i = 0

    while i < len(words):
        w_lower = words[i].lower()

        # 1) Verificar "qu'est-ce que/qui"
        if (i + 3 < len(words) and 
            w_lower in ("qu'", "que") and 
            words[i+1].lower() == "est" and 
            words[i+2].lower() in ("ce", "c'") and 
            words[i+3].lower() in ("que", "qui")):
            
            new_words.append("késs")
            new_words.append("ke" if words[i+3].lower() == "que" else "ki")
            i += 4  # Consumimos 4 tokens
            continue

        # 2) Verificar "est-ce-que/qui" (tudo junto com hífen)
        if w_lower in ("est-ce-que", "est-ce-qui"):
            new_words.append("éss")
            new_words.append("ke" if w_lower == "est-ce-que" else "ki")
            i += 1  # Consumimos 1 token
            continue

        # 3) Verificar "est-ce que/qui" (3 tokens separados)
        if (i + 2 < len(words) and 
            w_lower == "est" and 
            words[i+1].lower() in ("ce", "c'") and 
            words[i+2].lower() in ("que", "qui")):
            
            new_words.append("éss")
            new_words.append("ke" if words[i+2].lower() == "que" else "ki")
            i += 3  # Consumimos 3 tokens
            continue

        # 4) Verificar "est-ce" + "que/qui" (2 tokens)
        if (i + 1 < len(words) and 
            w_lower == "est-ce" and 
            words[i+1].lower() in ("que", "qui")):
            
            new_words.append("éss")
            new_words.append("ke" if words[i+1].lower() == "que" else "ki")
            i += 2  # Consumimos 2 tokens
            continue

        # Caso não seja nenhum desses cenários, não mexe
        new_words.append(words[i])
        i += 1

    return new_words


def handle_numbers_pronunciation(words):
    """
    Processa números em francês para melhorar a pronúncia.
    
    Regras:
    - Números terminados em "s" mudo (deux, trois, six, dix) + vogal → liaison
    - "un" antes de substantivo → "ẽ"
    - "vingt" + consoante → "vẽ"
    - "vingt" + vogal → "vẽt"
    - "cent" + consoante → "sã"
    - "cent" + vogal → "sãt"
    """
    number_map = {
        "un": {"default": "ẽ", "before_vowel": "ẽn"},
        "deux": {"default": "deu", "before_vowel": "deuz"},
        "trois": {"default": "trua", "before_vowel": "truaz"},
        "quatre": {"default": "katr"},
        "cinq": {"default": "sẽk", "before_vowel": "sẽk"},
        "six": {"default": "sis", "before_vowel": "siz"},
        "sept": {"default": "sét"},
        "huit": {"default": "üit", "before_vowel": "üit"},
        "neuf": {"default": "neuf", "before_vowel": "neuv"},
        "dix": {"default": "dis", "before_vowel": "diz"},
        "vingt": {"default": "vẽ", "before_vowel": "vẽt"},
        "trente": {"default": "trãt"},
        "quarante": {"default": "karãt"},
        "cinquante": {"default": "sẽkãt"},
        "soixante": {"default": "suasãt"},
        "quatre-vingt": {"default": "katr-vẽ", "before_vowel": "katr-vẽt"},
        "cent": {"default": "sã", "before_vowel": "sãt"},
        "mille": {"default": "mil"},
    }
    
    result = []
    i = 0
    
    while i < len(words):
        word = words[i].lower()
        
        # Verificar se é um número conhecido
        if word in number_map:
            # Verificar se a próxima palavra começa com vogal
            if i < len(words) - 1:
                next_word = words[i + 1].lower()
                next_word_alpha = re.sub(r"[^a-zA-Zàâêîôûéèëïüÿæœ']", '', next_word)
                
                if next_word_alpha and re.match(r'^[aeiouhâêîôûéèëïüÿæœ]', next_word_alpha):
                    # Próxima palavra começa com vogal
                    result.append(number_map[word].get("before_vowel", number_map[word]["default"]))
                else:
                    # Próxima palavra começa com consoante ou não há próxima palavra
                    result.append(number_map[word]["default"])
            else:
                # Último token
                result.append(number_map[word]["default"])
        else:
            # Não é um número conhecido
            result.append(words[i])
        
        i += 1
    
    return result


def handle_common_liaisons(words):
    """
    Processa liaisons comuns em francês.
    
    Regras:
    - Artigos (les, des, mes, tes, ses, nos, vos, leurs) + vogal → liaison
    - Preposições (dans, chez, sans) + vogal → liaison
    - Pronomes (nous, vous, ils, elles) + vogal → liaison
    - Adjetivos antes de substantivos
    """
    liaison_words = {
        # Artigos e determinantes
        "les": {"default": "lé", "before_vowel": "léz"},
        "des": {"default": "dé", "before_vowel": "déz"},
        "mes": {"default": "mé", "before_vowel": "méz"},
        "tes": {"default": "té", "before_vowel": "téz"},
        "ses": {"default": "sé", "before_vowel": "séz"},
        "nos": {"default": "no", "before_vowel": "noz"},
        "vos": {"default": "vo", "before_vowel": "voz"},
        "leurs": {"default": "leur", "before_vowel": "leurz"},
        
        # Preposições
        "dans": {"default": "dã", "before_vowel": "dãz"},
        "chez": {"default": "ché", "before_vowel": "chéz"},
        "sans": {"default": "sã", "before_vowel": "sãz"},
        
        # Pronomes
        "nous": {"default": "nu", "before_vowel": "nuz"},
        "vous": {"default": "vu", "before_vowel": "vuz"},
        "ils": {"default": "il", "before_vowel": "ilz"},
        "elles": {"default": "él", "before_vowel": "élz"},
        
        # Outros casos comuns
        "tout": {"default": "tu", "before_vowel": "tut"},
        "grand": {"default": "grã", "before_vowel": "grãt"},
        "petit": {"default": "peti", "before_vowel": "petit"},
        "gros": {"default": "gro", "before_vowel": "groz"},
    }
    
    result = []
    i = 0
    
    while i < len(words):
        word = words[i].lower()
        
        # Verificar se é uma palavra com possível liaison
        if word in liaison_words:
            # Verificar se a próxima palavra começa com vogal
            if i < len(words) - 1:
                next_word = words[i + 1].lower()
                next_word_alpha = re.sub(r"[^a-zA-Zàâêîôûéèëïüÿæœ']", '', next_word)
                
                if next_word_alpha and re.match(r'^[aeiouhâêîôûéèëïüÿæœ]', next_word_alpha):
                    # Próxima palavra começa com vogal
                    result.append(liaison_words[word]["before_vowel"])
                else:
                    # Próxima palavra começa com consoante
                    result.append(liaison_words[word]["default"])
            else:
                # Último token
                result.append(liaison_words[word]["default"])
        else:
            # Não é uma palavra com liaison conhecida
            result.append(words[i])
        
        i += 1
    
    return result


def handle_apostrophes(words_list):
    """
    Processa palavras com apóstrofo em francês.
    
    Regras aprimoradas:
    - Contrações comuns (l', d', j', qu', n', m', c') → juntar com palavra seguinte
    - Caso especial "s'en" → manter como token único
    - Outros casos → separar em tokens distintos
    """
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
            elif prefix.lower() == "s" and suffix.lower() == "il":
                # Caso especial para s'il
                new_words.append("sil")
            elif prefix.lower() == "s" and suffix.lower() == "ils":
                # Caso especial para s'ils
                new_words.append("sils")
            else:
                # Caso não seja uma contração comum, mantém separado
                new_words.append(prefix)
                new_words.append(suffix)
        else:
            new_words.append(word)
    return new_words
