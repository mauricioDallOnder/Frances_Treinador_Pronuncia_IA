import re


def handle_plus_pronunciation(index, words):
    """
    Decide como pronunciar "plus" de acordo com o contexto.
    Retorna a string de IPA aproximada (ex.: 'plys', 'ply', 'plyz').
    """

    # A palavra atual é "plus"
    # 1) Verificar se há construção negativa "ne ... plus" (ou "n' ... plus")
    #    Simplesmente checando a palavra anterior (index-1) se existe
    is_negative = False
    if index > 0:
        prev_word_lower = words[index - 1].lower()
        if prev_word_lower in ("ne", "n'"):  
            # Exemplo simplificado: "je ne veux plus"
            is_negative = True

    # 2) Verificar próxima palavra
    plus_pron = "plys"  # valor padrão = "plys" (caso signifique "mais" e seja seguido de substantivo)
    if index < len(words) - 1:
        next_word = words[index + 1]
        # Remove pontuação e pega só letras iniciais para ver se é vogal
        next_word_alpha = re.sub(r"[^a-zA-Zàâêîôûéèëïüÿæœ']", '', next_word.lower())

        if is_negative:
            # Constr. negativa + plus
            # a) se a próxima palavra começa com vogal -> liaison => "plyz"
            if next_word_alpha and re.match(r'^[aeiouhâêîôûéèëïüÿæœ]', next_word_alpha):
                plus_pron = "plyz"
            else:
                # b) se não tem próxima vogal => "ply"
                plus_pron = "ply"
        else:
            # Se NÃO é negativo, supõe-se significado de "mais"
            # e normalmente pronunciamos "plys" ao vir "de" ou "du" ou substantivo
            # (Você pode refinar. Aqui é apenas um exemplo.)
            # Exemplo: Se a próxima palavra é “de”, “plus de ...” => "plys də"
            if next_word_alpha in ("de", "du", "des"):
                plus_pron = "plys"
            else:
                # default = "plys"
                plus_pron = "plys"
    else:
        # "plus" é a última palavra da frase?
        # Se for negativo => "ply", senão => "plys".
        plus_pron = "ply" if is_negative else "plys"

    return plus_pron


def handle_est_pronunciation(index, words):
    """
    Decide como pronunciar "est" de acordo com o contexto.
    Retorna 'ɛ' quando for verbo (il est / elle est / c’est)
    Retorna 'ɛst' quando for outra acepção (ex.: leste/direção).
    """
    # Pegamos a palavra atual:
    current_word = words[index].lower()

    # Checar se é “est” no sentido de verbo “être”.
    # Heurística simples: se a palavra anterior for 'il', 'elle', 'c'' ou 'ce',
    # ou se estiver sozinha num chunk "il est ..." => usaremos /ɛ/.
    # Caso contrário, usaremos /ɛst/.
    # Você pode refinar essa lógica conforme a gramática que quiser tratar.
    
    # Valor padrão
    est_pron = "ɛst"

    if index > 0:
        prev_word = words[index - 1].lower()
        # Remove pontuação
        prev_word_alpha = re.sub(r"[^a-zA-Zàâêîôûéèëïüÿæœ']", '', prev_word)

        # Se a anterior é "il", "elle", "on", "ce", "c'":
        if prev_word_alpha in ("il", "elle", "on", "ce") or prev_word_alpha.startswith("c'"):
            est_pron = "ɛ"  # Ex.: "il est" => "il ɛ"
    else:
        pass

    return est_pron

import re

import re

def handle_est_ce_que(words):
    """
    Detecta 'est-ce que' e 'est-ce-que' para substituir por tokens
    de pronúncia desejada:
      - Se for 'est' + 'ce' + 'que' => vira ['és', 'ke']
      - Se for 'est-ce-que' (tudo em 1 token) => vira ['ést-s', 'ke']
        (exemplo de pronúncia pedida)
    """
    new_words = []
    i = 0

    while i < len(words):
        w_lower = words[i].lower()

        # 1) Se for "est-ce-que" tudo junto (com hífen)
        if w_lower == "est-ce-que":
            # Usuário quer "ést-s ke"
            new_words.append("éss")
            new_words.append("ke")
            i += 1  # Consumimos 1 token
            continue

        # 2) Se for 'est' e ainda tivermos mais 2 palavras: 'ce' e 'que'
        if (w_lower == "est"
            and i + 2 < len(words)
            and words[i+1].lower() == "ce"
            and words[i+2].lower() == "que"):
            # Usuário quer "és ke"
            new_words.append("éss")
            new_words.append("ke")
            i += 3  # Consumimos 3 tokens
            continue

        # 3) Se for 'est-ce' + 'que' (apenas 2 tokens) - caso intermediário
        if (w_lower == "est-ce"
            and i + 1 < len(words)
            and words[i+1].lower() == "que"):
            # Decide como pronunciar: por ex. "és" + "ke"
            new_words.append("éss")
            new_words.append("ke")
            i += 2
            continue

        # Caso não seja nenhum desses cenários, não mexe
        new_words.append(words[i])
        i += 1

    return new_words

