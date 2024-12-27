import re
import random

COLOR_LIST = [
    '#FF0000', '#FF6600', '#CC00FF', '#FFCC00', '#0099FF',
    '#FF9900', '#0033CC', '#666600', '#33CC33', '#990000',
    '#339900', '#336600', '#CC3333', '#003366', '#336600',
    '#FF3399', '#99FF00', '#FF0033', '#CC3300', '#00CCCC',
    '#336633', '#9900CC', '#006600', '#FF3300', '#CC33CC',
    '#333300', '#6600CC', '#CC00CC', '#0033FF', '#009966',
    '#CC0066', '#33CC00', '#CC6666', '#999900', 'Tomato',
    '#336666', '#669966', 'SlateBlue', '#33FF00', '#666600',
    '#FF0066', '#CCCC33', '#33CC66', '#0033CC', '#660099',
    '#CC0033', '#009966', '#FF0000', '#33CCCC', '#0000FF'
]


def get_pronunciation_hints(word):
    """
    Analisa a palavra em francês e retorna a palavra com trechos destacados
    e explicações sobre a pronúncia de cada trecho.

    Retorna um dicionário com:
        - 'word': palavra original
        - 'highlighted_word': palavra com trechos destacados (com spans coloridos)
        - 'explanations': lista de explicações para cada trecho destacado
    """

    front_vowels = 'iéeèêëïyæœ'
    all_vowels = 'aeiouéêèëíóôúãõœæy'
    all_consonants = 'bcdfgjklmnpqrstvwxzʃʒɲŋçh'

    found_matches = []
    lower_case_word = word.lower()

    def add_match_object(match_object, explanation_template):
        """
        Adiciona um trecho destacado (encontrado pelo match_object) à lista de matches,
        incluindo a cor escolhida aleatoriamente e a explicação formatada.
        """
        if not match_object:
            # Se for None, não faz nada para evitar erro
            return

        start_position, end_position = match_object.start(), match_object.end()
        matched_text = match_object.group(0)
        chosen_color = random.choice(COLOR_LIST)

        explanation = explanation_template.format(
            match=f'<span style="color:{chosen_color}; font-weight:bold">{matched_text}</span>'
        )
        found_matches.append((start_position, end_position, matched_text, explanation, chosen_color))

    # -------------------------------------------------------------------------
    # Casos especiais (artigos, palavras específicas) que exigem regras únicas
    # -------------------------------------------------------------------------

    # Exemplo "j'ai"
    if lower_case_word == "j'ai":
        found_index = word.find('ai')
        if found_index != -1:
            start_position = found_index
            end_position = found_index + 2  # 'ai' tem 2 caracteres
            matched_text = word[start_position:end_position]
            explanation_text = 'Em "j\'ai", a pronúncia fica aproximadamente como "jê".'

            # Em vez de passar re.match diretamente, fazemos um re.search para confirmar
            partial_substring = word[start_position:end_position]  # deve ser "ai"
            match_object = re.search(r'^ai$', partial_substring)
            add_match_object(match_object, explanation_text)

    # Exemplo para "le"
    if lower_case_word == 'le':
        if word.endswith('e'):
            start_position = len(word) - 1
            end_position = len(word)
            matched_text = word[start_position:end_position]
            explanation_text = (
                'No artigo "le", o "e" é muito curto, parecido com "luh" rápido, não "lê". '
                'Ex: "le chat" → "luh chá".'
            )
            # Ao invés de re.match(r'e$', substring), verifique via re.search:
            partial_substring = word[start_position:end_position]  # "e"
            match_object = re.search(r'^e$', partial_substring)
            add_match_object(match_object, explanation_text)

    # Exemplo para "les"
    if lower_case_word == 'les':
        if word.endswith('es'):
            start_position = len(word) - 2
            end_position = len(word)
            matched_text = word[start_position:end_position]
            explanation_text = (
                'No artigo "les", soa como "lê", diferente de "le" (lu). Ex: "les chats" → "lê chá".'
            )
            partial_substring = word[start_position:end_position]  # "es"
            match_object = re.search(r'^es$', partial_substring)
            add_match_object(match_object, explanation_text)

    # Exemplo para palavras terminadas em "chats"
    if lower_case_word.endswith('chats'):
        search_object = re.search(r'(s)$', word)
        if search_object:
            add_match_object(
                search_object,
                'Em "chats", o "{match}" final é mudo, então "chats" soa como "chá".'
            )

    # Exemplo para "grand" ou "quand" (liaison do 'd')
    if re.search(r'(grand|quand)$', lower_case_word):
        search_object = re.search(r'(d)$', word)
        if search_object:
            add_match_object(
                search_object,
                'Na ligação (liaison), o "{match}" final soa como "t" antes de vogal. '
                'Ex: "grand arbre" → "gran_t arbre".'
            )

    # -------------------------------------------------------------------------
    # Padrões genéricos (sem AFI completo), usando comparações com o português
    # -------------------------------------------------------------------------
    regex_pattern_explanations = [
        (
            r'(am|em|an|en)(?=[bdfgjklpqrstvwxzʃʒɲŋç])',
            'A sequência {match} indica um som nasal parecido com "ãn". Ex: {match} ~ "ãn".'
        ),
        (
            r'(in|im|yn|ym|ein|ain|ien|aim)(?=[bdfgjklpqrstvwxzʃʒɲŋç])',
            'A sequência {match} representa um som nasal tipo "iñ" ou "iãn", lembrando "im" nasalizado.'
        ),
        (
            r'(on|om)(?=[bdfgjklpqrstvwxzʃʒɲŋç])',
            'A sequência {match} dá um som nasal parecido com "õ", como em "põe".'
        ),
        (
            r'(un|um)(?=[bdfgjklpqrstvwxzʃʒɲŋç])',
            'A sequência {match} produz um som nasal parecido com "œ̃", próximo de "ãn" com os lábios arredondados. '
            'Pense em {match} como "ãn" mais fechado.'
        ),
        (
            r'(au|aux|eau|eaux)',
            'A sequência {match} é pronunciada aproximadamente como "ô".'
        ),
        (
            r'(oy)',
            '{match} soa como "uai", semelhante ao mineiro "uai".'
        ),
        (
            r'(x)(?=[' + all_consonants + '])',
            '"{match}" antes de consoante soa como "ks".'
        ),
        (
            r'(y)(?=[' + all_vowels + '])',
            '"{match}" antes de vogal soa como o "i" deslizado, tipo "ia" → "iá".'
        ),
        (
            r'(c)(?=[' + front_vowels + '])',
            '"{match}" antes de vogal frontal soa como "s".'
        ),
        (
            r'(ch)',
            '"{match}" soa como "x" em "xarope".'
        ),
        (
            r'(j|g)(?=[eiy])',
            '"{match}" soa como o "j" de "jogar" antes de e, i ou y.'
        ),
        (
            r'(gn)',
            '"{match}" soa como "nh" em português.'
        ),
        (
            r'(e|es)$',
            'No final da palavra, "{match}" geralmente não é pronunciado.'
        ),
        (
            r'(oi)',
            'A sequência {match} soa como "uá".'
        ),
        (
            r'(ou)',
            'A sequência {match} soa como "u" fechado.'
        ),
        (
            r'(ille)',
            '"{match}" soa como "iê".'
        ),
        (
            r'(eu)',
            '"{match}" tem um som semelhante a "eu" fechado, algo entre "e" e "u".'
        ),
        (
            r'(é)',
            '"{match}" soa como "ê" mais fechado.'
        ),
        (
            r'(è|ê|ai|ei)',
            'A combinação {match} soa como "é" aberto.'
        ),
        (
            r'(er)$',
            'No final da palavra, "{match}" soa como "ê" fechado.'
        ),
        (
            r'(qu)',
            '"{match}" é pronunciado como "k".'
        ),
        (
            r'(ais|ait|aient)$',
            'Ao final, "{match}" costuma soar como "é".'
        ),
        (
            r'(h)',
            '"{match}" geralmente é mudo. H aspirado não se liga à vogal seguinte, mas não altera muito o som.'
        ),
        (
            r'(ge)$',
            'No final, "{match}" costuma soar como "je" (o j de "jogar").'
        ),
        (
            r'(ail)',
            '"{match}" soa como "ai" (ái).'
        ),
        (
            r'(eil)',
            '"{match}" soa como "ei" fechado.'
        ),
        (
            r'(euil)',
            '"{match}" soa algo como "õe", um som entre "e" e "u" nasalizado.'
        ),
        (
            r'(œil)',
            '"{match}" soa semelhante a "ói" curto, com os lábios arredondados.'
        ),
        (
            r'(ien)',
            '"{match}" soa como "iã" nasalizado.'
        ),
        (
            r'(ion)',
            '"{match}" soa como "iõ" nasalizado.'
        ),
        (
            r'(tion)$',
            'No final, "{match}" soa como "siõ" (s + iõ nasal).'
        ),
        (
            r'(ier)$',
            'No final, "{match}" soa como "iê".'
        ),
        (
            r'(iez)$',
            'No final da forma verbal, "{match}" soa como "iê".'
        ),
        (
            r'(oin)',
            'A sequência {match} soa como "uã" nasalizado.'
        ),
        (
            r'(ui)',
            'A sequência {match} soa como "üi", algo como "wi" em inglês.'
        ),
        (
            r'(œu)',
            '"{match}" soa entre "eu" e "éu" com lábios arredondados.'
        ),
        (
            r'(œ)',
            '"{match}" soa como um "é" com lábios arredondados, algo entre "é" e "eu".'
        ),
        (
            r'(cc)(?=[eiy])',
            '"{match}" pode soar como "ks".'
        ),
        (
            r'(ç)',
            '"{match}" é pronunciado como "s".'
        ),
        (
            r'(â)',
            '"{match}" indica um "a" mais aberto, semelhante ao "á".'
        ),
        (
            r'(î)',
            '"{match}" soa como "i" normal.'
        ),
        (
            r'(ô)',
            '"{match}" soa como "ô" fechado.'
        ),
        (
            r'(û)',
            '"{match}" soa como um "u" mais fechado, lembrando o "u" francês puxado para os lábios arredondados.'
        ),
        (
            r'(pt)$',
            'No final, "{match}" não é pronunciado.'
        ),
        (
            r'^(ps)',
            'No início, "{match}" muitas vezes se reduz a "s". Ex: "psychologie" → "ssicologie".'
        ),
        (
            r'(mn)$',
            'Ao final, "{match}" muitas vezes simplifica o som, soando mais próximo de "m".'
        ),
        (
            r'(ieux)$',
            'Ao final, "{match}" soa como "iô" ou "iêu" curto.'
        ),
        (
            r'(amment)$',
            'Em advérbios, "{match}" soa como "amã".'
        ),
        (
            r'(emment)$',
            'Em advérbios, "{match}" soa também como "amã".'
        ),
        (
            r'(ti)(?=[aeiouy])',
            'Antes de vogal, "{match}" às vezes soa como "tsi".'
        ),
        (
            rf'(?<=[{all_vowels}])(si)(?=[{all_vowels}])',
            'Entre vogais, "{match}" pode soar como "zi".'
        ),
        (
            r'(ll)(?=[eiy])',
            '"{match}" pode soar como "lh" ou um "i" palatalizado, ex: "fille" → "fii".'
        )
    ]

    regex_pattern_matches = []

    for regex_pattern, explanation_template in regex_pattern_explanations:
        for match_object in re.finditer(regex_pattern, word):
            start_position, end_position = match_object.start(), match_object.end()
            matched_text = match_object.group(0)
            chosen_color = random.choice(COLOR_LIST)
            explanation = explanation_template.format(
                match=f'<span style="color:{chosen_color}">{matched_text}</span>'
            )
            regex_pattern_matches.append(
                (start_position, end_position, matched_text, explanation, chosen_color)
            )

    # Capturar caso de "e" + 2 (ou mais) consoantes (exemplo: "e" fechado)
    match_object_consonants = re.search(r'e' + all_consonants + '{2,}', word)
    if match_object_consonants:
        matched_text = match_object_consonants.group(0)
        chosen_color = random.choice(COLOR_LIST)
        explanation_text = (
            'Quando "e" é seguido de duas ou mais consoantes '
            f'(<span style="color:{chosen_color}">{matched_text}</span>), '
            'tende a ficar mais fechado, soando quase como "é".'
        )
        regex_pattern_matches.append((
            match_object_consonants.start(),
            match_object_consonants.end(),
            matched_text,
            explanation_text,
            chosen_color
        ))

    # Caso de "ph"
    if 'ph' in word:
        for match_object in re.finditer(r'(ph)', word):
            matched_text = match_object.group(0)
            chosen_color = random.choice(COLOR_LIST)
            explanation_text = (
                f'"<span style="color:{chosen_color}">{matched_text}</span>" '
                'soa como "f". Ex: "photo" → "fôto".'
            )
            regex_pattern_matches.append((
                match_object.start(),
                match_object.end(),
                matched_text,
                explanation_text,
                None  # Marcaremos cor mais tarde se preciso
            ))

    # Caso de "th"
    if 'th' in word:
        for match_object in re.finditer(r'(th)', word):
            matched_text = match_object.group(0)
            chosen_color = random.choice(COLOR_LIST)
            explanation_text = (
                f'"<span style="color:{chosen_color}">{matched_text}</span>" '
                'costuma ser pronunciado como "t".'
            )
            regex_pattern_matches.append((
                match_object.start(),
                match_object.end(),
                matched_text,
                explanation_text,
                None
            ))

    def overlaps_with_existing(start_position, end_position, existing_matches):
        """
        Verifica se o intervalo [start_position, end_position) se sobrepõe
        com algum intervalo já presente em existing_matches.
        """
        for (existing_start, existing_end, _, _, _) in existing_matches:
            # Se não for totalmente antes nem totalmente depois, há sobreposição
            if not (end_position <= existing_start or start_position >= existing_end):
                return True
        return False

    # Ordena e mescla as regras padrão com os casos especiais
    regex_pattern_matches.sort(key=lambda x: x[0])
    final_matches = found_matches[:]

    for match_tuple in regex_pattern_matches:
        start_position, end_position, matched_text, explanation, chosen_color = match_tuple
        if not overlaps_with_existing(start_position, end_position, final_matches):
            final_matches.append(match_tuple)

    # Se absolutamente nada foi destacado, retorne sem destaques
    if not final_matches:
        return {
            "word": word,
            "highlighted_word": word,
            "explanations": []
        }

    # Ordena pelo início para reconstruir a palavra com spans
    final_matches.sort(key=lambda x: x[0])

    result_string = ""
    previous_end = 0
    explanations_list = []

    for (start_position, end_position, matched_text, explanation, chosen_color) in final_matches:
        result_string += word[previous_end:start_position]

        if chosen_color:
            highlight_color = chosen_color
        else:
            highlight_color = random.choice(COLOR_LIST)

        # Adiciona o trecho destacado
        result_string += f'<span style="color:{highlight_color}">{word[start_position:end_position]}</span>'
        previous_end = end_position

        explanations_list.append(explanation)

    # Adiciona o restante do texto
    result_string += word[previous_end:]

    return {
        "word": word,
        "highlighted_word": result_string,
        "explanations": explanations_list
    }
