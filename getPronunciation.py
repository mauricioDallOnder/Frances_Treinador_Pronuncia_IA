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
        - 'word': a palavra original
        - 'highlighted_word': a palavra com trechos destacados (<span> coloridos)
        - 'explanations': lista de explicações para cada trecho destacado
    """

    front_vowels = 'iéeèêëïyæœ'
    all_vowels = 'aeiouéêèëíóôúãõœæy'
    all_consonants = 'bcdfgjklmnpqrstvwxzʃʒɲŋçh'

    found_matches = []
    lower_case_word = word.lower()

    def add_match_object(match_object, explanation_template):
        """
        Adiciona um trecho destacado (match_object) e a explicação
        formatada (explanation_template) na lista found_matches.
        """
        if not match_object:
            return

        start_position = match_object.start()
        end_position = match_object.end()
        matched_text = match_object.group(0)
        chosen_color = random.choice(COLOR_LIST)

        explanation_formatted = explanation_template.format(
            match=f'<span style="color:{chosen_color}; font-weight:bold">{matched_text}</span>'
        )
        found_matches.append((
            start_position, 
            end_position, 
            matched_text, 
            explanation_formatted, 
            chosen_color
        ))

    # -------------------------------------------------------------
    # Casos especiais (artigos, expressões fixas etc.)
    # -------------------------------------------------------------

    # Exemplo "j'ai"
    if lower_case_word == "j'ai":
        index_ai = word.find('ai')
        if index_ai != -1:
            substring_ai = word[index_ai:index_ai + 2]  # "ai"
            match_for_ai = re.search(r'^ai$', substring_ai)
            if match_for_ai:
                explanation_text = (
                    'Em "j\'ai", pronuncia-se aproximadamente "jê" '
                    '(verbo "avoir" no presente).'
                )
                add_match_object(match_for_ai, explanation_text)

   

    # Exemplo para palavras terminadas em "chats"
    if lower_case_word.endswith('chats'):
        search_s = re.search(r'(s)$', word)
        if search_s:
            add_match_object(
                search_s,
                'Em "chats", o "{match}" final é mudo, então "chats" soa como "chá".'
            )

    # Exemplo para "grand" ou "quand" (liaison do 'd')
    if re.search(r'(grand|quand)$', lower_case_word):
        match_d = re.search(r'(d)$', word)
        if match_d:
            add_match_object(
                match_d,
                'Na liaison, o "{match}" final soa como "t" antes de vogal. '
                'Exemplo: "grand arbre" → "gran_t arbre".'
            )

    # ------------------------------------------------------------
    # REGRA ESPECIAL PARA "j'en" / "n'en"
    # ------------------------------------------------------------
    # Se quisermos captar "j'en" ou "n'en" como uma só unidade,
    # que normalmente soa "jã" / "nã". Ex: "j'en ai" → "jã né"
    # ou "je n'en ai" → "je nã né".
    # Vamos usar a regex (?:j|n)'en\b para pegar "j'en", "n'en" no final de palavra.
    # Se quiser também pegar "j'en," (com vírgula), pode fazer algo como:
    # (?:j|n)'en(?=[\s\.,;!?]|$) -- mas aqui use \b que em geral funciona bem.
    special_en_pattern = r"(?:j|n)'en\b"
    match_special_en = re.search(special_en_pattern, lower_case_word)
    if match_special_en:
        explanation_text = (
            'No pronome "{match}", pronuncia-se algo como "jã" ou "nã". '
            'Exemplo: "j\'en ai" → "jã né". "je n\'en ai" → "je nã né".'
        )
        add_match_object(match_special_en, explanation_text)

    # ------------------------------------------------------------
    # Conjunto principal de padrões (dicas de condicional, futuro, etc.)
    # ------------------------------------------------------------
    regex_pattern_explanations_list = [
        #
       #
        # (E) Condicional / Imparfait
        #
        (
            r'(ais|ait|aient)$',
            'No condicional ou no imparfait, {match} soa como "é".'
        ),
        #
        # (F) Futuro simples (mais amplo)
        #
        (
            r'(erai|eras|era|erons|erez|eront|'
            r'irai|iras|ira|irons|irez|iront|'
            r'rai|ras|ra|rons|rez|ront)$',
            'No futuro simples, {match} soa como "rê" (ex.: "erê", "irê").'
        ),
        #
        # (G) e ou es no final, seguido de pontuação, espaço ou fim
        #

        (
            r'\bh(?:aspiré)?\b',
            'O {match} aspirado não se liga à vogal seguinte, criando uma pequena pausa. '
            'Exemplo: "le héros" → não há liaison em "h aspiré".'
        ),
        #
        # (B) Artigos 'le' e 'les'
        #
        (
            r'\ble\b',
            'No artigo {match}, o "e" é muito curto, tipo "luh" (não "lê").'
        ),
        (
            r'\bles\b',
            'No artigo {match}, soa como "lê" (diferente de "le" = "luh").'
        ),
         #
        # (C) "gn" => "nh"
        #
        (
            r'gn',
            'A sequência {match} soa como "nh" em português.'
        ),
        #
        # (D) Pronome "j'en" / "n'en" (com espaço/pontuação ou fim após)
        #
        (
            r"(?:j|n)'en(?=[\s\.,;!?]|$)",
            'O pronome {match} pronuncia-se algo como "jã" ou "nã". Ex.: "j\'en ai" → "jã né".'
        ),

        #
        # 3) Regras genéricas de nasalização, vogais, etc.
        #
        (
            r'(am|em|an|en)(?=[bdfgjklpqrstvwxzʃʒɲŋç])',
            'A sequência {match} indica um som nasal tipo "ãn".'
        ),
        (
            r'(in|im|yn|ym|ein|ain|ien|aim)(?=[bdfgjklpqrstvwxzʃʒɲŋç])',
            'A sequência {match} representa um som nasal "iñ" ou "iãn".'
        ),
        (
            r'(on|om)(?=[bdfgjklpqrstvwxzʃʒɲŋç])',
            'A sequência {match} soa como "õ".'
        ),
        (
            r'(un|um)(?=[bdfgjklpqrstvwxzʃʒɲŋç])',
            'A sequência {match} dá um som nasal "œ̃", parecido com "ãn" mas arredondado.'
        ),
        (
            r'(au|aux|eau|eaux)',
            'A sequência {match} normalmente soa como "ô".'
        ),
        (
            r'(oy)',
            '{match} soa como "uai".'
        ),
        (
            r'(x)(?=[' + all_consonants + '])',
            '"{match}" antes de consoante soa "ks".'
        ),
        (
            r'(y)(?=[' + all_vowels + '])',
            '"{match}" antes de vogal soa como "i" deslizado.'
        ),
        (
            r'(c)(?=[' + front_vowels + '])',
            '"{match}" antes de vogal frontal soa como "s".'
        ),
        (
            r'(ch)',
            '"{match}" soa como "x" (xarope).'
        ),
        (
            r'(j|g)(?=[eiy])',
            '"{match}" soa como "j" de "jogar" antes de e, i ou y.'
        ),
        (
            r'(gn)',
            '"{match}" soa como "nh".'
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
            '"{match}" soa como "eu" fechado (entre "e" e "u").'
        ),
        (
            r'(é)',
            '"{match}" soa como "ê" fechado.'
        ),

        #
        # 4) A regra genérica (è|ê|ai|ei)
        #
        (
            r'(è|ê|ai|ei)',
            'A combinação {match} soa como "é" aberto.'
        ),

        (
            r'(er)$',
            'No final, "{match}" soa como "ê" (ex.: "parler" → "parlê").'
        ),
        (
            r'(qu)',
            '"{match}" pronuncia-se "k".'
        ),
        (
            r'(h)',
            '"{match}" geralmente é mudo (salvo "h aspiré").'
        ),
        (
            r'(ge)$',
            'No final, "{match}" soa como "je" (j de "jogar").'
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
            '"{match}" soa algo como "õe", um híbrido de "e" e "u" nasal.'
        ),
        (
            r'(œil)',
            '"{match}" soa como "ói" curto, com lábios arredondados.'
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
            '"{match}" no final soa como "iê".'
        ),
        (
            r'(iez)$',
            '"{match}" soa como "iê". Ex: "disiez" → "disiê".'
        ),
        (
            r'(oin)',
            'A sequência {match} soa como "uã" nasalizado.'
        ),
        (
            r'(ui)',
            'A sequência {match} soa como "üi" (semelhante a "wi").'
        ),
        (
            r'(œu)',
            '"{match}" soa entre "eu" e "éu" com lábios arredondados.'
        ),
        (
            r'(œ)',
            '"{match}" soa como "é" com lábios arredondados.'
        ),
        (
            r'(cc)(?=[eiy])',
            '"{match}" soa "ks".'
        ),
        (
            r'(ç)',
            '"{match}" é pronunciado como "s".'
        ),
        (
            r'(â)',
            '"{match}" indica um "a" mais aberto (á).'
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
            '"{match}" soa como "u" mais fechado.'
        ),
        (
            r'(pt)$',
            '"{match}" não se pronuncia no final.'
        ),
        (
            r'^(ps)',
            'No início, "{match}" vira "s". Exemplo: "psychologie" → "ssicologie".'
        ),
        (
            r'(mn)$',
            'Ao final, "{match}" simplifica, soando como "m".'
        ),
        (
            r'(ieux)$',
            '"{match}" soa como "iô" ou "iêu" curto.'
        ),
        (
            r'(amment)$',
            '"{match}" soa como "amã" em advérbios (ex.: "franchement").'
        ),
        (
            r'(emment)$',
            '"{match}" soa como "amã" em advérbios.'
        ),
        (
            r'(ti)(?=[aeiouy])',
            'Antes de vogal, "{match}" pode soar "tsi".'
        ),
        (
            rf'(?<=[{all_vowels}])(si)(?=[{all_vowels}])',
            'Entre vogais, "{match}" pode soar como "zi".'
        ),
        (
            r'(ll)(?=[eiy])',
            '"{match}" soa como "lh" ou "i" palatal. Exemplo: "fille" → "fii".'
        ),
    ]

    regex_pattern_matches_list = []

    # Aplica as regex definidas acima
    for (regex_pattern, explanation_template) in regex_pattern_explanations_list:
        for match_object in re.finditer(regex_pattern, word):
            start_position = match_object.start()
            end_position = match_object.end()
            matched_text = match_object.group(0)
            chosen_color = random.choice(COLOR_LIST)

            explanation_formatted = explanation_template.format(
                match=f'<span style="color:{chosen_color}">{matched_text}</span>'
            )
            regex_pattern_matches_list.append((
                start_position, 
                end_position, 
                matched_text, 
                explanation_formatted, 
                chosen_color
            ))

    # Captura "e" + 2 ou mais consoantes consecutivas
    match_object_e_consonants = re.search(r'e' + all_consonants + '{2,}', word)
    if match_object_e_consonants:
        matched_text = match_object_e_consonants.group(0)
        chosen_color = random.choice(COLOR_LIST)
        explanation_text = (
            f'Quando "e" é seguido de 2 ou mais consoantes '
            f'(<span style="color:{chosen_color}">{matched_text}</span>), '
            'tende a ficar mais fechado, quase "é".'
        )
        regex_pattern_matches_list.append((
            match_object_e_consonants.start(),
            match_object_e_consonants.end(),
            matched_text,
            explanation_text,
            chosen_color
        ))

    # Captura "ph"
    if 'ph' in word:
        for match_object in re.finditer(r'(ph)', word):
            matched_text = match_object.group(0)
            chosen_color = random.choice(COLOR_LIST)
            explanation_text = (
                f'"<span style="color:{chosen_color}">{matched_text}</span>" '
                'soa como "f". Exemplo: "photo" → "fôto".'
            )
            regex_pattern_matches_list.append((
                match_object.start(),
                match_object.end(),
                matched_text,
                explanation_text,
                chosen_color
            ))

    # Captura "th"
    if 'th' in word:
        for match_object in re.finditer(r'(th)', word):
            matched_text = match_object.group(0)
            chosen_color = random.choice(COLOR_LIST)
            explanation_text = (
                f'"<span style="color:{chosen_color}">{matched_text}</span>" '
                'soa como "t".'
            )
            regex_pattern_matches_list.append((
                match_object.start(),
                match_object.end(),
                matched_text,
                explanation_text,
                chosen_color
            ))

    def overlaps_with_existing_matches(
        candidate_start, candidate_end, existing_matches_list
    ):
        """
        Retorna True se [candidate_start, candidate_end) se sobrepõe
        a algum intervalo em existing_matches_list.
        """
        for (existing_start, existing_end, *_rest) in existing_matches_list:
            # Se não for totalmente antes ou totalmente depois, há sobreposição
            if not (candidate_end <= existing_start or candidate_start >= existing_end):
                return True
        return False

    # 1) Ordenar priorizando: posição inicial ASC e, em caso de sobreposição,
    #    prioriza o match mais longo (Longest Match First).
    #    Assim, se "erai" e "ai" competirem, "erai" prevalece.
    regex_pattern_matches_list.sort(
        key=lambda x: (x[0], -(x[1] - x[0]))
    )

    final_matches_list = []

    # 2) Selecionar os matches sem sobreposição,
    #    dando prioridade para o match maior quando houver conflito.
    for match_tuple in regex_pattern_matches_list:
        (start_position, end_position, matched_text, 
         explanation_formatted, chosen_color) = match_tuple

        if not overlaps_with_existing_matches(start_position, end_position, final_matches_list):
            final_matches_list.append(match_tuple)

    # Se não encontrou nenhum trecho a destacar, retorne sem destaques
    if not final_matches_list:
        return {
            "word": word,
            "highlighted_word": word,
            "explanations": []
        }

    # 3) Ordenar novamente apenas pela posição inicial, 
    #    para reconstruir o texto em ordem (da esquerda para a direita).
    final_matches_list.sort(key=lambda x: x[0])

    result_string = ""
    previous_end_position = 0
    explanations_list = []

    for (
        start_position, 
        end_position, 
        matched_text, 
        explanation_formatted, 
        chosen_color
    ) in final_matches_list:

        # Adiciona o pedaço do texto antes do match atual
        result_string += word[previous_end_position:start_position]

        highlight_color = chosen_color or random.choice(COLOR_LIST)
        # Destaca o trecho correspondente
        result_string += (
            f'<span style="color:{highlight_color}">'
            f'{word[start_position:end_position]}'
            '</span>'
        )

        explanations_list.append(explanation_formatted)
        previous_end_position = end_position

    # Adiciona o restante do texto depois do último match
    result_string += word[previous_end_position:]

    return {
        "word": word,
        "highlighted_word": result_string,
        "explanations": explanations_list
    }
