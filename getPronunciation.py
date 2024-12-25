# Lista de cores para destaque
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
        - 'highlighted_word': palavra com trechos destacados
        - 'explanations': lista de explicações para cada destaque
    """
    front_vowels = 'iéeèêëïyæœ'
    vowels = 'aeiouéêèëíóôúãõœæy'
    consonants = 'bcdfgjklmnpqrstvwxzʃʒɲŋçh'

    matches_found = []
    lower_word = word.lower()

    def add_match_object(m, explanation_template):
        start, end = m.start(), m.end()
        matched_text = m.group(0)
        # Seleciona uma cor aleatória para este match
        color = random.choice(COLOR_LIST)
        # Formata a explicação com o match destacado
        expl = explanation_template.format(
            match=f'<span style="color:{color}; font-weight:bold">{matched_text}</span>'
        )
        matches_found.append((start, end, matched_text, expl, color))

    # Casos especiais ---------------------------------------------------------
    if lower_word == "j'ai":
        idx = word.find('ai')
        if idx != -1:
            start, end = idx, idx+2
            matched_text = word[start:end]
            expl = 'Em "j\'ai", a pronúncia fica aproximadamente como "jê".'
            add_match_object(re.match(r'ai', word[idx:]), 'Em "j\'ai", a pronúncia fica aproximadamente como "jê".')
    
    if lower_word == 'le':
        if word.endswith('e'):
            start = len(word)-1
            end = len(word)
            matched_text = word[start:end]
            expl = 'No artigo "le", o "e" é muito curto, parecido com "luh" rápido, não "lê". Ex: "le chat" → "luh chá".'
            add_match_object(re.match(r'e$', word[start:end]), 'No artigo "le", o "e" é muito curto, parecido com "luh" rápido, não "lê". Ex: "le chat" → "luh chá".')
    
    if lower_word == 'les':
        if word.endswith('es'):
            start = len(word)-2
            end = len(word)
            matched_text = word[start:end]
            expl = 'No artigo "les", soa como "lê", diferente de "le" (lu). Ex: "les chats" → "lê chá".'
            add_match_object(re.match(r'es$', word[start:end]), 'No artigo "les", soa como "lê", diferente de "le" (lu). Ex: "les chats" → "lê chá".')
    
    if lower_word.endswith('chats'):
        m = re.search(r'(s)$', word)
        if m:
            add_match_object(m, 'Em "chats", o "{match}" final é mudo, então "chats" soa como "chá".')
    
    if re.search(r'(grand|quand)$', lower_word):
        m = re.search(r'(d)$', word)
        if m:
            add_match_object(m, 'Na ligação (liaison), o "{match}" final soa como "t" antes de vogal. Ex: "grand arbre" → "gran_t arbre".')
    
    # Padrões genéricos sem AFI, usando comparações com o português:
    patterns = [
        (r'(am|em|an|en)(?=[bdfgjklpqrstvwxzʃʒɲŋç])', 
         'A sequência {match} indica um som nasal parecido com \'ãn\'. Ex: {match} ~ \'ãn\'.'),
        (r'(in|im|yn|ym|ein|ain|ien|aim)(?=[bdfgjklpqrstvwxzʃʒɲŋç])',
         'A sequência {match} representa um som nasal tipo \'iñ\' ou \'iãn\', lembrando \'im\' nasalizado.'),
        (r'(on|om)(?=[bdfgjklpqrstvwxzʃʒɲŋç])', 
         'A sequência {match} dá um som nasal parecido com \'õ\', como em \'põe\'.'),
        (r'(un|um)(?=[bdfgjklpqrstvwxzʃʒɲŋç])', 
         'A sequência {match} produz um som nasal parecido com \'œ̃\', próximo de \'ãn\' com os lábios arredondados. Pense em {match} como \'ãn\' mais fechado.'),
        (r'(au|aux|eau|eaux)', 
         'A sequência {match} é pronunciada aproximadamente como \'ô\'.'),
        (r'(oy)',
         '{match} soa como \'uai\', semelhante ao mineiro \'uai\'.'),
        (r'(x)(?=[' + consonants + '])', 
         '"{match}" antes de consoante soa como "ks".'),
        (r'(y)(?=[' + vowels + '])', 
         '"{match}" antes de vogal soa como o "i" deslizado, tipo "ia" → "iá".'),
        (r'(c)(?=[' + front_vowels + '])', 
         '"{match}" antes de vogal frontal soa como "s".'),
        (r'(ch)', 
         '"{match}" soa como "x" em "xarope".'),
        (r'(j|g)(?=[eiy])', 
         '"{match}" soa como o "j" de "jogar" antes de e, i ou y.'),
        (r'(gn)', 
         '"{match}" soa como "nh" em português.'),
        (r'(e|es)$', 
         'No final da palavra, "{match}" geralmente não é pronunciado.'),
        (r'(oi)', 
         'A sequência {match} soa como "uá".'),
        (r'(ou)', 
         'A sequência {match} soa como "u" fechado.'),
        (r'(ille)', 
         '"{match}" soa como "iê".'),
        (r'(eu)', 
         '"{match}" tem um som semelhante a "eu" fechado, algo entre "e" e "u".'),
        (r'(é)', 
         '"{match}" soa como "ê" mais fechado.'),
        (r'(è|ê|ai|ei)', 
         'A combinação {match} soa como "é" aberto.'),
        (r'(er)$', 
         'No final da palavra, "{match}" soa como "ê" fechado.'),
        (r'(qu)', 
         '"{match}" é pronunciado como "k".'),
        (r'(ais|ait|aient)$', 
         'Ao final, "{match}" costuma soar como "é".'),
        (r'(h)', 
         '"{match}" geralmente é mudo. H aspirado não se liga à vogal seguinte, mas não altera muito o som.'),
        (r'(ge)$', 
         'No final, "{match}" costuma soar como "je" (o j de "jogar").'),
        (r'(ail)', 
         '"{match}" soa como "ai" (ái).'),
        (r'(eil)', 
         '"{match}" soa como "ei" fechado.'),
        (r'(euil)', 
         '"{match}" soa algo como "õe", um som entre "e" e "u" nasalizado.'),
        (r'(œil)', 
         '"{match}" soa semelhante a "ói" curto, com os lábios arredondados.'),
        (r'(ien)', 
         '"{match}" soa como "iã" nasalizado.'),
        (r'(ion)', 
         '"{match}" soa como "iõ" nasalizado.'),
        (r'(tion)$', 
         'No final, "{match}" soa como "siõ" (s + iõ nasal).'),
        (r'(ier)$', 
         'No final, "{match}" soa como "iê".'),
        (r'(iez)$', 
         'No final da forma verbal, "{match}" soa como "iê".'),
        (r'(oin)', 
         'A sequência {match} soa como "uã" nasalizado.'),
        (r'(ui)', 
         'A sequência {match} soa como "üi", algo como "wi" em inglês.'),
        (r'(œu)', 
         '"{match}" soa entre "eu" e "éu" com lábios arredondados.'),
        (r'(œ)', 
         '"{match}" soa como um "é" com lábios arredondados, algo entre "é" e "eu".'),
        (r'(cc)(?=[eiy])',
         '"{match}" pode soar como "ks".'),
        (r'(ç)', 
         '"{match}" é pronunciado como "s".'),
        (r'(â)', 
         '"{match}" indica um "a" mais aberto, semelhante ao "á".'),
        (r'(î)', 
         '"{match}" soa como "i" normal.'),
        (r'(ô)', 
         '"{match}" soa como "ô" fechado.'),
        (r'(û)', 
         '"{match}" soa como um "u" mais fechado, lembrando o "u" francês puxado para os lábios arredondados.'),
        (r'(pt)$', 
         'No final, "{match}" não é pronunciado.'),
        (r'^(ps)', 
         'No início, "{match}" muitas vezes se reduz a "s". Ex: "psychologie" → "ssicologie".'),
        (r'(mn)$', 
         'Ao final, "{match}" muitas vezes simplifica o som, soando mais próximo de "m".'),
        (r'(ieux)$', 
         'Ao final, "{match}" soa como "iô" ou "iêu" curto.'),
        (r'(amment)$',
         'Em advérbios, "{match}" soa como "amã".'),
        (r'(emment)$',
         'Em advérbios, "{match}" soa também como "amã".'),
        (r'(ti)(?=[aeiouy])', 
         'Antes de vogal, "{match}" às vezes soa como "tsi".'),
        (r'(?<=[' + vowels + '])(si)(?=[' + vowels + '])', 
         'Entre vogais, "{match}" pode soar como "zi".'),
        (r'(ll)(?=[eiy])',
         '"{match}" pode soar como "lh" ou um "i" palatalizado, ex: "fille" → "fii".')
    ]

    pattern_matches = []
    for pattern, explanation in patterns:
        for m in re.finditer(pattern, word):
            start, end = m.start(), m.end()
            matched_text = m.group(0)
            # Seleciona uma cor aleatória para este match
            color = random.choice(COLOR_LIST)
            # Formata a explicação com o match destacado
            expl = explanation.format(
                match=f'<span style="color:{color}">{matched_text}</span>'
            )
            pattern_matches.append((start, end, matched_text, expl, color))

    # 'e' + 2 ou mais consoantes
    m = re.search(r'e' + consonants + '{2,}', word)
    if m:
        matched_text = m.group(0)
        expl = f'Quando "e" é seguido de duas ou mais consoantes (<span style="color:{"red"}">{matched_text}</span>), tende a ficar mais fechado, soando quase como "é".'
        # Para consistência, selecione uma cor aleatória
        color = random.choice(COLOR_LIST)
        expl = f'Quando "e" é seguido de duas ou mais consoantes (<span style="color:{color}">{matched_text}</span>), tende a ficar mais fechado, soando quase como "é".'
        pattern_matches.append((m.start(), m.end(), matched_text, expl, color))

    # ph
    if 'ph' in word:
        for m in re.finditer(r'(ph)', word):
            matched_text = m.group(0)
            expl = f'"<span style="color:{random.choice(COLOR_LIST)}">{matched_text}</span>" soa como "f". Ex: "photo" → "fôto".'
            pattern_matches.append((m.start(), m.end(), matched_text, expl, None))

    # th
    if 'th' in word:
        for m in re.finditer(r'(th)', word):
            matched_text = m.group(0)
            expl = f'"<span style="color:{random.choice(COLOR_LIST)}">{matched_text}</span>" costuma ser pronunciado como "t".'
            pattern_matches.append((m.start(), m.end(), matched_text, expl, None))

    def overlaps_with_existing(start, end, existing_matches):
        for (s, e, _, _, _) in existing_matches:
            if not (end <= s or start >= e):
                return True
        return False

    # Filtra matches que não se sobrepõem
    pattern_matches.sort(key=lambda x: x[0])
    final_matches = matches_found[:]

    for match in pattern_matches:
        start, end, matched_text, expl, color = match
        if not overlaps_with_existing(start, end, final_matches):
            final_matches.append(match)

    if not final_matches:
        return {
            "word": word,
            "highlighted_word": word,
            "explanations": []
        }

    final_matches.sort(key=lambda x: x[0])

    result_str = ""
    prev_end = 0
    explanations = []
    for (start, end, matched_text, expl, color) in final_matches:
        result_str += word[prev_end:start]
        
        # Verifica se uma cor foi fornecida, caso contrário, seleciona uma
        if color:
            highlight_color = color
        else:
            highlight_color = random.choice(COLOR_LIST)
        
        result_str += f'<span style="color:{highlight_color}">{word[start:end]}</span>'
        prev_end = end
        explanations.append(expl)
    result_str += word[prev_end:]

    return {
        "word": word,
        "highlighted_word": result_str,
        "explanations": explanations
    }