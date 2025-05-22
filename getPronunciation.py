import re
import random
import html
from typing import List, Tuple, Dict, Pattern, Optional

# -------------------------------------------------------------
# Variáveis de apoio
# -------------------------------------------------------------
front_vowels = 'iéeèêëïyæœ'
all_vowels = 'aeiouéêèëíìîóòôúùûãõœæyâêîôû'
all_consonants = 'bcdfgjklmnpqrstvwxzʃʒɲŋçh'

# -------------------------------------------------------------
# Paleta de cores organizada por categorias fonéticas - Cores mais vibrantes e contrastantes
# -------------------------------------------------------------
COLOR_CATEGORIES = {
    'vowels': ['#FF0000', '#FF3333', '#FF6666', '#FF9999', '#FF0066', '#FF3399', '#FF66CC'],  # vermelhos mais vibrantes
    'consonants': ['#0000FF', '#3333FF', '#6666FF', '#9999FF', '#0066FF', '#3399FF', '#66CCFF'],  # azuis mais vibrantes
    'nasals': ['#9900CC', '#CC00FF', '#CC33FF', '#CC66FF', '#CC99FF', '#9933CC', '#6600CC'],  # roxos mais vibrantes
    'diphthongs': ['#00CC00', '#33FF33', '#66FF66', '#99FF99', '#00FF66', '#33FF99', '#66FFCC'],  # verdes mais vibrantes
    'silent': ['#999999', '#AAAAAA', '#BBBBBB', '#CCCCCC', '#888888', '#777777', '#666666'],  # cinzas mais distintos
    'special': ['#FF9900', '#FFCC00', '#FFFF00', '#FF00FF', '#00FFFF', '#FF6600', '#00FF00', '#FF0099']  # cores especiais vibrantes
}

# Função para obter cor por categoria
def get_color(category: str) -> str:
    if category in COLOR_CATEGORIES:
        return random.choice(COLOR_CATEGORIES[category])
    return random.choice(COLOR_CATEGORIES['special'])

# -------------------------------------------------------------
# Estilos CSS para melhorar a visualização
# -------------------------------------------------------------
CSS_STYLES = """
<style>
.pronunciation-hint {
    margin-bottom: 12px;
    line-height: 1.6;
    font-size: 16px;
}
.highlight {
    font-weight: bold;
    padding: 2px 4px;
    border-radius: 3px;
    margin: 0 2px;
    display: inline-block;
    box-shadow: 0 1px 2px rgba(0,0,0,0.2);
}
.vowel-highlight { background-color: rgba(255,0,0,0.1); }
.consonant-highlight { background-color: rgba(0,0,255,0.1); }
.nasal-highlight { background-color: rgba(153,0,204,0.1); }
.diphthong-highlight { background-color: rgba(0,204,0,0.1); }
.silent-highlight { background-color: rgba(153,153,153,0.1); }
.special-highlight { background-color: rgba(255,153,0,0.1); }
.pronunciation-list {
    list-style-type: none;
    padding-left: 0;
}
.pronunciation-list li {
    padding: 8px 0;
    border-bottom: 1px solid #eee;
}
.pronunciation-list li:last-child {
    border-bottom: none;
}
.pronunciation-category {
    font-weight: bold;
    margin-top: 16px;
    margin-bottom: 8px;
    font-size: 18px;
}
</style>
"""

# -------------------------------------------------------------
# Padrões gerais pré‑compilados e suas explicações
# -------------------------------------------------------------
GENERAL_PATTERNS: List[Tuple[Pattern, str, str]] = [
    # Terminações verbais
    (re.compile(r'(ais|ait|aient)$'),
     'No condicional/imparfait, {match} soa como "é".',
     'vowels'),

    (re.compile(r'(erai|eras|era|erons|erez|eront|'
                r'irai|iras|ira|irons|irez|iront|'
                r'rai|ras|ra|rons|rez|ront)$'),
     'No futuro simples, {match} soa como "rê".',
     'vowels'),

    # Artigos e H aspirado
    (re.compile(r'\bh(?:aspiré)?\b'),
     'O {match} aspirado não cria liaison e bloqueia elisão.',
     'consonants'),

    (re.compile(r'\ble\b'),
     'No artigo {match}, o "e" é muito curto (tipo "luh").',
     'vowels'),
    (re.compile(r'\bles\b'),
     'No artigo {match}, soa como "lê".',
     'vowels'),

    # Grupos consonantais específicos
    (re.compile(r'(gn)'),
     '"{match}" soa como "nh" (como em "banho").',
     'consonants'),

    # Pronomes e contrações
    (re.compile(r"(?:j|n)'en(?=[\s\.,;!?]|$)"),
     'No pronome {match}, soa como "jã"/"nã".',
     'nasals'),

    # Nasalizações
    (re.compile(r'(am|em|an|en)(?=[bdfgjklpqrstvwxzʃʒɲŋç]|$)'),
     'A sequência {match} indica som nasal "ã".',
     'nasals'),
    (re.compile(r'(in|im|yn|ym|ein|ain|ien|aim)(?=[bdfgjklpqrstvwxzʃʒɲŋç]|$)'),
     'A sequência {match} representa som nasal "ẽ".',
     'nasals'),
    (re.compile(r'(on|om)(?=[bdfgjklpqrstvwxzʃʒɲŋç]|$)'),
     'A sequência {match} soa como "õ".',
     'nasals'),
    (re.compile(r'(un|um)(?=[bdfgjklpqrstvwxzʃʒɲŋç]|$)'),
     'A sequência {match} dá som nasal "ẽ" (com lábios arredondados).',
     'nasals'),

    # Combinações de vogais
    (re.compile(r'(au|aux|eau|eaux)'),
     'A sequência {match} soa como "ô" fechado.',
     'diphthongs'),
    (re.compile(r'(oy)'),
     '{match} soa como "uá" + "i" (uai).',
     'diphthongs'),

    # Consoantes especiais
    (re.compile(r'(x)(?=[' + all_consonants + '])'),
     '"{match}" antes de consoante soa "ks".',
     'consonants'),
    (re.compile(r'(y)(?=[' + all_vowels + '])'),
     '"{match}" antes de vogal soa como "i" deslizado (semivogal).',
     'vowels'),
    (re.compile(r'(c)(?=[' + front_vowels + '])'),
     '"{match}" antes de vogal frontal (e, i, y) soa como "s".',
     'consonants'),
    (re.compile(r'(c)(?=[aouàâôû])'),
     '"{match}" antes de a, o, u soa como "k".',
     'consonants'),
    (re.compile(r'(ch)'),
     '"{match}" soa como "x" (como em "xarope").',
     'consonants'),
    (re.compile(r'(j)'),
     '"{match}" soa como "j" (como em "já").',
     'consonants'),
    (re.compile(r'(g)(?=[eiy])'),
     '"{match}" antes de e, i ou y soa como "j" (como em "já").',
     'consonants'),
    (re.compile(r'(g)(?=[aouàâôû])'),
     '"{match}" antes de a, o, u soa como "g" (como em "gato").',
     'consonants'),

    # Vogais/Finais silenciosos ou modificados
    (re.compile(r'(e|es)(?=[\s\.,;!?]|$)'),
     'No final, {match} geralmente não é pronunciado.',
     'silent'),
    (re.compile(r'(oi)'),
     'A sequência {match} soa como "uá".',
     'diphthongs'),
    (re.compile(r'(ou)'),
     'A sequência {match} soa como "u" fechado (como em "uva").',
     'diphthongs'),
    (re.compile(r'(ille)'),
     '"{match}" soa como "i" + "lh" + "e" mudo (aproximadamente "ilh").',
     'consonants'),
    (re.compile(r'(eu)'),
     '"{match}" soa como "ö" (vogal fechada, arredondada, entre "ê" e "ô").',
     'diphthongs'),
    (re.compile(r'(é)'),
     '"{match}" soa como "é" fechado.',
     'vowels'),
    (re.compile(r'(è|ê|ai|ei)'),
     'A combinação {match} soa como "é" aberto.',
     'vowels'),
    (re.compile(r'(er)$'),
     'No final, {match} soa como "ê".',
     'vowels'),
    (re.compile(r'(qu)'),
     '"{match}" pronuncia-se "k".',
     'consonants'),
    (re.compile(r'(h)'),
     '"{match}" geralmente é mudo, exceto em "h aspirado".',
     'silent'),
    (re.compile(r'(ge)$'),
     'No final, {match} soa como "je".',
     'consonants'),
    (re.compile(r'(ail)'),
     '"{match}" soa como "a" + "lh" (aproximadamente "alh").',
     'diphthongs'),
    (re.compile(r'(eil)'),
     '"{match}" soa como "é" + "lh" (aproximadamente "élh").',
     'diphthongs'),
    (re.compile(r'(euil)'),
     '"{match}" soa como "ö" + "lh" (aproximadamente "ölh").',
     'diphthongs'),
    (re.compile(r'(œil)'),
     '"{match}" soa como "ö" + "lh" (aproximadamente "ölh").',
     'diphthongs'),
    (re.compile(r'(ien)'),
     '"{match}" soa como "iẽ" (nasalizado).',
     'nasals'),
    (re.compile(r'(ion)'),
     '"{match}" soa como "iõ" (nasalizado).',
     'nasals'),
    (re.compile(r'(tion)$'),
     'No final, {match} soa como "siõ".',
     'nasals'),
    (re.compile(r'(ier)$'),
     '{match} no final soa como "iê".',
     'vowels'),
    (re.compile(r'(iez)$'),
     '{match} soa como "iê".',
     'vowels'),
    (re.compile(r'(oin)'),
     'A sequência {match} soa como "uẽ" (nasalizado).',
     'nasals'),
    (re.compile(r'(ui)'),
     'A sequência {match} soa como "üi" (u francês + i).',
     'diphthongs'),
    (re.compile(r'(œu)'),
     '{match} soa como "ö" (vogal fechada, arredondada).',
     'diphthongs'),
    (re.compile(r'(œ)'),
     '{match} soa como "ö" (vogal aberta, arredondada).',
     'vowels'),
    (re.compile(r'(cc)(?=[eiy])'),
     '{match} soa "ks" antes de e, i, y.',
     'consonants'),
    (re.compile(r'(ç)'),
     '{match} é pronunciado como "s".',
     'consonants'),
    (re.compile(r'(â)'),
     '{match} indica um "a" mais aberto e longo.',
     'vowels'),
    (re.compile(r'(î)'),
     '{match} soa como "i" longo.',
     'vowels'),
    (re.compile(r'(ô)'),
     '{match} soa como "ô" fechado e longo.',
     'vowels'),
    (re.compile(r'(û)'),
     '{match} soa como "u" mais fechado e longo.',
     'vowels'),
    (re.compile(r'(pt)$'),
     '{match} no final geralmente não se pronuncia.',
     'silent'),
    (re.compile(r'^(ps)'),
     'No início, {match} simplifica para "s".',
     'consonants'),
    (re.compile(r'(mn)$'),
     '{match} final simplifica para "m" ou "n".',
     'consonants'),
    (re.compile(r'(ieux)$'),
     '{match} soa como "iö".',
     'diphthongs'),
    (re.compile(r'(amment)$'),
     '{match} soa como "amã".',
     'nasals'),
    (re.compile(r'(emment)$'),
     '{match} soa como "amã".',
     'nasals'),
    (re.compile(r'(ti)(?=[aeiouy])'),
     '{match} pode soar "tsi" ou "ti" dependendo da palavra.',
     'consonants'),
    (re.compile(r'(?<=[aeiouœæyâêîôû])(si)(?=[aeiouœæyâêîôû])'),
     '{match} entre vogais pode soar "zi".',
     'consonants'),
    (re.compile(r'(ll)(?=[eiy])'),
     '{match} pode soar "l" ou "lh" dependendo da palavra.',
     'consonants'),
    (re.compile(r'(ph)'),
     '{match} soa como "f".',
     'consonants'),
    (re.compile(r'(th)'),
     '{match} soa como "t".',
     'consonants'),
    (re.compile(r'e(?=[' + all_consonants + ']{2,})'),
     'Quando "e" é seguido de 2+ consoantes, tende a ficar mais fechado.',
     'vowels'),
    
    # Novas regras adicionadas
    (re.compile(r'(s)(?=[aeiouéèêëàâîïôûœæy])'),
     '{match} entre vogais geralmente soa como "z".',
     'consonants'),
    (re.compile(r'(s)$'),
     '{match} no final geralmente é mudo, exceto em liaison.',
     'silent'),
    (re.compile(r'(ent)$'),
     'A terminação verbal {match} é muda na 3ª pessoa do plural.',
     'silent'),
    (re.compile(r'(ez)$'),
     'A terminação {match} soa como "ê".',
     'vowels'),
    (re.compile(r'(et)$'),
     'No final, {match} soa como "ê".',
     'vowels'),
    (re.compile(r'(ê)'),
     '{match} soa como "ê" fechado.',
     'vowels'),
    (re.compile(r'(ë)'),
     '{match} (trema) indica que a vogal deve ser pronunciada separadamente.',
     'vowels'),
    (re.compile(r'(ï)'),
     '{match} (trema) indica que a vogal deve ser pronunciada separadamente.',
     'vowels'),
    (re.compile(r'(ü)'),
     '{match} (trema) indica que a vogal deve ser pronunciada separadamente.',
     'vowels'),
    (re.compile(r'(ille)$'),
     'No final, {match} soa como "i" + "lh" + "e" mudo (aproximadamente "ilh").',
     'consonants'),
    (re.compile(r'(ail|eil|euil|œil)$'),
     'No final, {match} tem som de "lh".',
     'consonants'),
    (re.compile(r'(gu)(?=[eiy])'),
     '{match} antes de e, i, y soa como "g" (como em "guerra").',
     'consonants'),
    (re.compile(r'(ain|ein|in)$'),
     'No final, {match} soa como "ẽ" (nasalizado).',
     'nasals'),
    (re.compile(r'(oin)$'),
     'No final, {match} soa como "uẽ" (nasalizado).',
     'nasals'),
    (re.compile(r'(en|an|em|am)$'),
     'No final, {match} soa como "ã" (nasalizado).',
     'nasals'),
    (re.compile(r'(on|om)$'),
     'No final, {match} soa como "õ" (nasalizado).',
     'nasals'),
    (re.compile(r'(un|um)$'),
     'No final, {match} soa como "ẽ" com lábios arredondados (nasalizado).',
     'nasals'),
    (re.compile(r'(ien)$'),
     'No final, {match} soa como "iẽ" (nasalizado).',
     'nasals'),
    (re.compile(r'(x)$'),
     '{match} no final geralmente é mudo, exceto em liaison.',
     'silent'),
    (re.compile(r'(z)$'),
     '{match} no final geralmente é mudo, exceto em liaison.',
     'silent'),
    (re.compile(r'(d)$'),
     '{match} no final geralmente é mudo, exceto em liaison.',
     'silent'),
    (re.compile(r'(t)$'),
     '{match} no final geralmente é mudo, exceto em liaison.',
     'silent'),
    (re.compile(r'(p)$'),
     '{match} no final geralmente é mudo.',
     'silent'),
    (re.compile(r'(g)$'),
     '{match} no final geralmente é mudo.',
     'silent'),
    (re.compile(r'(c)$'),
     '{match} no final geralmente é mudo, exceto em algumas palavras.',
     'silent'),
    (re.compile(r'(r)$'),
     '{match} no final pode ser pronunciado levemente ou omitido, dependendo do dialeto.',
     'consonants'),
    (re.compile(r'(l)$'),
     '{match} no final geralmente é pronunciado.',
     'consonants'),
    (re.compile(r'(f)$'),
     '{match} no final geralmente é pronunciado.',
     'consonants'),
    (re.compile(r'(que)$'),
     'No final, {match} soa como "k".',
     'consonants'),
    (re.compile(r'(ue)$'),
     'No final, {match} soa como "ü".',
     'vowels'),
    (re.compile(r'(eau)$'),
     'No final, {match} soa como "ô".',
     'vowels'),
    (re.compile(r'(au)$'),
     'No final, {match} soa como "ô".',
     'vowels'),
    (re.compile(r'(ou)$'),
     'No final, {match} soa como "u".',
     'vowels'),
    (re.compile(r'(oi)$'),
     'No final, {match} soa como "uá".',
     'diphthongs'),
    (re.compile(r'(ui)$'),
     'No final, {match} soa como "üi".',
     'diphthongs'),
    (re.compile(r'(ie)$'),
     'No final, {match} soa como "i".',
     'vowels'),
    (re.compile(r'(ée)$'),
     'No final, {match} soa como "ê".',
     'vowels'),
    (re.compile(r'(y)$'),
     'No final, {match} soa como "i".',
     'vowels'),
]

# -------------------------------------------------------------
# Palavras com pronúncia específica
# -------------------------------------------------------------
SPECIFIC_WORDS = {
    "plus": "Em contexto negativo (ne...plus), pronuncia-se 'plü'. Em outros contextos, 'plüs'.",
    "tous": "Pronuncia-se 'tu' quando é pronome/adjetivo, 'tus' quando é substantivo.",
    "est": "Como verbo 'ser', pronuncia-se 'é'. Como ponto cardeal (leste), 'ést'.",
    "fils": "Pronuncia-se 'fis' (sem o 'l').",
    "monsieur": "Pronuncia-se 'mesiê'.",
    "messieurs": "Pronuncia-se 'mésiê'.",
    "femme": "Pronuncia-se 'fam'.",
    "oignon": "Pronuncia-se 'onhõ'.",
    "ville": "Pronuncia-se 'vil'.",
    "fille": "Pronuncia-se 'filh'.",
    "famille": "Pronuncia-se 'familh'.",
    "gentil": "Pronuncia-se 'jãti'.",
    "oeil": "Pronuncia-se 'ölh'.",
    "yeux": "Pronuncia-se 'iö'.",
    "second": "Pronuncia-se 'segõ' ou 'sekõ'.",
    "dix": "Isolado ou final: 'dis'. Antes de vogal: 'diz'. Antes de consoante: 'di'.",
    "six": "Isolado ou final: 'sis'. Antes de vogal: 'siz'. Antes de consoante: 'si'.",
    "huit": "Pronuncia-se 'üit' (com h aspirado).",
    "neuf": "Como número 9: 'nöf'. Antes de vogal em data: 'növ'.",
    "oeuf": "Singular: 'öf'. Plural (oeufs): 'ö'.",
    "boeuf": "Singular: 'böf'. Plural (boeufs): 'bö'.",
    "os": "Singular: 'os'. Plural: 'ô'.",
    "gens": "Pronuncia-se 'jã'.",
    "août": "Pronuncia-se 'ut' ou 'aut'.",
    "pays": "Pronuncia-se 'péi'.",
    "fils": "Pronuncia-se 'fis'.",
    "clé": "Pronuncia-se 'clê'.",
    "pied": "Pronuncia-se 'piê'.",
    "nez": "Pronuncia-se 'nê'.",
    "et": "Pronuncia-se 'ê'.",
    "des": "Pronuncia-se 'dê'.",
    "les": "Pronuncia-se 'lê'.",
    "mes": "Pronuncia-se 'mê'.",
    "tes": "Pronuncia-se 'tê'.",
    "ses": "Pronuncia-se 'sê'.",
    "est-ce que": "Pronuncia-se 'és ke'.",
    "qu'est-ce que": "Pronuncia-se 'kés ke'.",
}

# -------------------------------------------------------------
# Funções Auxiliares
# -------------------------------------------------------------
def make_highlight(m: re.Match, template: str, category: str) -> Tuple[int, int, str, str]:
    color = get_color(category)
    text = html.escape(m.group(0))
    span = f'<span class="highlight {category}-highlight" style="color:{color}; font-weight:bold; font-size:1.1em;">{text}</span>'
    explanation = template.format(match=span)
    return m.start(), m.end(), span, explanation

def find_special_cases(word: str) -> List[Tuple[re.Match, str, str]]:
    lc = word.lower()
    specials: List[Tuple[re.Match, str, str]] = []

    # Verificar se a palavra está no dicionário de palavras específicas
    if lc in SPECIFIC_WORDS:
        m = re.search(r"^.*$", word)
        if m:
            specials.append((m, SPECIFIC_WORDS[lc], 'special'))
        return specials

    # j'ai → pronúncia "jê"
    if lc == "j'ai":
        m = re.search(r"ai", word)
        if m:
            specials.append((m, 'Em "j\'ai", pronuncia-se aproximadamente "jê".', 'vowels'))

    # chats → s mudo
    if lc.endswith('chats'):
        m = re.search(r's$', word)
        if m:
            specials.append((m, 'Em "chats", o {match} final é mudo → "chá".', 'silent'))

    # grand|quand → liaison 'd'
    if re.search(r'(grand|quand)$', lc):
        m = re.search(r'd$', word)
        if m:
            specials.append((m, 'Em liaison, o {match} final soa "t".', 'consonants'))

    # Liaison em números
    if lc in ['deux', 'trois', 'six', 'dix']:
        m = re.search(r'x$', word)
        if m:
            specials.append((m, 'Em liaison, o {match} final soa "z".', 'consonants'))

    # Liaison em "vingt"
    if lc == 'vingt':
        m = re.search(r't$', word)
        if m:
            specials.append((m, 'Em liaison, o {match} final é pronunciado.', 'consonants'))

    # Liaison em "cent"
    if lc == 'cent':
        m = re.search(r't$', word)
        if m:
            specials.append((m, 'Em liaison, o {match} final é pronunciado.', 'consonants'))

    return specials

# -------------------------------------------------------------
# Fluxo Principal de Matching
# -------------------------------------------------------------
def get_pronunciation_hints(word: str) -> Dict[str, List[str] or str]:
    raw_matches: List[Tuple[int, int, str, str]] = []

    # 1) especiais
    for m, tmpl, category in find_special_cases(word):
        raw_matches.append(make_highlight(m, tmpl, category))

    # 2) gerais
    for pattern, tmpl, category in GENERAL_PATTERNS:
        for m in pattern.finditer(word):
            raw_matches.append(make_highlight(m, tmpl, category))

    # 3) remover sobreposições
    raw_matches.sort(key=lambda x: (x[0], -(x[1] - x[0])))
    final_matches: List[Tuple[int, int, str, str]] = []
    for start, end, span, expl in raw_matches:
        if not any(not (end <= s or start >= e) for s, e, *_ in final_matches):
            final_matches.append((start, end, span, expl))

    # 4) sem matches
    if not final_matches:
        return {"word": word, "highlighted_word": html.escape(word), "explanations": []}

    # 5) construir resultado
    final_matches.sort(key=lambda x: x[0])
    highlighted = []
    last = 0
    explanations: List[str] = []
    for start, end, span, expl in final_matches:
        highlighted.append(html.escape(word[last:start]))
        highlighted.append(span)
        explanations.append(expl)
        last = end
    highlighted.append(html.escape(word[last:]))

    return {
        "word": word,
        "highlighted_word": ''.join(highlighted),
        "explanations": explanations
    }

# -------------------------------------------------------------
# Função para obter dicas de pronúncia para frases inteiras
# -------------------------------------------------------------
def get_sentence_pronunciation_hints(sentence: str) -> List[Dict[str, List[str] or str]]:
    """
    Obtém dicas de pronúncia para cada palavra em uma frase.
    
    Args:
        sentence: A frase para analisar
        
    Returns:
        Lista de dicionários com dicas de pronúncia para cada palavra
    """
    # Limpar a frase e dividir em palavras
    cleaned_sentence = re.sub(r'[^\w\s\'-]', ' ', sentence)
    words = cleaned_sentence.split()
    
    # Obter dicas para cada palavra
    hints = []
    for word in words:
        if word.strip():
            hints.append(get_pronunciation_hints(word))
    
    return hints

# -------------------------------------------------------------
# Função para obter dicas de liaison entre palavras
# -------------------------------------------------------------
def get_liaison_hints(sentence: str) -> List[Dict[str, str]]:
    """
    Identifica possíveis liaisons entre palavras em uma frase.
    
    Args:
        sentence: A frase para analisar
        
    Returns:
        Lista de dicionários com informações sobre liaisons
    """
    # Limpar a frase e dividir em palavras
    cleaned_sentence = re.sub(r'[^\w\s\'-]', ' ', sentence)
    words = cleaned_sentence.split()
    
    liaison_hints = []
    
    # Padrões de liaison
    for i in range(len(words) - 1):
        current_word = words[i].lower()
        next_word = words[i + 1].lower()
        
        # Verificar se a próxima palavra começa com vogal ou h mudo
        if re.match(r'^[aeiouhéèêëàâîïôûœæy]', next_word):
            # Artigos e determinantes
            if current_word in ['les', 'des', 'mes', 'tes', 'ses', 'nos', 'vos', 'leurs'] and current_word.endswith('s'):
                liaison_hints.append({
                    "word1": words[i],
                    "word2": words[i + 1],
                    "type": "obrigatória",
                    "sound": "z",
                    "explanation": f"Liaison obrigatória: o 's' final de '{words[i]}' soa como 'z' antes de '{words[i + 1]}'."
                })
            
            # Pronomes
            elif current_word in ['nous', 'vous', 'ils', 'elles'] and current_word.endswith('s'):
                liaison_hints.append({
                    "word1": words[i],
                    "word2": words[i + 1],
                    "type": "obrigatória",
                    "sound": "z",
                    "explanation": f"Liaison obrigatória: o 's' final de '{words[i]}' soa como 'z' antes de '{words[i + 1]}'."
                })
            
            # Preposições e advérbios
            elif current_word in ['dans', 'chez', 'sans', 'très', 'plus']:
                sound = "z" if current_word.endswith('s') else "n" if current_word.endswith('n') else ""
                if sound:
                    liaison_hints.append({
                        "word1": words[i],
                        "word2": words[i + 1],
                        "type": "obrigatória",
                        "sound": sound,
                        "explanation": f"Liaison obrigatória: o final de '{words[i]}' soa como '{sound}' antes de '{words[i + 1]}'."
                    })
            
            # Números
            elif current_word in ['un', 'deux', 'trois', 'six', 'dix', 'vingt', 'cent']:
                sound = "z" if current_word in ['deux', 'trois', 'six', 'dix'] else "t" if current_word in ['vingt', 'cent'] else "n"
                liaison_hints.append({
                    "word1": words[i],
                    "word2": words[i + 1],
                    "type": "obrigatória",
                    "sound": sound,
                    "explanation": f"Liaison obrigatória: o final de '{words[i]}' soa como '{sound}' antes de '{words[i + 1]}'."
                })
            
            # Adjetivos antes de substantivos
            elif i > 0 and current_word.endswith(('d', 't', 's', 'x', 'z', 'n')):
                # Verificar se é provavelmente um adjetivo (simplificado)
                if len(current_word) > 2:  # Evitar falsos positivos com artigos
                    sound = "t" if current_word.endswith('d') else "z" if current_word.endswith(('s', 'x', 'z')) else "n"
                    liaison_hints.append({
                        "word1": words[i],
                        "word2": words[i + 1],
                        "type": "comum",
                        "sound": sound,
                        "explanation": f"Liaison comum: o final de '{words[i]}' pode soar como '{sound}' antes de '{words[i + 1]}'."
                    })
    
    return liaison_hints

# -------------------------------------------------------------
# Função para formatar as dicas de pronúncia em HTML
# -------------------------------------------------------------
def format_pronunciation_hints_html(hints: Dict[str, List[str] or str]) -> str:
    """
    Formata as dicas de pronúncia em HTML com estilos melhorados.
    
    Args:
        hints: Dicionário com dicas de pronúncia
        
    Returns:
        String HTML formatada
    """
    word = hints["word"]
    highlighted_word = hints["highlighted_word"]
    explanations = hints["explanations"]
    
    html_output = []
    html_output.append(CSS_STYLES)
    html_output.append(f'<div class="pronunciation-container">')
    html_output.append(f'<h3 class="pronunciation-word">{word}</h3>')
    html_output.append(f'<div class="pronunciation-highlighted">{highlighted_word}</div>')
    
    if explanations:
        html_output.append('<h4 class="pronunciation-category">Dicas de pronúncia:</h4>')
        html_output.append('<ul class="pronunciation-list">')
        for explanation in explanations:
            html_output.append(f'<li class="pronunciation-hint">{explanation}</li>')
        html_output.append('</ul>')
    
    html_output.append('</div>')
    
    return ''.join(html_output)

# -------------------------------------------------------------
# Função para formatar as dicas de liaison em HTML
# -------------------------------------------------------------
def format_liaison_hints_html(liaisons: List[Dict[str, str]]) -> str:
    """
    Formata as dicas de liaison em HTML com estilos melhorados.
    
    Args:
        liaisons: Lista de dicionários com informações sobre liaisons
        
    Returns:
        String HTML formatada
    """
    if not liaisons:
        return ""
    
    html_output = []
    html_output.append(CSS_STYLES)
    html_output.append('<div class="liaison-container">')
    html_output.append('<h3 class="pronunciation-category">Dicas de liaison:</h3>')
    html_output.append('<ul class="pronunciation-list">')
    
    for liaison in liaisons:
        word1 = liaison["word1"]
        word2 = liaison["word2"]
        sound = liaison["sound"]
        type_liaison = liaison["type"]
        explanation = liaison["explanation"]
        
        color = "#FF0000" if type_liaison == "obrigatória" else "#0000FF"
        
        html_output.append(f'<li class="pronunciation-hint">')
        html_output.append(f'<strong style="color:{color}">{word1}</strong> + ')
        html_output.append(f'<strong>{word2}</strong>: ')
        html_output.append(f'{explanation}')
        html_output.append('</li>')
    
    html_output.append('</ul>')
    html_output.append('</div>')
    
    return ''.join(html_output)

# -------------------------------------------------------------
# Função para formatar as dicas de pronúncia para uma frase inteira em HTML
# -------------------------------------------------------------
def format_sentence_pronunciation_html(sentence: str) -> str:
    """
    Formata as dicas de pronúncia para uma frase inteira em HTML.
    
    Args:
        sentence: A frase para analisar
        
    Returns:
        String HTML formatada
    """
    word_hints = get_sentence_pronunciation_hints(sentence)
    liaison_hints = get_liaison_hints(sentence)
    
    html_output = []
    html_output.append(CSS_STYLES)
    html_output.append('<div class="sentence-pronunciation-container">')
    html_output.append(f'<h2 class="sentence-original">Frase: {html.escape(sentence)}</h2>')
    
    # Seção de palavras
    html_output.append('<div class="words-section">')
    html_output.append('<h3 class="pronunciation-category">Pronúncia por palavra:</h3>')
    
    for word_hint in word_hints:
        word = word_hint["word"]
        highlighted_word = word_hint["highlighted_word"]
        explanations = word_hint["explanations"]
        
        if explanations:
            html_output.append('<div class="word-container" style="margin-bottom: 20px; padding: 10px; border: 1px solid #eee; border-radius: 5px;">')
            html_output.append(f'<h4 class="word-title" style="margin-bottom: 10px;">{word}</h4>')
            html_output.append(f'<div class="word-highlighted" style="font-size: 1.2em; margin-bottom: 10px;">{highlighted_word}</div>')
            
            html_output.append('<ul class="pronunciation-list">')
            for explanation in explanations:
                html_output.append(f'<li class="pronunciation-hint">{explanation}</li>')
            html_output.append('</ul>')
            html_output.append('</div>')
    
    html_output.append('</div>')  # Fim da seção de palavras
    
    # Seção de liaison
    if liaison_hints:
        html_output.append('<div class="liaison-section">')
        html_output.append('<h3 class="pronunciation-category">Liaisons na frase:</h3>')
        html_output.append('<ul class="pronunciation-list">')
        
        for liaison in liaison_hints:
            word1 = liaison["word1"]
            word2 = liaison["word2"]
            sound = liaison["sound"]
            type_liaison = liaison["type"]
            explanation = liaison["explanation"]
            
            color = "#FF0000" if type_liaison == "obrigatória" else "#0000FF"
            
            html_output.append(f'<li class="pronunciation-hint" style="margin-bottom: 15px;">')
            html_output.append(f'<strong style="color:{color}; font-size: 1.1em;">{word1}</strong> + ')
            html_output.append(f'<strong style="font-size: 1.1em;">{word2}</strong>: ')
            html_output.append(f'{explanation}')
            html_output.append('</li>')
        
        html_output.append('</ul>')
        html_output.append('</div>')  # Fim da seção de liaison
    
    html_output.append('</div>')  # Fim do container principal
    
    return ''.join(html_output)
