import re
import random
import html
from typing import List, Tuple, Dict, Pattern

# -------------------------------------------------------------
# Variáveis de apoio
# -------------------------------------------------------------
front_vowels = 'iéeèêëïyæœ'
all_vowels = 'aeiouéêèëíóôúãõœæy'
all_consonants = 'bcdfgjklmnpqrstvwxzʃʒɲŋçh'

# -------------------------------------------------------------
# Paleta de cores única (tuple para imutabilidade)
# -------------------------------------------------------------
COLOR_LIST: Tuple[str, ...] = tuple({
    '#FF0000', '#FF6600', '#CC00FF', '#FFCC00', '#0099FF',
    '#FF9900', '#0033CC', '#666600', '#33CC33', '#990000',
    '#339900', '#336600', '#CC3333', '#003366', '#FF3399',
    '#99FF00', '#FF0033', '#CC3300', '#00CCCC', '#336633',
    '#9900CC', '#006600', '#FF3300', '#CC33CC', '#333300',
    '#6600CC', '#0033FF', '#009966', '#CC0066', '#33CC00',
    '#CC6666', '#999900', 'Tomato', '#336666', '#669966',
    'SlateBlue', '#33FF00', '#FF0066', '#CCCC33', '#33CC66',
    '#660099', '#33CCCC', '#0000FF'
})

# -------------------------------------------------------------
# Padrões gerais pré‑compilados e suas explicações
# -------------------------------------------------------------
GENERAL_PATTERNS: List[Tuple[Pattern, str]] = [
    (re.compile(r'(ais|ait|aient)$'),
     'No condicional/imparfait, {match} soa como "é".'),

    (re.compile(r'(erai|eras|era|erons|erez|eront|'
                r'irai|iras|ira|irons|irez|iront|'
                r'rai|ras|ra|rons|rez|ront)$'),
     'No futuro simples, {match} soa como "rê".'),

    (re.compile(r'\bh(?:aspiré)?\b'),
     'O {match} aspirado não cria liaison.'),

    (re.compile(r'\ble\b'),
     'No artigo {match}, o "e" é muito curto (tipo "luh").'),
    (re.compile(r'\bles\b'),
     'No artigo {match}, soa como "lê".'),

    (re.compile(r'(gn)'),
     '"{match}" soa como "nh".'),

    (re.compile(r"(?:j|n)'en(?=[\s\.,;!?]|$)"),
     'No pronome {match}, soa como "jã"/"nã".'),

    # Nasalizações
    (re.compile(r'(am|em|an|en)(?=[bdfgjklpqrstvwxzʃʒɲŋç])'),
     'A sequência {match} indica som nasal "ãn".'),
    (re.compile(r'(in|im|yn|ym|ein|ain|ien|aim)(?=[bdfgjklpqrstvwxzʃʒɲŋç])'),
     'A sequência {match} representa som nasal "iñ"/"iãn".'),
    (re.compile(r'(on|om)(?=[bdfgjklpqrstvwxzʃʒɲŋç])'),
     'A sequência {match} soa como "õ".'),
    (re.compile(r'(un|um)(?=[bdfgjklpqrstvwxzʃʒɲŋç])'),
     'A sequência {match} dá som nasal "œ̃".'),

    # Combinações de vogais
    (re.compile(r'(au|aux|eau|eaux)'),
     'A sequência {match} soa como "ô".'),
    (re.compile(r'(oy)'),
     '{match} soa como "uai".'),

    # Consoantes especiais
    (re.compile(r'(x)(?=[' + all_consonants + '])'),
     '"{match}" antes de consoante soa "ks".'),
    (re.compile(r'(y)(?=[' + all_vowels + '])'),
     '"{match}" antes de vogal soa como "i" deslizado.'),
    (re.compile(r'(c)(?=[' + front_vowels + '])'),
     '"{match}" antes de vogal frontal soa como "s".'),
    (re.compile(r'(ch)'),
     '"{match}" soa como "x" (xarope).'),
    (re.compile(r'(j|g)(?=[eiy])'),
     '"{match}" soa como "j" antes de e, i ou y.'),

    # Vogais/Finais silenciosos ou modificados
    (re.compile(r'(e|es)(?=[\s\.,;!?]|$)'),
     'No final, {match} geralmente não é pronunciado.'),
    (re.compile(r'(oi)'),
     'A sequência {match} soa como "uá".'),
    (re.compile(r'(ou)'),
     'A sequência {match} soa como "u" fechado.'),
    (re.compile(r'(ille)'),
     '"{match}" soa como "iê".'),
    (re.compile(r'(eu)'),
     '"{match}" soa como "eu" fechado."'),
    (re.compile(r'(é)'),
     '"{match}" soa como "ê" fechado."'),
    (re.compile(r'(è|ê|ai|ei)'),
     'A combinação {match} soa como "é" aberto."'),
    (re.compile(r'(er)$'),
     'No final, {match} soa como "ê".'),
    (re.compile(r'(qu)'),
     '"{match}" pronuncia-se "k".'),
    (re.compile(r'(h)'),
     '"{match}" geralmente é mudo."'),
    (re.compile(r'(ge)$'),
     'No final, {match} soa como "je".'),
    (re.compile(r'(ail)'),
     '"{match}" soa como "ai".'),
    (re.compile(r'(eil)'),
     '"{match}" soa como "ei" fechado."'),
    (re.compile(r'(euil)'),
     '"{match}" soa como "õe".'),
    (re.compile(r'(œil)'),
     '"{match}" soa como "ói".'),
    (re.compile(r'(ien)'),
     '"{match}" soa como "iã" nasalizado."'),
    (re.compile(r'(ion)'),
     '"{match}" soa como "iõ" nasalizado."'),
    (re.compile(r'(tion)$'),
     'No final, {match} soa como "siõ".'),
    (re.compile(r'(ier)$'),
     '{match} no final soa como "iê".'),
    (re.compile(r'(iez)$'),
     '{match} soa como "iê".'),
    (re.compile(r'(oin)'),
     'A sequência {match} soa como "uã".'),
    (re.compile(r'(ui)'),
     'A sequência {match} soa como "üi".'),
    (re.compile(r'(œu)'),
     '{match} soa entre "eu" e "éu".'),
    (re.compile(r'(œ)'),
     '{match} soa como "é" arredondado.'),
    (re.compile(r'(cc)(?=[eiy])'),
     '{match} soa "ks".'),
    (re.compile(r'(ç)'),
     '{match} é pronunciado como "s".'),
    (re.compile(r'(â)'),
     '{match} indica um "a" mais aberto.'),
    (re.compile(r'(î)'),
     '{match} soa como "i".'),
    (re.compile(r'(ô)'),
     '{match} soa como "ô" fechado.'),
    (re.compile(r'(û)'),
     '{match} soa como "u" mais fechado.'),
    (re.compile(r'(pt)$'),
     '{match} não se pronuncia no final.'),
    (re.compile(r'^(ps)'),
     'No início, {match} vira "s".'),
    (re.compile(r'(mn)$'),
     '{match} final simplifica para "m".'),
    (re.compile(r'(ieux)$'),
     '{match} soa como "iô".'),
    (re.compile(r'(amment)$'),
     '{match} soa como "amã".'),
    (re.compile(r'(emment)$'),
     '{match} soa como "amã".'),
    (re.compile(r'(ti)(?=[aeiouy])'),
     '{match} pode soar "tsi".'),
    (re.compile(r'(?<=[aeiouœæyâêîôû])(si)(?=[aeiouœæyâêîôû])'),
     '{match} pode soar "zi".'),
    (re.compile(r'(ll)(?=[eiy])'),
     '{match} soa "lh".'),
    (re.compile(r'(ph)'),
     '{match} soa como "f".'),
    (re.compile(r'(th)'),
     '{match} soa como "t".'),
    (re.compile(r'e(?=[' + all_consonants + ']{2,})'),
     'Quando "e" é seguido de 2+ consoantes, tende a ficar mais fechado.'),
    # ... adicione outros padrões conforme necessário ...
]

# -------------------------------------------------------------
# Funções Auxiliares
# -------------------------------------------------------------
def make_highlight(m: re.Match, template: str) -> Tuple[int, int, str, str]:
    color = random.choice(COLOR_LIST)
    text = html.escape(m.group(0))
    span = f'<span class="highlight" style="color:{color}">{text}</span>'
    explanation = template.format(match=span)
    return m.start(), m.end(), span, explanation

def find_special_cases(word: str) -> List[Tuple[re.Match, str]]:
    lc = word.lower()
    specials: List[Tuple[re.Match, str]] = []

    # j'ai → pronúncia "jê"
    if lc == "j'ai":
        m = re.search(r"ai", word)
        if m:
            specials.append((m, 'Em "j\'ai", pronuncia-se aproximadamente "jê".'))

    # chats → s mudo
    if lc.endswith('chats'):
        m = re.search(r's$', word)
        if m:
            specials.append((m, 'Em "chats", o {match} final é mudo → "chá".'))

    # grand|quand → liaison 'd'
    if re.search(r'(grand|quand)$', lc):
        m = re.search(r'd$', word)
        if m:
            specials.append((m, 'Em liaison, o {match} final soa "t".'))

    return specials

# -------------------------------------------------------------
# Fluxo Principal de Matching
# -------------------------------------------------------------
def get_pronunciation_hints(word: str) -> Dict[str, List[str] or str]:
    raw_matches: List[Tuple[int, int, str, str]] = []

    # 1) especiais
    for m, tmpl in find_special_cases(word):
        raw_matches.append(make_highlight(m, tmpl))

    # 2) gerais
    for pattern, tmpl in GENERAL_PATTERNS:
        for m in pattern.finditer(word):
            raw_matches.append(make_highlight(m, tmpl))

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
