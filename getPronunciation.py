import re
import random
import html
from typing import List, Tuple, Dict, Pattern, Optional, Any

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
    "c'est": "Pronuncia-se 'sé'.",
    "j'aie": "Pronuncia-se 'je' (e fechado).",
}

# -------------------------------------------------------------
# Funções Auxiliares
# -------------------------------------------------------------
def make_highlight(m: re.Match, template: str, category: str) -> Tuple[int, int, str, str]:
    color = get_color(category)
    text = html.escape(m.group(0))
    span = f'<span class="highlight" style="color:{color}; font-weight:bold;">{text}</span>'
    explanation = template.format(match=text)
    return (m.start(), m.end(), span, explanation)

def find_patterns_in_word(word: str) -> List[Tuple[int, int, str, str]]:
    """Encontra padrões fonéticos em uma palavra."""
    results = []
    
    # Verificar palavras específicas primeiro
    word_lower = word.lower()
    if word_lower in SPECIFIC_WORDS:
        # Para palavras específicas, destacamos a palavra inteira
        color = get_color('special')
        span = f'<span class="highlight" style="color:{color}; font-weight:bold;">{html.escape(word)}</span>'
        explanation = SPECIFIC_WORDS[word_lower]
        results.append((0, len(word), span, explanation))
        return results
    
    # Verificar padrões gerais
    for pattern, template, category in GENERAL_PATTERNS:
        for m in pattern.finditer(word):
            results.append(make_highlight(m, template, category))
    
    # Ordenar por posição inicial
    results.sort(key=lambda x: x[0])
    return results

def apply_highlights(word: str, highlights: List[Tuple[int, int, str, str]]) -> Tuple[str, List[str]]:
    """Aplica destaques ao texto e retorna explicações."""
    if not highlights:
        return word, []
    
    # Ordenar destaques por posição inicial (decrescente)
    highlights.sort(key=lambda x: x[0], reverse=True)
    
    # Aplicar destaques
    result = word
    explanations = []
    for start, end, span, explanation in highlights:
        result = result[:start] + span + result[end:]
        if explanation not in explanations:
            explanations.append(explanation)
    
    return result, explanations

# -------------------------------------------------------------
# Função principal para obter dicas de pronúncia
# -------------------------------------------------------------
def get_pronunciation_hints(word: str) -> Dict[str, Any]:
    """
    Obtém dicas de pronúncia para uma palavra.
    
    Args:
        word: A palavra para analisar
        
    Returns:
        Um dicionário com a palavra, explicações e palavra destacada
    """
    word = word.strip()
    if not word:
        return {"word": "", "explanations": [], "highlighted": ""}
    
    # Remover pontuação no início e fim
    word = re.sub(r'^[^\w]+|[^\w]+$', '', word)
    if not word:
        return {"word": "", "explanations": [], "highlighted": ""}
    
    # Encontrar padrões
    highlights = find_patterns_in_word(word)
    
    # Aplicar destaques
    highlighted, explanations = apply_highlights(word, highlights)
    
    # Garantir que as explicações contenham HTML com cores
    colored_explanations = []
    for explanation in explanations:
        # Verificar se já contém HTML
        if '<span' in explanation:
            colored_explanations.append(explanation)
        else:
            # Procurar por padrões entre aspas para colorir
            matches = re.finditer(r'"([^"]+)"', explanation)
            colored_exp = explanation
            offset = 0
            for m in matches:
                start, end = m.span(1)
                start += offset
                end += offset
                color = get_color('special')
                span = f'<span class="highlight" style="color:{color}; font-weight:bold;">{m.group(1)}</span>'
                colored_exp = colored_exp[:start-1] + span + colored_exp[end+1:]
                offset += len(span) - (end - start + 2)  # +2 para as aspas
            colored_explanations.append(colored_exp)
    
    return {
        "word": word,
        "explanations": colored_explanations,
        "highlighted": highlighted
    }

# -------------------------------------------------------------
# Funções adicionais para análise de frases completas
# -------------------------------------------------------------
def get_sentence_pronunciation_hints(sentence: str) -> List[Dict[str, Any]]:
    """
    Obtém dicas de pronúncia para todas as palavras em uma frase.
    
    Args:
        sentence: A frase para analisar
        
    Returns:
        Uma lista de dicionários com dicas para cada palavra
    """
    words = re.findall(r'\b\w+\b', sentence)
    return [get_pronunciation_hints(word) for word in words if word.strip()]

def get_liaison_hints(sentence: str) -> List[Dict[str, Any]]:
    """
    Detecta possíveis liaisons entre palavras em uma frase.
    
    Args:
        sentence: A frase para analisar
        
    Returns:
        Uma lista de dicionários com informações sobre liaisons
    """
    words = re.findall(r'\b\w+\b', sentence)
    liaisons = []
    
    for i in range(len(words) - 1):
        current_word = words[i].lower()
        next_word = words[i+1].lower()
        
        # Verificar se a palavra atual termina com consoante muda
        if re.search(r'[sxztn]$', current_word):
            # Verificar se a próxima palavra começa com vogal ou h mudo
            if re.match(r'^[aeiouéèêëàâîïôûœæyh]', next_word) and next_word not in ["huit", "hache", "honte"]:
                liaison_type = "obrigatória" if current_word in ["les", "des", "ces", "mes", "tes", "ses", "nos", "vos", "aux", "est", "sont"] else "opcional"
                
                # Colorir as palavras
                color1 = get_color('special')
                color2 = get_color('special')
                colored_word1 = f'<span class="highlight" style="color:{color1}; font-weight:bold;">{current_word}</span>'
                colored_word2 = f'<span class="highlight" style="color:{color2}; font-weight:bold;">{next_word}</span>'
                
                liaisons.append({
                    "word1": current_word,
                    "word2": next_word,
                    "type": liaison_type,
                    "explanation": f"Liaison {liaison_type} entre {colored_word1} e {colored_word2}"
                })
    
    return liaisons

# -------------------------------------------------------------
# Funções para formatação HTML
# -------------------------------------------------------------
def format_pronunciation_hints_html(hints: Dict[str, Any]) -> str:
    """
    Formata dicas de pronúncia como HTML.
    
    Args:
        hints: Dicionário com dicas de pronúncia
        
    Returns:
        String HTML formatada
    """
    if not hints["explanations"]:
        return ""
    
    html_output = f"""
    {CSS_STYLES}
    <div class="pronunciation-hint">
        <strong>{hints["word"]}</strong>: {", ".join(hints["explanations"])}
    </div>
    """
    
    return html_output

def format_sentence_pronunciation_html(sentence: str) -> str:
    """
    Formata dicas de pronúncia para uma frase completa como HTML.
    
    Args:
        sentence: A frase para analisar
        
    Returns:
        String HTML formatada
    """
    hints_list = get_sentence_pronunciation_hints(sentence)
    liaisons = get_liaison_hints(sentence)
    
    # Filtrar apenas palavras com explicações
    hints_list = [h for h in hints_list if h["explanations"]]
    
    if not hints_list and not liaisons:
        return "<p>Nenhuma dica de pronúncia encontrada.</p>"
    
    html_output = f"""
    {CSS_STYLES}
    <div class="pronunciation-container">
    """
    
    if hints_list:
        html_output += """
        <h3 class="pronunciation-category">Dicas de pronúncia:</h3>
        <ul class="pronunciation-list">
        """
        
        for hint in hints_list:
            html_output += f"""
            <li><strong>{hint["word"]}</strong>: {", ".join(hint["explanations"])}</li>
            """
        
        html_output += "</ul>"
    
    if liaisons:
        html_output += """
        <h3 class="pronunciation-category">Liaisons:</h3>
        <ul class="pronunciation-list">
        """
        
        for liaison in liaisons:
            html_output += f"""
            <li>{liaison["explanation"]}</li>
            """
        
        html_output += "</ul>"
    
    html_output += "</div>"
    
    return html_output

def format_liaison_hints_html(liaisons: List[Dict[str, Any]]) -> str:
    """
    Formata dicas de liaison como HTML.
    
    Args:
        liaisons: Lista de dicionários com informações sobre liaisons
        
    Returns:
        String HTML formatada
    """
    if not liaisons:
        return "<p>Nenhuma liaison detectada.</p>"
    
    html_output = f"""
    {CSS_STYLES}
    <div class="pronunciation-container">
        <h3 class="pronunciation-category">Liaisons:</h3>
        <ul class="pronunciation-list">
    """
    
    for liaison in liaisons:
        html_output += f"""
        <li>{liaison["explanation"]}</li>
        """
    
    html_output += """
        </ul>
    </div>
    """
    
    return html_output
