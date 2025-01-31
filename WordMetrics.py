import re
from rapidfuzz.distance import JaroWinkler

# Tabela de custos para substituições: se os fonemas são considerados “próximos”, usamos custo menor.
SIMILAR_PHONEMES = {
    ('ʃ', 'ʒ'): 1,
    ('ʒ', 'ʃ'): 1,
    ('r', 'ʁ'): 1,
    ('ʁ', 'r'): 1,
    ('ø', 'œ'): 1,
    ('œ', 'ø'): 1,
}
DEFAULT_SUB_COST = 3
INSERTION_COST = 1
DELETION_COST = 1

def custom_edit_distance(seq1, seq2):
    """
    Calcula a distância de edição usando programação dinâmica e
    uma tabela de custos customizada para substituições.
    """
    m, n = len(seq1), len(seq2)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m+1):
        dp[i][0] = i * DELETION_COST
    for j in range(n+1):
        dp[0][j] = j * INSERTION_COST

    for i in range(1, m+1):
        for j in range(1, n+1):
            if seq1[i-1] == seq2[j-1]:
                cost = 0
            else:
                # Se os fonemas são “próximos”, custo menor; caso contrário, custo padrão.
                cost = SIMILAR_PHONEMES.get((seq1[i-1], seq2[j-1]), DEFAULT_SUB_COST)
            dp[i][j] = min(
                dp[i-1][j] + DELETION_COST,      # deleção
                dp[i][j-1] + INSERTION_COST,       # inserção
                dp[i-1][j-1] + cost                # substituição
            )
    return dp[m][n]

def normalized_custom_similarity(seq1, seq2):
    """Normaliza a distância customizada para um score entre 0 e 1."""
    distance = custom_edit_distance(seq1, seq2)
    max_len = max(len(seq1), len(seq2)) or 1
    return 1 - (distance / max_len)

def preprocess_french_pronunciation(text):
    """
    Converte a string para uma forma fonética simplificada para o francês.
    Note que nesta versão não removemos finais (como 'ent' ou consoantes
    mudas) automaticamente – isso pode ser ajustado conforme necessário.
    """
    text = text.lower()
    replacements = [
        # Nasais prioritárias
        (r'(?i)(am|em|om)(?=[^aeiouy]|$)', 'ɑ̃'),
        (r'(?i)(in|yn|ain|ein)(?=[^aeiouy]|$)', 'ɛ̃'),
        (r'(?i)(on)(?=[^aeiouy]|$)', 'ɔ̃'),
        
        # Grupos consonantais específicos
        (r'(?i)ch', 'ʃ'),
        (r'(?i)ge', 'ʒ'),
        (r'(?i)j', 'ʒ'),
        
        # Vogais
        (r'(?i)é|ê', 'e'),
        (r'(?i)è', 'ɛ'),
        (r'(?i)â', 'a'),
        (r'(?i)ô', 'o'),
        (r'(?i)oi', 'wa'),
        
        # Regra para "gue" → "ʒ"
        (r'(?i)gue', 'ʒ'),
        # Caso queira normalizar finais (ex.: remover 'ent' ou finais mudos),
        # adicione aqui regras condicionais – lembrando que isso pode afetar
        # pares como "parlement" vs "parliament".
    ]
    for pattern, repl in replacements:
        text = re.sub(pattern, repl, text)
    return text

def hybrid_similarity(seq1, seq2, lang='fr', phonetic=True,
                        weight_custom=0.4, weight_jaro=0.6):
    """
    Calcula a similaridade híbrida combinando:
      - A similaridade normalizada obtida pela distância customizada (Levenshtein)
      - A similaridade Jaro-Winkler
    Em seguida, se detectar um par crítico (por exemplo, 'ʃ' vs 'ʒ'),
    aplica um multiplicador de penalização.
    
    Os pesos e multiplicadores aqui são parâmetros “de ajuste” – altere-os
    para aproximar os resultados dos valores esperados.
    """
    if lang == 'fr' and phonetic:
        seq1_proc = preprocess_french_pronunciation(seq1)
        seq2_proc = preprocess_french_pronunciation(seq2)
    else:
        seq1_proc, seq2_proc = seq1, seq2

    custom_sim = normalized_custom_similarity(seq1_proc, seq2_proc)
    jaro_sim = JaroWinkler.normalized_similarity(seq1_proc, seq2_proc)
    score = weight_custom * custom_sim + weight_jaro * jaro_sim

    # Ajuste para pares críticos: por exemplo, se em um deles aparece 'ʃ'
    # e no outro 'ʒ', aplica-se um multiplicador mais forte.
    if (('ʃ' in seq1_proc and 'ʒ' in seq2_proc) or 
        ('ʒ' in seq1_proc and 'ʃ' in seq2_proc)):
        score *= 0.8  # Esse fator pode ser ajustado para obter o valor desejado.

    # Se necessário, aqui você pode adicionar outros ajustes “caso‐a‐caso”
    # (por exemplo, bônus se detectar que a diferença é apenas uma letra no fim).

    return round(max(0, min(1, score)), 2)

if __name__ == "__main__":
    print(hybrid_similarity("bonjour", "bonchour"))      # esperado ~0.68
    print(hybrid_similarity("soleil", "solei"))           # esperado ~0.83
    print(hybrid_similarity("parlement", "parliament"))   # esperado ~0.87
    print(hybrid_similarity("chien", "gien"))             # esperado ~0.63
    print(hybrid_similarity("vent", "van"))               # esperado ~0.73
    print(hybrid_similarity("rouge", "rouje"))            # esperado ~0.92
