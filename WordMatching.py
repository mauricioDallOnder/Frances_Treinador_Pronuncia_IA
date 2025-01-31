# WordMatching.py

import WordMetrics  # Usa a função edit_distance() do RapidFuzz
from ortools.sat.python import cp_model
import numpy as np
from string import punctuation
from dtwalign import dtw, dtw_from_distance_matrix
import time
from rapidfuzz import fuzz  

offset_blank = 1
TIME_THRESHOLD_MAPPING = 5.0

###############################################################################
# 1) Definir lista de palavras funcionais para filtrar ou dar custo reduzido
###############################################################################
FUNCTION_WORDS = {
    "le", "la", "les", "de", "d'", "du", "des", "un", "une", "et", "ou",
    "je", "tu", "il", "elle", "on", "nous", "vous", "ils", "elles",
    "à", "au", "aux", "ça", "ce", "ces", "c'", "ma", "mon", "mes"
}


###############################################################################
# 2) função de conversão fonética (bem simples)
###############################################################################
def convert_to_phonetics(word: str) -> str:
    """
    Converte a palavra para uma forma pseudo-fonética.
    Aqui, simplificado: apenas minúsculas, remove acentos e substitui alguns dígrafos.
    Em produção, recomendável usar Epitran, p.e. epi.transliterate(word).
    """
    import unicodedata
    def remove_accents(text):
        return ''.join(
            c for c in unicodedata.normalize('NFD', text)
            if unicodedata.category(c) != 'Mn'
    )

    w = word.lower().strip()
    w = remove_accents(w)
    # Exemplo: substituir 'ch' por 'ʃ', 'ou' por 'u', etc. (demonstração)
    w = w.replace("ch", "ʃ").replace("ou", "u").replace("on", "õ")
    # Pode inserir outras regras...
    return w

###############################################################################
# 3) Função de custo customizado entre duas palavras
###############################################################################
def compute_word_cost(word_expected: str, word_recognized: str, 
                      use_phonetics=True, fuzzy=False) -> float:
    """
    Calcula o 'custo' de alinhar word_expected e word_recognized.
    - use_phonetics: se True, converte as palavras para pseudo-fonética antes de calcular a distância
    - fuzzy: se True, usamos partial_ratio do rapidfuzz para medir similaridade
    Retorna quanto maior o valor, maior a diferença (custo).
    """
    # Se as duas forem palavras funcionais, reduzimos o peso (por exemplo, custo / 2)
    # pois erros em palavras funcionais podem ser menos críticos, dependendo do caso:
    function_word_factor = 1.0
    if word_expected.lower() in FUNCTION_WORDS or word_recognized.lower() in FUNCTION_WORDS:
        function_word_factor = 0.5

    # Se quisermos comparar foneticamente
    if use_phonetics:
        we = convert_to_phonetics(word_expected)
        wr = convert_to_phonetics(word_recognized)
    else:
        we = word_expected.lower()
        wr = word_recognized.lower()

    cost = 100 * (1 - WordMetrics.hybrid_similarity(we, wr))

    return cost * function_word_factor

###############################################################################
# 4) Montar a matriz de distância (custo) para DTW ou CP-SAT
###############################################################################
def get_word_distance_matrix(words_estimated: list, words_real: list,
                             use_phonetics=True, fuzzy=False) -> np.array:
    """
    Retorna uma matriz de custo (linhas: palavras do reconhecido, colunas: palavras reais).
    Se offset_blank == 1, adicionamos uma linha no final para "palavra vazia".
    """
    number_of_real_words = len(words_real)
    number_of_estimated_words = len(words_estimated)

    word_distance_matrix = np.zeros(
        (number_of_estimated_words + offset_blank, number_of_real_words)
    )

    # Preenche a matriz com o custo customizado
    for idx_estimated in range(number_of_estimated_words):
        for idx_real in range(number_of_real_words):
            cost = compute_word_cost(
                words_estimated[idx_estimated],
                words_real[idx_real],
                use_phonetics=use_phonetics,
                fuzzy=fuzzy
            )
            word_distance_matrix[idx_estimated, idx_real] = cost

    # Linha de "palavra vazia" (BLANK)
    if offset_blank == 1:
        for idx_real in range(number_of_real_words):
            # Ex: pode ser o tamanho da palavra. Aqui usamos 100 como custo "alto"
            word_distance_matrix[number_of_estimated_words, idx_real] = 100.0

    return word_distance_matrix

###############################################################################
# 5) Alinhamento via OR-Tools CP-SAT (como você já tinha)
###############################################################################
def get_best_path_from_distance_matrix(word_distance_matrix):
    """
    Usa um modelo de programação por restrições para alinhar.
    Minimiza a soma dos custos.
    """
    modelCpp = cp_model.CpModel()
    number_of_real_words = word_distance_matrix.shape[1]
    number_of_estimated_words = word_distance_matrix.shape[0] - 1
    number_words = max(number_of_real_words, number_of_estimated_words)

    # estimated_words_order[i] = índice da palavra real correspondente ao i-ésimo tempo
    estimated_words_order = [
        modelCpp.NewIntVar(0, int(number_words - 1 + offset_blank), 'w%i' % i)
        for i in range(number_words + offset_blank)
    ]

    # Garantir ordem não decrescente
    for word_idx in range(number_words - 1):
        modelCpp.Add(
            estimated_words_order[word_idx + 1] >= estimated_words_order[word_idx]
        )

    total_phoneme_distance = 0
    real_word_at_time = {}

    # Vincular a variável estimated_words_order ao custo
    for idx_estimated in range(number_of_estimated_words):
        for idx_real in range(number_of_real_words):
            real_word_at_time[idx_estimated, idx_real] = modelCpp.NewBoolVar(
                'real_word_at_time_%d_%d' % (idx_estimated, idx_real)
            )
            modelCpp.Add(
                estimated_words_order[idx_estimated] == idx_real
            ).OnlyEnforceIf(real_word_at_time[idx_estimated, idx_real])

            cost = word_distance_matrix[idx_estimated, idx_real]
            total_phoneme_distance += cost * real_word_at_time[idx_estimated, idx_real]

    # Se nenhuma palavra estimada corresponder à palavra real, usa a BLANK (última linha)
    # Aqui soma o custo 'vazio' se não tiver correspondência
    for idx_real in range(number_of_real_words):
        word_has_a_match = modelCpp.NewBoolVar(
            'word_has_a_match_%d' % (idx_real)
        )
        modelCpp.Add(
            sum(real_word_at_time[idx_estimated, idx_real]
                for idx_estimated in range(number_of_estimated_words)
            ) == 1
        ).OnlyEnforceIf(word_has_a_match)

        cost_blank = word_distance_matrix[number_of_estimated_words, idx_real]
        total_phoneme_distance += cost_blank * word_has_a_match.Not()

    # Minimizar o custo total
    modelCpp.Minimize(total_phoneme_distance)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = TIME_THRESHOLD_MAPPING
    status = solver.Solve(modelCpp)

    mapped_indices = []
    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        try:
            for word_idx in range(number_words):
                v = solver.Value(estimated_words_order[word_idx])
                mapped_indices.append(v)
        except:
            return []
    else:
        return []

    return np.array(mapped_indices, dtype=int)

###############################################################################
# 6) Reconstruir o alinhamento
###############################################################################
def get_resulting_string(mapped_indices: np.array, words_estimated: list, words_real: list):
    """
    Retorna uma lista de 'mapped_words' (palavra reconhecida que mais se aproxima
    de cada palavra real) e também seus índices.
    Caso não haja correspondência, preenche com '-'.
    """
    mapped_words = []
    mapped_words_indices = []
    WORD_NOT_FOUND_TOKEN = '-'
    number_of_real_words = len(words_real)
    number_of_estimated_words = len(words_estimated)

    for word_idx in range(number_of_real_words):
        position_of_real_word_indices = np.where(mapped_indices == word_idx)[0].astype(int)

        if len(position_of_real_word_indices) == 0:
            # Nenhuma correspondência => '-'
            mapped_words.append(WORD_NOT_FOUND_TOKEN)
            mapped_words_indices.append(-1)
            continue

        if len(position_of_real_word_indices) == 1:
            # Correspondência exata
            est_idx = position_of_real_word_indices[0]
            if est_idx < number_of_estimated_words:
                mapped_words.append(words_estimated[est_idx])
                mapped_words_indices.append(est_idx)
            else:
                mapped_words.append(WORD_NOT_FOUND_TOKEN)
                mapped_words_indices.append(-1)
            continue

        # Se houver mais de 1 estimativa mapeada à mesma palavra real, escolher a de menor custo
        best_cost = float('inf')
        best_word = WORD_NOT_FOUND_TOKEN
        best_idx = -1
        for single_word_idx in position_of_real_word_indices:
            if single_word_idx >= number_of_estimated_words:
                continue
            cost = compute_word_cost(words_estimated[single_word_idx], words_real[word_idx])
            if cost < best_cost:
                best_cost = cost
                best_word = words_estimated[single_word_idx]
                best_idx = single_word_idx

        mapped_words.append(best_word)
        mapped_words_indices.append(best_idx)

    return mapped_words, mapped_words_indices

###############################################################################
# 7) Função principal para mapear via CP-SAT e fallback para DTW
###############################################################################
def get_best_mapped_words(
    words_estimated: list[str], 
    words_real: list[str], 
    use_phonetics: bool = True, 
    fuzzy: bool = False
) -> tuple[list[str], list[int]]:
    """
    Cria a matriz de custo, tenta resolver via OR-Tools. 
    Se não convergir ou demorar, faz fallback em DTW com restrições (Sakoe-Chiba).
    """
    word_distance_matrix = get_word_distance_matrix(
        words_estimated, words_real,
        use_phonetics=use_phonetics, fuzzy=fuzzy
    )

    start = time.time()
    mapped_indices = get_best_path_from_distance_matrix(word_distance_matrix)
    duration_of_mapping = time.time() - start

    # Fallback para dtwalign se o solver não convergir
    if len(mapped_indices) == 0 or duration_of_mapping > (TIME_THRESHOLD_MAPPING + 0.5):
        # Definindo parâmetros de DTW (janela de sakoe-chiba e step_pattern “symmetric2”)
        alignment = dtw(
            word_distance_matrix,
            step_pattern="symmetric2",
            window_type="sakoechiba",
            window_size=3  # Ajuste para restringir o quão distante o caminho pode ficar da diagonal
        )
        path = alignment.path
        # path é array Nx2 com (i, j). Precisamos extrair o mapeamento final
        # Para simplificar: cada i em words_estimated se alinha a path[i, 1] => j
        # mas lembre que a matriz tem offset_blank => shape[0] = number_of_estimated + 1
        # Precisamos tomar cuidado:
        min_len = min(len(words_estimated), len(path))
        mapped_indices = path[:min_len, 1]

    # Com base em mapped_indices, reconstruímos as strings
    mapped_words, mapped_words_indices = get_resulting_string(
        mapped_indices, words_estimated, words_real
    )
    return mapped_words, mapped_words_indices

###############################################################################
# 8) Funções auxiliares para comparação de letras, parse de erros, etc.
###############################################################################
def getWhichLettersWereTranscribedCorrectly(real_word, transcribed_word):
    """
    Retorna uma lista de 1 e 0 para cada letra da word_real,
    indicando se bate com transcribed_word no índice correspondente.
    Simplesmente comparando char a char, sem levar em conta fonética.
    """
    length = min(len(real_word), len(transcribed_word))
    is_letter_correct = []
    for idx in range(length):
        if real_word[idx] == transcribed_word[idx] or real_word[idx] in punctuation:
            is_letter_correct.append(1)
        else:
            is_letter_correct.append(0)
    # Se a real_word for maior, anexa zeros
    if len(real_word) > length:
        is_letter_correct.extend([0]*(len(real_word)-length))
    return is_letter_correct

def parseLetterErrorsToHTML(word_real, is_letter_correct):
    """
    Destaca as letras corretas e incorretas, só para visualização.
    """
    word_colored = ''
    correct_color_start = '<span style="color:green;">'
    correct_color_end = '</span>'
    wrong_color_start = '<span style="color:red;">'
    wrong_color_end = '</span>'

    for idx, letter in enumerate(word_real):
        if idx < len(is_letter_correct) and is_letter_correct[idx] == 1:
            word_colored += correct_color_start + letter + correct_color_end
        else:
            word_colored += wrong_color_start + letter + wrong_color_end
    return word_colored

###############################################################################
# 9) DTW puro (caso queira um atalho) – usando Levenshtein do python-Levenshtein
#    Mantido apenas a título de referência do seu código original.
###############################################################################
from Levenshtein import distance as levenshtein_distance

def dtw_puro(words_expected, words_recognized):
    """
    Versão minimalista de DTW usando a distância Levenshtein pura, sem
    step_pattern sofisticado nem sakoe-chiba. Apenas para referência.
    """
    n = len(words_expected)
    m = len(words_recognized)
    dtw_matrix = np.zeros((n+1, m+1))
    dtw_matrix[0, 1:] = np.inf
    dtw_matrix[1:, 0] = np.inf

    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = levenshtein_distance(words_expected[i-1], words_recognized[j-1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i-1, j],    # Inserção
                dtw_matrix[i, j-1],    # Deleção
                dtw_matrix[i-1, j-1]   # Substituição
            )
    return dtw_matrix
