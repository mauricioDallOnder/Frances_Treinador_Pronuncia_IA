from rapidfuzz.distance import Levenshtein

def edit_distance(seq1, seq2):
    return Levenshtein.distance(seq1, seq2)

