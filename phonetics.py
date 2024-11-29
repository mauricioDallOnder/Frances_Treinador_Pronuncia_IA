# phonetics.py

# Mapeamento de fonemas francês para português com regras contextuais
# Mapeamento de fonemas francês para português com regras contextuais
french_to_portuguese_phonemes = {
    # Vogais orais
    'i': {'default': 'i'},
    'e': {'default': 'e'},
    'ɛ': {'default': 'é', 'before_nasal': 'ê'},
    'a': {'default': 'a'},
    'ɑ': {'default': 'a'},
    'ɔ': {'default': 'ó'},
    'o': {'default': 'ô'},
    'u': {'default': 'u'},
    'y': {'default': 'u'},
    'ø': {'default': 'eu'},
    'œ': {'default': 'é'},
    'ə': {'default': 'e'},

    # Vogais nasais
    'ɛ̃': {'default': 'ẽ'},
    'ɑ̃': {'default': 'ã'},
    'ɔ̃': {'default': 'õ'},
    'œ̃': {'default': 'ũ'},

    # Semivogais
    'j': {'default': 'i'},
    'w': {'default': 'u'},
    'ɥ': {'default': 'u'},

    # Consoantes
    'b': {'default': 'b'},
    'd': {'default': 'd', 'before_i': 'dj'},
    'f': {'default': 'f'},
    'g': {'default': 'g'},
    'ʒ': {'default': 'j', 'word_initial': 'j', 'after_nasal': 'j'},
    'k': {'default': 'k', 'before_front_vowel': 'qu'},
    'l': {'default': 'l'},
    'm': {'default': 'm'},
    'n': {'default': 'n'},
    'p': {'default': 'p'},
    'ʁ': {'default': 'r', 'word_initial': 'h', 'after_vowel': 'rr', 'after_consonant': 'r'},
    's': {'default': 's', 'between_vowels': 'z'},
    't': {'default': 't', 'before_i': 'tch'},
    'v': {'default': 'v'},
    'z': {'default': 'z'},
    'ʃ': {'default': 'ch'},
    'dʒ': {'default': 'dj'},
    'tʃ': {'default': 'tch'},
    'ɲ': {'default': 'nh'},
    'ŋ': {'default': 'ng'},
    'ç': {'default': 's'},

    # Fonemas compostos
    'sj': {'default': 'ch'},  # Exemplo: "attention"
    'ks': {'default': 'x'},   # Exemplo: "exact"

    # Outros fonemas
    'x': {'default': 'x'},
    'ʎ': {'default': 'lh'},
    'ʔ': {'default': ''},  # Fonema glotal stop geralmente não tem equivalente em português
    'θ': {'default': 't'},  # Adapte conforme necessário
    'ð': {'default': 'd'},  # Adapte conforme necessário
    'ɾ': {'default': 'r'},
    'ʕ': {'default': 'r'},  # Adapte conforme necessário
}

def split_into_phonemes(pronunciation):
    phonemes = []
    idx = 0
    while idx < len(pronunciation):
        matched = False
        for phoneme in sorted(french_to_portuguese_phonemes.keys(), key=len, reverse=True):
            length = len(phoneme)
            if pronunciation[idx:idx + length] == phoneme:
                phonemes.append(phoneme)
                idx += length
                matched = True
                break
        if not matched:
            # Caso o fonema não seja encontrado, adiciona o caractere atual
            phonemes.append(pronunciation[idx])
            idx += 1
    return phonemes