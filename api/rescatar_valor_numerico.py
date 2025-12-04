import re
from cateogries import CATEGORIES 

# --- utilidades ---
def strip_punct(token: str) -> str:
    # quita signos al inicio/fin (mantiene letras acentuadas y dígitos)
    return re.sub(r'^[^\w\dáéíóúüñÁÉÍÓÚÜÑ]+|[^\w\dáéíóúüñÁÉÍÓÚÜÑ]+$', '', token)

def levenshtein(a: str, b: str) -> int:
    a, b = a.lower(), b.lower()
    m, n = len(a), len(b)
    if m > n:
        a, b = b, a
        m, n = n, m

    prev = list(range(m + 1))
    curr = [0] * (m + 1)

    for j in range(1, n + 1):
        curr[0] = j
        for i in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[i] = min(
                prev[i] + 1,
                curr[i - 1] + 1,
                prev[i - 1] + cost
            )
        prev, curr = curr, prev
    return prev[m]

# --- función principal que separa texto y valor ---
def separar_texto_valor(texto: str):

    nombre_gasto, cat, number = texto.split()
    
    number = float(number)

    lower_text = cat.lower()

    # 1. Check prefix matches
    prefix_matches = [
        c for c in CATEGORIES
        if c.lower().startswith(lower_text)
    ]

    if len(prefix_matches) == 1:
        best_cat = prefix_matches[0]
    else:
        # 2. Fallback: closest by Levenshtein distance
        best_cat = min(CATEGORIES, key=lambda c: levenshtein(lower_text, c))

    return nombre_gasto, best_cat, number

