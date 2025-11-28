import re

# --- Diccionarios (añadí "un" y variantes sin tilde) ---
UNIDADES = {
    "cero":0, "uno":1, "una":1, "un":1,
    "dos":2, "tres":3, "cuatro":4, "cinco":5,
    "seis":6, "siete":7, "ocho":8, "nueve":9, "diez":10, "once":11,
    "doce":12, "trece":13, "catorce":14, "quince":15, "dieciséis":16,
    "dieciseis":16, "diecisiete":17, "dieciocho":18, "diecinueve":19,
    "veinte":20, "veintiuno":21, "veintiún":21, "veintiun":21, "veintidos":22, "veintidós":22, "veintitrés":23,
    "veintitres":23, "veinticuatro":24, "veinticinco":25, "veintiseis":26,
    "veintiséis":26, "veintisiete":27, "veintiocho":28, "veintinueve":29
}

DECENAS = {
    "treinta":30, "cuarenta":40, "cincuenta":50, "sesenta":60, "setenta":70,
    "ochenta":80, "noventa":90
}

CENTENAS = {
    "cien":100, "ciento":100, "doscientos":200, "trescientos":300,
    "cuatrocientos":400, "quinientos":500, "seiscientos":600,
    "setecientos":700, "ochocientos":800, "novecientos":900
}

MULTIPLICADORES = {
    "mil":1000, "millon":1000000, "millón":1000000, "millones":1000000,
    "millonesde":1000000  # por si se pega "millonesde" al tokenizar mal
}

# --- utilidades ---
def strip_punct(token: str) -> str:
    # quita signos al inicio/fin (mantiene letras acentuadas y dígitos)
    return re.sub(r'^[^\w\dáéíóúüñÁÉÍÓÚÜÑ]+|[^\w\dáéíóúüñÁÉÍÓÚÜÑ]+$', '', token)

def parse_numeric_token(token: str):
    """Intenta convertir un token que contiene dígitos a int/float
       con heurísticas para separadores (., y ,). Devuelve None si no es número."""
    s = token.strip()
    if not re.search(r'\d', s):
        return None
    s = re.sub(r'[^\d,.\-]', '', s)  # deja sólo dígitos, coma, punto, signo
    if not s:
        return None

    # Si hay ambos '.' y ',' usamos la posición para decidir
    if '.' in s and ',' in s:
        if s.find('.') < s.find(','):          # "1.234,56" => pone decimal con '.'
            s_norm = s.replace('.', '').replace(',', '.')
        else:                                  # "1,234.56"
            s_norm = s.replace(',', '')
    elif ',' in s:
        parts = s.split(',')
        # heurística: si la parte final tiene 3 dígitos y hay más de una parte -> miles
        if len(parts) > 1 and len(parts[-1]) == 3:
            s_norm = ''.join(parts)
        else:
            s_norm = s.replace(',', '.')
    elif '.' in s:
        parts = s.split('.')
        if len(parts) > 1 and len(parts[-1]) == 3:
            s_norm = ''.join(parts)
        else:
            s_norm = s  # "1234.56" ya es ok
    else:
        s_norm = s

    try:
        val = float(s_norm)
    except:
        return None
    if val.is_integer():
        return int(val)
    return val

# --- parser de palabras a número mejorado ---
def palabras_a_numero(texto: str):
    texto = texto.lower().replace('-', ' ')
    tokens_raw = re.split(r'\s+', texto)
    tokens = [strip_punct(t) for t in tokens_raw if t.strip()]
    total = 0
    current = 0
    i = 0
    n = len(tokens)

    while i < n:
        t = tokens[i]
        if not t:
            i += 1
            continue

        if t in UNIDADES:
            current += UNIDADES[t]
            i += 1
            continue
        if t in DECENAS:
            current += DECENAS[t]
            i += 1
            continue
        if t in CENTENAS:
            current += CENTENAS[t]
            i += 1
            continue

        # Si encontramos un multiplicador, miramos si hay una secuencia de multiplicadores
        if t in MULTIPLICADORES:
            prod = MULTIPLICADORES[t]
            j = i + 1
            while j < n and tokens[j] in MULTIPLICADORES:
                prod *= MULTIPLICADORES[tokens[j]]
                j += 1
            if current == 0:
                current = 1
            current *= prod
            total += current
            current = 0
            i = j
            continue

        # intentar convertir token numérico (ej "$25,000", "1.234,56", "25000")
        num = parse_numeric_token(t)
        if num is not None:
            current += num
            i += 1
            continue

        # token no numérico -> ignorar (por ejemplo "de", "pesos", "clp")
        i += 1

    total += current
    return total

# --- función principal que separa texto y valor ---
def separar_texto_valor(texto: str):
    total = palabras_a_numero(texto)

    tokens_raw = re.split(r'[\s-]+', texto)
    cleaned_tokens = []
    for w in tokens_raw:
        if not w.strip():
            continue
        w_clean = strip_punct(w)
        wl = w_clean.lower()
        if not wl:
            continue
        # si es palabra numérica conocida, la quitamos
        if wl in UNIDADES or wl in DECENAS or wl in CENTENAS or wl in MULTIPLICADORES:
            continue
        # si es un número o contiene dígitos lo quitamos
        if parse_numeric_token(wl) is not None or re.search(r'\d', wl):
            continue
        cleaned_tokens.append(w_clean)
    texto_final = ' '.join(cleaned_tokens).strip()
    return texto_final, total