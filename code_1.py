import pandas as pd
import numpy as np

datos = pd.read_csv("all-datos-first-sense-limpio.csv", sep=";", encoding="utf-8", header=0, names=["Kanji", "Reading", "Sense"]) 

# entropía marginal
def H(columna):
    columna = columna.dropna()  # Eliminar valores NaN
    columna = columna[(columna != "") & (columna != " ")]  # Filtrar valores vacíos o espacios
    p_relativo = columna.value_counts(normalize=True)
    H = -np.sum(p_relativo * np.log2(p_relativo))
    return H

#  entropía conjunta
def H_conjunta(*columnas):
    df = pd.DataFrame({f"col{i}": col for i, col in enumerate(columnas, start=1)}).dropna()
    for col in df.columns:
        df = df[df[col] != ""]
    p_relativo = df.groupby(list(df.columns)).size() / len(df)
    p_relativo = p_relativo[p_relativo > 0]
    H = -np.sum(p_relativo * np.log2(p_relativo))
    return H

#  entropía condicional
def H_condicional(columna1, *otras_columnas):
    H_conj = H_conjunta(columna1, *otras_columnas)
    H_otros = H_conjunta(*otras_columnas)
    return H_conj - H_otros

# información mutua
def I_mutua(columna1, columna2):
    H1 = H(columna1)
    H2 = H(columna2)
    H_conj = H_conjunta(columna1, columna2)
    return H1 + H2 - H_conj

# información mutua triple
def I_mutua_triple(col1, col2, col3):
    H1 = H(col1)
    H2 = H(col2)
    H3 = H(col3)
    H12 = H_conjunta(col1, col2)
    H13 = H_conjunta(col1, col3)
    H23 = H_conjunta(col2, col3)
    H123 = H_conjunta(col1, col2, col3)
    return H1 + H2 + H3 - H12 - H13 - H23 + H123

# Función para información mutua condicional
def I_condicional(col1, col2, *condicionales):
    H_col1_cond = H_condicional(col1, *condicionales)
    H_col1_col2_cond = H_condicional(col1, col2, *condicionales)
    return H_col1_cond - H_col1_col2_cond

kanjis = datos["Kanji"]
readings = datos["Reading"]
senses = datos["Sense"]

# Cálculos

print("### ENTROPÍAS MARGINALES ###")
print(f"H(K) = {H(kanjis)}")
print(f"H(R) = {H(readings)}")
print(f"H(S) = {H(senses)}")

print("\n### ENTROPÍAS CONJUNTAS ###")
print(f"H(K, R) = {H_conjunta(kanjis, readings)}")
print(f"H(K, S) = {H_conjunta(kanjis, senses)}")
print(f"H(R, S) = {H_conjunta(readings, senses)}")
print(f"H(K, R, S) = {H_conjunta(kanjis, readings, senses)}")

print("\n### ENTROPÍAS CONDICIONALES ###")
print(f"H(K | R) = {H_condicional(kanjis, readings)}")
print(f"H(K | S) = {H_condicional(kanjis, senses)}")
print(f"H(R | K) = {H_condicional(readings, kanjis)}")
print(f"H(R | S) = {H_condicional(readings, senses)}")
print(f"H(S | K) = {H_condicional(senses, kanjis)}")
print(f"H(S | R) = {H_condicional(senses, readings)}")
print(f"H(S | K, R) = {H_condicional(senses, kanjis, readings)}")
print(f"H(K | R, S) = {H_condicional(kanjis, readings, senses)}")
print(f"H(R | K, S) = {H_condicional(readings, kanjis, senses)}")

print("\n### INFORMACIÓN MUTUA ###")
print(f"I(K; R) = {I_mutua(kanjis, readings)}")
print(f"I(K; S) = {I_mutua(kanjis, senses)}")
print(f"I(R; S) = {I_mutua(readings, senses)}")
print(f"I(K; R; S) = {I_mutua_triple(kanjis, readings, senses)}")

print(0.1)

