# Tarea 1 - Pregunta 3

Este notebook resuelve únicamente la **Pregunta 3** de la Tarea 1.

La pregunta solicita analizar la base `Datos Tarea 1.sav`, que contiene los resultados de un experimento con tres grupos:

1. **CSR Externa**
2. **CSR Interna**
3. **Información sin CSR**, grupo de control

En el experimento se midió la confianza en la empresa antes y después de entregar información.

Para cada prueba se reporta:

- hipótesis nula e hipótesis alternativa;
- estadístico calculado;
- grados de libertad, cuando corresponde;
- cálculo o lógica de los grados de libertad;
- estadístico crítico al 95% de confianza;
- decisión respecto de la hipótesis nula;
- interpretación sustantiva del resultado.

---

## 0. Paquetes necesarios

Este notebook usa:

- `pandas` para manipular la base de datos;
- `numpy` para cálculos numéricos;
- `scipy` para pruebas t, chi-cuadrado y ANOVA;
- `statsmodels` para la prueba Z de proporciones y Tukey HSD;
- `pyreadstat` para leer archivos SPSS `.sav`;
- `openpyxl` para exportar resultados a Excel.

Si estás en Google Colab o en un entorno sin paquetes instalados, descomenta la siguiente celda.

---

```python
# %pip install pandas numpy scipy statsmodels pyreadstat openpyxl
```

---

## 1. Importar librerías

Primero cargamos las librerías necesarias. Las funciones estadísticas principales provienen de `scipy.stats`.

---

```python
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat

from scipy import stats
from scipy.stats import chi2, f, t

from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.proportion import proportions_ztest
```

---

## 2. Configuración del análisis

Trabajamos con un nivel de significancia de 5%.

Esto equivale a un nivel de confianza de 95%.

La regla de decisión general será:

- si `p-value < 0.05`, se rechaza la hipótesis nula;
- si `p-value >= 0.05`, no se rechaza la hipótesis nula.

---

```python
ALPHA = 0.05
CONFIANZA = 0.95

ruta_base = Path("Datos Tarea 1.sav")

if not ruta_base.exists():
    ruta_base = Path("Tarea1") / "Datos Tarea 1.sav"

print("Ruta de la base:", ruta_base.resolve())
print("Nivel de significancia:", ALPHA)
print("Nivel de confianza:", CONFIANZA)
```

---

## 3. Cargar la base SPSS

La base viene en formato SPSS, por eso usamos `pyreadstat.read_sav`.

El argumento `apply_value_formats=False` mantiene los códigos numéricos originales, lo cual facilita el cálculo estadístico.

---

```python
df, meta = pyreadstat.read_sav(ruta_base, apply_value_formats=False)

print("Dimensiones de la base:", df.shape)
print("Columnas:")
print(df.columns.tolist())

df.head()
```

---

## 4. Revisar metadata del archivo SPSS

Cuando usamos `pyreadstat.read_sav`, no solo se carga la base en `df`.

También se carga un objeto llamado `meta`, que contiene la metadata del archivo SPSS.

Esto es importante porque `df` muestra los códigos numéricos, por ejemplo `Grupo = 1`, pero `meta` indica qué significa cada código.

En esta base, las etiquetas de variables categóricas vienen guardadas dentro del archivo `.sav`.

---

```python
print("Etiquetas descriptivas de las columnas:")
for nombre, etiqueta in zip(meta.column_names, meta.column_labels):
    print(f"{nombre}: {etiqueta}")

print("\nRelación entre variables y conjuntos de etiquetas:")
print(meta.variable_to_label)

print("\nEtiquetas de valores guardadas en el archivo SPSS:")
for nombre_etiqueta, etiquetas in meta.value_labels.items():
    print(f"\n{nombre_etiqueta}:")
    for codigo, texto in etiquetas.items():
        print(f"  {codigo}: {texto}")
```

---

La salida anterior justifica el mapeo que se hace más adelante.

Por ejemplo, `meta.variable_to_label` indica que la variable `Grupo` usa el conjunto de etiquetas `labels0`.

Luego, `meta.value_labels["labels0"]` muestra:

- `1 = CSR Externa`
- `2 = CSR Interna`
- `3 = Info sin CSR`

Por eso sabemos que esos son los nombres correctos de los grupos.

---

## 5. Revisión inicial de la base

Antes de aplicar pruebas inferenciales, verificamos:

- tipo de datos;
- valores perdidos;
- cantidad de valores únicos.

Esto ayuda a detectar problemas antes de calcular pruebas estadísticas.

---

```python
estructura = pd.DataFrame({
    "tipo": df.dtypes,
    "valores_perdidos": df.isna().sum(),
    "valores_unicos": df.nunique()
})

estructura
```

---

## 6. Etiquetar variables categóricas

La base usa códigos numéricos para las variables categóricas.

Para interpretar mejor las tablas, creamos columnas con etiquetas.

Estas etiquetas salen de la metadata SPSS revisada en la sección anterior.

---

```python
labels_grupo = {
    1: "CSR Externa",
    2: "CSR Interna",
    3: "Info sin CSR"
}

labels_genero = {
    1: "Masculino",
    2: "Femenino"
}

labels_ocupacion = {
    1: "Estudiante",
    2: "Trabajador dependiente",
    3: "Trabajador independiente",
    4: "Otro"
}

labels_ingresos = {
    1: "Menos de 200 mil",
    2: "Entre 200 y 500 mil",
    3: "Entre 501 y 1000 mil",
    4: "Entre 1 millón y 2 millones",
    5: "Más de 2 millones"
}

df["Grupo_label"] = df["Grupo"].map(labels_grupo)
df["Genero_label"] = df["Genero"].map(labels_genero)
df["Ocupacion_label"] = df["Ocupación"].map(labels_ocupacion)
df["Ingresos_label"] = df["Ingresos"].map(labels_ingresos)

df[["Grupo", "Grupo_label", "Genero", "Genero_label", "Ocupación", "Ocupacion_label", "Ingresos", "Ingresos_label"]].head()
```

---

## 7. Crear variable de cambio en confianza

La pregunta 3.g pide calcular una nueva variable que mida el cambio en confianza producto de la información.

Definimos:

$$
\text{Cambio confianza} = \text{Confianza después} - \text{Confianza antes}
$$

Interpretación:

- valor positivo: la confianza aumentó;
- valor cero: la confianza no cambió;
- valor negativo: la confianza disminuyó.

---

```python
df["Cambio_confianza"] = df["Confianza_después"] - df["Confianza_antes"]

df[["Grupo_label", "Confianza_antes", "Confianza_después", "Cambio_confianza"]].head()
```

---

## 8. Funciones auxiliares

Estas funciones permiten imprimir decisiones e interpretaciones de manera homogénea.

---

```python
def decision_hipotesis(p_value, alpha=ALPHA):
    if p_value < alpha:
        return "Se rechaza H0"
    return "No se rechaza H0"


def interpretar_significancia(p_value, alpha=ALPHA):
    if p_value < alpha:
        return "Existe evidencia estadísticamente significativa al 95% de confianza."
    return "No existe evidencia estadísticamente significativa al 95% de confianza."


def imprimir_resultado(nombre_prueba, estadistico, p_value, gl=None, critico=None):
    print(nombre_prueba)
    print("-" * len(nombre_prueba))
    print(f"Estadístico calculado: {estadistico:.4f}")
    if gl is not None:
        print(f"Grados de libertad: {gl}")
    if critico is not None:
        print(f"Estadístico crítico al 95%: {critico}")
    print(f"p-value: {p_value:.6f}")
    print("Decisión:", decision_hipotesis(p_value))
    print("Interpretación:", interpretar_significancia(p_value))


def resumen_numerico_por_grupo(variable):
    return (
        df.groupby("Grupo_label")[variable]
        .agg(n="count", media="mean", desviacion_estandar="std", minimo="min", maximo="max")
        .round(4)
    )
```

---

## 9. Estadísticos descriptivos generales

Antes de las pruebas, observamos el tamaño de cada grupo y los principales descriptivos.

Esto permite entender el patrón inicial de los datos.

---

```python
print("Frecuencia por grupo:")
display(df["Grupo_label"].value_counts().rename("n").to_frame())

print("\nEdad por grupo:")
display(resumen_numerico_por_grupo("Edad"))

print("\nConfianza antes por grupo:")
display(resumen_numerico_por_grupo("Confianza_antes"))

print("\nConfianza después por grupo:")
display(resumen_numerico_por_grupo("Confianza_después"))

print("\nCambio en confianza por grupo:")
display(resumen_numerico_por_grupo("Cambio_confianza"))
```

---

# 3.a Prueba t de dos muestras para edad entre tratamientos

## Objetivo

Comparar si la edad promedio difiere entre los dos grupos de tratamiento:

- CSR Externa
- CSR Interna

El grupo de control no se considera en esta prueba.

## Hipótesis

$$
H_0: \mu_{\text{Edad, CSR Externa}} = \mu_{\text{Edad, CSR Interna}}
$$

$$
H_1: \mu_{\text{Edad, CSR Externa}} \neq \mu_{\text{Edad, CSR Interna}}
$$

## Prueba usada

Se usa una prueba t de dos muestras independientes.

Aplicamos la versión de Welch porque no exige asumir igualdad de varianzas.

Los grados de libertad de Welch se calculan como:

$$
gl =
\frac{
\left(\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}\right)^2
}{
\frac{\left(\frac{s_1^2}{n_1}\right)^2}{n_1 - 1}
+
\frac{\left(\frac{s_2^2}{n_2}\right)^2}{n_2 - 1}
}
$$

---

```python
edad_externa = df.loc[df["Grupo"] == 1, "Edad"].dropna()
edad_interna = df.loc[df["Grupo"] == 2, "Edad"].dropna()

n1 = len(edad_externa)
n2 = len(edad_interna)

media1 = edad_externa.mean()
media2 = edad_interna.mean()

sd1 = edad_externa.std(ddof=1)
sd2 = edad_interna.std(ddof=1)

print("Descriptivos")
print("------------")
print(f"CSR Externa: n={n1}, media={media1:.4f}, sd={sd1:.4f}")
print(f"CSR Interna: n={n2}, media={media2:.4f}, sd={sd2:.4f}")
```

---

```python
t_edad_stat, p_edad_value = stats.ttest_ind(
    edad_externa,
    edad_interna,
    equal_var=False
)

var1 = sd1 ** 2
var2 = sd2 ** 2

gl_edad_welch = ((var1 / n1 + var2 / n2) ** 2) / (
    ((var1 / n1) ** 2) / (n1 - 1) + ((var2 / n2) ** 2) / (n2 - 1)
)

t_edad_critico = t.ppf(1 - ALPHA / 2, gl_edad_welch)

imprimir_resultado(
    nombre_prueba="Prueba t de Welch para edad entre tratamientos",
    estadistico=t_edad_stat,
    p_value=p_edad_value,
    gl=round(gl_edad_welch, 4),
    critico=f"±{t_edad_critico:.4f}"
)
```

---

## Conclusión 3.a

El p-value es mayor que 0.05, por lo tanto **no se rechaza H0**.

No existe evidencia estadísticamente significativa para afirmar que la edad promedio difiera entre CSR Externa y CSR Interna.

En términos del experimento, esto es favorable porque sugiere que ambos tratamientos son comparables en edad.

---

# 3.b Prueba t para muestras relacionadas: confianza antes y después

## Objetivo

Evaluar si la confianza cambia antes vs. después dentro de cada grupo.

Se realiza una prueba separada para:

- CSR Externa
- CSR Interna
- Info sin CSR

## Hipótesis para cada grupo

$$
H_0: \mu_{\text{después} - \text{antes}} = 0
$$

$$
H_1: \mu_{\text{después} - \text{antes}} \neq 0
$$

## Prueba usada

Se usa una prueba t pareada porque se mide a los mismos participantes antes y después.

Los grados de libertad se calculan como:

$$
gl = n - 1
$$

---

```python
resultados_t_pareada = []

for codigo_grupo, nombre_grupo in labels_grupo.items():
    pares = (
        df.loc[df["Grupo"] == codigo_grupo, ["Confianza_antes", "Confianza_después"]]
        .dropna()
    )
    
    diferencias = pares["Confianza_después"] - pares["Confianza_antes"]
    
    n = len(diferencias)
    media_antes = pares["Confianza_antes"].mean()
    media_despues = pares["Confianza_después"].mean()
    media_cambio = diferencias.mean()
    sd_cambio = diferencias.std(ddof=1)
    
    t_stat, p_value = stats.ttest_rel(
        pares["Confianza_después"],
        pares["Confianza_antes"]
    )
    
    gl = n - 1
    t_critico = t.ppf(1 - ALPHA / 2, gl)
    
    print("\n" + "=" * 80)
    print(f"Grupo: {nombre_grupo}")
    print("=" * 80)
    print(f"N pares: {n}")
    print(f"Media antes: {media_antes:.4f}")
    print(f"Media después: {media_despues:.4f}")
    print(f"Media cambio: {media_cambio:.4f}")
    print(f"Desviación estándar del cambio: {sd_cambio:.4f}")
    
    imprimir_resultado(
        nombre_prueba="Prueba t pareada",
        estadistico=t_stat,
        p_value=p_value,
        gl=gl,
        critico=f"±{t_critico:.4f}"
    )
    
    resultados_t_pareada.append({
        "grupo": nombre_grupo,
        "n": n,
        "media_antes": media_antes,
        "media_despues": media_despues,
        "media_cambio": media_cambio,
        "sd_cambio": sd_cambio,
        "t": t_stat,
        "gl": gl,
        "t_critico": t_critico,
        "p_value": p_value,
        "decision": decision_hipotesis(p_value)
    })

tabla_t_pareada = pd.DataFrame(resultados_t_pareada)
tabla_t_pareada.round(4)
```

---

## Conclusión 3.b

En CSR Externa y CSR Interna el p-value es menor que 0.05, por lo tanto se rechaza H0.

Esto indica que la confianza aumentó significativamente después de recibir información de CSR.

En el grupo Info sin CSR, el p-value es mayor que 0.05, por lo tanto no se rechaza H0.

Esto indica que el grupo de control no presenta un cambio estadísticamente significativo en confianza.

---

# 3.c Prueba Z de porcentajes para género entre tratamientos

## Objetivo

Comparar la proporción de participantes de género femenino entre los dos grupos de tratamiento.

No se considera el grupo de control.

## Hipótesis

$$
H_0: p_{\text{Femenino, CSR Externa}} = p_{\text{Femenino, CSR Interna}}
$$

$$
H_1: p_{\text{Femenino, CSR Externa}} \neq p_{\text{Femenino, CSR Interna}}
$$

## Prueba usada

Se usa una prueba Z para dos proporciones.

El valor crítico bilateral al 95% es aproximadamente:

$$
Z_c = \pm 1.96
$$

---

```python
tratamientos = df[df["Grupo"].isin([1, 2])].copy()

grupo_externa = tratamientos[tratamientos["Grupo"] == 1]
grupo_interna = tratamientos[tratamientos["Grupo"] == 2]

evento_femenino = 2

x1 = (grupo_externa["Genero"] == evento_femenino).sum()
x2 = (grupo_interna["Genero"] == evento_femenino).sum()

n1 = grupo_externa["Genero"].notna().sum()
n2 = grupo_interna["Genero"].notna().sum()

p1 = x1 / n1
p2 = x2 / n2

print(f"Evento analizado: género femenino = {evento_femenino}")
print(f"CSR Externa: {x1} de {n1} = {p1:.4f}")
print(f"CSR Interna: {x2} de {n2} = {p2:.4f}")
```

---

```python
count = np.array([x1, x2])
nobs = np.array([n1, n2])

z_genero_stat, p_genero_value = proportions_ztest(
    count=count,
    nobs=nobs,
    alternative="two-sided"
)

z_genero_critico = stats.norm.ppf(1 - ALPHA / 2)

imprimir_resultado(
    nombre_prueba="Prueba Z de dos proporciones para género",
    estadistico=z_genero_stat,
    p_value=p_genero_value,
    gl=None,
    critico=f"±{z_genero_critico:.4f}"
)
```

---

## Conclusión 3.c

El p-value es mayor que 0.05, por lo tanto **no se rechaza H0**.

No existe evidencia estadísticamente significativa para afirmar que la proporción de mujeres sea distinta entre CSR Externa y CSR Interna.

Esto sugiere que los grupos de tratamiento son comparables en género.

---

# 3.d Chi-cuadrado para ocupación entre tratamientos

## Objetivo

Evaluar si la distribución de ocupación difiere entre los dos grupos de tratamiento.

No se considera el grupo de control.

## Hipótesis

$$
H_0: \text{ocupación y grupo de tratamiento son independientes}
$$

$$
H_1: \text{ocupación y grupo de tratamiento no son independientes}
$$

## Prueba usada

Se usa una prueba de chi-cuadrado de independencia.

Los grados de libertad se calculan como:

$$
gl = (r - 1)(c - 1)
$$

donde `r` es el número de filas y `c` el número de columnas.

---

```python
tabla_ocupacion = pd.crosstab(
    tratamientos["Grupo_label"],
    tratamientos["Ocupacion_label"]
)

tabla_ocupacion
```

---

```python
chi_ocupacion_stat, p_ocupacion_value, gl_ocupacion, expected_ocupacion = stats.chi2_contingency(tabla_ocupacion)
chi_ocupacion_critico = chi2.ppf(1 - ALPHA, gl_ocupacion)

frecuencias_esperadas_ocupacion = pd.DataFrame(
    expected_ocupacion,
    index=tabla_ocupacion.index,
    columns=tabla_ocupacion.columns
)

print("Frecuencias esperadas:")
display(frecuencias_esperadas_ocupacion.round(4))

imprimir_resultado(
    nombre_prueba="Chi-cuadrado de independencia para ocupación",
    estadistico=chi_ocupacion_stat,
    p_value=p_ocupacion_value,
    gl=gl_ocupacion,
    critico=f"{chi_ocupacion_critico:.4f}"
)
```

---

## Conclusión 3.d

El p-value es mayor que 0.05, por lo tanto **no se rechaza H0**.

No existe evidencia estadísticamente significativa para afirmar que la ocupación esté asociada al grupo de tratamiento.

Esto sugiere que CSR Externa y CSR Interna son comparables en ocupación.

---

# 3.e ANOVA para confianza antes y después entre los tres grupos

## Objetivo

Comparar las medias de confianza entre los tres grupos:

- CSR Externa
- CSR Interna
- Info sin CSR

Se hacen dos ANOVA:

1. Confianza antes de recibir la información.
2. Confianza después de recibir la información.

## Hipótesis general del ANOVA

$$
H_0: \mu_1 = \mu_2 = \mu_3
$$

$$
H_1: \text{al menos una media de grupo es diferente}
$$

## Grados de libertad

Para un ANOVA de una vía:

$$
gl_{\text{entre}} = k - 1
$$

$$
gl_{\text{dentro}} = N - k
$$

donde `k` es el número de grupos y `N` es el total de observaciones.

## Post hoc

Si el ANOVA es significativo, se aplica Tukey HSD para identificar qué pares de grupos difieren.

---

```python
def anova_una_via(variable):
    grupos = []
    
    print("\n" + "=" * 80)
    print(f"ANOVA para: {variable}")
    print("=" * 80)
    
    for codigo_grupo, nombre_grupo in labels_grupo.items():
        valores = df.loc[df["Grupo"] == codigo_grupo, variable].dropna()
        grupos.append(valores)
        print(
            f"{nombre_grupo}: "
            f"n={len(valores)}, media={valores.mean():.4f}, sd={valores.std(ddof=1):.4f}"
        )
    
    f_stat, p_value = stats.f_oneway(*grupos)
    
    k = len(grupos)
    N = sum(len(g) for g in grupos)
    
    gl_entre = k - 1
    gl_dentro = N - k
    
    f_critico = f.ppf(1 - ALPHA, gl_entre, gl_dentro)
    
    imprimir_resultado(
        nombre_prueba=f"ANOVA de una vía para {variable}",
        estadistico=f_stat,
        p_value=p_value,
        gl=f"{gl_entre}, {gl_dentro}",
        critico=f"{f_critico:.4f}"
    )
    
    tukey_resultado = None
    
    if p_value < ALPHA:
        print("\nPost hoc Tukey HSD:")
        temp = df[["Grupo_label", variable]].dropna()
        tukey_resultado = pairwise_tukeyhsd(
            endog=temp[variable],
            groups=temp["Grupo_label"],
            alpha=ALPHA
        )
        print(tukey_resultado)
    else:
        print("\nNo se aplica Tukey HSD porque el ANOVA no es significativo.")
    
    return {
        "variable": variable,
        "F": f_stat,
        "gl_entre": gl_entre,
        "gl_dentro": gl_dentro,
        "F_critico": f_critico,
        "p_value": p_value,
        "decision": decision_hipotesis(p_value),
        "tukey": tukey_resultado
    }
```

---

```python
resultado_anova_antes = anova_una_via("Confianza_antes")
resultado_anova_despues = anova_una_via("Confianza_después")

tabla_anova_confianza = pd.DataFrame([
    {k: v for k, v in resultado_anova_antes.items() if k != "tukey"},
    {k: v for k, v in resultado_anova_despues.items() if k != "tukey"}
])

tabla_anova_confianza.round(4)
```

---

## Conclusión 3.e

Para **Confianza antes**, el p-value es mayor que 0.05, por lo tanto no se rechaza H0.

Esto indica que antes de la intervención no existen diferencias estadísticamente significativas en confianza entre los tres grupos. En términos experimentales, esto es importante porque sugiere que los grupos partían desde niveles comparables de confianza.

Para **Confianza después**, el p-value es menor que 0.05, por lo tanto se rechaza H0.

Esto indica que después de recibir la información existen diferencias significativas de confianza entre los grupos.

La prueba post hoc Tukey muestra que:

- CSR Externa difiere significativamente del grupo Info sin CSR;
- CSR Interna difiere significativamente del grupo Info sin CSR;
- CSR Externa y CSR Interna no difieren significativamente entre sí al 5%.

En términos sustantivos, los grupos que recibieron información de CSR presentan mayor confianza posterior que el grupo de control.

---

# 3.f Chi-cuadrado para ingresos en los tres grupos

## Objetivo

Evaluar si la distribución de ingresos difiere entre los tres grupos.

## Hipótesis

$$
H_0: \text{ingresos y grupo son independientes}
$$

$$
H_1: \text{ingresos y grupo no son independientes}
$$

## Prueba usada

Se usa una prueba de chi-cuadrado de independencia.

Los grados de libertad se calculan como:

$$
gl = (r - 1)(c - 1)
$$

---

```python
tabla_ingresos = pd.crosstab(
    df["Grupo_label"],
    df["Ingresos_label"]
)

tabla_ingresos
```

---

```python
chi_ingresos_stat, p_ingresos_value, gl_ingresos, expected_ingresos = stats.chi2_contingency(tabla_ingresos)
chi_ingresos_critico = chi2.ppf(1 - ALPHA, gl_ingresos)

frecuencias_esperadas_ingresos = pd.DataFrame(
    expected_ingresos,
    index=tabla_ingresos.index,
    columns=tabla_ingresos.columns
)

print("Frecuencias esperadas:")
display(frecuencias_esperadas_ingresos.round(4))

imprimir_resultado(
    nombre_prueba="Chi-cuadrado de independencia para ingresos",
    estadistico=chi_ingresos_stat,
    p_value=p_ingresos_value,
    gl=gl_ingresos,
    critico=f"{chi_ingresos_critico:.4f}"
)
```

---

## Conclusión 3.f

El p-value es mayor que 0.05, por lo tanto **no se rechaza H0**.

No existe evidencia estadísticamente significativa para afirmar que la distribución de ingresos difiera entre los tres grupos.

Esto sugiere que los grupos son comparables en ingresos.

---

# 3.g ANOVA para cambio en confianza entre los tres grupos

## Objetivo

Comparar si el cambio promedio en confianza difiere entre los tres grupos.

La variable analizada es:

$$
\text{Cambio confianza} = \text{Confianza después} - \text{Confianza antes}
$$

## Hipótesis

$$
H_0: \mu_{\Delta,1} = \mu_{\Delta,2} = \mu_{\Delta,3}
$$

$$
H_1: \text{al menos un grupo tiene un cambio promedio diferente}
$$

## Prueba usada

Se usa ANOVA de una vía.

Si el ANOVA es significativo, se aplica Tukey HSD.

---

```python
resultado_anova_cambio = anova_una_via("Cambio_confianza")

tabla_anova_cambio = pd.DataFrame([
    {k: v for k, v in resultado_anova_cambio.items() if k != "tukey"}
])

tabla_anova_cambio.round(4)
```

---

## Conclusión 3.g

El p-value es menor que 0.05, por lo tanto **se rechaza H0**.

Esto indica que el cambio promedio en confianza difiere significativamente entre los tres grupos.

La prueba post hoc Tukey muestra que los tres pares de grupos difieren significativamente entre sí:

- CSR Externa difiere de CSR Interna;
- CSR Externa difiere de Info sin CSR;
- CSR Interna difiere de Info sin CSR.

En términos sustantivos, la información de CSR Externa produce el mayor aumento promedio en confianza, seguida por CSR Interna. El grupo de control presenta el menor cambio.

Este resultado es central para la interpretación del experimento, porque muestra que no solo cambia la confianza dentro de algunos grupos, sino que el tamaño del cambio depende del tipo de información recibida.

---

# 9. Tabla resumen final

Esta tabla resume las pruebas solicitadas en la pregunta 3, con estadístico, grados de libertad, p-value y decisión.

---

```python
filas_resumen = []

filas_resumen.append({
    "pregunta": "3.a",
    "prueba": "t de Welch",
    "comparacion": "Edad: CSR Externa vs CSR Interna",
    "estadistico": "t",
    "valor": t_edad_stat,
    "gl": round(gl_edad_welch, 4),
    "critico_95": f"±{t_edad_critico:.4f}",
    "p_value": p_edad_value,
    "decision": decision_hipotesis(p_edad_value)
})

for fila in resultados_t_pareada:
    filas_resumen.append({
        "pregunta": "3.b",
        "prueba": "t pareada",
        "comparacion": f"Confianza antes vs después: {fila['grupo']}",
        "estadistico": "t",
        "valor": fila["t"],
        "gl": fila["gl"],
        "critico_95": f"±{fila['t_critico']:.4f}",
        "p_value": fila["p_value"],
        "decision": fila["decision"]
    })

filas_resumen.append({
    "pregunta": "3.c",
    "prueba": "Z proporciones",
    "comparacion": "Género femenino: CSR Externa vs CSR Interna",
    "estadistico": "Z",
    "valor": z_genero_stat,
    "gl": "",
    "critico_95": f"±{z_genero_critico:.4f}",
    "p_value": p_genero_value,
    "decision": decision_hipotesis(p_genero_value)
})

filas_resumen.append({
    "pregunta": "3.d",
    "prueba": "Chi-cuadrado",
    "comparacion": "Ocupación entre tratamientos",
    "estadistico": "Chi2",
    "valor": chi_ocupacion_stat,
    "gl": gl_ocupacion,
    "critico_95": f"{chi_ocupacion_critico:.4f}",
    "p_value": p_ocupacion_value,
    "decision": decision_hipotesis(p_ocupacion_value)
})

for pregunta, resultado in [("3.e", resultado_anova_antes), ("3.e", resultado_anova_despues), ("3.g", resultado_anova_cambio)]:
    filas_resumen.append({
        "pregunta": pregunta,
        "prueba": "ANOVA",
        "comparacion": resultado["variable"],
        "estadistico": "F",
        "valor": resultado["F"],
        "gl": f"{resultado['gl_entre']}, {resultado['gl_dentro']}",
        "critico_95": f"{resultado['F_critico']:.4f}",
        "p_value": resultado["p_value"],
        "decision": resultado["decision"]
    })

filas_resumen.append({
    "pregunta": "3.f",
    "prueba": "Chi-cuadrado",
    "comparacion": "Ingresos entre los tres grupos",
    "estadistico": "Chi2",
    "valor": chi_ingresos_stat,
    "gl": gl_ingresos,
    "critico_95": f"{chi_ingresos_critico:.4f}",
    "p_value": p_ingresos_value,
    "decision": decision_hipotesis(p_ingresos_value)
})

tabla_resumen = pd.DataFrame(filas_resumen)
tabla_resumen.round(4)
```

---

# 10. Conclusión general de la pregunta 3

Los resultados indican que los grupos de tratamiento son comparables en edad, género, ocupación e ingresos, ya que no se detectan diferencias estadísticamente significativas en estas variables.

Además, antes de la intervención no existen diferencias significativas en confianza entre los tres grupos, lo cual sugiere que partían de niveles similares.

Después de recibir información, sí aparecen diferencias significativas entre los grupos. Tanto CSR Externa como CSR Interna presentan mayores niveles de confianza posterior que el grupo de control.

El análisis de cambio en confianza muestra que la información recibida afecta de manera distinta a los grupos. CSR Externa genera el mayor aumento promedio de confianza, seguida por CSR Interna, mientras que el grupo Info sin CSR presenta el menor cambio.

En conjunto, la evidencia sugiere que la información sobre CSR aumenta la confianza en la empresa, y que la información de CSR Externa tiene el efecto promedio más fuerte en esta base.

---

# 11. Exportar resultados

La siguiente celda exporta las tablas principales a Excel.

---

```python
tabla_descriptivos = df.groupby("Grupo_label").agg(
    n=("ID", "count"),
    edad_media=("Edad", "mean"),
    edad_sd=("Edad", "std"),
    confianza_antes_media=("Confianza_antes", "mean"),
    confianza_antes_sd=("Confianza_antes", "std"),
    confianza_despues_media=("Confianza_después", "mean"),
    confianza_despues_sd=("Confianza_después", "std"),
    cambio_confianza_media=("Cambio_confianza", "mean"),
    cambio_confianza_sd=("Cambio_confianza", "std")
).reset_index()

nombre_excel = "resultados_pregunta_3_tarea_1.xlsx"

with pd.ExcelWriter(nombre_excel, engine="openpyxl") as writer:
    df.to_excel(writer, sheet_name="base_con_cambio", index=False)
    tabla_descriptivos.to_excel(writer, sheet_name="descriptivos", index=False)
    tabla_t_pareada.to_excel(writer, sheet_name="t_pareada", index=False)
    tabla_anova_confianza.to_excel(writer, sheet_name="anova_confianza", index=False)
    tabla_anova_cambio.to_excel(writer, sheet_name="anova_cambio", index=False)
    tabla_ocupacion.to_excel(writer, sheet_name="chi_ocupacion")
    tabla_ingresos.to_excel(writer, sheet_name="chi_ingresos")
    tabla_resumen.to_excel(writer, sheet_name="resumen_final", index=False)

print(f"Archivo exportado: {nombre_excel}")
```
