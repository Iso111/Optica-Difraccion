"""
Código 20 — Patrón analítico de difracción de Fraunhofer 2D.

Punto 1 del Parcial 4: abertura COMPUESTA formada por dos aberturas separadas una
distancia D (ver figura del enunciado):

    · Marco rectangular  = rectángulo exterior (a × b)  MENOS  rectángulo interior
      (c × d) centrado  → una "ventana".
    · Círculo sólido de radio R.

La amplitud de Fraunhofer es ∝ la Transformada de Fourier espacial de la transmisión
de la abertura. Con  fx = x'/(λz),  fy = y'/(λz),  ρ = √(fx²+fy²):

    A_marco  = a·b·sinc(a·fx)·sinc(b·fy) − c·d·sinc(c·fx)·sinc(d·fy)
    A_círc   = πR² · 2·J₁(2πRρ)/(2πRρ)            (disco de Airy)

Las dos aberturas, centradas en x = ∓D/2, introducen por el teorema de desplazamiento
una diferencia de fase 2πD·fx, de modo que la INTENSIDAD analítica es:

    I(x',y') = A_marco² + A_círc² + 2·A_marco·A_círc·cos(2πD·fx)

El término coseno son las franjas de interferencia debidas a la separación D,
moduladas por las dos envolventes de difracción.

El programa SIEMPRE calcula Fraunhofer, pero declara explícitamente el régimen
(número de Fresnel N_F = D_char²/λz) para no dar una respuesta falsa en campo cercano;
en ese caso se debe usar el Código 22 (Fresnel).

GUI interactivo (tkinter) al estilo del simulador del interferómetro de Michelson.
"""

import tkinter as tk
from tkinter import ttk

import numpy as np
from scipy.special import j1
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.patches import Rectangle, Circle
from matplotlib.colors import LogNorm, PowerNorm


# Umbral del número de Fresnel para considerar válida la aproximación de Fraunhofer.
# Equivale al criterio z ≥ 2·D_char²/λ  (N_F = D_char²/λz ≤ 0.5).
NF_UMBRAL = 0.5


# =============================================================================
# 1. NÚCLEO FÍSICO  (funciones puras, sin GUI)
# =============================================================================

def amplitud_marco(fx, fy, a, b, c, d):
    """
    Amplitud de Fraunhofer del marco rectangular = rectángulo exterior (a×b)
    menos rectángulo interior (c×d), ambos centrados.

    ∫∫ rect e^{-i2π(fx·x+fy·y)} dx dy = area · sinc(a·fx) · sinc(b·fy),
    con la sinc normalizada de numpy: sinc(u)=sin(πu)/(πu).
    """
    exterior = a * b * np.sinc(a * fx) * np.sinc(b * fy)
    interior = c * d * np.sinc(c * fx) * np.sinc(d * fy)
    return exterior - interior


def amplitud_circulo(rho, R):
    """
    Amplitud de Fraunhofer de un círculo de radio R (disco de Airy):

        A = πR² · 2·J₁(2πRρ)/(2πRρ),   con límite πR² cuando ρ→0.
    """
    if R <= 0.0:
        return np.zeros_like(rho)
    x = 2.0 * np.pi * R * rho
    area = np.pi * R ** 2
    # np.where evita la división 0/0 en el origen; el límite es 1 → A = πR².
    factor = np.where(x == 0.0, 1.0, 2.0 * j1(x) / np.where(x == 0.0, 1.0, x))
    return area * factor


def intensidad(X, Y, p):
    """
    Intensidad analítica normalizada del patrón compuesto sobre la malla (X, Y)
    del plano de observación [m]. `p` es el dict de parámetros en unidades SI.

        I = A_marco² + A_círc² + 2·A_marco·A_círc·cos(2πD·fx)

    Normalizada por I0 = (a·b − c·d + πR²)²  (valor en el origen, x'=y'=0).
    """
    lz = p["lam"] * p["z"]
    fx = X / lz
    fy = Y / lz
    rho = np.hypot(fx, fy)

    A_m = amplitud_marco(fx, fy, p["a"], p["b"], p["c"], p["d"])
    A_c = amplitud_circulo(rho, p["R"])

    I = A_m ** 2 + A_c ** 2 + 2.0 * A_m * A_c * np.cos(2.0 * np.pi * p["D"] * fx)

    A0 = p["a"] * p["b"] - p["c"] * p["d"] + np.pi * p["R"] ** 2
    I0 = A0 ** 2
    if I0 > 0.0:
        I = I / I0
    return np.clip(I, 0.0, None)


def d_caracteristica(p):
    """
    Mayor extensión lineal de la estructura completa = diagonal del bounding box.
    span_x abarca desde el borde izquierdo del marco hasta el borde derecho del
    círculo (aberturas centradas en ∓D/2); el alto es max(b, 2R).
    """
    span_x = p["D"] + p["a"] / 2.0 + p["R"]
    alto = max(p["b"], 2.0 * p["R"])
    return float(np.hypot(span_x, alto))


def regimen(p):
    """
    Devuelve (D_char, z_min, N_F, es_fraunhofer) para el plano de observación.
        z_min = 2·D_char²/λ      (criterio de Fraunhofer)
        N_F   = D_char²/(λ·z)    (número de Fresnel)
    Es Fraunhofer si N_F ≤ NF_UMBRAL  (equivalente a z ≥ z_min).
    """
    D_char = d_caracteristica(p)
    z_min = 2.0 * D_char ** 2 / p["lam"]
    N_F = D_char ** 2 / (p["lam"] * p["z"])
    return D_char, z_min, N_F, (N_F <= NF_UMBRAL)


# =============================================================================
# 1B. NÚCLEO FÍSICO — EJERCICIOS DE TALLER (pestañas 2-5)
# =============================================================================
#
# Cuatro aberturas de ejercicios de taller, cada una con su propia sección de
# amplitud/intensidad. A diferencia de la Pestaña 0 (que normaliza la
# intensidad para que I(0,0)/I0=1, útil solo para ver la FORMA del patrón),
# aquí se sigue la MISMA convención analítica (amplitud = integral de área,
# independiente de z/λ) salvo en "Círculo con muesca", donde el enunciado pide
# un VALOR ABSOLUTO de irradiancia que sí depende de z — ahí se usa la fórmula
# completa I(0,0) = I0·(Área/λz)², sin normalizar esa dependencia.

def regimen_generico(D_char, lam, z):
    """
    Igual que `regimen()` de la Pestaña 0, pero para un D_char ya calculado
    (evita repetir la construcción del dict de parámetros de esa pestaña).
    Devuelve (z_min, N_F, es_fraunhofer).
    """
    z_min = 2.0 * D_char ** 2 / lam
    N_F = D_char ** 2 / (lam * z)
    return z_min, N_F, (N_F <= NF_UMBRAL)


# ---- Rectángulo rotado (rectángulo a×b girado un ángulo θ en el plano) -----

def amplitud_rectangulo_rotado(fx, fy, a, b, theta):
    """
    TF de Fraunhofer de un rectángulo de lados a×b ROTADO un ángulo θ en el
    plano de la abertura (no es un paralelogramo cizallado: los lados siguen a
    90°). Rotar la abertura un ángulo θ equivale a rotar el plano de
    frecuencias el mismo ángulo — la TF de f(R_θ·r) es F(R_θ·k):

        f_a =  fx·cosθ + fy·senθ      (frecuencia a lo largo del lado a)
        f_b = -fx·senθ + fy·cosθ      (frecuencia a lo largo del lado b)
        A   = a·b · sinc(a·f_a) · sinc(b·f_b)

    En θ=0 se reduce al rectángulo alineado (a en x, b en y). El patrón sinc²
    del rectángulo simplemente GIRA el mismo ángulo θ que la abertura.
    """
    c, s = np.cos(theta), np.sin(theta)
    f_a = fx * c + fy * s
    f_b = -fx * s + fy * c
    return a * b * np.sinc(a * f_a) * np.sinc(b * f_b)


def intensidad_rectangulo_rotado(X, Y, a, b, theta, lam, z):
    fx = X / (lam * z)
    fy = Y / (lam * z)
    A = amplitud_rectangulo_rotado(fx, fy, a, b, theta)
    I0 = (a * b) ** 2
    I = A ** 2 / I0 if I0 > 0.0 else A ** 2
    return np.clip(I, 0.0, None)


# ---- Cruz (unión de dos barras — inclusión-exclusión de conjuntos) --------

def amplitud_cruz(fx, fy, L, a):
    """
    Cruz = barra horizontal (L×a) UNIÓN barra vertical (a×L), ambas centradas
    en el origen y compartiendo el cuadrado central a×a. El indicador de la
    unión es exactamente  1_h + 1_v − 1_(h∩v)  (para no contar dos veces el
    solape), y como la TF es lineal, la amplitud de la cruz es la misma
    combinación de las TF de los tres rectángulos.

    Validación: si a=L (los brazos llenan todo el largo), la cruz degenera en
    un cuadrado sólido L×L y A_cruz se reduce algebraicamente a
    L²·sinc(Lfx)·sinc(Lfy).
    """
    A_h = L * a * np.sinc(L * fx) * np.sinc(a * fy)
    A_v = a * L * np.sinc(a * fx) * np.sinc(L * fy)
    A_ov = a * a * np.sinc(a * fx) * np.sinc(a * fy)
    return A_h + A_v - A_ov


def intensidad_cruz(X, Y, L, a, lam, z):
    fx = X / (lam * z)
    fy = Y / (lam * z)
    A = amplitud_cruz(fx, fy, L, a)
    area = 2.0 * L * a - a * a          # área real de la cruz (sin doble conteo)
    I0 = area ** 2
    I = A ** 2 / I0 if I0 > 0.0 else A ** 2
    return np.clip(I, 0.0, None)


# ---- Dos semicírculos (mitad superior r1, mitad inferior r2) ---------------

def area_dos_semicirculos(r1, r2):
    """
    Área de la abertura formada por un semicírculo SUPERIOR (y≥0) de radio r1
    unido a un semicírculo INFERIOR (y<0) de radio r2, compartiendo el diámetro
    horizontal:

        Área = π·r1²/2 + π·r2²/2 = π·(r1² + r2²)/2

    Casos límite:  r1=r2 → círculo completo (π·r²);  r2=0 → un solo semicírculo.
    """
    return np.pi * (r1 ** 2 + r2 ** 2) / 2.0


def irradiancia_axial_relativa(area, lam, z):
    """
    I(0,0)/I0 en el eje óptico (x'=y'=0): en ese punto fx=fy=0 y toda
    sinc/Bessel de la abertura vale 1, así que la integral de difracción se
    reduce al ÁREA TOTAL de la abertura:

        I(0,0) = I0 · (Área/(λz))²

    A diferencia de la normalización de la Pestaña 0 (que oculta el factor
    1/(λz)² porque solo le importa la FORMA del patrón), aquí se necesita el
    valor absoluto pedido por el enunciado, así que el factor (1/λz)² se deja
    explícito.
    """
    return (area / (lam * z)) ** 2


def mascara_dos_semicirculos(X, Y, r1, r2):
    """
    Máscara booleana (True=transparente) de la abertura de dos semicírculos:
    semicírculo superior (y≥0) de radio r1 y semicírculo inferior (y<0) de
    radio r2, sobre la malla (X,Y) del PLANO DE LA ABERTURA [m]. Se usa solo
    para la visualización NUMÉRICA del patrón 2D (FFT); el valor axial cerrado
    se calcula aparte con `area_dos_semicirculos` (método analítico).
    """
    superior = (Y >= 0.0) & (X ** 2 + Y ** 2 <= r1 ** 2)
    inferior = (Y < 0.0) & (X ** 2 + Y ** 2 <= r2 ** 2)
    return superior | inferior


RESOLUCION_ABERTURA = 60  # muestras deseadas a través de max(r1, r2)


def patron_fft_semicirculos(r1, r2, lam, z, xmax, N):
    """
    Patrón de Fraunhofer NUMÉRICO (no analítico) de la abertura de dos
    semicírculos, vía `fft2`.

    La resolución en el plano de OBSERVACIÓN es  dx' = λz/L_ap  — depende
    solo de la ventana física de la abertura `L_ap`, NO de N. Para que el
    slider N sirva realmente para "ver más fino" (en vez de solo extender el
    rango), se fija primero una resolución de muestreo de la ABERTURA
    (dx_ap = max(r1,r2)/RESOLUCION_ABERTURA) y se hace crecer la ventana
    con N:  L_ap = N·dx_ap  (equivale a "zero-padding": más N ⇒ ventana más
    grande ⇒ dx' más fino, la interpolación estándar de la FFT). Se impone
    además un piso  L_ap ≥ 4×max(r1,r2)  para no recortar la abertura
    cuando N está en su valor mínimo.

    El slider `xmax` solo RECORTA el rango ya calculado — pedir un `xmax`
    mayor al disponible simplemente se satura al máximo alcanzable (nunca se
    inventa una extensión mayor a la realmente calculada).

    Devuelve (x_obs [m] recortado, I2D_rel recortado, máscara, x_ap [m]).
    """
    r_max = max(r1, r2)
    dx_ap_objetivo = r_max / RESOLUCION_ABERTURA
    margen_min = 4.0 * r_max
    L_ap = max(margen_min, N * dx_ap_objetivo)

    x_ap = np.linspace(-L_ap / 2, L_ap / 2, N, endpoint=False)
    dx_ap = x_ap[1] - x_ap[0]
    Xap, Yap = np.meshgrid(x_ap, x_ap)
    mascara = mascara_dos_semicirculos(Xap, Yap, r1, r2).astype(float)

    U = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(mascara)))
    U *= dx_ap * dx_ap / (lam * z)          # escala de Riemann + prefactor 1/(λz)
    I2D = np.abs(U) ** 2                    # = I/I0 (I0 se deja implícito, =1)

    freq = np.fft.fftshift(np.fft.fftfreq(N, d=dx_ap))
    x_obs = freq * lam * z

    # Recorte a la ventana de visualización solicitada (saturada al rango
    # realmente disponible: (N/2)·dx').
    xmax_efectivo = min(xmax, x_obs[-1])
    dentro = np.abs(x_obs) <= xmax_efectivo
    idx = np.where(dentro)[0]
    x_obs_c = x_obs[idx]
    I2D_c = I2D[np.ix_(idx, idx)]
    return x_obs_c, I2D_c, mascara, x_ap


# ---- Doble cuadrado (lados a y 3a, separados 4a centro-a-centro) ----------

def intensidad_doble_cuadrado(X, Y, a, lam, z):
    """
    Cuadrado pequeño (lado a) y grande (lado 3a), separados 2a borde-a-borde
    en la figura del enunciado → separación centro-a-centro
    D = a/2 + 2a + 3a/2 = 4a (geométrica, no es un parámetro libre).

    Misma estructura de "dos aberturas + interferencia" que la Pestaña 0
    (teorema de desplazamiento), escrita de nuevo aquí para no tocar
    `intensidad()` ya validada (CLAUDE.md: cambios quirúrgicos).

    Validación universal: I(0,0) = (Área_total)² = (a²+9a²)² = (10a²)².
    """
    D = 4.0 * a
    fx = X / (lam * z)
    fy = Y / (lam * z)
    A1 = a * a * np.sinc(a * fx) * np.sinc(a * fy)
    A2 = (3.0 * a) ** 2 * np.sinc(3.0 * a * fx) * np.sinc(3.0 * a * fy)
    I = A1 ** 2 + A2 ** 2 + 2.0 * A1 * A2 * np.cos(2.0 * np.pi * D * fx)
    area_total = a ** 2 + (3.0 * a) ** 2
    I0 = area_total ** 2
    I = I / I0 if I0 > 0.0 else I
    return np.clip(I, 0.0, None), D


# ---- Rendija(s): 1 a N ranuras (rendija simple y red de difracción) --------

def intensidad_rendijas(sin_theta, a_lam, d_lam, N):
    """
    Patrón de Fraunhofer de N rendijas verticales idénticas (idealizadas como
    infinitas en y), de ancho de ranura a y período d (separación centro-a-
    centro), en función de sinθ (adimensional). Todo se parametriza por los
    cocientes a/λ y d/λ, así el mismo modelo sirve para LUZ o SONIDO sin
    importar la escala absoluta:

        u = (a/λ)·sinθ                              (ancho en longitudes de onda)
        v = (d/λ)·sinθ                              (período en longitudes de onda)
        I = sinc²(u) · [ sin(Nπv) / (N·sin(πv)) ]²

    · Primer factor: envolvente de UNA rendija → mínimos en sinθ = m·λ/a.
    · Segundo factor: factor de red → máximos principales en sinθ = m·λ/d
      (orden m). Para N=1 vale 1 y queda la rendija simple.

    Normalizada a I(0)=1.
    """
    u = a_lam * sin_theta
    env = np.sinc(u) ** 2                        # sinc normalizada de numpy
    if N <= 1:
        return np.clip(env, 0.0, None)
    v = d_lam * sin_theta
    num = np.sin(N * np.pi * v)
    den = N * np.sin(np.pi * v)
    # En los máximos principales den→0 y el cociente → ±1 (límite de L'Hôpital).
    den_safe = np.where(np.abs(den) < 1e-9, 1.0, den)
    red = np.where(np.abs(den) < 1e-9, 1.0, (num / den_safe) ** 2)
    return np.clip(env * red, 0.0, None)


def angulos_minimos_rendija(a_lam, m_max=3):
    """Ángulos (grados) de los primeros mínimos de una rendija: sinθ=m·λ/a."""
    angs = []
    for m in range(1, m_max + 1):
        s = m / a_lam
        if s <= 1.0:
            angs.append((m, np.degrees(np.arcsin(s))))
    return angs


def ordenes_red(d_lam, sin_max=1.0, m_max=20):
    """Órdenes principales visibles de la red: sinθ=m·λ/d con |sinθ|≤sin_max."""
    ordenes = []
    for m in range(0, m_max + 1):
        s = m / d_lam
        if s <= sin_max:
            ordenes.append((m, np.degrees(np.arcsin(s))))
        else:
            break
    return ordenes


# ---- Escalón de Michelson (pto 15): red de N peldaños de vidrio -------------

def intensidad_escalon(sin_theta, s, h, n, lam, N):
    """
    Patrón de Fraunhofer del escalón de Michelson: una "escalera" de N láminas
    de vidrio (índice n, espesor h, saliente de ancho s). Cada peldaño actúa
    como una rendija de ancho s; entre peldaños vecinos el desfase de camino
    óptico tiene dos aportes:

        · geométrico (como red normal):  s·senθ
        · vidrio extra (un espesor h más por peldaño):  (n−1)·h
        ⇒  Δ = (n−1)·h + s·senθ

        I(θ) = sinc²(s·senθ/λ) · [ sin(Nπ·Δ/λ) / (N·sin(π·Δ/λ)) ]²

    · sinc²: envolvente de difracción de UN peldaño (mínimos en s·senθ=mλ).
    · factor de red: máximos principales en Δ=mλ. El término (n−1)h (≈mm,
      constante) empuja los órdenes a m enormes → altísima dispersión.

    El lóbulo central de la envolvente abarca u=s·senθ/λ ∈ (−1,1), o sea Δu=2 =
    2 espaciados de orden → caben ≈2 máximos principales por máximo de
    difracción (según el desfase fraccional (n−1)h/λ se ven 1 ó 2). Para N=1
    se reduce a la envolvente sinc². Normalizada a la envolvente en θ=0.
    """
    u = s * sin_theta / lam
    env = np.sinc(u) ** 2
    if N <= 1:
        return np.clip(env, 0.0, None)
    Delta = (n - 1.0) * h + s * sin_theta
    v = Delta / lam
    num = np.sin(N * np.pi * v)
    den = N * np.sin(np.pi * v)
    den_safe = np.where(np.abs(den) < 1e-9, 1.0, den)
    red = np.where(np.abs(den) < 1e-9, 1.0, (num / den_safe) ** 2)
    return np.clip(env * red, 0.0, None)


# ---- Doble círculo / zonas de Fresnel (pto 19) -----------------------------

def mascara_doble_circulo(X, Y, r1, r2):
    """
    Máscara (True=transparente) de la abertura del pto 19: disco de radio r1 en
    3 cuadrantes, extendido a r2 SOLO en el cuadrante superior-derecho (x≥0,
    y≥0). Con z=2m y λ=500nm: r1=√(λz)=1mm (1ª zona de Fresnel) y r2=√(2λz)=
    1.414mm (2ª zona) → toda la 1ª zona + ¼ de la 2ª zona.
    """
    q_sup_der = (X >= 0.0) & (Y >= 0.0) & (X ** 2 + Y ** 2 <= r2 ** 2)
    resto = (X ** 2 + Y ** 2 <= r1 ** 2)
    return q_sup_der | resto


def area_doble_circulo(r1, r2):
    """Área de la abertura del pto 19: ¾ de disco r1 + ¼ de disco r2."""
    return 0.75 * np.pi * r1 ** 2 + 0.25 * np.pi * r2 ** 2


def fresnel_propagate(U0, dx, lam, z):
    """
    Difracción de Fresnel de campo cercano por FFT único (motor reutilizable
    por el Código 22). Propaga el campo complejo `U0` (malla N×N, paso `dx` [m])
    una distancia `z` [m] en la aproximación de Fresnel:

        U(x') = 1/(iλz) · e^{iπx'²/λz} · 𝓕{ U0(x)·e^{iπx²/λz} }

    El paso en el plano de observación es dx' = λz/(N·dx). Válido mientras el
    "chirp" cuadrático esté bien muestreado: L=N·dx < √(N·λz). Con onda plana
    incidente de amplitud 1, el campo queda normalizado (una abertura de una
    zona de Fresnel da |U|=2 en el eje, I=4; la abertura del pto 19 da |U|=1.5,
    I=2.25 — validado headless).

    Devuelve (U complejo en el plano de obs, dx2 paso de obs [m]).
    """
    N = U0.shape[0]
    x = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(x, x)
    Q1 = np.exp(1j * np.pi / (lam * z) * (X ** 2 + Y ** 2))
    A = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(U0 * Q1)))
    dx2 = lam * z / (N * dx)
    x2 = (np.arange(N) - N // 2) * dx2
    X2, Y2 = np.meshgrid(x2, x2)
    Q2 = np.exp(1j * np.pi / (lam * z) * (X2 ** 2 + Y2 ** 2))
    U = (1.0 / (1j * lam * z)) * Q2 * A * dx * dx
    return U, dx2


def patrones_doble_circulo(r1, r2, lam, z, N):
    """
    Calcula ambos patrones 2D de la abertura del pto 19 sobre la MISMA máscara
    rasterizada: Fraunhofer (campo lejano, |𝓕|²) y Fresnel (campo cercano, vía
    `fresnel_propagate`). Devuelve un dict con máscara, ejes y ambos patrones
    normalizados a la intensidad incidente I (onda plana de amplitud 1).

    La ventana de la abertura se elige L = min(6·r2, 0.8·√(Nλz)) para (a)
    contener la abertura con margen y (b) respetar el muestreo del chirp de
    Fresnel. dx_obs difiere entre ambos métodos: Fresnel usa λz/(N·dx);
    Fraunhofer usa el mismo mapeo (idéntico dx', ya que es el mismo FFT-único
    con z→∞ en el chirp de salida).
    """
    r_max = max(r1, r2)
    L = min(6.0 * r_max, 0.8 * np.sqrt(N * lam * z))
    x = (np.arange(N) - N // 2) * (L / N)
    dx = L / N
    X, Y = np.meshgrid(x, x)
    mascara = mascara_doble_circulo(X, Y, r1, r2).astype(complex)

    # Fresnel (campo cercano)
    U_fr, dx2 = fresnel_propagate(mascara, dx, lam, z)
    I_fresnel = np.abs(U_fr) ** 2

    # Fraunhofer (campo lejano): misma TF pero sin el chirp de salida.
    A = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(mascara)))
    U_fh = A * dx * dx / (lam * z)          # amplitud ∝ TF, mismo mapeo dx2
    I_fraunhofer = np.abs(U_fh) ** 2         # I/I0 con I0=(área/λz)² en el eje

    x_obs = (np.arange(N) - N // 2) * dx2
    return {
        "mascara": mascara.real, "x_ap": x, "x_obs": x_obs,
        "I_fresnel": I_fresnel, "I_fraunhofer": I_fraunhofer,
        "U_fresnel_axial": U_fr[N // 2, N // 2],
    }


# ---- Dos redes de difracción en cascada, con desplazamiento (pto 6) ---------

def transmision_redes(x, a, d, N, s):
    """
    Transmisión t₁(x)·t₂(x−s) de dos redes binarias idénticas apiladas: cada
    red tiene N ranuras de ancho `a` y período `d` (=3a por el enunciado),
    centradas en el origen. La red 2 está desplazada `s`. Como el producto de
    dos binarias solo transmite donde AMBAS abren, el resultado es de nuevo una
    red binaria: alineadas (s=0) → t·t=t (una sola red); al desplazar, el
    solape de las ranuras se estrecha y desaparece por tramos.
    """
    centros = (np.arange(N) - (N - 1) / 2.0) * d
    g1 = np.zeros_like(x)
    g2 = np.zeros_like(x)
    for c in centros:
        g1 += (x >= c - a / 2) & (x <= c + a / 2)
        g2 += (x >= c - a / 2 + s) & (x <= c + a / 2 + s)
    return g1 * g2


def patron_redes_cascada(a, N, s, lam, M=16384, pad=4.0):
    """
    Patrón de Fraunhofer 1D de las dos redes en cascada (pto 6). Rasteriza la
    transmisión t₁·t₂(−s) y hace FFT 1D → I(senθ). Período fijo d=3a (ancho de
    ranura a, hueco 2a). El slider de desplazamiento llega hasta s=N·d = 3aN,
    donde las dos redes finitas dejan de solaparse (campo oscuro total).

    Normalizada al pico central (orden 0) de la configuración alineada (s=0),
    de modo que la CAÍDA de intensidad al desplazar es visible: s=d → red de
    N−1 ranuras (factor ((N−1)/N)²), s∈[a,2a] (mod d) → oscuro, s=Nd → 0.

    Devuelve (sen_theta, I_norm, d, W=3aN).
    """
    d = 3.0 * a
    W = N * d
    L = pad * (W + s + d)
    x = (np.arange(M) - M // 2) * (L / M)
    dx = L / M
    t = transmision_redes(x, a, d, N, s)
    F = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(t)))
    f = np.fft.fftshift(np.fft.fftfreq(M, dx))
    sen_theta = lam * f
    ref = (N * a / dx) ** 2                      # pico central en s=0
    I = np.abs(F) ** 2 / ref if ref > 0 else np.abs(F) ** 2
    return sen_theta, I, d, W


# =============================================================================
# 2. INTERFAZ GRÁFICA
# =============================================================================

NX_MAX = 900  # tope de resolución del patrón para mantener fluido el recálculo


def crear_slider(parent, label, frm, to, init, on_change, fmt="{:.2f}"):
    """
    Fila etiqueta + Scale + Entry editable, reutilizable por cualquier pestaña.
    El Entry permite teclear un valor exacto (Enter o perder el foco): se
    valida, se recorta a [frm, to], se mueve el slider y se llama a
    `on_change()`. El slider también llama a `on_change()` al soltar el ratón
    (no en cada píxel del arrastre). Devuelve la DoubleVar asociada.
    """
    row = ttk.Frame(parent)
    row.pack(fill="x", pady=1)
    ttk.Label(row, text=label, width=15).pack(side="left")

    var = tk.DoubleVar(value=init)
    entry_var = tk.StringVar(value=fmt.format(init))
    entry = ttk.Entry(row, textvariable=entry_var, width=8, justify="right")
    entry.pack(side="right")

    def on_move(_=None):
        entry_var.set(fmt.format(var.get()))

    def commit(_=None):
        try:
            v = float(entry_var.get().replace(",", "."))
        except ValueError:
            entry_var.set(fmt.format(var.get()))
            return
        v = max(frm, min(to, v))
        var.set(v)
        entry_var.set(fmt.format(v))
        on_change()

    entry.bind("<Return>", commit)
    entry.bind("<FocusOut>", commit)

    scale = ttk.Scale(row, from_=frm, to=to, variable=var,
                      orient="horizontal", command=on_move)
    scale.pack(side="left", fill="x", expand=True, padx=4)
    scale.bind("<ButtonRelease-1>", lambda _=None: on_change())

    return var


# =============================================================================
# 3. DIBUJO — helpers compartidos por los render de cada abertura
# =============================================================================

def _escala_norm(I, modo):
    """Devuelve (datos, norm) según la escala de visualización elegida
    ('lineal'/'gamma'/'log'). Función libre reutilizada por todos los render y
    también importada por el Código 22."""
    if modo == "log":
        piso = I.max() * 1e-5 if I.max() > 0 else 1e-9
        return I + piso, LogNorm(vmin=piso, vmax=max(I.max(), piso * 10))
    if modo == "gamma":
        return I, PowerNorm(gamma=0.4, vmin=0.0, vmax=max(I.max(), 1e-12))
    return I, None  # lineal


def _status_regimen(D_char, z_min, N_F, es_fh, z, metodo="sinc²"):
    """Texto y color de la caja de régimen (verde=Fraunhofer, rojo=Fresnel).
    Devuelve (txt, color) — el host los escribe en el widget de status."""
    if es_fh:
        color = "#127a12"
        etiqueta = f"Régimen: FRAUNHOFER — cálculo válido ({metodo})"
    else:
        color = "#c00000"
        etiqueta = (f"Régimen: FRESNEL (campo cercano)\n"
                    f"El patrón {metodo} NO es válido aquí.\n"
                    f"→ usar el Código 22 (Fresnel).")
    txt = (
        f"D_char = {D_char*1e6:8.1f} µm\n"
        f"z_min  = {z_min:8.3f} m   (2·D²/λ)\n"
        f"z      = {z:8.3f} m\n"
        f"N_F    = {N_F:8.3f}   (= D²/λz)\n"
        f"─────────────────────────\n"
        f"{etiqueta}"
    )
    return txt, color


# =============================================================================
# 4. RENDER POR ABERTURA
#    Cada función arma su propio gridspec (layout heterogéneo), calcula la
#    física (funciones del núcleo, sin tocar) y dibuja. Recibe:
#       fig  : la Figure compartida (se limpia con fig.clf())
#       p    : dict {clave: valor en SI}  (ya convertido por el host)
#       modo : escala de intensidad ('lineal'/'gamma'/'log')
#    Devuelve (status_txt, status_color).
# =============================================================================

def render_marco_circulo(fig, p, modo):
    Npx = int(p["N"])
    x = np.linspace(-p["xmax"], p["xmax"], Npx)
    X, Y = np.meshgrid(x, x)
    I = intensidad(X, Y, p)

    fig.clf()
    gs = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.28, height_ratios=[1.5, 1])
    ax_ap = fig.add_subplot(gs[0, 0])
    ax_pat = fig.add_subplot(gs[0, 1])
    ax_prof = fig.add_subplot(gs[1, :])

    um = 1e6
    a, b, c, d = p["a"] * um, p["b"] * um, p["c"] * um, p["d"] * um
    R, D = p["R"] * um, p["D"] * um
    xm, xc = -D / 2.0, +D / 2.0
    ax_ap.set_facecolor("#111111")
    if a > 0 and b > 0:
        ax_ap.add_patch(Rectangle((xm - a / 2, -b / 2), a, b,
                                  facecolor="white", edgecolor="none"))
    if c > 0 and d > 0:
        ax_ap.add_patch(Rectangle((xm - c / 2, -d / 2), c, d,
                                  facecolor="#111111", edgecolor="none"))
    if R > 0:
        ax_ap.add_patch(Circle((xc, 0.0), R, facecolor="white", edgecolor="none"))
    if D > 0:
        ytop = max(b, 2 * R) / 2 * 1.15 + 5
        ax_ap.annotate("", xy=(xc, ytop), xytext=(xm, ytop),
                       arrowprops=dict(arrowstyle="<->", color="orange"))
        ax_ap.text(0.0, ytop + 4, "D", color="orange", ha="center",
                   va="bottom", fontsize=11)
    semi = max(a, c, 2 * R) / 2 + D / 2 + 10
    semiy = max(b, 2 * R) / 2 * 1.15 + 20
    ax_ap.set_xlim(-semi, semi)
    ax_ap.set_ylim(-semiy, semiy)
    ax_ap.set_aspect("equal")
    ax_ap.set_xlabel("x̃  [µm]")
    ax_ap.set_ylabel("ỹ  [µm]")
    ax_ap.set_title("Plano de la abertura (a escala)")

    ext = p["xmax"] * 1e3
    datos, norm = _escala_norm(I, modo)
    ax_pat.imshow(datos, extent=[-ext, ext, -ext, ext], origin="lower",
                  cmap="inferno", norm=norm,
                  vmax=(None if norm is not None else max(I.max(), 1e-12)))
    ax_pat.set_xlabel("x'  [mm]")
    ax_pat.set_ylabel("y'  [mm]")
    ax_pat.set_title("Fraunhofer analítico — sinc² (marco) + Airy (círculo)")

    j = I.shape[0] // 2
    ax_prof.plot(x * 1e3, I[j, :], color="crimson", lw=1.0)
    ax_prof.fill_between(x * 1e3, I[j, :], alpha=0.20, color="crimson")
    ax_prof.set_xlim(-p["xmax"] * 1e3, p["xmax"] * 1e3)
    ax_prof.set_ylim(bottom=0.0)
    ax_prof.set_xlabel("x'  [mm]   (perfil horizontal en y'=0)")
    ax_prof.set_ylabel("I / I₀")
    ax_prof.set_title("Perfil de intensidad  I(x', 0)")

    D_char, z_min, N_F, es_fh = regimen(p)
    return _status_regimen(D_char, z_min, N_F, es_fh, p["z"], metodo="sinc²/Airy")


def render_rectangulo_rotado(fig, p, modo):
    a, b, theta = p["a"], p["b"], p["theta"]
    lam, z = p["lam"], p["z"]
    xmax, Npx = p["xmax"], int(p["N"])
    x = np.linspace(-xmax, xmax, Npx)
    X, Y = np.meshgrid(x, x)
    I = intensidad_rectangulo_rotado(X, Y, a, b, theta, lam, z)

    fig.clf()
    gs = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.28, height_ratios=[1.5, 1])
    ax_ap = fig.add_subplot(gs[0, 0])
    ax_pat = fig.add_subplot(gs[0, 1])
    ax_prof = fig.add_subplot(gs[1, :])

    from matplotlib.patches import Polygon
    um = 1e6
    av, bv = a * um, b * um
    cth, sth = np.cos(theta), np.sin(theta)
    Rm = np.array([[cth, -sth], [sth, cth]])
    esquinas = np.array([(-av / 2, -bv / 2), (av / 2, -bv / 2),
                         (av / 2, bv / 2), (-av / 2, bv / 2)])
    verts = esquinas @ Rm.T
    ax_ap.set_facecolor("#111111")
    ax_ap.add_patch(Polygon(verts, closed=True, facecolor="white", edgecolor="none"))
    semi = max(av, bv) * 1.1
    ax_ap.set_xlim(-semi, semi)
    ax_ap.set_ylim(-semi, semi)
    ax_ap.set_aspect("equal")
    ax_ap.set_xlabel("x̃  [µm]")
    ax_ap.set_ylabel("ỹ  [µm]")
    ax_ap.set_title("Plano de la abertura (a escala)")

    ext = xmax * 1e3
    datos, norm = _escala_norm(I, modo)
    ax_pat.imshow(datos, extent=[-ext, ext, -ext, ext], origin="lower",
                  cmap="inferno", norm=norm,
                  vmax=(None if norm is not None else max(I.max(), 1e-12)))
    ax_pat.set_xlabel("x'  [mm]")
    ax_pat.set_ylabel("y'  [mm]")
    ax_pat.set_title("Fraunhofer analítico — rectángulo rotado (sinc² girado θ)")

    j = I.shape[0] // 2
    ax_prof.plot(x * 1e3, I[j, :], color="crimson", lw=1.0)
    ax_prof.fill_between(x * 1e3, I[j, :], alpha=0.20, color="crimson")
    ax_prof.set_xlim(-xmax * 1e3, xmax * 1e3)
    ax_prof.set_ylim(bottom=0.0)
    ax_prof.set_xlabel("x'  [mm]   (perfil horizontal en y'=0)")
    ax_prof.set_ylabel("I / I₀")
    ax_prof.set_title("Perfil de intensidad  I(x', 0)")

    D_char = np.hypot(a, b)
    z_min, N_F, es_fh = regimen_generico(D_char, lam, z)
    return _status_regimen(D_char, z_min, N_F, es_fh, z)


def render_cruz(fig, p, modo):
    L, a = p["L"], min(p["a"], p["L"])
    lam, z = p["lam"], p["z"]
    xmax, Npx = p["xmax"], int(p["N"])
    x = np.linspace(-xmax, xmax, Npx)
    X, Y = np.meshgrid(x, x)
    I = intensidad_cruz(X, Y, L, a, lam, z)

    fig.clf()
    gs = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.28, height_ratios=[1.5, 1])
    ax_ap = fig.add_subplot(gs[0, 0])
    ax_pat = fig.add_subplot(gs[0, 1])
    ax_prof = fig.add_subplot(gs[1, :])

    um = 1e6
    Lv, av = L * um, a * um
    ax_ap.set_facecolor("#111111")
    ax_ap.add_patch(Rectangle((-Lv / 2, -av / 2), Lv, av,
                              facecolor="white", edgecolor="none"))
    ax_ap.add_patch(Rectangle((-av / 2, -Lv / 2), av, Lv,
                              facecolor="white", edgecolor="none"))
    semi = Lv * 0.65
    ax_ap.set_xlim(-semi, semi)
    ax_ap.set_ylim(-semi, semi)
    ax_ap.set_aspect("equal")
    ax_ap.set_xlabel("x̃  [µm]")
    ax_ap.set_ylabel("ỹ  [µm]")
    ax_ap.set_title("Plano de la abertura (a escala)")

    ext = xmax * 1e3
    datos, norm = _escala_norm(I, modo)
    ax_pat.imshow(datos, extent=[-ext, ext, -ext, ext], origin="lower",
                  cmap="inferno", norm=norm,
                  vmax=(None if norm is not None else max(I.max(), 1e-12)))
    ax_pat.set_xlabel("x'  [mm]")
    ax_pat.set_ylabel("y'  [mm]")
    ax_pat.set_title("Fraunhofer analítico — cruz (unión de sinc²)")

    j = I.shape[0] // 2
    ax_prof.plot(x * 1e3, I[j, :], color="crimson", lw=1.0)
    ax_prof.fill_between(x * 1e3, I[j, :], alpha=0.20, color="crimson")
    ax_prof.set_xlim(-xmax * 1e3, xmax * 1e3)
    ax_prof.set_ylim(bottom=0.0)
    ax_prof.set_xlabel("x'  [mm]   (perfil horizontal en y'=0)")
    ax_prof.set_ylabel("I / I₀")
    ax_prof.set_title("Perfil de intensidad  I(x', 0)")

    D_char = L * np.sqrt(2.0)
    z_min, N_F, es_fh = regimen_generico(D_char, lam, z)
    return _status_regimen(D_char, z_min, N_F, es_fh, z)


def render_dos_semicirculos(fig, p, modo):
    r1, r2 = p["r1"], p["r2"]
    lam, z = p["lam"], p["z"]
    xmax, N = p["xmax"], int(p["N"])

    fig.clf()
    gs = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.28, height_ratios=[1.5, 1])
    ax_ap = fig.add_subplot(gs[0, 0])
    ax_pat = fig.add_subplot(gs[0, 1])
    ax_prof = fig.add_subplot(gs[1, :])

    if max(r1, r2) <= 0.0:
        return "Sin abertura (r1 = r2 = 0)", "#c00000"

    area = area_dos_semicirculos(r1, r2)
    I0_axial = irradiancia_axial_relativa(area, lam, z)
    x_obs, I2D, mascara, x_ap = patron_fft_semicirculos(r1, r2, lam, z, xmax, N)

    j0 = np.argmin(np.abs(x_obs))
    I_fft_centro = I2D[j0, j0] if I2D.size else float("nan")

    ext_ap = x_ap[-1] * 1e3
    ax_ap.imshow(mascara, extent=[-ext_ap, ext_ap, -ext_ap, ext_ap],
                 origin="lower", cmap="gray", vmin=0, vmax=1)
    ax_ap.axhline(0.0, color="orange", lw=0.6, ls=":")
    ax_ap.set_xlabel("x̃  [mm]")
    ax_ap.set_ylabel("ỹ  [mm]")
    ax_ap.set_title("Máscara de la abertura (rasterizada)")
    rlim = 3.5 * max(r1, r2) * 1e3
    ax_ap.set_xlim(-rlim, rlim)
    ax_ap.set_ylim(-rlim, rlim)

    ext = x_obs[-1] * 1e3 if x_obs.size else xmax * 1e3
    datos, norm = _escala_norm(I2D, modo)
    ax_pat.imshow(datos, extent=[-ext, ext, -ext, ext], origin="lower",
                  cmap="inferno", norm=norm,
                  vmax=(None if norm is not None else max(I2D.max(), 1e-12)))
    ax_pat.set_xlabel("x'  [mm]")
    ax_pat.set_ylabel("y'  [mm]")
    ax_pat.set_title("Patrón NUMÉRICO (FFT de la máscara) — no analítico")

    j = I2D.shape[0] // 2 if I2D.size else 0
    if I2D.size:
        ax_prof.plot(x_obs * 1e3, I2D[j, :], color="crimson", lw=1.0)
        ax_prof.fill_between(x_obs * 1e3, I2D[j, :], alpha=0.20, color="crimson")
    ax_prof.set_xlim(-xmax * 1e3, xmax * 1e3)
    ax_prof.set_ylim(bottom=0.0)
    ax_prof.set_xlabel("x'  [mm]   (perfil horizontal en y'=0)")
    ax_prof.set_ylabel("I / I₀")
    ax_prof.set_title("Perfil de intensidad  I(x', 0)  [FFT]")

    if I_fft_centro > 0:
        diff_pct = 100.0 * abs(I_fft_centro - I0_axial) / I0_axial
    else:
        diff_pct = float("nan")
    txt = (
        f"Área abertura = {area*1e6:8.4f} mm²\n"
        f"λ = {lam*1e9:.0f} nm    z = {z:.2f} m\n"
        f"─────────────────────────────\n"
        f"I(0,0)/I₀ [cerrado]  = {I0_axial:10.4e}\n"
        f"I(0,0)/I₀ [FFT centro] = {I_fft_centro:10.4e}\n"
        f"diferencia relativa   = {diff_pct:6.2f} %"
    )
    return txt, ("#127a12" if diff_pct < 5 else "#c00000")


def render_doble_cuadrado(fig, p, modo):
    a = p["a"]
    lam, z = p["lam"], p["z"]
    xmax, Npx = p["xmax"], int(p["N"])
    x = np.linspace(-xmax, xmax, Npx)
    X, Y = np.meshgrid(x, x)
    I, D = intensidad_doble_cuadrado(X, Y, a, lam, z)

    fig.clf()
    gs = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.28, height_ratios=[1.5, 1])
    ax_ap = fig.add_subplot(gs[0, 0])
    ax_pat = fig.add_subplot(gs[0, 1])
    ax_prof = fig.add_subplot(gs[1, :])

    um = 1e6
    av, Dv = a * um, D * um
    x1, x2 = -Dv / 2.0, +Dv / 2.0
    ax_ap.set_facecolor("#111111")
    ax_ap.add_patch(Rectangle((x1 - av / 2, -av / 2), av, av,
                              facecolor="white", edgecolor="none"))
    ax_ap.add_patch(Rectangle((x2 - 3 * av / 2, -3 * av / 2), 3 * av, 3 * av,
                              facecolor="white", edgecolor="none"))
    semi = Dv / 2.0 + 2.5 * av
    ax_ap.set_xlim(-semi, semi)
    ax_ap.set_ylim(-semi, semi)
    ax_ap.set_aspect("equal")
    ax_ap.set_xlabel("x̃  [µm]")
    ax_ap.set_ylabel("ỹ  [µm]")
    ax_ap.set_title(f"Plano de la abertura (D={Dv:.1f} µm, a escala)")

    ext = xmax * 1e3
    datos, norm = _escala_norm(I, modo)
    ax_pat.imshow(datos, extent=[-ext, ext, -ext, ext], origin="lower",
                  cmap="inferno", norm=norm,
                  vmax=(None if norm is not None else max(I.max(), 1e-12)))
    ax_pat.set_xlabel("x'  [mm]")
    ax_pat.set_ylabel("y'  [mm]")
    ax_pat.set_title("Fraunhofer analítico — doble cuadrado (sinc² + interferencia)")

    j = I.shape[0] // 2
    ax_prof.plot(x * 1e3, I[j, :], color="crimson", lw=1.0)
    ax_prof.fill_between(x * 1e3, I[j, :], alpha=0.20, color="crimson")
    ax_prof.set_xlim(-xmax * 1e3, xmax * 1e3)
    ax_prof.set_ylim(bottom=0.0)
    ax_prof.set_xlabel("x'  [mm]   (perfil horizontal en y'=0)")
    ax_prof.set_ylabel("I / I₀")
    ax_prof.set_title("Perfil de intensidad  I(x', 0)")

    D_char = D + a / 2.0 + 1.5 * a
    z_min, N_F, es_fh = regimen_generico(D_char, lam, z)
    return _status_regimen(D_char, z_min, N_F, es_fh, z)


def render_rendijas(fig, p, modo):
    N = int(p["N"])
    a_lam = p["a_lam"]
    d_a = p["d_a"]
    d_lam = d_a * a_lam
    tmax = p["tmax"]

    theta = np.linspace(-tmax, tmax, 4000)
    st = np.sin(theta)
    I = intensidad_rendijas(st, a_lam, d_lam, N)
    theta_deg = np.degrees(theta)

    fig.clf()
    gs = fig.add_gridspec(2, 1, hspace=0.32, height_ratios=[1, 1.2])
    ax_pat = fig.add_subplot(gs[0, 0])
    ax_prof = fig.add_subplot(gs[1, 0])

    img = np.tile(I, (60, 1))
    ax_pat.imshow(img, extent=[theta_deg[0], theta_deg[-1], -1, 1],
                  origin="lower", cmap="inferno", aspect="auto",
                  vmax=max(I.max(), 1e-12))
    ax_pat.set_yticks([])
    ax_pat.set_xlabel("θ  [grados]")
    titulo = ("Rendija simple (N=1)" if N <= 1
              else f"Red de {N} rendijas (a/λ={a_lam:.2f}, d/a={d_a:.1f})")
    ax_pat.set_title("Fraunhofer — " + titulo)

    ax_prof.plot(theta_deg, I, color="crimson", lw=1.0, label="I(θ)")
    env = np.sinc(a_lam * st) ** 2
    ax_prof.plot(theta_deg, env, color="steelblue", lw=0.9, ls="--",
                 label="envolvente de 1 rendija")
    ax_prof.fill_between(theta_deg, I, alpha=0.15, color="crimson")
    ax_prof.set_xlim(theta_deg[0], theta_deg[-1])
    ax_prof.set_ylim(bottom=0.0)
    ax_prof.set_xlabel("θ  [grados]")
    ax_prof.set_ylabel("I / I₀")
    ax_prof.legend(fontsize=8, loc="upper right")
    ax_prof.set_title("Perfil de intensidad  I(θ)")

    if N <= 1:
        mins = angulos_minimos_rendija(a_lam, m_max=4)
        lineas = "\n".join(f"  mín m={m}: θ = ±{ang:5.2f}°" for m, ang in mins)
        for _, ang in mins:
            for signo in (+1, -1):
                ax_prof.axvline(signo * ang, color="navy", lw=0.6, ls=":")
        txt = (f"RENDIJA SIMPLE (N=1)\n"
               f"a/λ = {a_lam:.3f}\n"
               f"Mínimos  sinθ = m·λ/a:\n{lineas}")
    else:
        ords = ordenes_red(d_lam, sin_max=np.sin(tmax))
        lineas = "\n".join(f"  orden m={m}: θ = ±{ang:5.2f}°" for m, ang in ords)
        for _, ang in ords:
            for signo in (+1, -1):
                ax_prof.axvline(signo * ang, color="seagreen", lw=0.5, ls=":")
        txt = (f"RED de N={N} rendijas\n"
               f"a/λ = {a_lam:.2f}   d/λ = {d_lam:.2f}\n"
               f"Máx. principales  sinθ = m·λ/d:\n{lineas}")
    return txt, "#333333"


def render_escalon(fig, p, modo):
    N = int(p["N"])
    s, h = p["s"], p["h"]
    n, lam = p["n"], p["lam"]
    umax = p["umax"]

    u = np.linspace(-umax, umax, 8000)
    sin_theta = u * lam / s
    I = intensidad_escalon(sin_theta, s, h, n, lam, N)

    fig.clf()
    gs = fig.add_gridspec(2, 1, hspace=0.32, height_ratios=[1, 1.2])
    ax_pat = fig.add_subplot(gs[0, 0])
    ax_prof = fig.add_subplot(gs[1, 0])

    ax_pat.imshow(np.tile(I, (60, 1)), extent=[-umax, umax, -1, 1],
                  origin="lower", cmap="inferno", aspect="auto",
                  vmax=max(I.max(), 1e-12))
    ax_pat.set_yticks([])
    ax_pat.set_xlabel("u = s·senθ/λ")
    ax_pat.set_title(f"Escalón de Michelson — {N} peldaños")

    ax_prof.plot(u, I, color="crimson", lw=1.0, label="I(u)")
    ax_prof.plot(u, np.sinc(u) ** 2, color="steelblue", lw=0.9, ls="--",
                 label="envolvente de 1 peldaño")
    ax_prof.fill_between(u, I, alpha=0.15, color="crimson")
    for k in range(-int(umax), int(umax) + 1):
        if k != 0:
            ax_prof.axvline(k, color="navy", lw=0.5, ls=":")
    ax_prof.set_xlim(-umax, umax)
    ax_prof.set_ylim(bottom=0.0)
    ax_prof.set_xlabel("u = s·senθ/λ    (ceros de difracción en u entero)")
    ax_prof.set_ylabel("I / I_env")
    ax_prof.legend(fontsize=8, loc="upper right")
    ax_prof.set_title("Perfil — máx. principales (peine) bajo la envolvente")

    m0 = (n - 1.0) * h / lam
    R = m0 * N
    txt = (
        f"Orden central  m₀=(n−1)h/λ = {m0:,.0f}\n"
        f"Poder resolvente  R=m₀·N ≈ {R:,.0f}\n"
        f"─────────────────────────────\n"
        f"Máximos principales por máximo\n"
        f"de difracción ≈ 2  (Δu=2 = 2\n"
        f"espaciados de orden; se ven 1–2\n"
        f"según el desfase (n−1)h/λ)"
    )
    return txt, "#333333"


def render_doble_circulo(fig, p, modo):
    r1, r2 = p["r1"], p["r2"]
    lam, z = p["lam"], p["z"]
    N = int(p["N"])

    d = patrones_doble_circulo(r1, r2, lam, z, N)
    c = N // 2
    I_fr_axial = d["I_fresnel"][c, c]
    I_fh_axial = d["I_fraunhofer"][c, c]

    xmax = p["xmax"]
    sel = np.abs(d["x_obs"]) <= xmax
    x_obs = d["x_obs"][sel]
    I_fr = d["I_fresnel"][np.ix_(sel, sel)]
    I_fh = d["I_fraunhofer"][np.ix_(sel, sel)]

    fig.clf()
    gs = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.28)
    ax_ap = fig.add_subplot(gs[0, 0])
    ax_fh = fig.add_subplot(gs[0, 1])
    ax_fr = fig.add_subplot(gs[1, 0])
    ax_prof = fig.add_subplot(gs[1, 1])

    ext_ap = d["x_ap"][-1] * 1e3
    ax_ap.imshow(d["mascara"], extent=[-ext_ap, ext_ap, -ext_ap, ext_ap],
                 origin="lower", cmap="gray", vmin=0, vmax=1)
    ax_ap.set_title("Abertura (3 cuad. r₁ + ¼ r₂)")
    ax_ap.set_xlabel("x̃ [mm]")
    ax_ap.set_ylabel("ỹ [mm]")
    rlim = 1.6 * max(r1, r2) * 1e3
    ax_ap.set_xlim(-rlim, rlim)
    ax_ap.set_ylim(-rlim, rlim)

    ext = xmax * 1e3
    datos, norm = _escala_norm(I_fh, modo)
    ax_fh.imshow(datos, extent=[-ext, ext, -ext, ext], origin="lower",
                 cmap="inferno", norm=norm,
                 vmax=(None if norm is not None else I_fh.max()))
    ax_fh.set_title("Fraunhofer (campo lejano)")
    ax_fh.set_xlabel("x' [mm]")

    datos, norm = _escala_norm(I_fr, modo)
    ax_fr.imshow(datos, extent=[-ext, ext, -ext, ext], origin="lower",
                 cmap="inferno", norm=norm,
                 vmax=(None if norm is not None else I_fr.max()))
    ax_fr.set_title("Fresnel (campo cercano)")
    ax_fr.set_xlabel("x' [mm]")
    ax_fr.set_ylabel("y' [mm]")

    xo = x_obs * 1e3
    cc = I_fr.shape[0] // 2
    ax_prof.plot(xo, I_fr[cc, :], color="crimson", lw=1.0, label="Fresnel")
    ax_prof.plot(xo, I_fh[cc, :], color="steelblue", lw=0.9, label="Fraunhofer")
    ax_prof.set_xlim(xo[0], xo[-1])
    ax_prof.set_ylim(bottom=0.0)
    ax_prof.set_xlabel("x' [mm]  (perfil en y'=0)")
    ax_prof.set_ylabel("I / I_inc")
    ax_prof.legend(fontsize=8, loc="upper right")
    ax_prof.set_title("Perfiles I(x',0)")

    D_char = 2.0 * max(r1, r2)
    z_min, N_F, es_fh = regimen_generico(D_char, lam, z)
    regimen_txt = ("FRAUNHOFER válido" if es_fh else "FRESNEL (campo cercano)")
    txt = (
        f"λ={lam*1e9:.0f}nm  z={z:.2f}m\n"
        f"r₁={r1*1e3:.3f}  r₂={r2*1e3:.3f} mm\n"
        f"─────────────────────────\n"
        f"Fresnel   |U|/A={np.sqrt(I_fr_axial):5.3f}  I/I={I_fr_axial:6.3f}\n"
        f"Fraunhofer          I/I={I_fh_axial:7.3f}\n"
        f"─────────────────────────\n"
        f"Régimen real: {regimen_txt}\n"
        f"z_min=2D²/λ={z_min:.1f}m  N_F={N_F:.2f}"
    )
    return txt, ("#127a12" if es_fh else "#c00000")


def render_redes_cascada(fig, p, modo):
    a = p["a"]
    N = int(p["N"])
    d = 3.0 * a
    W = N * d
    s = min(p["s"], W)
    lam, z = p["lam"], p["z"]

    sen, I, d, W = patron_redes_cascada(a, N, s, lam)
    smax = 4.0 * lam / a
    sel = np.abs(sen) <= smax
    sen_v, I_v = sen[sel], I[sel]

    fig.clf()
    gs = fig.add_gridspec(3, 1, hspace=0.45, height_ratios=[0.7, 0.6, 1.3])
    ax_red = fig.add_subplot(gs[0, 0])
    ax_pat = fig.add_subplot(gs[1, 0])
    ax_prof = fig.add_subplot(gs[2, 0])

    xr = np.linspace(-2.5 * d, 2.5 * d, 3000)
    centros = (np.arange(N) - (N - 1) / 2.0) * d
    r1 = np.zeros_like(xr)
    r2 = np.zeros_like(xr)
    for cc in centros:
        r1 += (xr >= cc - a / 2) & (xr <= cc + a / 2)
        r2 += (xr >= cc - a / 2 + s) & (xr <= cc + a / 2 + s)
    solape = (r1 * r2).clip(0, 1)
    xr_mm = xr * 1e3
    ax_red.fill_between(xr_mm, 2.4, 3.2, where=r1 > 0, color="#3b6", step="mid")
    ax_red.fill_between(xr_mm, 1.3, 2.1, where=r2 > 0, color="#38c", step="mid")
    ax_red.fill_between(xr_mm, 0.2, 1.0, where=solape > 0, color="#e33", step="mid")
    ax_red.text(xr_mm[0], 2.8, " red 1", va="center", fontsize=8)
    ax_red.text(xr_mm[0], 1.7, " red 2", va="center", fontsize=8)
    ax_red.text(xr_mm[0], 0.6, " solape", va="center", fontsize=8)
    ax_red.set_ylim(0, 3.4)
    ax_red.set_yticks([])
    ax_red.set_xlim(xr_mm[0], xr_mm[-1])
    ax_red.set_xlabel("x̃ [mm]  (transmisión efectiva = solape)")
    ax_red.set_title("Redes en espacio real (red 2 desplazada s)")

    ax_pat.imshow(np.tile(I_v, (40, 1)),
                  extent=[sen_v[0], sen_v[-1], -1, 1], origin="lower",
                  cmap="inferno", aspect="auto", vmin=0, vmax=1.0)
    ax_pat.set_yticks([])
    ax_pat.set_xlabel("senθ")
    ax_pat.set_title("Patrón de difracción")

    ax_prof.plot(sen_v, I_v, color="crimson", lw=1.0, label="I(senθ)")
    ax_prof.plot(sen_v, np.sinc(a * sen_v / lam) ** 2, color="steelblue",
                 lw=0.9, ls="--", label="envolvente de 1 ranura")
    ax_prof.fill_between(sen_v, I_v, alpha=0.15, color="crimson")
    m = 1
    while m * lam / d <= smax:
        for signo in (+1, -1):
            col = "navy" if m % 3 else "gray"
            ax_prof.axvline(signo * m * lam / d, color=col, lw=0.5, ls=":")
        m += 1
    ax_prof.set_xlim(-smax, smax)
    ax_prof.set_ylim(0, max(1.05, I_v.max() * 1.1))
    ax_prof.set_xlabel("senθ   (órdenes en mλ/d; grises = múltiplos de 3, faltan)")
    ax_prof.set_ylabel("I / I₀(s=0)")
    ax_prof.legend(fontsize=8, loc="upper right")
    ax_prof.set_title("Perfil de intensidad")

    s_mm, W_mm = s * 1e3, W * 1e3
    s_frac = s % d
    oscuro = a <= s_frac <= 2 * a
    n_perdidas = int(s // d)
    obs = "0, ±1, ±2, ±4, ±5, ±7 …  (faltan ±3, ±6: múltiplos de d/a=3)"
    txt = (
        f"d = 3a = {d*1e3:.3f} mm\n"
        f"s_sep (dejan de solapar) = N·d = {W_mm:.2f} mm\n"
        f"─────────────────────────────\n"
        f"Alineadas (s=0): órdenes\n  {obs}\n"
        f"─────────────────────────────\n"
        f"s = {s_mm:.3f} mm   ({'CAMPO OSCURO' if oscuro else 'con solape'})\n"
        f"ranuras que aún solapan ≈ {max(N - n_perdidas, 0)}\n"
        f"Al desplazar: posiciones de orden\n"
        f"FIJAS (d cte); reaparecen los\n"
        f"múltiplos de 3; I total ↓ → 0 en\n"
        f"s∈[a,2a] (mod d) y en s=N·d."
    )
    return txt, ("#c00000" if oscuro else "#333333")


# =============================================================================
# 4B. PERFIL DE INTENSIDAD ABSOLUTA (sin normalizar) — corte y'=0
#    abs_*(p) → (x' [m], I_abs, I_norm), con I_abs = |U|² relativa a la onda
#    plana incidente (amplitud 1): centro Fraunhofer = (Área/λz)². Para las
#    analíticas se escala la I normalizada del núcleo por (Área/λz)²; para las
#    numéricas (FFT) el patrón ya viene en esa escala absoluta.
# =============================================================================

def abs_marco_circulo(p):
    x = np.linspace(-p["xmax"], p["xmax"], int(p["N"]))
    I_norm = intensidad(x, np.zeros_like(x), p)
    A0 = p["a"] * p["b"] - p["c"] * p["d"] + np.pi * p["R"] ** 2
    k = (A0 / (p["lam"] * p["z"])) ** 2
    return x, I_norm * k, I_norm


def abs_rectangulo_rotado(p):
    x = np.linspace(-p["xmax"], p["xmax"], int(p["N"]))
    I_norm = intensidad_rectangulo_rotado(x, np.zeros_like(x), p["a"], p["b"],
                                          p["theta"], p["lam"], p["z"])
    k = ((p["a"] * p["b"]) / (p["lam"] * p["z"])) ** 2
    return x, I_norm * k, I_norm


def abs_cruz(p):
    L, a = p["L"], min(p["a"], p["L"])
    x = np.linspace(-p["xmax"], p["xmax"], int(p["N"]))
    I_norm = intensidad_cruz(x, np.zeros_like(x), L, a, p["lam"], p["z"])
    area = 2.0 * L * a - a * a
    k = (area / (p["lam"] * p["z"])) ** 2
    return x, I_norm * k, I_norm


def abs_doble_cuadrado(p):
    x = np.linspace(-p["xmax"], p["xmax"], int(p["N"]))
    I_norm, _ = intensidad_doble_cuadrado(x, np.zeros_like(x), p["a"],
                                          p["lam"], p["z"])
    area = 10.0 * p["a"] ** 2
    k = (area / (p["lam"] * p["z"])) ** 2
    return x, I_norm * k, I_norm


def abs_dos_semicirculos(p):
    x_obs, I2D, _, _ = patron_fft_semicirculos(
        p["r1"], p["r2"], p["lam"], p["z"], p["xmax"], int(p["N"]))
    if not I2D.size:
        z = np.zeros(1)
        return z, z, z
    j = I2D.shape[0] // 2
    I_abs = I2D[j, :]                       # ya absoluto (|U|², I0 incidente=1)
    axial = I_abs[np.argmin(np.abs(x_obs))]
    I_norm = I_abs / axial if axial > 0 else I_abs
    return x_obs, I_abs, I_norm


def abs_doble_circulo(p):
    d = patrones_doble_circulo(p["r1"], p["r2"], p["lam"], p["z"], int(p["N"]))
    xo = d["x_obs"]
    sel = np.abs(xo) <= p["xmax"]
    xo = xo[sel]
    Ifh = d["I_fraunhofer"][np.ix_(sel, sel)]  # ya absoluto (campo lejano)
    cc = Ifh.shape[0] // 2
    I_abs = Ifh[cc, :]
    N = int(p["N"])
    axial = d["I_fraunhofer"][N // 2, N // 2]
    I_norm = I_abs / axial if axial > 0 else I_abs
    return xo, I_abs, I_norm


# =============================================================================
# 5. REGISTRO DE ABERTURAS
#    nombre → dict(titulo, sliders, usa_escala, render, expr [, status_titulo, abs])
#    sliders: lista de (clave, label, frm, to, init, escala_a_SI, fmt)
#    render(fig, p, modo) → (status_txt, status_color);  p = {clave: valor SI}
#    expr: fórmula analítica de I (dato; su visualización es el cambio C)
#    abs (opcional): perfil de intensidad absoluta → usado por la pestaña 2
# =============================================================================

_ESC_RAD = np.pi / 180.0  # grados → radianes (multiplicador para el modelo escala_SI)

APERTURAS_20 = {
    "Dos aberturas (marco + círculo) — pto 1": {
        "titulo": "Difracción de Fraunhofer",
        "usa_escala": True,
        "sliders": [
            ("a", "a  ext. ancho", 0.0, 400.0, 120.0, 1e-6, "{:.1f}"),
            ("b", "b  ext. alto", 0.0, 400.0, 160.0, 1e-6, "{:.1f}"),
            ("c", "c  int. ancho", 0.0, 400.0, 60.0, 1e-6, "{:.1f}"),
            ("d", "d  int. alto", 0.0, 400.0, 90.0, 1e-6, "{:.1f}"),
            ("R", "R  radio", 0.0, 300.0, 80.0, 1e-6, "{:.1f}"),
            ("D", "D  separación", 0.0, 1500.0, 400.0, 1e-6, "{:.1f}"),
            ("lam", "λ (nm)", 380.0, 1000.0, 633.0, 1e-9, "{:.0f}"),
            ("z", "z (m)", 0.05, 20.0, 5.0, 1.0, "{:.2f}"),
            ("xmax", "x'_max (mm)", 1.0, 80.0, 25.0, 1e-3, "{:.1f}"),
            ("N", "N (px)", 200.0, float(NX_MAX), 500.0, 1.0, "{:.0f}"),
        ],
        "render": render_marco_circulo,
        "abs": abs_marco_circulo,
        "expr": ("I = A_marco² + A_círc² + 2·A_marco·A_círc·cos(2πD·fx)\n"
                 "A_marco = ab·sinc(a·fx)sinc(b·fy) − cd·sinc(c·fx)sinc(d·fy)\n"
                 "A_círc  = πR²·2J₁(2πRρ)/(2πRρ)"),
    },
    "Rectángulo rotado — pto 3": {
        "titulo": "Rectángulo rotado (lados a×b, giro θ)",
        "usa_escala": True,
        "sliders": [
            ("a", "a (lado en x)", 1.0, 50.0, 10.0, 1e-6, "{:.2f}"),
            ("b", "b (lado en y)", 1.0, 50.0, 5.0, 1e-6, "{:.2f}"),
            ("theta", "θ giro (grados)", 0.0, 180.0, 60.0, _ESC_RAD, "{:.1f}"),
            ("lam", "λ (nm)", 380.0, 1000.0, 633.0, 1e-9, "{:.0f}"),
            ("z", "z (m)", 0.05, 5.0, 1.0, 1.0, "{:.2f}"),
            ("xmax", "x'_max (mm)", 5.0, 300.0, 100.0, 1e-3, "{:.0f}"),
            ("N", "N (px)", 200.0, float(NX_MAX), 500.0, 1.0, "{:.0f}"),
        ],
        "render": render_rectangulo_rotado,
        "abs": abs_rectangulo_rotado,
        "expr": ("I = sinc²(a·f_a)·sinc²(b·f_b)\n"
                 "f_a = fx·cosθ + fy·senθ    f_b = −fx·senθ + fy·cosθ"),
    },
    "Cruz — pto 5": {
        "titulo": "Abertura en cruz",
        "usa_escala": True,
        "sliders": [
            ("L", "L  largo total", 20.0, 500.0, 200.0, 1e-6, "{:.1f}"),
            ("a", "a  ancho brazo", 5.0, 500.0, 50.0, 1e-6, "{:.1f}"),
            ("lam", "λ (nm)", 380.0, 1000.0, 633.0, 1e-9, "{:.0f}"),
            ("z", "z (m)", 0.05, 5.0, 1.0, 1.0, "{:.2f}"),
            ("xmax", "x'_max (mm)", 5.0, 300.0, 60.0, 1e-3, "{:.0f}"),
            ("N", "N (px)", 200.0, float(NX_MAX), 500.0, 1.0, "{:.0f}"),
        ],
        "render": render_cruz,
        "abs": abs_cruz,
        "expr": ("I = |La·sinc(L·fx)sinc(a·fy) + aL·sinc(a·fx)sinc(L·fy)\n"
                 "     − a²·sinc(a·fx)sinc(a·fy)|² / (2La − a²)²"),
    },
    "Dos semicírculos — pto 9": {
        "titulo": "Dos semicírculos (r1 arriba, r2 abajo)",
        "usa_escala": True,
        "status_titulo": "Resultado pedido — irradiancia axial",
        "sliders": [
            ("r1", "r1  (superior)", 0.0, 5.0, 2.0, 1e-3, "{:.3f}"),
            ("r2", "r2  (inferior)", 0.0, 5.0, 1.41421, 1e-3, "{:.3f}"),
            ("lam", "λ (nm)", 380.0, 1000.0, 500.0, 1e-9, "{:.0f}"),
            ("z", "z (m)", 0.1, 20.0, 4.0, 1.0, "{:.2f}"),
            ("xmax", "x'_max (mm)", 0.5, 100.0, 4.0, 1e-3, "{:.2f}"),
            ("N", "N (px, FFT)", 128.0, float(NX_MAX), 800.0, 1.0, "{:.0f}"),
        ],
        "render": render_dos_semicirculos,
        "abs": abs_dos_semicirculos,
        "expr": ("Numérico (FFT de la máscara). Axial:\n"
                 "I(0,0)/I₀ = (Área/λz)²,  Área = π(r1²+r2²)/2"),
    },
    "Doble cuadrado — pto 14": {
        "titulo": "Doble cuadrado (a y 3a, separación 2a)",
        "usa_escala": True,
        "sliders": [
            ("a", "a  (3a automático)", 5.0, 200.0, 50.0, 1e-6, "{:.1f}"),
            ("lam", "λ (nm)", 380.0, 1000.0, 633.0, 1e-9, "{:.0f}"),
            ("z", "z (m)", 0.05, 5.0, 2.0, 1.0, "{:.2f}"),
            ("xmax", "x'_max (mm)", 5.0, 300.0, 60.0, 1e-3, "{:.0f}"),
            ("N", "N (px)", 200.0, float(NX_MAX), 500.0, 1.0, "{:.0f}"),
        ],
        "render": render_doble_cuadrado,
        "abs": abs_doble_cuadrado,
        "expr": ("I = A1² + A2² + 2A1A2·cos(2πD·fx),  D = 4a\n"
                 "A1 = a²·sinc(a·fx)sinc(a·fy)\n"
                 "A2 = 9a²·sinc(3a·fx)sinc(3a·fy)"),
    },
    "Rendija(s) 1..N — ptos 2/6": {
        "titulo": "Rendija(s): 1 a N ranuras",
        "usa_escala": False,
        "status_titulo": "Mínimos / órdenes",
        "sliders": [
            ("N", "N (nº ranuras)", 1.0, 40.0, 1.0, 1.0, "{:.0f}"),
            ("a_lam", "a/λ (ancho)", 0.5, 100.0, 3.18, 1.0, "{:.2f}"),
            ("d_a", "d/a (período/ancho)", 1.0, 10.0, 3.0, 1.0, "{:.2f}"),
            ("tmax", "θ_max (grados)", 5.0, 90.0, 80.0, _ESC_RAD, "{:.0f}"),
        ],
        "render": render_rendijas,
        "expr": "I = sinc²(a·senθ/λ)·[sin(Nπd·senθ/λ)/(N·sin(πd·senθ/λ))]²",
    },
    "Escalón de Michelson — pto 15": {
        "titulo": "Escalón de Michelson (N peldaños)",
        "usa_escala": False,
        "status_titulo": "Resultado",
        "sliders": [
            ("N", "N (láminas)", 2.0, 30.0, 10.0, 1.0, "{:.0f}"),
            ("s", "s saliente (mm)", 0.1, 5.0, 1.0, 1e-3, "{:.3f}"),
            ("h", "h espesor (mm)", 0.5, 30.0, 10.0, 1e-3, "{:.2f}"),
            ("n", "n vidrio", 1.3, 2.0, 1.5, 1.0, "{:.3f}"),
            ("lam", "λ (nm)", 380.0, 1000.0, 500.0, 1e-9, "{:.0f}"),
            ("umax", "u_max (=s·senθ/λ)", 1.5, 5.0, 2.5, 1.0, "{:.1f}"),
        ],
        "render": render_escalon,
        "expr": ("I = sinc²(s·senθ/λ)·[sin(NπΔ/λ)/(N·sin(πΔ/λ))]²\n"
                 "Δ = (n−1)h + s·senθ"),
    },
    "Doble círculo (Fh+Fr) — pto 19": {
        "titulo": "Doble círculo — Fraunhofer y Fresnel",
        "usa_escala": True,
        "status_titulo": "Valores axiales en P'",
        "sliders": [
            ("r1", "r1 (3 cuadrantes)", 0.2, 3.0, 1.0, 1e-3, "{:.3f}"),
            ("r2", "r2 (cuad. sup-der)", 0.2, 4.0, 1.41, 1e-3, "{:.3f}"),
            ("lam", "λ (nm)", 380.0, 1000.0, 500.0, 1e-9, "{:.0f}"),
            ("z", "z (m)", 0.5, 40.0, 2.0, 1.0, "{:.2f}"),
            ("xmax", "x'_max (mm)", 1.0, 60.0, 8.0, 1e-3, "{:.1f}"),
            ("N", "N (px, FFT)", 256.0, 2048.0, 1024.0, 1.0, "{:.0f}"),
        ],
        "render": render_doble_circulo,
        "abs": abs_doble_circulo,
        "expr": ("Numérico (FFT). Fraunhofer axial (Área/λz)²;\n"
                 "Fresnel vía zonas. Área = ¾πr1² + ¼πr2²"),
    },
    "Redes en cascada — pto 6": {
        "titulo": "Dos redes en cascada (d=3a)",
        "usa_escala": False,
        "status_titulo": "Órdenes y desplazamiento",
        "sliders": [
            ("a", "a ranura (mm)", 0.05, 1.0, 0.2, 1e-3, "{:.3f}"),
            ("N", "N ranuras", 2.0, 20.0, 8.0, 1.0, "{:.0f}"),
            ("s", "s (mm)", 0.0, 5.0, 0.0, 1e-3, "{:.3f}"),
            ("lam", "λ (nm)", 380.0, 1000.0, 633.0, 1e-9, "{:.0f}"),
            ("z", "z (m)", 0.1, 20.0, 1.0, 1.0, "{:.2f}"),
        ],
        "render": render_redes_cascada,
        "expr": "I = |𝓕{t₁(x)·t₂(x−s)}|²   (FFT 1D de la transmisión)",
    },
}


# =============================================================================
# 6. HOST — una sola ventana con selector de abertura
# =============================================================================

class HostFraunhofer:
    """Envoltura única: Combobox de abertura + sliders reconstruidos por
    abertura + una figura cuyo layout arma el render de cada caso."""

    def __init__(self, parent):
        self.parent = parent
        self._pv = []           # [(clave, DoubleVar, escala_a_SI), ...]
        self._build_controls()
        self._build_figure()
        self._on_aperture()     # construye sliders + primer cálculo

    def _build_controls(self):
        panel = ttk.Frame(self.parent, padding=8)
        panel.pack(side="left", fill="y")

        self.titulo_var = tk.StringVar(value="")
        ttk.Label(panel, textvariable=self.titulo_var,
                  font=("", 11, "bold")).pack(anchor="w", pady=(0, 6))

        sel = ttk.LabelFrame(panel, text="Abertura", padding=6)
        sel.pack(fill="x", pady=4)
        self.aper = tk.StringVar(value=list(APERTURAS_20)[0])
        cb = ttk.Combobox(sel, textvariable=self.aper, state="readonly",
                          values=list(APERTURAS_20), width=30)
        cb.pack(fill="x")
        cb.bind("<<ComboboxSelected>>", self._on_aperture)

        self.param_frame = ttk.LabelFrame(panel, text="Parámetros", padding=6)
        self.param_frame.pack(fill="x", pady=4)

        self.escala_frame = ttk.LabelFrame(panel, text="Escala de intensidad",
                                           padding=6)
        self.escala = tk.StringVar(value="gamma")
        for txt, val in (("Lineal", "lineal"), ("γ (0.4)", "gamma"),
                         ("Log", "log")):
            ttk.Radiobutton(self.escala_frame, text=txt, variable=self.escala,
                            value=val, command=self.recompute).pack(side="left")

        self.status_frame = ttk.LabelFrame(panel, text="Régimen del cálculo",
                                           padding=6)
        self.status_frame.pack(fill="x", pady=4)
        self.status = tk.StringVar(value="")
        self.status_lbl = ttk.Label(self.status_frame, textvariable=self.status,
                                    justify="left", font=("Consolas", 9))
        self.status_lbl.pack(anchor="w")

        ex = ttk.LabelFrame(panel, text="Expresión de I", padding=6)
        ex.pack(fill="x", pady=4)
        self.expr = tk.StringVar(value="")
        ttk.Label(ex, textvariable=self.expr, justify="left",
                  font=("Consolas", 8)).pack(anchor="w")

    def _build_figure(self):
        right = ttk.Frame(self.parent)
        right.pack(side="left", fill="both", expand=True)
        self.fig = Figure(figsize=(10.5, 8.0))
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        NavigationToolbar2Tk(self.canvas, right).update()

    def _on_aperture(self, *_):
        entry = APERTURAS_20[self.aper.get()]
        for w in self.param_frame.winfo_children():
            w.destroy()
        self._pv = []
        for (clave, label, frm, to, init, esc, fmt) in entry["sliders"]:
            var = crear_slider(self.param_frame, label, frm, to, init,
                               self.recompute, fmt)
            self._pv.append((clave, var, esc))

        if entry["usa_escala"]:
            self.escala_frame.pack(fill="x", pady=4, before=self.status_frame)
        else:
            self.escala_frame.pack_forget()

        self.titulo_var.set(entry["titulo"])
        self.status_frame.configure(
            text=entry.get("status_titulo", "Régimen del cálculo"))
        self.expr.set(entry.get("expr", ""))
        self.recompute()

    def recompute(self):
        entry = APERTURAS_20[self.aper.get()]
        p = {clave: var.get() * esc for (clave, var, esc) in self._pv}
        txt, color = entry["render"](self.fig, p, self.escala.get())
        self.status.set(txt)
        self.status_lbl.configure(foreground=color)
        self.canvas.draw_idle()


# =============================================================================
# 7. PESTAÑA DE INTENSIDAD ABSOLUTA (sin normalizar) vs x'
# =============================================================================

# Solo las aberturas con plano de observación físico x' (excluye las angulares:
# rendijas, escalón, redes en cascada).
ABS_NAMES = [n for n, e in APERTURAS_20.items() if "abs" in e]


class TabAbsoluto:
    """Muestra I_abs(x',0) = |U|² relativa al incidente (centro = (Área/λz)²),
    con la normalizada debajo para comparar. Selector limitado a ABS_NAMES."""

    def __init__(self, parent):
        self.parent = parent
        self._pv = []
        self._build_controls()
        self._build_figure()
        self._on_aperture()

    def _build_controls(self):
        panel = ttk.Frame(self.parent, padding=8)
        panel.pack(side="left", fill="y")

        self.titulo_var = tk.StringVar(value="")
        ttk.Label(panel, textvariable=self.titulo_var,
                  font=("", 11, "bold")).pack(anchor="w", pady=(0, 6))

        sel = ttk.LabelFrame(panel, text="Abertura", padding=6)
        sel.pack(fill="x", pady=4)
        self.aper = tk.StringVar(value=ABS_NAMES[0])
        cb = ttk.Combobox(sel, textvariable=self.aper, state="readonly",
                          values=ABS_NAMES, width=30)
        cb.pack(fill="x")
        cb.bind("<<ComboboxSelected>>", self._on_aperture)

        self.param_frame = ttk.LabelFrame(panel, text="Parámetros", padding=6)
        self.param_frame.pack(fill="x", pady=4)

        self.status_frame = ttk.LabelFrame(panel, text="Intensidad absoluta",
                                           padding=6)
        self.status_frame.pack(fill="x", pady=4)
        self.status = tk.StringVar(value="")
        self.status_lbl = ttk.Label(self.status_frame, textvariable=self.status,
                                    justify="left", font=("Consolas", 9))
        self.status_lbl.pack(anchor="w")

        ex = ttk.LabelFrame(panel, text="Expresión de I", padding=6)
        ex.pack(fill="x", pady=4)
        self.expr = tk.StringVar(value="")
        ttk.Label(ex, textvariable=self.expr, justify="left",
                  font=("Consolas", 8)).pack(anchor="w")

    def _build_figure(self):
        right = ttk.Frame(self.parent)
        right.pack(side="left", fill="both", expand=True)
        self.fig = Figure(figsize=(10.5, 8.0))
        gs = self.fig.add_gridspec(2, 1, hspace=0.32)
        self.ax_abs = self.fig.add_subplot(gs[0, 0])
        self.ax_norm = self.fig.add_subplot(gs[1, 0])
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        NavigationToolbar2Tk(self.canvas, right).update()

    def _on_aperture(self, *_):
        entry = APERTURAS_20[self.aper.get()]
        for w in self.param_frame.winfo_children():
            w.destroy()
        self._pv = []
        for (clave, label, frm, to, init, esc, fmt) in entry["sliders"]:
            var = crear_slider(self.param_frame, label, frm, to, init,
                               self.recompute, fmt)
            self._pv.append((clave, var, esc))
        self.titulo_var.set(entry["titulo"])
        self.expr.set(entry.get("expr", ""))
        self.recompute()

    def recompute(self):
        entry = APERTURAS_20[self.aper.get()]
        p = {clave: var.get() * esc for (clave, var, esc) in self._pv}
        x, I_abs, I_norm = entry["abs"](p)
        xmm = x * 1e3

        ax = self.ax_abs
        ax.clear()
        ax.plot(xmm, I_abs, color="darkorange", lw=1.1)
        ax.fill_between(xmm, I_abs, alpha=0.20, color="darkorange")
        ax.set_xlim(xmm[0], xmm[-1])
        ax.set_ylim(bottom=0.0)
        ax.set_xlabel("x'  [mm]   (perfil en y'=0)")
        ax.set_ylabel("I / I_inc  (absoluta)")
        ax.set_title("Intensidad absoluta  I(x',0) = |U|²  (centro = (Área/λz)²)")

        ax = self.ax_norm
        ax.clear()
        ax.plot(xmm, I_norm, color="crimson", lw=1.0)
        ax.fill_between(xmm, I_norm, alpha=0.15, color="crimson")
        ax.set_xlim(xmm[0], xmm[-1])
        ax.set_ylim(bottom=0.0)
        ax.set_xlabel("x'  [mm]")
        ax.set_ylabel("I / I₀  (normalizada)")
        ax.set_title("Misma curva normalizada (comparación)")

        axial = I_abs[np.argmin(np.abs(x))] if x.size else float("nan")
        self.status.set(
            f"I_abs(x'=0) = {axial:.4e}\n"
            f"(= (Área/λz)², relativa a la\n"
            f" onda incidente amplitud 1)")
        self.status_lbl.configure(foreground="#333333")
        self.canvas.draw_idle()


def main():
    root = tk.Tk()
    root.title("Código 20 — Difracción de Fraunhofer analítica 2D")
    nb = ttk.Notebook(root)
    nb.pack(fill="both", expand=True)

    t1 = ttk.Frame(nb)
    nb.add(t1, text="Patrón (normalizado)")
    HostFraunhofer(t1)

    t2 = ttk.Frame(nb)
    nb.add(t2, text="Intensidad absoluta")
    TabAbsoluto(t2)

    root.mainloop()


if __name__ == "__main__":
    main()
