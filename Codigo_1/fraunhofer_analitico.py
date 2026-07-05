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


# ---- Paralelogramo (lado a ‖ eje x, lado b a ángulo θ respecto de a) -------

def amplitud_paralelogramo(fx, fy, a, b, theta):
    """
    TF de Fraunhofer de un paralelogramo con lado a a lo largo del eje x y
    lado b formando ángulo θ con el lado a (fórmula genérica del
    Contexto_códigos.md, Código 20). Si θ=90° se reduce al rectángulo a×b
    estándar (producto de sinc independientes) — caso ya validado en la
    Pestaña 0.
    """
    return a * b * np.sinc(a * fx) * np.sinc(
        b * (fx * np.cos(theta) + fy * np.sin(theta)))


def intensidad_paralelogramo(X, Y, a, b, theta, lam, z):
    fx = X / (lam * z)
    fy = Y / (lam * z)
    A = amplitud_paralelogramo(fx, fy, a, b, theta)
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


# ---- Círculo con muesca (segmento circular cortado) ------------------------

def area_circulo_con_muesca(R, ancho_muesca):
    """
    Área del círculo de radio R al que se le remueve el segmento circular
    inferior cuya cuerda tiene ancho `ancho_muesca` (≤ 2R).

    Geometría: la cuerda a altura y0=-h (h>0) tiene semi-ancho
    √(R²−h²) = ancho_muesca/2  →  h = √(R² − (ancho_muesca/2)²).
    Ángulo central del segmento: φ = 2·arccos(h/R).
    Área del segmento = R²/2·(φ − sen φ).  Área final = πR² − segmento.

    Con R=1mm y ancho_muesca=√2 mm≈1.414mm (caso del enunciado) da φ=π/2
    exactamente (la cuerda está a 45°) y Área ≈ 2.8562 mm².
    """
    semi = ancho_muesca / 2.0
    if semi <= 0.0 or semi >= R:
        return np.pi * R ** 2
    h = np.sqrt(max(R ** 2 - semi ** 2, 0.0))
    phi = 2.0 * np.arccos(np.clip(h / R, -1.0, 1.0))
    segmento = 0.5 * R ** 2 * (phi - np.sin(phi))
    return np.pi * R ** 2 - segmento


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


def mascara_circulo_muesca(X, Y, R, ancho_muesca):
    """
    Máscara booleana (True=transparente) del círculo con el segmento inferior
    recortado, sobre la malla (X,Y) del PLANO DE LA ABERTURA [m]. Se usa solo
    para la visualización NUMÉRICA del patrón 2D (FFT) — el valor axial
    cerrado se calcula aparte con `area_circulo_con_muesca` (método analítico).
    """
    semi = ancho_muesca / 2.0
    h = np.sqrt(max(R ** 2 - semi ** 2, 0.0)) if semi < R else 0.0
    dentro_circulo = (X ** 2 + Y ** 2) <= R ** 2
    fuera_muesca = Y >= -h
    return dentro_circulo & fuera_muesca


RESOLUCION_ABERTURA = 60  # muestras deseadas a través de max(R, ancho_muesca)


def patron_fft_muesca(R, ancho_muesca, lam, z, xmax, N):
    """
    Patrón de Fraunhofer NUMÉRICO (no analítico) de la abertura círculo-con-
    muesca, vía `fft2`.

    La resolución en el plano de OBSERVACIÓN es  dx' = λz/L_ap  — depende
    solo de la ventana física de la abertura `L_ap`, NO de N. Para que el
    slider N sirva realmente para "ver más fino" (en vez de solo extender el
    rango), se fija primero una resolución de muestreo de la ABERTURA
    (dx_ap = max(R,muesca)/RESOLUCION_ABERTURA) y se hace crecer la ventana
    con N:  L_ap = N·dx_ap  (equivale a "zero-padding": más N ⇒ ventana más
    grande ⇒ dx' más fino, la interpolación estándar de la FFT). Se impone
    además un piso  L_ap ≥ 4×max(R,muesca)  para no recortar la abertura
    cuando N está en su valor mínimo.

    El slider `xmax` solo RECORTA el rango ya calculado — pedir un `xmax`
    mayor al disponible simplemente se satura al máximo alcanzable (nunca se
    inventa una extensión mayor a la realmente calculada).

    Devuelve (x_obs [m] recortado, I2D_rel recortado, máscara, x_ap [m]).
    """
    dx_ap_objetivo = max(R, ancho_muesca) / RESOLUCION_ABERTURA
    margen_min = 4.0 * max(R, ancho_muesca)
    L_ap = max(margen_min, N * dx_ap_objetivo)

    x_ap = np.linspace(-L_ap / 2, L_ap / 2, N, endpoint=False)
    dx_ap = x_ap[1] - x_ap[0]
    Xap, Yap = np.meshgrid(x_ap, x_ap)
    mascara = mascara_circulo_muesca(Xap, Yap, R, ancho_muesca).astype(float)

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


class FraunhoferGUI:

    def __init__(self, parent):
        self.parent = parent

        self._build_controls()
        self._build_figure()
        self.recompute()

    # ------------------------------------------------------------------ UI
    def _slider(self, parent, label, frm, to, init, fmt="{:.2f}"):
        var = crear_slider(parent, label, frm, to, init, self.recompute, fmt)
        self._sliders.append(var)
        return var

    def _build_controls(self):
        self._sliders = []
        panel = ttk.Frame(self.parent, padding=8)
        panel.pack(side="left", fill="y")

        ttk.Label(panel, text="Difracción de Fraunhofer",
                  font=("", 11, "bold")).pack(anchor="w", pady=(0, 6))

        # Geometría de la abertura (µm)
        f1 = ttk.LabelFrame(panel, text="Abertura — marco (µm)", padding=6)
        f1.pack(fill="x", pady=4)
        self.a = self._slider(f1, "a  ext. ancho", 0.0, 400.0, 120.0, "{:.1f}")
        self.b = self._slider(f1, "b  ext. alto",  0.0, 400.0, 160.0, "{:.1f}")
        self.c = self._slider(f1, "c  int. ancho", 0.0, 400.0, 60.0, "{:.1f}")
        self.d = self._slider(f1, "d  int. alto",  0.0, 400.0, 90.0, "{:.1f}")

        f2 = ttk.LabelFrame(panel, text="Abertura — círculo y separación (µm)",
                            padding=6)
        f2.pack(fill="x", pady=4)
        self.R = self._slider(f2, "R  radio",      0.0, 300.0, 80.0, "{:.1f}")
        self.D = self._slider(f2, "D  separación", 0.0, 1500.0, 400.0, "{:.1f}")

        # Fuente
        f3 = ttk.LabelFrame(panel, text="Fuente", padding=6)
        f3.pack(fill="x", pady=4)
        self.lam = self._slider(f3, "λ (nm)", 380.0, 1000.0, 633.0, "{:.0f}")

        # Plano de observación
        f4 = ttk.LabelFrame(panel, text="Plano de observación", padding=6)
        f4.pack(fill="x", pady=4)
        self.z = self._slider(f4, "z (m)", 0.05, 20.0, 5.0, "{:.2f}")
        self.xmax = self._slider(f4, "x'_max (mm)", 1.0, 80.0, 25.0, "{:.1f}")
        self.N = self._slider(f4, "N (px)", 200.0, float(NX_MAX), 500.0, "{:.0f}")

        # Visualización
        f5 = ttk.LabelFrame(panel, text="Escala de intensidad", padding=6)
        f5.pack(fill="x", pady=4)
        self.escala = tk.StringVar(value="gamma")
        for txt, val in (("Lineal", "lineal"), ("γ (0.4)", "gamma"),
                         ("Log", "log")):
            ttk.Radiobutton(f5, text=txt, variable=self.escala, value=val,
                            command=self.recompute).pack(side="left")

        # Estado / régimen
        st = ttk.LabelFrame(panel, text="Régimen del cálculo", padding=6)
        st.pack(fill="x", pady=4)
        self.status = tk.StringVar(value="")
        self.status_lbl = ttk.Label(st, textvariable=self.status, justify="left",
                                    font=("Consolas", 9))
        self.status_lbl.pack(anchor="w")

    def _build_figure(self):
        right = ttk.Frame(self.parent)
        right.pack(side="left", fill="both", expand=True)

        self.fig = Figure(figsize=(10.5, 8.0))
        gs = self.fig.add_gridspec(2, 2, hspace=0.30, wspace=0.28,
                                   height_ratios=[1.5, 1])
        self.ax_ap = self.fig.add_subplot(gs[0, 0])   # abertura a escala
        self.ax_pat = self.fig.add_subplot(gs[0, 1])  # patrón 2D
        self.ax_prof = self.fig.add_subplot(gs[1, :])  # perfil I(x',0)

        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        NavigationToolbar2Tk(self.canvas, right).update()

    # -------------------------------------------------------------- lectura
    def _leer(self):
        """Lee los sliders y convierte a unidades SI (µm→m, nm→m, mm→m)."""
        return dict(
            a=self.a.get() * 1e-6,
            b=self.b.get() * 1e-6,
            c=self.c.get() * 1e-6,
            d=self.d.get() * 1e-6,
            R=self.R.get() * 1e-6,
            D=self.D.get() * 1e-6,
            lam=self.lam.get() * 1e-9,
            z=self.z.get(),
            xmax=self.xmax.get() * 1e-3,
            N=int(self.N.get()),
        )

    # ----------------------------------------------------------- recálculo
    def recompute(self):
        p = self._leer()

        # Malla del plano de observación (X, Y) en metros.
        x = np.linspace(-p["xmax"], p["xmax"], p["N"])
        X, Y = np.meshgrid(x, x)
        I = intensidad(X, Y, p)

        self._draw_aperture(p)
        self._draw_pattern(I, p)
        self._draw_profile(x, I, p)
        self._update_status(p)

        self.canvas.draw_idle()

    def _draw_aperture(self, p):
        """Dibuja la abertura a escala (µm): marco (con hueco) + círculo."""
        ax = self.ax_ap
        ax.clear()
        um = 1e6  # m → µm para el dibujo

        a, b, c, d = p["a"] * um, p["b"] * um, p["c"] * um, p["d"] * um
        R, D = p["R"] * um, p["D"] * um
        # Centros: marco en -D/2, círculo en +D/2.
        xm, xc = -D / 2.0, +D / 2.0

        ax.set_facecolor("#111111")
        # Marco: rectángulo exterior claro + interior en color del fondo (hueco).
        if a > 0 and b > 0:
            ax.add_patch(Rectangle((xm - a / 2, -b / 2), a, b,
                                   facecolor="white", edgecolor="none"))
        if c > 0 and d > 0:
            ax.add_patch(Rectangle((xm - c / 2, -d / 2), c, d,
                                   facecolor="#111111", edgecolor="none"))
        if R > 0:
            ax.add_patch(Circle((xc, 0.0), R, facecolor="white", edgecolor="none"))

        # Cota de separación D.
        if D > 0:
            ytop = max(b, 2 * R) / 2 * 1.15 + 5
            ax.annotate("", xy=(xc, ytop), xytext=(xm, ytop),
                        arrowprops=dict(arrowstyle="<->", color="orange"))
            ax.text(0.0, ytop + 4, "D", color="orange", ha="center",
                    va="bottom", fontsize=11)

        semi = max(a, c, 2 * R) / 2 + D / 2 + 10
        semiy = max(b, 2 * R) / 2 * 1.15 + 20
        ax.set_xlim(-semi, semi)
        ax.set_ylim(-semiy, semiy)
        ax.set_aspect("equal")
        ax.set_xlabel("x̃  [µm]")
        ax.set_ylabel("ỹ  [µm]")
        ax.set_title("Plano de la abertura (a escala)")

    def _escala_norm(self, I):
        """Devuelve (datos, norm) según la escala de visualización elegida."""
        modo = self.escala.get()
        if modo == "log":
            piso = I.max() * 1e-5 if I.max() > 0 else 1e-9
            return I + piso, LogNorm(vmin=piso, vmax=max(I.max(), piso * 10))
        if modo == "gamma":
            return I, PowerNorm(gamma=0.4, vmin=0.0, vmax=max(I.max(), 1e-12))
        return I, None  # lineal

    def _draw_pattern(self, I, p):
        ax = self.ax_pat
        ax.clear()
        ext = p["xmax"] * 1e3  # m → mm
        datos, norm = self._escala_norm(I)
        ax.imshow(datos, extent=[-ext, ext, -ext, ext], origin="lower",
                  cmap="inferno", norm=norm,
                  vmax=(None if norm is not None else max(I.max(), 1e-12)))
        ax.set_xlabel("x'  [mm]")
        ax.set_ylabel("y'  [mm]")
        ax.set_title("Fraunhofer analítico — sinc² (marco) + Airy (círculo)")

    def _draw_profile(self, x, I, p):
        ax = self.ax_prof
        ax.clear()
        j = I.shape[0] // 2  # fila central y'=0
        ax.plot(x * 1e3, I[j, :], color="crimson", lw=1.0)
        ax.fill_between(x * 1e3, I[j, :], alpha=0.20, color="crimson")
        ax.set_xlim(-p["xmax"] * 1e3, p["xmax"] * 1e3)
        ax.set_ylim(bottom=0.0)
        ax.set_xlabel("x'  [mm]   (perfil horizontal en y'=0)")
        ax.set_ylabel("I / I₀")
        ax.set_title("Perfil de intensidad  I(x', 0)")

    def _update_status(self, p):
        D_char, z_min, N_F, es_fh = regimen(p)
        if es_fh:
            color = "#127a12"
            etiqueta = "Régimen: FRAUNHOFER — cálculo válido (sinc²/Airy)"
        else:
            color = "#c00000"
            etiqueta = ("Régimen: FRESNEL (campo cercano)\n"
                        "El patrón sinc²/Airy NO es válido aquí.\n"
                        "→ usar el Código 22 (Fresnel).")
        txt = (
            f"D_char = {D_char*1e6:8.1f} µm\n"
            f"z_min  = {z_min:8.3f} m   (2·D²/λ)\n"
            f"z      = {p['z']:8.3f} m\n"
            f"N_F    = {N_F:8.3f}   (= D²/λz)\n"
            f"─────────────────────────\n"
            f"{etiqueta}"
        )
        self.status.set(txt)
        self.status_lbl.configure(foreground=color)


# =============================================================================
# 3. PESTAÑAS DE TALLER
# =============================================================================

class TabParalelogramo:
    """Ejercicio: paralelogramo a=10µm, b=5µm, θ=60° (fórmula genérica Código 20)."""

    def __init__(self, parent):
        self.parent = parent
        self._build_controls()
        self._build_figure()
        self.recompute()

    def _build_controls(self):
        panel = ttk.Frame(self.parent, padding=8)
        panel.pack(side="left", fill="y")
        ttk.Label(panel, text="Paralelogramo (a ‖ x, b a ángulo θ)",
                  font=("", 11, "bold")).pack(anchor="w", pady=(0, 6))

        f1 = ttk.LabelFrame(panel, text="Abertura (µm)", padding=6)
        f1.pack(fill="x", pady=4)
        self.a = crear_slider(f1, "a", 1.0, 50.0, 10.0, self.recompute, "{:.2f}")
        self.b = crear_slider(f1, "b", 1.0, 50.0, 5.0, self.recompute, "{:.2f}")
        self.theta = crear_slider(f1, "θ (grados)", 0.0, 180.0, 60.0,
                                  self.recompute, "{:.1f}")

        f3 = ttk.LabelFrame(panel, text="Fuente / observación", padding=6)
        f3.pack(fill="x", pady=4)
        self.lam = crear_slider(f3, "λ (nm)", 380.0, 1000.0, 633.0,
                                self.recompute, "{:.0f}")
        self.z = crear_slider(f3, "z (m)", 0.05, 5.0, 1.0, self.recompute, "{:.2f}")
        self.xmax = crear_slider(f3, "x'_max (mm)", 5.0, 300.0, 100.0,
                                 self.recompute, "{:.0f}")
        self.N = crear_slider(f3, "N (px)", 200.0, float(NX_MAX), 500.0,
                              self.recompute, "{:.0f}")

        f4 = ttk.LabelFrame(panel, text="Escala de intensidad", padding=6)
        f4.pack(fill="x", pady=4)
        self.escala = tk.StringVar(value="gamma")
        for txt, val in (("Lineal", "lineal"), ("γ (0.4)", "gamma"), ("Log", "log")):
            ttk.Radiobutton(f4, text=txt, variable=self.escala, value=val,
                            command=self.recompute).pack(side="left")

        st = ttk.LabelFrame(panel, text="Régimen del cálculo", padding=6)
        st.pack(fill="x", pady=4)
        self.status = tk.StringVar(value="")
        self.status_lbl = ttk.Label(st, textvariable=self.status, justify="left",
                                    font=("Consolas", 9))
        self.status_lbl.pack(anchor="w")

    def _build_figure(self):
        right = ttk.Frame(self.parent)
        right.pack(side="left", fill="both", expand=True)
        self.fig = Figure(figsize=(10.5, 8.0))
        gs = self.fig.add_gridspec(2, 2, hspace=0.30, wspace=0.28,
                                   height_ratios=[1.5, 1])
        self.ax_ap = self.fig.add_subplot(gs[0, 0])
        self.ax_pat = self.fig.add_subplot(gs[0, 1])
        self.ax_prof = self.fig.add_subplot(gs[1, :])
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        NavigationToolbar2Tk(self.canvas, right).update()

    def recompute(self):
        a, b = self.a.get() * 1e-6, self.b.get() * 1e-6
        theta = np.radians(self.theta.get())
        lam, z = self.lam.get() * 1e-9, self.z.get()
        xmax, N = self.xmax.get() * 1e-3, int(self.N.get())

        x = np.linspace(-xmax, xmax, N)
        X, Y = np.meshgrid(x, x)
        I = intensidad_paralelogramo(X, Y, a, b, theta, lam, z)

        # --- Abertura a escala ---
        ax = self.ax_ap
        ax.clear()
        um = 1e6
        av, bv = a * um, b * um
        verts = np.array([(0, 0), (av, 0),
                          (av + bv * np.cos(theta), bv * np.sin(theta)),
                          (bv * np.cos(theta), bv * np.sin(theta))])
        centro = verts.mean(axis=0)
        verts = verts - centro
        ax.set_facecolor("#111111")
        from matplotlib.patches import Polygon
        ax.add_patch(Polygon(verts, closed=True, facecolor="white", edgecolor="none"))
        semi = max(av, bv) * 1.3
        ax.set_xlim(-semi, semi)
        ax.set_ylim(-semi, semi)
        ax.set_aspect("equal")
        ax.set_xlabel("x̃  [µm]")
        ax.set_ylabel("ỹ  [µm]")
        ax.set_title("Plano de la abertura (a escala)")

        # --- Patrón 2D ---
        ax = self.ax_pat
        ax.clear()
        ext = xmax * 1e3
        datos, norm = _escala_norm(I, self.escala.get())
        ax.imshow(datos, extent=[-ext, ext, -ext, ext], origin="lower",
                  cmap="inferno", norm=norm,
                  vmax=(None if norm is not None else max(I.max(), 1e-12)))
        ax.set_xlabel("x'  [mm]")
        ax.set_ylabel("y'  [mm]")
        ax.set_title("Fraunhofer analítico — paralelogramo (sinc²)")

        # --- Perfil ---
        ax = self.ax_prof
        ax.clear()
        j = I.shape[0] // 2
        ax.plot(x * 1e3, I[j, :], color="crimson", lw=1.0)
        ax.fill_between(x * 1e3, I[j, :], alpha=0.20, color="crimson")
        ax.set_xlim(-xmax * 1e3, xmax * 1e3)
        ax.set_ylim(bottom=0.0)
        ax.set_xlabel("x'  [mm]   (perfil horizontal en y'=0)")
        ax.set_ylabel("I / I₀")
        ax.set_title("Perfil de intensidad  I(x', 0)")

        # --- Régimen ---
        D_char = np.hypot(a, b)
        z_min, N_F, es_fh = regimen_generico(D_char, lam, z)
        _actualizar_status_regimen(self.status, self.status_lbl, D_char, z_min, N_F, es_fh, z)

        self.canvas.draw_idle()


class TabCruz:
    """Ejercicio: abertura en cruz, brazos de ancho a, longitud total L."""

    def __init__(self, parent):
        self.parent = parent
        self._build_controls()
        self._build_figure()
        self.recompute()

    def _build_controls(self):
        panel = ttk.Frame(self.parent, padding=8)
        panel.pack(side="left", fill="y")
        ttk.Label(panel, text="Abertura en cruz",
                  font=("", 11, "bold")).pack(anchor="w", pady=(0, 6))

        f1 = ttk.LabelFrame(panel, text="Abertura (µm)", padding=6)
        f1.pack(fill="x", pady=4)
        self.L = crear_slider(f1, "L  largo total", 20.0, 500.0, 200.0,
                              self.recompute, "{:.1f}")
        self.a = crear_slider(f1, "a  ancho brazo", 5.0, 500.0, 50.0,
                              self.recompute, "{:.1f}")

        f3 = ttk.LabelFrame(panel, text="Fuente / observación", padding=6)
        f3.pack(fill="x", pady=4)
        self.lam = crear_slider(f3, "λ (nm)", 380.0, 1000.0, 633.0,
                                self.recompute, "{:.0f}")
        self.z = crear_slider(f3, "z (m)", 0.05, 5.0, 1.0, self.recompute, "{:.2f}")
        self.xmax = crear_slider(f3, "x'_max (mm)", 5.0, 300.0, 60.0,
                                 self.recompute, "{:.0f}")
        self.N = crear_slider(f3, "N (px)", 200.0, float(NX_MAX), 500.0,
                              self.recompute, "{:.0f}")

        f4 = ttk.LabelFrame(panel, text="Escala de intensidad", padding=6)
        f4.pack(fill="x", pady=4)
        self.escala = tk.StringVar(value="gamma")
        for txt, val in (("Lineal", "lineal"), ("γ (0.4)", "gamma"), ("Log", "log")):
            ttk.Radiobutton(f4, text=txt, variable=self.escala, value=val,
                            command=self.recompute).pack(side="left")

        st = ttk.LabelFrame(panel, text="Régimen del cálculo", padding=6)
        st.pack(fill="x", pady=4)
        self.status = tk.StringVar(value="")
        self.status_lbl = ttk.Label(st, textvariable=self.status, justify="left",
                                    font=("Consolas", 9))
        self.status_lbl.pack(anchor="w")

    def _build_figure(self):
        right = ttk.Frame(self.parent)
        right.pack(side="left", fill="both", expand=True)
        self.fig = Figure(figsize=(10.5, 8.0))
        gs = self.fig.add_gridspec(2, 2, hspace=0.30, wspace=0.28,
                                   height_ratios=[1.5, 1])
        self.ax_ap = self.fig.add_subplot(gs[0, 0])
        self.ax_pat = self.fig.add_subplot(gs[0, 1])
        self.ax_prof = self.fig.add_subplot(gs[1, :])
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        NavigationToolbar2Tk(self.canvas, right).update()

    def recompute(self):
        L, a = self.L.get() * 1e-6, self.a.get() * 1e-6
        a = min(a, L)  # el brazo no puede ser más ancho que largo
        lam, z = self.lam.get() * 1e-9, self.z.get()
        xmax, N = self.xmax.get() * 1e-3, int(self.N.get())

        x = np.linspace(-xmax, xmax, N)
        X, Y = np.meshgrid(x, x)
        I = intensidad_cruz(X, Y, L, a, lam, z)

        # --- Abertura ---
        ax = self.ax_ap
        ax.clear()
        um = 1e6
        Lv, av = L * um, a * um
        ax.set_facecolor("#111111")
        ax.add_patch(Rectangle((-Lv / 2, -av / 2), Lv, av,
                               facecolor="white", edgecolor="none"))
        ax.add_patch(Rectangle((-av / 2, -Lv / 2), av, Lv,
                               facecolor="white", edgecolor="none"))
        semi = Lv * 0.65
        ax.set_xlim(-semi, semi)
        ax.set_ylim(-semi, semi)
        ax.set_aspect("equal")
        ax.set_xlabel("x̃  [µm]")
        ax.set_ylabel("ỹ  [µm]")
        ax.set_title("Plano de la abertura (a escala)")

        # --- Patrón 2D ---
        ax = self.ax_pat
        ax.clear()
        ext = xmax * 1e3
        datos, norm = _escala_norm(I, self.escala.get())
        ax.imshow(datos, extent=[-ext, ext, -ext, ext], origin="lower",
                  cmap="inferno", norm=norm,
                  vmax=(None if norm is not None else max(I.max(), 1e-12)))
        ax.set_xlabel("x'  [mm]")
        ax.set_ylabel("y'  [mm]")
        ax.set_title("Fraunhofer analítico — cruz (unión de sinc²)")

        # --- Perfil ---
        ax = self.ax_prof
        ax.clear()
        j = I.shape[0] // 2
        ax.plot(x * 1e3, I[j, :], color="crimson", lw=1.0)
        ax.fill_between(x * 1e3, I[j, :], alpha=0.20, color="crimson")
        ax.set_xlim(-xmax * 1e3, xmax * 1e3)
        ax.set_ylim(bottom=0.0)
        ax.set_xlabel("x'  [mm]   (perfil horizontal en y'=0)")
        ax.set_ylabel("I / I₀")
        ax.set_title("Perfil de intensidad  I(x', 0)")

        # --- Régimen ---
        D_char = L * np.sqrt(2.0)   # diagonal del bounding box L×L
        z_min, N_F, es_fh = regimen_generico(D_char, lam, z)
        _actualizar_status_regimen(self.status, self.status_lbl, D_char, z_min, N_F, es_fh, z)

        self.canvas.draw_idle()


class TabCirculoMuesca:
    """
    Ejercicio: círculo D=2mm (R=1mm) con muesca rectangular de ancho 1.414mm
    (≈R√2, cuerda a 45°). Pide la irradiancia AXIAL en z=4m, λ=500nm — se
    calcula de forma CERRADA (analítica) y además se visualiza el patrón 2D
    de forma NUMÉRICA (FFT de la máscara rasterizada), declarando en pantalla
    que ese panel usa un método distinto (numérico) al resto de pestañas.
    """

    def __init__(self, parent):
        self.parent = parent
        self._build_controls()
        self._build_figure()
        self.recompute()

    def _build_controls(self):
        panel = ttk.Frame(self.parent, padding=8)
        panel.pack(side="left", fill="y")
        ttk.Label(panel, text="Círculo con muesca",
                  font=("", 11, "bold")).pack(anchor="w", pady=(0, 6))

        f1 = ttk.LabelFrame(panel, text="Abertura (mm)", padding=6)
        f1.pack(fill="x", pady=4)
        self.R = crear_slider(f1, "R  radio", 0.1, 5.0, 1.0, self.recompute, "{:.3f}")
        self.muesca = crear_slider(f1, "ancho muesca", 0.05, 2.0, 1.41421,
                                   self.recompute, "{:.3f}")

        f3 = ttk.LabelFrame(panel, text="Fuente / observación", padding=6)
        f3.pack(fill="x", pady=4)
        self.lam = crear_slider(f3, "λ (nm)", 380.0, 1000.0, 500.0,
                                self.recompute, "{:.0f}")
        self.z = crear_slider(f3, "z (m)", 0.1, 20.0, 4.0, self.recompute, "{:.2f}")
        self.xmax = crear_slider(f3, "x'_max (mm)", 0.5, 100.0, 5.0,
                                 self.recompute, "{:.2f}")
        self.N = crear_slider(f3, "N (px, FFT)", 128.0, float(NX_MAX), 800.0,
                              self.recompute, "{:.0f}")

        f4 = ttk.LabelFrame(panel, text="Escala de intensidad", padding=6)
        f4.pack(fill="x", pady=4)
        self.escala = tk.StringVar(value="gamma")
        for txt, val in (("Lineal", "lineal"), ("γ (0.4)", "gamma"), ("Log", "log")):
            ttk.Radiobutton(f4, text=txt, variable=self.escala, value=val,
                            command=self.recompute).pack(side="left")

        st = ttk.LabelFrame(panel, text="Resultado pedido — irradiancia axial",
                            padding=6)
        st.pack(fill="x", pady=4)
        self.status = tk.StringVar(value="")
        self.status_lbl = ttk.Label(st, textvariable=self.status, justify="left",
                                    font=("Consolas", 9))
        self.status_lbl.pack(anchor="w")

    def _build_figure(self):
        right = ttk.Frame(self.parent)
        right.pack(side="left", fill="both", expand=True)
        self.fig = Figure(figsize=(10.5, 8.0))
        gs = self.fig.add_gridspec(2, 2, hspace=0.30, wspace=0.28,
                                   height_ratios=[1.5, 1])
        self.ax_ap = self.fig.add_subplot(gs[0, 0])
        self.ax_pat = self.fig.add_subplot(gs[0, 1])
        self.ax_prof = self.fig.add_subplot(gs[1, :])
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        NavigationToolbar2Tk(self.canvas, right).update()

    def recompute(self):
        R = self.R.get() * 1e-3
        muesca = min(self.muesca.get() * 1e-3, 2.0 * R * 0.999)
        lam, z = self.lam.get() * 1e-9, self.z.get()
        xmax, N = self.xmax.get() * 1e-3, int(self.N.get())

        area = area_circulo_con_muesca(R, muesca)
        I0_axial = irradiancia_axial_relativa(area, lam, z)
        x_obs, I2D, mascara, x_ap = patron_fft_muesca(R, muesca, lam, z, xmax, N)

        # Valor central del FFT (validación cruzada contra el cerrado).
        j0 = np.argmin(np.abs(x_obs))
        I_fft_centro = I2D[j0, j0] if I2D.size else float("nan")

        # --- Abertura (máscara rasterizada) ---
        ax = self.ax_ap
        ax.clear()
        ext_ap = x_ap[-1] * 1e3
        ax.imshow(mascara, extent=[-ext_ap, ext_ap, -ext_ap, ext_ap],
                  origin="lower", cmap="gray", vmin=0, vmax=1)
        ax.set_xlabel("x̃  [mm]")
        ax.set_ylabel("ỹ  [mm]")
        ax.set_title("Máscara de la abertura (rasterizada)")
        ax.set_xlim(-3.5 * R * 1e3, 3.5 * R * 1e3)
        ax.set_ylim(-3.5 * R * 1e3, 3.5 * R * 1e3)

        # --- Patrón 2D (NUMÉRICO, FFT) ---
        ax = self.ax_pat
        ax.clear()
        ext = x_obs[-1] * 1e3 if x_obs.size else xmax * 1e3
        datos, norm = _escala_norm(I2D, self.escala.get())
        ax.imshow(datos, extent=[-ext, ext, -ext, ext], origin="lower",
                  cmap="inferno", norm=norm,
                  vmax=(None if norm is not None else max(I2D.max(), 1e-12)))
        ax.set_xlabel("x'  [mm]")
        ax.set_ylabel("y'  [mm]")
        ax.set_title("Patrón NUMÉRICO (FFT de la máscara) — no analítico")

        # --- Perfil ---
        ax = self.ax_prof
        ax.clear()
        j = I2D.shape[0] // 2 if I2D.size else 0
        if I2D.size:
            ax.plot(x_obs * 1e3, I2D[j, :], color="crimson", lw=1.0)
            ax.fill_between(x_obs * 1e3, I2D[j, :], alpha=0.20, color="crimson")
        ax.set_xlim(-xmax * 1e3, xmax * 1e3)
        ax.set_ylim(bottom=0.0)
        ax.set_xlabel("x'  [mm]   (perfil horizontal en y'=0)")
        ax.set_ylabel("I / I₀")
        ax.set_title("Perfil de intensidad  I(x', 0)  [FFT]")

        # --- Resultado pedido + validación cruzada ---
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
        self.status.set(txt)
        self.status_lbl.configure(foreground="#127a12" if diff_pct < 5 else "#c00000")

        self.canvas.draw_idle()


class TabDobleCuadrado:
    """Ejercicio: cuadrados de lado a y 3a, separados 2a borde a borde (D=4a)."""

    def __init__(self, parent):
        self.parent = parent
        self._build_controls()
        self._build_figure()
        self.recompute()

    def _build_controls(self):
        panel = ttk.Frame(self.parent, padding=8)
        panel.pack(side="left", fill="y")
        ttk.Label(panel, text="Doble cuadrado (a y 3a, separación 2a)",
                  font=("", 11, "bold")).pack(anchor="w", pady=(0, 6))

        f1 = ttk.LabelFrame(panel, text="Abertura (µm)", padding=6)
        f1.pack(fill="x", pady=4)
        self.a = crear_slider(f1, "a  (3a automático)", 5.0, 200.0, 50.0,
                              self.recompute, "{:.1f}")

        f3 = ttk.LabelFrame(panel, text="Fuente / observación", padding=6)
        f3.pack(fill="x", pady=4)
        self.lam = crear_slider(f3, "λ (nm)", 380.0, 1000.0, 633.0,
                                self.recompute, "{:.0f}")
        self.z = crear_slider(f3, "z (m)", 0.05, 5.0, 2.0, self.recompute, "{:.2f}")
        self.xmax = crear_slider(f3, "x'_max (mm)", 5.0, 300.0, 60.0,
                                 self.recompute, "{:.0f}")
        self.N = crear_slider(f3, "N (px)", 200.0, float(NX_MAX), 500.0,
                              self.recompute, "{:.0f}")

        f4 = ttk.LabelFrame(panel, text="Escala de intensidad", padding=6)
        f4.pack(fill="x", pady=4)
        self.escala = tk.StringVar(value="gamma")
        for txt, val in (("Lineal", "lineal"), ("γ (0.4)", "gamma"), ("Log", "log")):
            ttk.Radiobutton(f4, text=txt, variable=self.escala, value=val,
                            command=self.recompute).pack(side="left")

        st = ttk.LabelFrame(panel, text="Régimen del cálculo", padding=6)
        st.pack(fill="x", pady=4)
        self.status = tk.StringVar(value="")
        self.status_lbl = ttk.Label(st, textvariable=self.status, justify="left",
                                    font=("Consolas", 9))
        self.status_lbl.pack(anchor="w")

    def _build_figure(self):
        right = ttk.Frame(self.parent)
        right.pack(side="left", fill="both", expand=True)
        self.fig = Figure(figsize=(10.5, 8.0))
        gs = self.fig.add_gridspec(2, 2, hspace=0.30, wspace=0.28,
                                   height_ratios=[1.5, 1])
        self.ax_ap = self.fig.add_subplot(gs[0, 0])
        self.ax_pat = self.fig.add_subplot(gs[0, 1])
        self.ax_prof = self.fig.add_subplot(gs[1, :])
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        NavigationToolbar2Tk(self.canvas, right).update()

    def recompute(self):
        a = self.a.get() * 1e-6
        lam, z = self.lam.get() * 1e-9, self.z.get()
        xmax, N = self.xmax.get() * 1e-3, int(self.N.get())

        x = np.linspace(-xmax, xmax, N)
        X, Y = np.meshgrid(x, x)
        I, D = intensidad_doble_cuadrado(X, Y, a, lam, z)

        # --- Abertura ---
        ax = self.ax_ap
        ax.clear()
        um = 1e6
        av, Dv = a * um, D * um
        x1, x2 = -Dv / 2.0, +Dv / 2.0
        ax.set_facecolor("#111111")
        ax.add_patch(Rectangle((x1 - av / 2, -av / 2), av, av,
                               facecolor="white", edgecolor="none"))
        ax.add_patch(Rectangle((x2 - 3 * av / 2, -3 * av / 2), 3 * av, 3 * av,
                               facecolor="white", edgecolor="none"))
        semi = Dv / 2.0 + 2.5 * av
        ax.set_xlim(-semi, semi)
        ax.set_ylim(-semi, semi)
        ax.set_aspect("equal")
        ax.set_xlabel("x̃  [µm]")
        ax.set_ylabel("ỹ  [µm]")
        ax.set_title(f"Plano de la abertura (D={Dv:.1f} µm, a escala)")

        # --- Patrón 2D ---
        ax = self.ax_pat
        ax.clear()
        ext = xmax * 1e3
        datos, norm = _escala_norm(I, self.escala.get())
        ax.imshow(datos, extent=[-ext, ext, -ext, ext], origin="lower",
                  cmap="inferno", norm=norm,
                  vmax=(None if norm is not None else max(I.max(), 1e-12)))
        ax.set_xlabel("x'  [mm]")
        ax.set_ylabel("y'  [mm]")
        ax.set_title("Fraunhofer analítico — doble cuadrado (sinc² + interferencia)")

        # --- Perfil ---
        ax = self.ax_prof
        ax.clear()
        j = I.shape[0] // 2
        ax.plot(x * 1e3, I[j, :], color="crimson", lw=1.0)
        ax.fill_between(x * 1e3, I[j, :], alpha=0.20, color="crimson")
        ax.set_xlim(-xmax * 1e3, xmax * 1e3)
        ax.set_ylim(bottom=0.0)
        ax.set_xlabel("x'  [mm]   (perfil horizontal en y'=0)")
        ax.set_ylabel("I / I₀")
        ax.set_title("Perfil de intensidad  I(x', 0)")

        # --- Régimen ---
        D_char = D + a / 2.0 + 1.5 * a
        z_min, N_F, es_fh = regimen_generico(D_char, lam, z)
        _actualizar_status_regimen(self.status, self.status_lbl, D_char, z_min, N_F, es_fh, z)

        self.canvas.draw_idle()


# =============================================================================
# 4. UTILIDADES COMPARTIDAS DE DIBUJO (usadas solo por las pestañas de taller)
# =============================================================================

def _escala_norm(I, modo):
    """Misma lógica que `FraunhoferGUI._escala_norm`, como función libre para
    que la reutilicen las pestañas de taller sin depender de esa clase."""
    if modo == "log":
        piso = I.max() * 1e-5 if I.max() > 0 else 1e-9
        return I + piso, LogNorm(vmin=piso, vmax=max(I.max(), piso * 10))
    if modo == "gamma":
        return I, PowerNorm(gamma=0.4, vmin=0.0, vmax=max(I.max(), 1e-12))
    return I, None


def _actualizar_status_regimen(status_var, status_lbl, D_char, z_min, N_F, es_fh, z):
    """Misma caja de estado verde/rojo de la Pestaña 0, reutilizada aquí."""
    if es_fh:
        color = "#127a12"
        etiqueta = "Régimen: FRAUNHOFER — cálculo válido (sinc²)"
    else:
        color = "#c00000"
        etiqueta = ("Régimen: FRESNEL (campo cercano)\n"
                    "El patrón sinc² NO es válido aquí.\n"
                    "→ usar el Código 22 (Fresnel).")
    txt = (
        f"D_char = {D_char*1e6:8.1f} µm\n"
        f"z_min  = {z_min:8.3f} m   (2·D²/λ)\n"
        f"z      = {z:8.3f} m\n"
        f"N_F    = {N_F:8.3f}   (= D²/λz)\n"
        f"─────────────────────────\n"
        f"{etiqueta}"
    )
    status_var.set(txt)
    status_lbl.configure(foreground=color)


def main():
    root = tk.Tk()
    root.title("Código 20 — Difracción de Fraunhofer analítica 2D")

    nb = ttk.Notebook(root)
    nb.pack(fill="both", expand=True)

    tab0 = ttk.Frame(nb)
    nb.add(tab0, text="Parcial 4 — Punto 1")
    FraunhoferGUI(tab0)

    tab1 = ttk.Frame(nb)
    nb.add(tab1, text="Taller — Paralelogramo")
    TabParalelogramo(tab1)

    tab2 = ttk.Frame(nb)
    nb.add(tab2, text="Taller — Cruz")
    TabCruz(tab2)

    tab3 = ttk.Frame(nb)
    nb.add(tab3, text="Taller — Círculo con muesca")
    TabCirculoMuesca(tab3)

    tab4 = ttk.Frame(nb)
    nb.add(tab4, text="Taller — Doble cuadrado")
    TabDobleCuadrado(tab4)

    root.mainloop()


if __name__ == "__main__":
    main()
