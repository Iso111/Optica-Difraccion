"""
Código 21 — Fraunhofer numérico vía Transformada de Fourier espacial 2D (FFT).

Punto 21 del taller: demostrar que el patrón de difracción de Fraunhofer es la
Transformada de Fourier 2D de la función de transmisión t(x,y) de la abertura:

    U(x', y') ∝ 𝓕{ t(x, y) }(f_x, f_y),      I(x', y') ∝ |U|²
    con  f_x = x'/(λz),  f_y = y'/(λz)   →   dx' = λz / L_ap

Es la contraparte NUMÉRICA del Código 20 (analítico): permite aberturas SIN
fórmula cerrada (corazón, hexágono, estrella, triángulo, compuestas…), al estilo
del simulador PhET de difracción (máscara a la izquierda, patrón a la derecha).

Idea (acordada con el usuario):
  · Se REUTILIZA el motor/ayudantes ya hechos del Código 20 (crear_slider,
    _escala_norm, regimen_generico, _status_regimen) — no se toca el Código 20.
  · El patrón se calcula rasterizando la máscara t(x,y) y aplicando fft2, con el
    mismo mapeo de muestreo VALIDADO en `patron_fft_semicirculos` del Código 20:
    dx' = λz/L_ap, ventana que crece con N ("zero-padding"), piso L_ap ≥ 4·tamaño.
  · GUI tkinter al estilo del Código 20/22 (Combobox + sliders + escala lin/γ/log).

Casos de validación (ver bloque de tests headless):
  · Rectángulo: el corte central del |FFT|² coincide con sinc²·sinc² (Código 20).
  · Círculo:    primer cero (disco de Airy) en 1.22·λz/diámetro.
"""

import os
import sys

import numpy as np
import tkinter as tk
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)

# --- Reutilización del Código 20 (helpers GUI + régimen, sin duplicar) --------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Codigo_1"))
from fraunhofer_analitico import (          # noqa: E402
    crear_slider, _escala_norm, regimen_generico, _status_regimen, _mascara_fina)


# =============================================================================
# 1. MÁSCARAS DE LAS ABERTURAS  (numpy booleano vectorizado; True = transparente)
#    Firma: mask(X, Y, p)  con p = lista de parámetros en SI y (X,Y) malla [m].
# =============================================================================

def mascara_circulo(X, Y, p):
    """Círculo/elipse de diámetro `diam` y excentricidad `ecc` (0=círculo).
    ry = rx·√(1−ecc²): a mayor ecc, más achatada en 'y'."""
    diam, ecc = p[0], p[1]
    rx = diam / 2.0
    ry = rx * np.sqrt(max(1.0 - ecc ** 2, 1e-6))
    return (X / rx) ** 2 + (Y / ry) ** 2 <= 1.0


def mascara_rectangulo(X, Y, p):
    """Rectángulo de ancho `w` y alto `h`."""
    w, h = p[0], p[1]
    return (np.abs(X) <= w / 2.0) & (np.abs(Y) <= h / 2.0)


def mascara_circulo_cuadrado(X, Y, p):
    """Círculo (diámetro `diam`) en x=−D/2 y cuadrado (lado `lado`) en x=+D/2,
    separados centro-a-centro `D` (abertura compuesta del PhET)."""
    diam, lado, D = p[0], p[1], p[2]
    R = diam / 2.0
    circ = (X + D / 2.0) ** 2 + Y ** 2 <= R ** 2
    cuad = (np.abs(X - D / 2.0) <= lado / 2.0) & (np.abs(Y) <= lado / 2.0)
    return circ | cuad


SEED_RED = 12345  # semilla fija → el desorden es reproducible y estable


def mascara_red_circulos(X, Y, p):
    """Malla Ng×Ng de círculos (diámetro `diam`, paso `paso`) con un jitter
    aleatorio de amplitud `desorden·paso` en cada centro (red de difracción 2D
    con desorden, como el PhET). Ng fijo = 4 (16 círculos)."""
    diam, paso, desorden = p[0], p[1], p[2]
    Ng = 4
    R = diam / 2.0
    rng = np.random.default_rng(SEED_RED)
    centros = (np.arange(Ng) - (Ng - 1) / 2.0) * paso
    m = np.zeros_like(X, dtype=bool)
    for cx in centros:
        for cy in centros:
            jx = (rng.random() - 0.5) * 2.0 * desorden * paso
            jy = (rng.random() - 0.5) * 2.0 * desorden * paso
            m |= (X - cx - jx) ** 2 + (Y - cy - jy) ** 2 <= R ** 2
    return m


def mascara_corazon(X, Y, p):
    """Corazón por curva implícita clásica (x²+y²−1)³ − x²·y³ ≤ 0, escalada por
    `s` (radio característico). El corazón queda con la punta hacia abajo."""
    s = p[0]
    x = X / s
    y = Y / s
    return (x ** 2 + y ** 2 - 1.0) ** 3 - x ** 2 * y ** 3 <= 0.0


def _rotar(X, Y, ang):
    """Rota la malla un ángulo `ang` [rad] (para hexágono/estrella/triángulo)."""
    c, s = np.cos(ang), np.sin(ang)
    return X * c + Y * s, -X * s + Y * c


def mascara_hexagono(X, Y, p):
    """Hexágono regular de radio (centro-a-vértice) `s`, girado `rot`. Definido
    como |proyección| ≤ apotema sobre las 3 direcciones normales a los lados."""
    s, rot = p[0], p[1]
    Xr, Yr = _rotar(X, Y, rot)
    apotema = s * np.sqrt(3.0) / 2.0
    dentro = np.ones_like(X, dtype=bool)
    for k in range(3):
        ang = np.pi * k / 3.0            # normales a 0°, 60°, 120°
        proy = Xr * np.cos(ang) + Yr * np.sin(ang)
        dentro &= np.abs(proy) <= apotema
    return dentro


def mascara_estrella(X, Y, p):
    """Estrella de `puntas` picos: polígono estrellado en polar, radio exterior
    `s` y radio interior `s·f_int`. Se compara el radio del punto contra r(θ)."""
    s, puntas = p[0], int(p[1])
    f_int = 0.42
    theta = np.arctan2(Y, X)
    r = np.hypot(X, Y)
    # r(θ) triangular entre r_ext (en cada punta) y r_int (entre puntas)
    ang_punta = 2.0 * np.pi / puntas
    fase = np.mod(theta, ang_punta) / ang_punta        # 0..1 dentro de un sector
    triang = 1.0 - 2.0 * np.abs(fase - 0.5)            # 0 en bordes, 1 al centro
    r_borde = s * (f_int + (1.0 - f_int) * triang)
    return r <= r_borde


def mascara_triangulo(X, Y, p):
    """Triángulo equilátero de radio (centro-a-vértice) `s`, girado `rot`.
    Intersección de 3 semiplanos. Su Fraunhofer es una estrella de 6 puntas."""
    s, rot = p[0], p[1]
    Xr, Yr = _rotar(X, Y, rot)
    apotema = s / 2.0                    # inradio del equilátero = s/2
    dentro = np.ones_like(X, dtype=bool)
    for k in range(3):
        ang = np.pi / 2.0 + 2.0 * np.pi * k / 3.0    # normales a los 3 lados
        proy = Xr * np.cos(ang) + Yr * np.sin(ang)
        dentro &= proy <= apotema
    return dentro


def mascara_corazon_hexagono(X, Y, p):
    """Corazón (radio `s`) en x=−D/2 y hexágono (radio `s`) en x=+D/2, separados
    centro-a-centro `D`. Su patrón muestra franjas de interferencia (por D)
    moduladas por la difracción de cada forma."""
    s, D = p[0], p[1]
    cor = mascara_corazon(X + D / 2.0, Y, [s])
    hexa = mascara_hexagono(X - D / 2.0, Y, [s, 0.0])
    return cor | hexa


# =============================================================================
# 2. MOTOR FFT GENÉRICO  (adaptado de patron_fft_semicirculos del Código 20)
# =============================================================================

RESOLUCION_ABERTURA = 60   # muestras a través del tamaño característico


def patron_fft(mask_fn, params, size_char, lam, z, xmax, N):
    """
    Patrón de Fraunhofer NUMÉRICO por FFT de la máscara `mask_fn(X,Y,params)`.

    Muestreo (idéntico al validado en el Código 20): la resolución en el plano
    de observación es dx' = λz/L_ap — depende de la ventana física L_ap, NO de N.
    Se fija dx_ap = size_char/RESOLUCION_ABERTURA y se hace crecer la ventana con
    N (L_ap = N·dx_ap ≡ zero-padding ⇒ dx' más fino), con piso L_ap ≥ 4·size_char
    para no recortar la abertura. `xmax` solo RECORTA el rango ya calculado.

    Devuelve (x_obs [m] recortado, I2D_rel recortado, máscara, x_ap [m]).
    """
    dx_ap = size_char / RESOLUCION_ABERTURA
    L_ap = max(4.0 * size_char, N * dx_ap)

    x_ap = np.linspace(-L_ap / 2.0, L_ap / 2.0, N, endpoint=False)
    dxa = x_ap[1] - x_ap[0]
    Xap, Yap = np.meshgrid(x_ap, x_ap)
    mascara = mask_fn(Xap, Yap, params).astype(float)

    U = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(mascara)))
    U *= dxa * dxa / (lam * z)              # regla de Riemann + prefactor 1/(λz)
    I2D = np.abs(U) ** 2                    # = I/I0 (I0 implícito = 1)

    x_obs = np.fft.fftshift(np.fft.fftfreq(N, d=dxa)) * lam * z

    xmax_ef = min(xmax, x_obs[-1])
    idx = np.where(np.abs(x_obs) <= xmax_ef)[0]
    return x_obs[idx], I2D[np.ix_(idx, idx)], mascara, x_ap


# =============================================================================
# 3. REGISTRO DE ABERTURAS
#    nombre → dict(titulo, sliders, mask, size, nota)
#    sliders: (clave, label, frm, to, init, escala_a_SI, fmt)
#    mask(X,Y,p): p = lista de valores en SI (solo params de forma)
#    size(p): tamaño característico [m] (ventana FFT + D_char del régimen)
# =============================================================================

# Sliders comunes de fuente/cálculo (λ, z, x'_max, N) que se anexan a todas.
# z llega a 50 m para poder mantener el régimen de Fraunhofer con aberturas
# grandes (que dan patrones de franjas finas más ricos, estilo "amiga").
_COMUNES = [
    ("lam", "λ (nm)", 380.0, 1000.0, 633.0, 1e-9, "{:.0f}"),
    ("z", "z (m)", 0.05, 50.0, 3.0, 1.0, "{:.2f}"),
    ("xmax", "x'_max (mm)", 1.0, 80.0, 20.0, 1e-3, "{:.1f}"),
    ("N", "N (px, FFT)", 200.0, 1400.0, 600.0, 1.0, "{:.0f}"),
]

_GRAD = np.pi / 180.0

APERTURAS_21 = {
    "Círculo / elipse": {
        "titulo": "Círculo / elipse — disco de Airy",
        "sliders": [("diam", "diámetro (µm)", 20.0, 600.0, 150.0, 1e-6, "{:.0f}"),
                    ("ecc", "excentricidad", 0.0, 0.99, 0.0, 1.0, "{:.2f}")],
        "mask": mascara_circulo,
        "size": lambda p: p[0],
        "nota": "F{círculo} = 2J1(πdρ)/(πdρ) → disco de Airy\nprimer cero en 1.22·λz/d",
    },
    "Rectángulo": {
        "titulo": "Rectángulo — sinc²·sinc² (coincide con Código 20)",
        "sliders": [("w", "ancho (µm)", 20.0, 600.0, 150.0, 1e-6, "{:.0f}"),
                    ("h", "alto (µm)", 20.0, 600.0, 150.0, 1e-6, "{:.0f}")],
        "mask": mascara_rectangulo,
        "size": lambda p: np.hypot(p[0], p[1]),
        "nota": "I = sinc²(w·x'/λz)·sinc²(h·y'/λz)\n(np.sinc normalizada)",
    },
    "Círculo + cuadrado": {
        "titulo": "Círculo + cuadrado separados (interferencia)",
        "sliders": [("diam", "diám. círculo (µm)", 20.0, 500.0, 120.0, 1e-6, "{:.0f}"),
                    ("lado", "lado cuadrado (µm)", 20.0, 500.0, 120.0, 1e-6, "{:.0f}"),
                    ("D", "separación D (µm)", 100.0, 1200.0, 350.0, 1e-6, "{:.0f}")],
        "mask": mascara_circulo_cuadrado,
        "size": lambda p: p[2] + max(p[0], p[1]),
        "nota": "Airy·+·sinc² con franjas cos(2πD·x'/λz)",
    },
    "Red de círculos (desorden)": {
        "titulo": "Red 4×4 de círculos con desorden",
        "sliders": [("diam", "diámetro (µm)", 10.0, 120.0, 50.0, 1e-6, "{:.0f}"),
                    ("paso", "paso (µm)", 50.0, 300.0, 140.0, 1e-6, "{:.0f}"),
                    ("des", "desorden (0–1)", 0.0, 1.0, 0.0, 1.0, "{:.2f}")],
        "mask": mascara_red_circulos,
        "size": lambda p: 4.0 * p[1] + p[0],
        "nota": "Red 2D: picos de Bragg bajo envolvente de Airy;\nel desorden difumina los órdenes altos",
    },
    "Corazón": {
        "titulo": "Corazón — (x²+y²−1)³ − x²y³ ≤ 0",
        "sliders": [("s", "tamaño (µm)", 20.0, 600.0, 150.0, 1e-6, "{:.0f}")],
        "mask": mascara_corazon,
        "size": lambda p: 2.2 * p[0],
        "nota": "Sin forma cerrada → FFT. Patrón con simetría\nespecular en x (el corazón la tiene)",
    },
    "Hexágono": {
        "titulo": "Hexágono regular",
        "sliders": [("s", "radio (µm)", 20.0, 600.0, 180.0, 1e-6, "{:.0f}"),
                    ("rot", "rotación (°)", 0.0, 360.0, 0.0, _GRAD, "{:.0f}")],
        "mask": mascara_hexagono,
        "size": lambda p: 2.0 * p[0],
        "nota": "Patrón con simetría de 6 (hexagonal)",
    },
    "Estrella": {
        "titulo": "Estrella de N puntas",
        "sliders": [("s", "radio (µm)", 30.0, 600.0, 180.0, 1e-6, "{:.0f}"),
                    ("puntas", "nº puntas", 3.0, 8.0, 5.0, 1.0, "{:.0f}")],
        "mask": mascara_estrella,
        "size": lambda p: 2.0 * p[0],
        "nota": "Estrella de p puntas → patrón con simetría 2p",
    },
    "Triángulo (→ estrella 6)": {
        "titulo": "Triángulo equilátero — Fraunhofer = estrella de 6 puntas",
        "sliders": [("s", "radio (µm)", 30.0, 600.0, 200.0, 1e-6, "{:.0f}"),
                    ("rot", "rotación (°)", 0.0, 360.0, 0.0, _GRAD, "{:.0f}")],
        "mask": mascara_triangulo,
        "size": lambda p: 2.0 * p[0],
        "nota": "Abertura de 3 lados → patrón de 6 puntas\n(resultado clásico contraintuitivo)",
    },
    "Corazón + hexágono": {
        "titulo": "Corazón + hexágono separados (interferencia)",
        "sliders": [("s", "tamaño (µm)", 20.0, 400.0, 120.0, 1e-6, "{:.0f}"),
                    ("D", "separación D (µm)", 150.0, 1200.0, 450.0, 1e-6, "{:.0f}")],
        "mask": mascara_corazon_hexagono,
        "size": lambda p: p[1] + 2.2 * p[0],
        "nota": "Dos formas distintas a distancia D → franjas\ncos(2πD·x'/λz) moduladas por cada difracción",
    },
}


# =============================================================================
# 4. INTERFAZ GRÁFICA  (espejo de HostFraunhofer del Código 20)
# =============================================================================

class HostFFT:
    """Vista única: Combobox de forma + sliders reconstruidos por forma + figura
    de 2 paneles (máscara | patrón FFT) + escala lin/γ/log + caja de régimen."""

    def __init__(self, parent):
        self.parent = parent
        self._pv = []            # [(clave, DoubleVar, escala_a_SI), ...]
        self._build_controls()
        self._build_figure()
        self._on_aperture()

    def _build_controls(self):
        panel = ttk.Frame(self.parent, padding=8)
        panel.pack(side="left", fill="y")

        self.titulo_var = tk.StringVar(value="")
        ttk.Label(panel, textvariable=self.titulo_var,
                  font=("", 11, "bold")).pack(anchor="w", pady=(0, 6))

        sel = ttk.LabelFrame(panel, text="Forma de la abertura", padding=6)
        sel.pack(fill="x", pady=4)
        self.aper = tk.StringVar(value=list(APERTURAS_21)[0])
        cb = ttk.Combobox(sel, textvariable=self.aper, state="readonly",
                          values=list(APERTURAS_21), width=28)
        cb.pack(fill="x")
        cb.bind("<<ComboboxSelected>>", self._on_aperture)

        self.param_frame = ttk.LabelFrame(panel, text="Parámetros", padding=6)
        self.param_frame.pack(fill="x", pady=4)

        esc = ttk.LabelFrame(panel, text="Escala de intensidad", padding=6)
        esc.pack(fill="x", pady=4)
        self.escala = tk.StringVar(value="gamma")
        fila = ttk.Frame(esc)
        fila.pack(fill="x")
        for txt, val in (("Lineal", "lineal"), ("γ (0.4)", "gamma"),
                         ("Log", "log")):
            ttk.Radiobutton(fila, text=txt, variable=self.escala, value=val,
                            command=self.recompute).pack(side="left")
        ttk.Label(esc, text="Tip: Log + abertura grande + z alto\nrevela las franjas finas del patrón.",
                  font=("Consolas", 7), foreground="#555").pack(anchor="w", pady=(3, 0))

        st = ttk.LabelFrame(panel, text="Régimen del cálculo", padding=6)
        st.pack(fill="x", pady=4)
        self.status = tk.StringVar(value="")
        self.status_lbl = ttk.Label(st, textvariable=self.status, justify="left",
                                    font=("Consolas", 9))
        self.status_lbl.pack(anchor="w")

        nt = ttk.LabelFrame(panel, text="Nota física", padding=6)
        nt.pack(fill="x", pady=4)
        self.nota = tk.StringVar(value="")
        ttk.Label(nt, textvariable=self.nota, justify="left",
                  font=("Consolas", 8)).pack(anchor="w")

    def _build_figure(self):
        right = ttk.Frame(self.parent)
        right.pack(side="left", fill="both", expand=True)
        self.fig = Figure(figsize=(10.5, 6.0))
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        NavigationToolbar2Tk(self.canvas, right).update()

    def _on_aperture(self, *_):
        entry = APERTURAS_21[self.aper.get()]
        for w in self.param_frame.winfo_children():
            w.destroy()
        self._pv = []
        sliders = entry["sliders"] + _COMUNES
        for (clave, label, frm, to, init, esc, fmt) in sliders:
            var = crear_slider(self.param_frame, label, frm, to, init,
                               self.recompute, fmt)
            self._pv.append((clave, var, esc))
        self.titulo_var.set(entry["titulo"])
        self.nota.set(entry.get("nota", ""))
        self.recompute()

    def recompute(self):
        entry = APERTURAS_21[self.aper.get()]
        p = {clave: var.get() * esc for (clave, var, esc) in self._pv}
        # separa params de forma (los del registro) de los comunes
        n_forma = len(entry["sliders"])
        forma = [var.get() * esc for (_, var, esc) in self._pv[:n_forma]]
        lam, z, xmax, N = p["lam"], p["z"], p["xmax"], int(p["N"])

        size = entry["size"](forma)
        x_obs, I2D, _, _ = patron_fft(
            entry["mask"], forma, size, lam, z, xmax, N)

        self._dibujar(entry["mask"], forma, x_obs, I2D, size)

        z_min, N_F, es_fh = regimen_generico(size, lam, z)
        txt, color = _status_regimen(size, z_min, N_F, es_fh, z, metodo="FFT2D")
        self.status.set(txt)
        self.status_lbl.configure(foreground=color)
        self.canvas.draw_idle()

    def _dibujar(self, mask_fn, params, x_obs, I2D, size):
        self.fig.clf()
        gs = self.fig.add_gridspec(1, 2, wspace=0.28)
        ax_ap = self.fig.add_subplot(gs[0, 0])
        ax_pat = self.fig.add_subplot(gs[0, 1])

        # Máscara DESACOPLADA de la FFT: malla fina solo para verse limpia, sin
        # depender de la resolución del FFT (que usa zero-padding y dejaría los
        # bordes pixelados). Visual — la física sale de la malla de patron_fft().
        rlim_m = 0.72 * size
        md = _mascara_fina(lambda X, Y: mask_fn(X, Y, params), rlim_m, npix=600)
        rl = rlim_m * 1e3
        ax_ap.imshow(md, extent=[-rl, rl, -rl, rl], origin="lower",
                     cmap="gray", vmin=0, vmax=1)
        ax_ap.set_xlabel("x̃  [mm]")
        ax_ap.set_ylabel("ỹ  [mm]")
        ax_ap.set_title("Máscara de la abertura t(x,y)")

        ext = x_obs[-1] * 1e3 if x_obs.size else 1.0
        datos, norm = _escala_norm(I2D, self.escala.get())
        ax_pat.imshow(datos, extent=[-ext, ext, -ext, ext], origin="lower",
                      cmap="inferno", norm=norm,
                      vmax=(None if norm is not None else max(I2D.max(), 1e-12)))
        ax_pat.set_xlabel("x'  [mm]")
        ax_pat.set_ylabel("y'  [mm]")
        ax_pat.set_title("Patrón de Fraunhofer  |F{t}|²  (FFT2D)")


def main():
    root = tk.Tk()
    root.title("Código 21 — Fraunhofer numérico vía FFT espacial 2D")
    marco = ttk.Frame(root)
    marco.pack(fill="both", expand=True)
    HostFFT(marco)
    root.mainloop()


if __name__ == "__main__":
    main()
