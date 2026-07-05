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
# 2. INTERFAZ GRÁFICA
# =============================================================================

NX_MAX = 900  # tope de resolución del patrón para mantener fluido el recálculo


class FraunhoferGUI:

    def __init__(self, root):
        self.root = root
        root.title("Código 20 — Difracción de Fraunhofer analítica 2D")

        self._build_controls()
        self._build_figure()
        self.recompute()

    # ------------------------------------------------------------------ UI
    def _slider(self, parent, label, frm, to, init, fmt="{:.2f}"):
        """
        Fila etiqueta + Scale + Entry editable. El Entry permite teclear un valor
        exacto (Enter o perder el foco): se valida, se recorta a [frm, to], se
        mueve el slider y se recalcula. Devuelve la variable DoubleVar.
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
            self.recompute()

        entry.bind("<Return>", commit)
        entry.bind("<FocusOut>", commit)

        scale = ttk.Scale(row, from_=frm, to=to, variable=var,
                          orient="horizontal", command=on_move)
        scale.pack(side="left", fill="x", expand=True, padx=4)
        # Recalcular al soltar el ratón (no en cada píxel del arrastre).
        scale.bind("<ButtonRelease-1>", lambda _=None: self.recompute())

        self._sliders.append(var)
        return var

    def _build_controls(self):
        self._sliders = []
        panel = ttk.Frame(self.root, padding=8)
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
        right = ttk.Frame(self.root)
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


def main():
    root = tk.Tk()
    FraunhoferGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
