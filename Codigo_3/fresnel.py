"""
Código 22 — Evolución del patrón de difracción de Fresnel → Fraunhofer.

Punto 22 del taller: graficar el patrón de difracción de FRESNEL (campo cercano)
y mostrar, en una misma gráfica, su evolución hacia el patrón de FRAUNHOFER
(campo lejano) al variar el número de Fresnel  N_F = D²/(λz).

Idea (acordada con el usuario):
  · Se reutiliza el motor de Fresnel `fresnel_propagate` (FFT único) y las
    máscaras circulares ya VALIDADAS del Código 20 (se importan; no se duplica
    el motor — ver CLAUDE.md).  Las máscaras rectangulares/rendijas, que en el
    Código 20 solo existen como amplitud de Fourier, se rasterizan aquí.
  · EJE ANGULAR común  senθ = x'/z = n·λ/(N·dx):  es INDEPENDIENTE de z, así
    que todas las curvas (Fresnel a cualquier z y la Fraunhofer límite) caen
    sobre el mismo eje y se superponen sin reescalar.  La Fraunhofer (|𝓕|² sin
    el chirp de salida) queda FIJA y las Fresnel convergen a ella cuando N_F→0.
  · Una sola pestaña con SELECTOR de las 6 aberturas del taller; el slider de
    N_F mueve la curva actual sobre un fondo de curvas fijas (Fresnel a varios
    N_F + Fraunhofer + la curva umbral z=z_min, N_F=0.5, en rayado).

GUI tkinter al estilo del Código 20.
"""

import os
import sys

import numpy as np
import tkinter as tk
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)

# --- Reutilización del Código 20 (motor Fresnel + máscaras + helpers GUI) -----
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Codigo_1"))
from fraunhofer_analitico import (          # noqa: E402
    fresnel_propagate, mascara_doble_circulo, mascara_dos_semicirculos,
    regimen_generico, crear_slider, _escala_norm)


# =============================================================================
# 1. MÁSCARAS REALES DE LAS ABERTURAS (las que faltan en el Código 20)
# =============================================================================

def mascara_marco_circulo(X, Y, a, b, c, d, R, D):
    """Marco rectangular (ext a×b − int c×d) centrado en x=−D/2, más círculo R
    en x=+D/2 (la abertura compuesta del Parcial 4, punto 1)."""
    ext = (np.abs(X + D / 2) <= a / 2) & (np.abs(Y) <= b / 2)
    inte = (np.abs(X + D / 2) <= c / 2) & (np.abs(Y) <= d / 2)
    marco = ext & ~inte
    circ = ((X - D / 2) ** 2 + Y ** 2) <= R ** 2
    return marco | circ


def mascara_cruz(X, Y, L, a):
    """Cruz: unión de barra horizontal (L×a) y barra vertical (a×L)."""
    horiz = (np.abs(X) <= L / 2) & (np.abs(Y) <= a / 2)
    vert = (np.abs(X) <= a / 2) & (np.abs(Y) <= L / 2)
    return horiz | vert


def mascara_doble_cuadrado(X, Y, a):
    """Cuadrado pequeño (lado a) y grande (lado 3a), separados centro-a-centro
    D=4a (misma geometría del pto 14 del Código 20)."""
    D = 4.0 * a
    q1 = (np.abs(X + D / 2) <= a / 2) & (np.abs(Y) <= a / 2)
    q2 = (np.abs(X - D / 2) <= 1.5 * a) & (np.abs(Y) <= 1.5 * a)
    return q1 | q2


def mascara_rendijas(X, Y, a, d, N):
    """N rendijas verticales de ancho a y período d, altas (ocupan toda la
    ventana en y) → el corte horizontal es la difracción de la red."""
    centros = (np.arange(N) - (N - 1) / 2.0) * d
    m = np.zeros_like(X, dtype=bool)
    for c in centros:
        m |= np.abs(X - c) <= a / 2
    return m


# =============================================================================
# 2. REGISTRO DE ABERTURAS
#    nombre → dict(sliders, mask, Dchar)
#    sliders: lista de (label, frm, to, init, escala_a_SI, fmt)
#    mask(X,Y,p): p = lista de valores en SI
#    Dchar(p): extensión característica [m] para N_F y z_min
# =============================================================================

APERTURAS = {
    "Doble círculo (pto 19)": {
        "sliders": [("r1 (mm)", 0.2, 3.0, 1.0, 1e-3, "{:.3f}"),
                    ("r2 (mm)", 0.2, 3.0, 1.41, 1e-3, "{:.3f}")],
        "mask": lambda X, Y, p: mascara_doble_circulo(X, Y, p[0], p[1]),
        "Dchar": lambda p: 2.0 * max(p[0], p[1]),
    },
    "Dos semicírculos (pto 9)": {
        "sliders": [("r1 sup (mm)", 0.2, 3.0, 2.0, 1e-3, "{:.3f}"),
                    ("r2 inf (mm)", 0.2, 3.0, 1.414, 1e-3, "{:.3f}")],
        "mask": lambda X, Y, p: mascara_dos_semicirculos(X, Y, p[0], p[1]),
        "Dchar": lambda p: 2.0 * max(p[0], p[1]),
    },
    "Cruz (pto 5)": {
        "sliders": [("L largo (mm)", 0.5, 6.0, 3.0, 1e-3, "{:.3f}"),
                    ("a ancho (mm)", 0.1, 2.0, 0.6, 1e-3, "{:.3f}")],
        "mask": lambda X, Y, p: mascara_cruz(X, Y, p[0], p[1]),
        "Dchar": lambda p: np.hypot(p[0], p[0]),
    },
    "Doble cuadrado (pto 14)": {
        "sliders": [("a lado peq (mm)", 0.1, 1.5, 0.5, 1e-3, "{:.3f}")],
        "mask": lambda X, Y, p: mascara_doble_cuadrado(X, Y, p[0]),
        "Dchar": lambda p: np.hypot(6.0 * p[0], 3.0 * p[0]),
    },
    "N rendijas (ptos 2/6)": {
        "sliders": [("a ranura (mm)", 0.05, 1.0, 0.15, 1e-3, "{:.3f}"),
                    ("d período (mm)", 0.1, 2.0, 0.45, 1e-3, "{:.3f}"),
                    ("N ranuras", 1.0, 12.0, 5.0, 1.0, "{:.0f}")],
        "mask": lambda X, Y, p: mascara_rendijas(X, Y, p[0], p[1], int(p[2])),
        "Dchar": lambda p: int(p[2]) * p[1],
    },
    "Marco + círculo (pto 1)": {
        "sliders": [("a marco (mm)", 0.3, 4.0, 2.0, 1e-3, "{:.3f}"),
                    ("b marco (mm)", 0.3, 4.0, 3.0, 1e-3, "{:.3f}"),
                    ("c hueco (mm)", 0.0, 2.0, 0.6, 1e-3, "{:.3f}"),
                    ("d hueco (mm)", 0.0, 2.0, 1.0, 1e-3, "{:.3f}"),
                    ("R círc (mm)", 0.2, 2.0, 1.0, 1e-3, "{:.3f}"),
                    ("D separ (mm)", 2.0, 10.0, 5.0, 1e-3, "{:.3f}")],
        "mask": lambda X, Y, p: mascara_marco_circulo(
            X, Y, p[0], p[1], p[2], p[3], p[4], p[5]),
        "Dchar": lambda p: np.hypot(p[5] + p[0] / 2 + p[4], max(p[1], 2 * p[4])),
    },
}


# =============================================================================
# 3. NÚCLEO: perfiles Fresnel/Fraunhofer sobre el eje angular común
# =============================================================================

def _campo(mask, dx, lam, z, quiere_2d=False):
    """Fresnel (campo cercano) y Fraunhofer límite del mismo `mask` complejo.
    Devuelve dict con el corte horizontal de ambos y (opcional) el 2D Fresnel."""
    U_fr, dx2 = fresnel_propagate(mask, dx, lam, z)
    N = mask.shape[0]
    c = N // 2
    # Fraunhofer límite: misma malla, TF sin el chirp de salida.
    A = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(mask)))
    U_fh = A * dx * dx / (lam * z)
    sen = ((np.arange(N) - c) * dx2) / z          # senθ = x'/z (indep. de z)
    out = {"sen": sen, "I_fr": np.abs(U_fr[c, :]) ** 2,
           "I_fh": np.abs(U_fh[c, :]) ** 2, "dx2": dx2}
    if quiere_2d:
        out["I_fr2d"] = np.abs(U_fr) ** 2
    return out


def evolucion(aper, params, lam, NF_actual, NF_fijos, N=512, pad=6.0):
    """
    Para la abertura `aper` (nombre en APERTURAS) con `params` en SI, calcula:
      · el fondo de curvas Fresnel a los N_F de `NF_fijos`,
      · la Fraunhofer límite,
      · la curva Fresnel a `NF_actual` (con su patrón 2D),
    todas sobre el mismo eje angular. Devuelve un dict con todo lo dibujable.
    """
    spec = APERTURAS[aper]
    D = spec["Dchar"](params)
    L = pad * D
    x = (np.arange(N) - N // 2) * (L / N)
    dx = L / N
    X, Y = np.meshgrid(x, x)
    mask = spec["mask"](X, Y, params).astype(complex)

    def curva(NF, dos_d=False):
        z = D ** 2 / (lam * NF)
        return _campo(mask, dx, lam, z, quiere_2d=dos_d), z

    fondo = []
    for NF in NF_fijos:
        c, z = curva(NF)
        fondo.append((NF, z, c["sen"], c["I_fr"]))
    cur, z_cur = curva(NF_actual, dos_d=True)
    return {"D": D, "fondo": fondo, "sen": cur["sen"],
            "I_fr": cur["I_fr"], "I_fh": cur["I_fh"], "I_fr2d": cur["I_fr2d"],
            "z_cur": z_cur, "dx2": cur["dx2"]}


# =============================================================================
# 4. INTERFAZ GRÁFICA
# =============================================================================

# N_F fijos del fondo (uno es 0.5 = z_min, se dibuja rayado).
NF_FONDO = [4.0, 2.0, 1.0, 0.5, 0.15]


class TabEvolucionFresnel:
    """Pestaña única con selector de abertura + evolución Fresnel→Fraunhofer."""

    def __init__(self, parent):
        self.parent = parent
        self.param_vars = []
        self._build_controls()
        self._build_figure()
        self._on_aperture()          # construye sliders + primer cálculo

    def _build_controls(self):
        panel = ttk.Frame(self.parent, padding=8)
        panel.pack(side="left", fill="y")
        ttk.Label(panel, text="Evolución Fresnel → Fraunhofer",
                  font=("", 11, "bold")).pack(anchor="w", pady=(0, 6))

        sel = ttk.LabelFrame(panel, text="Abertura", padding=6)
        sel.pack(fill="x", pady=4)
        self.aper = tk.StringVar(value=list(APERTURAS)[0])
        cb = ttk.Combobox(sel, textvariable=self.aper, state="readonly",
                          values=list(APERTURAS), width=24)
        cb.pack(fill="x")
        cb.bind("<<ComboboxSelected>>", lambda e: self._on_aperture())

        # Frame de parámetros (se reconstruye por abertura)
        self.param_frame = ttk.LabelFrame(panel, text="Parámetros", padding=6)
        self.param_frame.pack(fill="x", pady=4)

        f2 = ttk.LabelFrame(panel, text="Número de Fresnel (evolución)", padding=6)
        f2.pack(fill="x", pady=4)
        self.NF = crear_slider(f2, "N_F actual", 0.05, 10.0, 2.0,
                               self.recompute, "{:.2f}")

        f3 = ttk.LabelFrame(panel, text="Fuente / cálculo", padding=6)
        f3.pack(fill="x", pady=4)
        self.lam = crear_slider(f3, "λ (nm)", 380.0, 1000.0, 500.0,
                                self.recompute, "{:.0f}")
        self.zoom = crear_slider(f3, "zoom (×auto)", 0.3, 3.0, 1.0,
                                 self.recompute, "{:.2f}")
        self.N = crear_slider(f3, "N (px, FFT)", 256.0, 1024.0, 512.0,
                              self.recompute, "{:.0f}")

        f4 = ttk.LabelFrame(panel, text="Escala patrón 2D", padding=6)
        f4.pack(fill="x", pady=4)
        self.escala = tk.StringVar(value="gamma")
        for txt, val in (("Lineal", "lineal"), ("γ", "gamma"), ("Log", "log")):
            ttk.Radiobutton(f4, text=txt, variable=self.escala, value=val,
                            command=self.recompute).pack(side="left")

        st = ttk.LabelFrame(panel, text="Régimen", padding=6)
        st.pack(fill="x", pady=4)
        self.status = tk.StringVar(value="")
        self.status_lbl = ttk.Label(st, textvariable=self.status, justify="left",
                                    font=("Consolas", 9))
        self.status_lbl.pack(anchor="w")

    def _build_figure(self):
        right = ttk.Frame(self.parent)
        right.pack(side="left", fill="both", expand=True)
        self.fig = Figure(figsize=(10.5, 8.0))
        gs = self.fig.add_gridspec(2, 1, hspace=0.32, height_ratios=[0.9, 1.3])
        self.ax_2d = self.fig.add_subplot(gs[0, 0])
        self.ax_ev = self.fig.add_subplot(gs[1, 0])
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        NavigationToolbar2Tk(self.canvas, right).update()

    def _on_aperture(self):
        """Reconstruye los sliders de parámetros para la abertura elegida."""
        for w in self.param_frame.winfo_children():
            w.destroy()
        self.param_vars = []
        self.param_scales = []
        for (label, frm, to, init, escala, fmt) in APERTURAS[self.aper.get()]["sliders"]:
            var = crear_slider(self.param_frame, label, frm, to, init,
                               self.recompute, fmt)
            self.param_vars.append(var)
            self.param_scales.append(escala)
        self.recompute()

    def recompute(self):
        params = [v.get() * e for v, e in zip(self.param_vars, self.param_scales)]
        if not params:
            return
        lam = self.lam.get() * 1e-9
        NF_cur = self.NF.get()
        Npx = int(self.N.get())
        zoom = self.zoom.get()

        d = evolucion(self.aper.get(), params, lam, NF_cur, NF_FONDO, N=Npx)
        D = d["D"]
        z_min = 2.0 * D ** 2 / lam

        # Auto-ajuste del rango angular: hasta donde la Fraunhofer (límite) es
        # significativa (>1% del pico), con margen y el zoom manual encima.
        Ifh_n = d["I_fh"] / d["I_fh"].max() if d["I_fh"].max() > 0 else d["I_fh"]
        sig = np.abs(d["sen"])[Ifh_n > 0.01]
        base = sig.max() if sig.size else np.abs(d["sen"]).max()
        senmax = 1.3 * base / zoom

        # --- Patrón 2D Fresnel (N_F actual), recortado a senθ_max·z ---
        ax = self.ax_2d
        ax.clear()
        xmax = senmax * d["z_cur"]
        xo = (np.arange(Npx) - Npx // 2) * d["dx2"]
        sel2 = np.abs(xo) <= xmax
        I2 = d["I_fr2d"][np.ix_(sel2, sel2)]
        ext = xmax * 1e3
        datos, norm = _escala_norm(I2, self.escala.get())
        ax.imshow(datos, extent=[-ext, ext, -ext, ext], origin="lower",
                  cmap="inferno", norm=norm,
                  vmax=(None if norm is not None else max(I2.max(), 1e-12)))
        ax.set_xlabel("x' [mm]")
        ax.set_ylabel("y' [mm]")
        ax.set_title(f"Patrón de Fresnel 2D  (N_F = {NF_cur:.2f})")

        # --- Gráfica de evolución (eje angular común) ---
        ax = self.ax_ev
        ax.clear()
        smax3 = senmax * 1e3
        cmap = self.fig.cm if False else None
        colores = ["#c7e9ff", "#8ec9f0", "#5aa9e0", "#2f7fc0", "#1f5f9f"]
        for i, (NF, z, sen, I) in enumerate(d["fondo"]):
            senv = sen * 1e3
            m = np.abs(senv) <= smax3
            In = I[m] / I[m].max() if I[m].max() > 0 else I[m]
            if abs(NF - 0.5) < 1e-9:          # z = z_min → rayada
                ax.plot(senv[m], In, color="purple", lw=1.3, ls="--",
                        label=f"z=z_min (N_F={NF:g})")
            else:
                ax.plot(senv[m], In, color=colores[i % len(colores)], lw=0.9,
                        label=f"Fresnel N_F={NF:g}")
        # Fraunhofer límite (fija)
        senv = d["sen"] * 1e3
        m = np.abs(senv) <= smax3
        Ifh = d["I_fh"][m]; Ifh = Ifh / Ifh.max() if Ifh.max() > 0 else Ifh
        ax.plot(senv[m], Ifh, color="black", lw=1.6, label="Fraunhofer (límite)")
        # Curva actual (resaltada)
        Ifr = d["I_fr"][m]; Ifr = Ifr / Ifr.max() if Ifr.max() > 0 else Ifr
        ax.plot(senv[m], Ifr, color="crimson", lw=1.8, alpha=0.9,
                label=f"actual N_F={NF_cur:.2f}")
        ax.set_xlim(-smax3, smax3)
        ax.set_ylim(bottom=0.0)
        ax.set_xlabel("senθ = x'/z   [×10⁻³]  (eje angular común)")
        ax.set_ylabel("I / I_pico")
        ax.legend(fontsize=7, loc="upper right", ncol=2)
        ax.set_title("Evolución Fresnel → Fraunhofer vs número de Fresnel")

        # --- Régimen ---
        z = d["z_cur"]
        _, N_F, es_fh = regimen_generico(D, lam, z)
        txt = (
            f"D_char = {D*1e3:7.3f} mm\n"
            f"N_F actual = {NF_cur:6.2f}\n"
            f"z = {z:8.3f} m\n"
            f"z_min = {z_min:8.3f} m  (N_F=0.5)\n"
            f"z / z_min = {z/z_min:6.3f}\n"
            f"─────────────────────────\n"
            f"{'FRAUNHOFER (campo lejano)' if es_fh else 'FRESNEL (campo cercano)'}"
        )
        self.status.set(txt)
        self.status_lbl.configure(foreground="#127a12" if es_fh else "#c00000")

        self.canvas.draw_idle()


def main():
    root = tk.Tk()
    root.title("Código 22 — Evolución del patrón de Fresnel → Fraunhofer")
    nb = ttk.Notebook(root)
    nb.pack(fill="both", expand=True)
    tab = ttk.Frame(nb)
    nb.add(tab, text="Fresnel → Fraunhofer")
    TabEvolucionFresnel(tab)
    root.mainloop()


if __name__ == "__main__":
    main()
