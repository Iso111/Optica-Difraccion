# -*- coding: utf-8 -*-
"""
Validación física headless de los Códigos 20, 21 y 22.

Comprueba el NÚCLEO FÍSICO de cada simulador contra casos límite conocidos
(sinc², disco de Airy, espiral de Cornu, zonas de Fresnel, teorema de
desplazamiento, órdenes suprimidos de una red…). No abre ninguna ventana
(matplotlib "Agg"); se ejecuta con:

    python validacion_fisica.py

Salida ASCII pura (evita el crash de cp1252 en la consola de Windows).
Termina con código 0 si TODO pasa, o 1 si algún check falla.
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")

from scipy.special import fresnel as fres_int
from scipy.optimize import brentq

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE, "Codigo_1"))
sys.path.insert(0, os.path.join(BASE, "Codigo_2"))
sys.path.insert(0, os.path.join(BASE, "Codigo_3"))

import fraunhofer_analitico as c20   # noqa: E402
import fraunhofer_fft as c21         # noqa: E402
import fresnel as c22                # noqa: E402

OK, FAIL = [], []


def check(nombre, cond, detalle=""):
    (OK if cond else FAIL).append(nombre)
    print(("[OK]   " if cond else "[FAIL] ") + nombre + ("  " + detalle if detalle else ""))


# ============================================================
# CODIGO 20 - Fraunhofer analitico
# ============================================================
print("=" * 60)
print("CODIGO 20 - Fraunhofer analitico")
print("=" * 60)

# 1. Marco+circulo: I(0,0) = 1 (normalizacion en el eje optico)
p = dict(a=100e-6, b=150e-6, c=30e-6, d=50e-6, R=40e-6, D=300e-6,
         lam=633e-9, z=2.0)
I00 = c20.intensidad(np.array([[0.0]]), np.array([[0.0]]), p)[0, 0]
check("C20 marco+circ: I(0,0)=1", abs(I00 - 1.0) < 1e-12, "I00=%.15f" % I00)

# 2. Rectangulo puro (R=0, c=d=0): cero de sinc en x'=lam*z/a
p2 = dict(a=100e-6, b=150e-6, c=0.0, d=0.0, R=0.0, D=0.0, lam=633e-9, z=2.0)
x0 = p2["lam"] * p2["z"] / p2["a"]
Iz = c20.intensidad(np.array([[x0]]), np.array([[0.0]]), p2)[0, 0]
check("C20 rect puro: cero de sinc en x'=lam*z/a", Iz < 1e-20, "I=%.3e" % Iz)

# 3. Airy: primer cero de amplitud_circulo en 2piR*rho = 3.8317
R = 50e-6
rho_cero = brentq(lambda r: c20.amplitud_circulo(np.array([r]), R)[0],
                  0.5 / R, 0.7 / R)
rho_teo = 3.831705970207512 / (2 * np.pi * R)
check("C20 Airy: 1er cero rho=3.8317/(2piR)",
      abs(rho_cero - rho_teo) / rho_teo < 1e-9)
lam, z = 500e-9, 2.0
xp = rho_cero * lam * z
xp_teo = 1.2196698912665045 * lam * z / (2 * R)
check("C20 Airy: x'=1.22*lam*z/(2R)", abs(xp - xp_teo) / xp_teo < 1e-9)

# 4. Interferencia de dos aberturas: periodo de franjas = lam*z/D
p4 = dict(a=60e-6, b=60e-6, c=0.0, d=0.0, R=60e-6 / np.sqrt(np.pi), D=500e-6,
          lam=633e-9, z=2.0)
x_franja = p4["lam"] * p4["z"] / p4["D"]
xs = np.linspace(-8 * x_franja, 8 * x_franja, 32001)
I = c20.intensidad(xs[None, :], np.zeros((1, xs.size)), p4)[0]
picos = [xs[k] for k in range(1, len(I) - 1)
         if I[k] > I[k - 1] and I[k] > I[k + 1] and I[k] > 0.05 * I.max()]
per_med = np.median(np.diff(picos))
check("C20 franjas: periodo = lam*z/D", abs(per_med - x_franja) / x_franja < 0.02,
      "medido=%.4e teo=%.4e" % (per_med, x_franja))

# 5. Rectangulo rotado: theta=0 -> rect alineado; theta!=0 -> patron rota
fx = np.linspace(-2e4, 2e4, 101)
FX, FY = np.meshgrid(fx, fx)
a_, b_ = 80e-6, 40e-6
A0 = c20.amplitud_rectangulo_rotado(FX, FY, a_, b_, 0.0)
A0_ref = a_ * b_ * np.sinc(a_ * FX) * np.sinc(b_ * FY)
check("C20 rect rotado theta=0 == rect alineado", np.allclose(A0, A0_ref))
th = np.radians(30.0)
A_rot = c20.amplitud_rectangulo_rotado(FX, FY, a_, b_, th)
FXr = FX * np.cos(th) + FY * np.sin(th)
FYr = -FX * np.sin(th) + FY * np.cos(th)
A_ref = c20.amplitud_rectangulo_rotado(FXr, FYr, a_, b_, 0.0)
check("C20 rect rotado: patron gira con la abertura", np.allclose(A_rot, A_ref))

# 6. Cruz: a=L degenera en cuadrado LxL; A(0)=area real (sin doble conteo)
L_ = 100e-6
Ac = c20.amplitud_cruz(FX, FY, L_, L_)
Aq = L_ ** 2 * np.sinc(L_ * FX) * np.sinc(L_ * FY)
check("C20 cruz a=L -> cuadrado LxL", np.allclose(Ac, Aq))
Ac0 = c20.amplitud_cruz(np.array([0.0]), np.array([0.0]), 100e-6, 20e-6)[0]
check("C20 cruz: A(0)=area (2La-a^2)",
      abs(Ac0 - (2 * 100e-6 * 20e-6 - (20e-6) ** 2)) < 1e-18)

# 7. Dos semicirculos: area y caso limite r1=r2 -> circulo (patron = Airy)
check("C20 semicirc: area(r,r)=pi r^2",
      abs(c20.area_dos_semicirculos(1e-3, 1e-3) - np.pi * 1e-6) < 1e-15)
check("C20 semicirc: area(r,0)=pi r^2/2",
      abs(c20.area_dos_semicirculos(1e-3, 0.0) - np.pi * 1e-6 / 2) < 1e-15)
r_ = 0.5e-3
x_obs, I2D, _, _ = c20.patron_fft_semicirculos(r_, r_, 500e-9, 50.0, 20e-3, 1024)
j = I2D.shape[0] // 2
rho_o = np.abs(x_obs) / (500e-9 * 50.0)
I_airy = (c20.amplitud_circulo(rho_o, r_) / (500e-9 * 50.0)) ** 2
corr = np.corrcoef(I2D[j, :], I_airy)[0, 1]
check("C20 FFT semicirc r1=r2 vs Airy analitico", corr > 0.999, "corr=%.6f" % corr)

# 8. Doble cuadrado: I(0,0)=1 y D=4a (geometria del enunciado)
Idc, Ddc = c20.intensidad_doble_cuadrado(np.array([[0.0]]), np.array([[0.0]]),
                                         0.5e-3, 500e-9, 5.0)
check("C20 doble cuadrado: I(0,0)=1, D=4a",
      abs(Idc[0, 0] - 1.0) < 1e-12 and abs(Ddc - 2e-3) < 1e-15)

# 9. Rendijas: N=1 -> sinc^2 puro; red N=5: max ppal toca envolvente; d=3a suprime m=3
st = np.linspace(-0.5, 0.5, 20001)
I1 = c20.intensidad_rendijas(st, 8.0, 24.0, 1)
check("C20 rendija N=1 == sinc^2", np.allclose(I1, np.sinc(8.0 * st) ** 2))
I5 = c20.intensidad_rendijas(st, 8.0, 24.0, 5)
i_m1 = np.argmin(np.abs(st - 1.0 / 24.0))
env_m1 = np.sinc(8.0 / 24.0) ** 2
check("C20 red N=5: max ppal m=1 toca envolvente",
      abs(I5[i_m1] - env_m1) / env_m1 < 1e-3)
i_m3 = np.argmin(np.abs(st - 3.0 / 24.0))
check("C20 red d=3a: orden m=3 suprimido (missing order)", I5[i_m3] < 1e-6,
      "I=%.2e" % I5[i_m3])

# 10. Escalon de Michelson: N=1 -> sinc^2; max ppal en Delta=m*lam
s_, h_, n_ = 1e-4, 1e-3, 1.5
Ie1 = c20.intensidad_escalon(st, s_, h_, n_, 500e-9, 1)
check("C20 escalon N=1 == sinc^2",
      np.allclose(Ie1, np.sinc(s_ * st / 500e-9) ** 2))
lam_ = 500e-9
m_obj = round(((n_ - 1) * h_) / lam_) + 2
st_max = (m_obj * lam_ - (n_ - 1) * h_) / s_
Ie5 = c20.intensidad_escalon(np.array([st_max]), s_, h_, n_, lam_, 5)[0]
env = np.sinc(s_ * st_max / lam_) ** 2
check("C20 escalon: max ppal en Delta=m*lam toca envolvente",
      abs(Ie5 - env) / max(env, 1e-30) < 1e-6)

# 11. fresnel_propagate: 1 zona -> I_axial=4; 2 zonas -> I_axial~0
lam_, z_ = 500e-9, 2.0
r1z = np.sqrt(lam_ * z_)
Nn = 1024
L_ = min(6 * r1z, 0.8 * np.sqrt(Nn * lam_ * z_))
dx_ = L_ / Nn
x_ = (np.arange(Nn) - Nn // 2) * dx_
X_, Y_ = np.meshgrid(x_, x_)
m1 = (X_ ** 2 + Y_ ** 2 <= r1z ** 2).astype(complex)
U_, _ = c20.fresnel_propagate(m1, dx_, lam_, z_)
check("C20 Fresnel: 1 zona -> I_axial=4",
      abs(abs(U_[Nn // 2, Nn // 2]) ** 2 - 4.0) < 0.1)
r2z = np.sqrt(2 * lam_ * z_)
m2 = (X_ ** 2 + Y_ ** 2 <= r2z ** 2).astype(complex)
U2_, _ = c20.fresnel_propagate(m2, dx_, lam_, z_)
check("C20 Fresnel: 2 zonas -> I_axial~0",
      abs(U2_[Nn // 2, Nn // 2]) ** 2 < 0.1)

# 12. Doble circulo pto 19: |U|=1.5 (3/4 z1 + 1/4 z2) -> I=2.25
d19 = c20.patrones_doble_circulo(r1z, r2z, lam_, z_, 1024)
check("C20 pto19: |U|=1.5 axial", abs(abs(d19["U_fresnel_axial"]) - 1.5) < 0.08,
      "|U|=%.4f" % abs(d19["U_fresnel_axial"]))

# 13. Redes en cascada: s=0 -> I(0)=1; s=d -> ((N-1)/N)^2; s in [a,2a] -> oscuro
a6 = 50e-6
stc, Ic, d6, _ = c20.patron_redes_cascada(a6, 6, 0.0, 633e-9)
check("C20 redes cascada s=0: I(0)=1",
      abs(Ic[np.argmin(np.abs(stc))] - 1.0) < 0.02)
stc2, Ic2, _, _ = c20.patron_redes_cascada(a6, 6, d6, 633e-9)
check("C20 redes cascada s=d: I(0)=((N-1)/N)^2",
      abs(Ic2[np.argmin(np.abs(stc2))] - (5 / 6) ** 2) < 0.02)
_, Ic3, _, _ = c20.patron_redes_cascada(a6, 6, 1.5 * a6, 633e-9)
check("C20 redes cascada s=1.5a: campo oscuro", Ic3.max() < 1e-6)

# 14. Regimen: z_min=2D^2/lam <-> N_F=0.5
D_ = 1e-3
zmin_, _, _ = c20.regimen_generico(D_, 500e-9, 1.0)
check("C20 regimen: z_min=2D^2/lam", abs(zmin_ - 2 * D_ ** 2 / 500e-9) < 1e-9)
_, NF_en_zmin, esfh2 = c20.regimen_generico(D_, 500e-9, zmin_)
check("C20 regimen: N_F(z_min)=0.5 y es Fraunhofer",
      abs(NF_en_zmin - 0.5) < 1e-12 and esfh2)


# ============================================================
# CODIGO 21 - Fraunhofer numerico via FFT 2D
# ============================================================
print("=" * 60)
print("CODIGO 21 - Fraunhofer numerico (FFT 2D)")
print("=" * 60)

lam_, z_ = 633e-9, 3.0

# 1. Rectangulo: corte central vs sinc^2 analitico (forma) y I(0) absoluta
w_, h_ = 150e-6, 150e-6
x_obs, I2D, mask, x_ap = c21.patron_fft(c21.mascara_rectangulo, [w_, h_],
                                        np.hypot(w_, h_), lam_, z_, 20e-3, 800)
j = I2D.shape[0] // 2
I_num = I2D[j, :] / I2D[j, :].max()
I_ana = np.sinc(w_ * x_obs / (lam_ * z_)) ** 2
corr = np.corrcoef(I_num, I_ana / I_ana.max())[0, 1]
check("C21 rectangulo vs sinc^2 (forma)", corr > 0.999, "corr=%.6f" % corr)
# I(0) absoluta = (area/(lam z))^2, comparada con el AREA RASTERIZADA (la que
# ve el FFT): el motor debe reproducirla exactamente; la unica desviacion
# frente al area exacta w*h es la pixelacion de la mascara.
dxa = x_ap[1] - x_ap[0]
area_raster = mask.sum() * dxa * dxa
teo_raster = (area_raster / (lam_ * z_)) ** 2
check("C21 rectangulo: I(0)=(area_raster/lam z)^2 exacto",
      abs(I2D[j, j] - teo_raster) / teo_raster < 1e-6)

# 2. Circulo -> Airy: primer cero en 1.22 lam z / d
d_ = 150e-6
x_obs, I2D, _, _ = c21.patron_fft(c21.mascara_circulo, [d_, 0.0],
                                  d_, lam_, z_, 40e-3, 1200)
j = I2D.shape[0] // 2
I_c, x_c = I2D[j, j:], x_obs[j:]
k_min = next(k for k in range(1, len(I_c) - 1)
             if I_c[k] < I_c[k - 1] and I_c[k] <= I_c[k + 1])
x_teo = 1.2196698912665045 * lam_ * z_ / d_
check("C21 circulo: 1er cero Airy 1.22 lam z/d",
      abs(x_c[k_min] - x_teo) / x_teo < 0.03,
      "err=%.4f" % (abs(x_c[k_min] - x_teo) / x_teo))

# 3. Mapeo dx' = lam z / L_ap exacto
N_, size_ = 600, 150e-6
L_ap = max(4 * size_, N_ * (size_ / c21.RESOLUCION_ABERTURA))
x_obs, _, _, _ = c21.patron_fft(c21.mascara_rectangulo, [size_, size_],
                                size_, lam_, z_, 1.0, N_)
check("C21 mapeo dx'=lam z/L_ap",
      abs((x_obs[1] - x_obs[0]) - lam_ * z_ / L_ap) / (x_obs[1] - x_obs[0]) < 1e-12)

# 4. Simetrias: triangulo -> patron de 6 (invariante a rot 60deg); corazon espejo-x
from scipy.ndimage import rotate   # noqa: E402
_, I2D, _, _ = c21.patron_fft(c21.mascara_triangulo, [200e-6, 0.0],
                              400e-6, lam_, z_, 30e-3, 600)
I_rot = rotate(I2D, 60.0, reshape=False, order=1)
mv = I_rot > I2D.max() * 1e-4
check("C21 triangulo: simetria 6 (rot 60deg)",
      np.corrcoef(I2D[mv], I_rot[mv])[0, 1] > 0.98)
_, I2D, _, _ = c21.patron_fft(c21.mascara_corazon, [150e-6],
                              330e-6, lam_, z_, 30e-3, 600)
check("C21 corazon: espejo en x",
      np.corrcoef(I2D.ravel(), I2D[:, ::-1].ravel())[0, 1] > 0.999)

# 5. Circulo+cuadrado: franjas de interferencia con periodo lam z/D
diam_, lado_, D_ = 120e-6, 120e-6, 350e-6
x_obs, I2D, _, _ = c21.patron_fft(c21.mascara_circulo_cuadrado,
                                  [diam_, lado_, D_], D_ + 120e-6,
                                  lam_, z_, 15e-3, 1000)
j = I2D.shape[0] // 2
I_c = I2D[j, :]
picos = [x_obs[k] for k in range(1, len(I_c) - 1)
         if I_c[k] > I_c[k - 1] and I_c[k] > I_c[k + 1] and I_c[k] > 0.1 * I_c.max()]
check("C21 circ+cuad: franjas periodo lam z/D",
      abs(np.median(np.diff(picos)) - lam_ * z_ / D_) / (lam_ * z_ / D_) < 0.05)


# ============================================================
# CODIGO 22 - Evolucion Fresnel -> Fraunhofer
# ============================================================
print("=" * 60)
print("CODIGO 22 - Evolucion Fresnel -> Fraunhofer")
print("=" * 60)

a_sl, lam_ = 0.15e-3, 500e-9

# 1. Espiral de Cornu: rendija simple N_F=2 vs integrales de Fresnel de scipy
NF = 2.0
z_ = a_sl ** 2 / (lam_ * NF)
params = [a_sl, 0.45e-3, 1]
d22 = c22.evolucion("N rendijas (ptos 2/6)", params, lam_, NF, [0.5], N=1024)
sen, I_fr = d22["sen"], d22["I_fr"]
xg = sen * z_
s2 = np.sqrt(2.0 / (lam_ * z_))
S1, C1 = fres_int(s2 * (-a_sl / 2 - xg))
S2, C2 = fres_int(s2 * (a_sl / 2 - xg))
I_teo = 0.5 * ((C2 - C1) ** 2 + (S2 - S1) ** 2)
sel = np.abs(sen) < np.abs(sen).max() * 0.5
corr = np.corrcoef(I_fr[sel] / I_fr[sel].max(), I_teo[sel] / I_teo[sel].max())[0, 1]
check("C22 Cornu rendija N_F=2 vs scipy.fresnel", corr > 0.999, "corr=%.6f" % corr)

# 2. Convergencia Fresnel -> Fraunhofer a N_F=0.1
d22b = c22.evolucion("N rendijas (ptos 2/6)", params, lam_, 0.1, [0.5], N=1024)
Ifr = d22b["I_fr"] / d22b["I_fr"].max()
Ifh = d22b["I_fh"] / d22b["I_fh"].max()
check("C22 convergencia N_F=0.1: Fresnel==Fraunhofer",
      np.corrcoef(Ifr, Ifh)[0, 1] > 0.999)

# 3. Fraunhofer limite de la rendija coincide con sinc^2
I_sinc = np.sinc(a_sl * d22b["sen"] / lam_) ** 2
check("C22 Fraunhofer limite rendija vs sinc^2",
      np.corrcoef(Ifh, I_sinc / I_sinc.max())[0, 1] > 0.999)

# 4. _pad_muestreo: cumple Nyquist y detecta submuestreo
pad, ok = c22._pad_muestreo(1024, 2.0)
check("C22 _pad_muestreo: cumple Nyquist con margen",
      pad <= np.sqrt(1024 / (1.3 * 2.0)) + 1e-9 and ok)
_, ok_hi = c22._pad_muestreo(256, 80.0)
check("C22 _pad_muestreo: detecta submuestreo (N=256, NF=80)", not ok_hi)

# 5. Doble circulo pto 19 via C22: intensidad axial = 2.25; z_cur=D^2/(lam NF)
r1z, r2z = np.sqrt(lam_ * 2.0), np.sqrt(2 * lam_ * 2.0)
par19 = [r1z, r2z]
NF19 = (2 * r2z) ** 2 / (lam_ * 2.0)
d19 = c22.evolucion("Doble círculo (pto 19)", par19, lam_, NF19, [0.5], N=1024)
Npx = d19["I_fr2d"].shape[0]
check("C22 pto19 axial: I=2.25", abs(d19["I_fr2d"][Npx // 2, Npx // 2] - 2.25) < 0.2,
      "I=%.4f" % d19["I_fr2d"][Npx // 2, Npx // 2])
z_esp = d19["D"] ** 2 / (lam_ * NF19)
check("C22 z_cur = D^2/(lam N_F)", abs(d19["z_cur"] - z_esp) / z_esp < 1e-12)


# ============================================================
print("=" * 60)
print("RESULTADO: %d OK, %d FAIL" % (len(OK), len(FAIL)))
if FAIL:
    print("FALLOS:")
    for f in FAIL:
        print("  - " + f)
sys.exit(1 if FAIL else 0)
