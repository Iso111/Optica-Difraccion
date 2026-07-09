# HANDOFF — Contexto estable del proyecto de simuladores de difracción óptica

> Este documento guarda lo que **git y el código no registran por sí solos**: arquitectura,
> dependencias entre archivos, convenciones físicas aprendidas, e ideas abiertas. Para el estado
> actual (qué está commiteado, qué cambió último) usa git directamente — nunca confíes en un
> resumen de estado escrito a mano en este archivo, se desactualiza:
>
> ```
> git log --oneline -10
> git status
> ```
>
> Las reglas de comportamiento (piensa primero, simplicidad, cambios quirúrgicos, AskUserQuestion,
> etc.) viven en `CLAUDE.md` (proyecto) y `~/.claude/CLAUDE.md` (global) — no se repiten aquí.

---

## 1. Qué es este proyecto

Tres simuladores interactivos (tkinter + numpy/scipy/matplotlib) para el Taller 4 y Parcial 4 del
curso Óptica (UNAL), **una app autónoma por archivo**:

- **Código 20** — `Codigo_1/fraunhofer_analitico.py` — Fraunhofer **analítico** 2D (fórmulas
  sinc²/Bessel + algunos casos por FFT). 9 aberturas.
- **Código 21** — `Codigo_2/fraunhofer_fft.py` — Fraunhofer **numérico** vía FFT2D de la máscara
  (estilo PhET). 9 máscaras, sin fórmula cerrada.
- **Código 22** — `Codigo_3/fresnel.py` — Evolución Fresnel→Fraunhofer vs número de Fresnel.
  6 aberturas.

Documentos de referencia en el repo: `Contexto_códigos.md` (spec matemática), `taller4.pdf`,
`Parcial4.pdf`, `Fraunhofer_Transfomada_Fourier_espacial.pdf`, `Difraccion_Fresnel_Clotoide.pdf`,
`Documentacion_Fisica.pdf`/`.tex` (documentación LaTeX de la física + funciones principales),
`validacion_fisica.py` (suite headless de 40 checks — correr antes de cerrar cualquier cambio
físico), `Contexto_codigo2 (1..5).png` (capturas del PhET que inspiran el Código 21).

**Estado general:** los tres códigos están completos, validados (40/40 checks) y documentados.
Cualquier trabajo nuevo es incremental sobre esta base — ver §5 para ideas abiertas.

---

## 2. Arquitectura y dependencias entre archivos (crítico)

**C21 y C22 importan helpers del C20** vía `sys.path.insert` + `from fraunhofer_analitico import (...)`.
Renombrar o cambiar la firma de cualquiera de estos rompe los otros dos códigos sin previo aviso:

| Helper (definido en C20) | Usado también en |
|---|---|
| `crear_slider(parent,label,frm,to,init,on_change,fmt)` | C21, C22 |
| `_escala_norm(I,modo)` (lineal/γ/log) | C21, C22 |
| `regimen_generico(D_char,lam,z)` | C21, C22 |
| `_status_regimen(D_char,z_min,N_F,es_fh,z,metodo)` | C21, C22 |
| `_mascara_fina(mask_xy,rlim,npix)` | C21, C22 |
| `fresnel_propagate(U0,dx,lam,z)` | C22 |
| `mascara_doble_circulo`, `mascara_dos_semicirculos` | C22 |

Antes de tocar cualquiera de estas funciones en el C20, revisar los tres archivos.

**Reparto de responsabilidades C20 vs C22:** ambos resuelven la MISMA abertura en regímenes
distintos (analítico vs Fresnel numérico). No replicar un motor de Fresnel completo en el C20 ni
un motor analítico en el C22 — cada uno remite al otro cuando corresponde (el indicador de régimen
del C20 ya hace esto).

### Código 20 — estructura interna
Una clase `HostFraunhofer` + `ttk.Combobox` + registro `APERTURAS_20` (9 aberturas, cada una con
`{titulo, usa_escala, sliders, render, expr}`). Cada `render(fig,p,modo)` arma su propio gridspec.
Vista única (sin `ttk.Notebook`).

### Código 21 — estructura interna
Clase `HostFFT` + registro `APERTURAS_21` (9 máscaras). Motor genérico `patron_fft(mask_fn, params,
size_char, lam, z, xmax, N)`. Vista única: máscara (izq.) + patrón FFT (der.).

### Código 22 — estructura interna
Clase `TabEvolucionFresnel` + registro `APERTURAS` (6 aberturas, todas en mm). Núcleo `_campo` +
`evolucion` + `_pad_muestreo` (pad adaptativo Nyquist). Slider z↔N_F sincronizado con flag
`_z_locked`.

---

## 3. Convenciones físicas aprendidas (para no re-descubrirlas)

- **SI internamente, GUI en unidades prácticas:** sliders en µm/nm/mm; el núcleo físico
  (`intensidad`, `amplitud_*`, `regimen`) recibe y devuelve todo en metros. La conversión ocurre
  solo en la lectura/dibujo, nunca dentro de las funciones físicas puras.
- **`np.sinc` normalizada:** `sinc(u)=sin(πu)/(πu)`. No multiplicar por π extra al traducir fórmulas
  de papers/PDFs.
- **Convención z_min:** `z_min = 2D²/λ ⟺ N_F=0.5` (NO `N_F=1`). La curva rayada morada del C22 es
  esa.
- **⚠ Punto de confusión frecuente: `N_F` grande = Fresnel (cerca); `N_F` pequeño = Fraunhofer
  (lejos).** Para reproducir un patrón del C20 en el C22 hay que poner `N_F ≤ 0.5` (o mirar la
  curva roja rayada "Fraunhofer límite").
- **Unidades mm vs µm:** C20 y C22 usan mm de forma consistente en sus aberturas (mín. 0.001 mm)
  salvo las rendijas, cuyo modelo es adimensional (`a/λ`, `d/a`). C21 sigue en µm (ver idea abierta
  en §5) pero muestra su unidad explícitamente, así que no es una inconsistencia silenciosa.
- **`ttk.Scale` no dispara `command` en `.set()` programático** — si un slider se actualiza por
  código y su Entry debe reflejarlo, usar `var.trace_add("write", ...)` (patrón ya aplicado en
  `crear_slider` y `crear_slider_log`).
- **Máscara vs física:** `_mascara_fina` rasteriza la abertura en malla fina (500–600 px) SOLO para
  visualización — es independiente de la malla del FFT (que usa zero-padding y dejaría bordes
  escalonados). Nunca toca el cálculo físico.
- **Patrón "rico" (franjas finas visibles):** sale de abertura grande + escala **Log** + N alto.
  Trade-off: aberturas grandes a z modesto caen en Fresnel (la caja de régimen lo marca en rojo);
  el patrón `|FFT|²` se dibuja igual, pero para que sea Fraunhofer válido hay que subir z.

---

## 4. Historial de decisiones no obvias

- La pestaña de **intensidad absoluta** (C20 y C22) se implementó y luego se **revirtió** por
  decisión explícita del usuario — la consideró innecesaria. Ambos códigos volvieron a vista única
  sin `ttk.Notebook`. No reintroducirla sin que se pida.
- El Código 20 pasó de 9 clases `Tab*` (Notebook) a una clase host `HostFraunhofer` +
  `ttk.Combobox` — refactor explícitamente pedido, no espontáneo.
- Las unidades de las aberturas del C20 que estaban en µm (marco, rectángulo rotado, cruz, doble
  cuadrado) pasaron a mm preservando el tamaño físico exacto de cada default — ver §3.

---

## 5. Ideas / trabajos abiertos (NO pedidas aún — confirmar antes)

- **Unificar el Código 21 a mm** (hoy en µm) para consistencia total con C20/C22. No es urgente:
  el usuario pidió mm específicamente "en el Código 20"; C21 sí muestra su unidad, así que no es
  una inconsistencia oculta.
- Revisar si algún subtítulo interno de eje en los `render_*` del C20 sigue en µm (los sliders ya
  son mm; las etiquetas de los ejes de la máscara/patrón son cosméticas, no afectan el cálculo).
