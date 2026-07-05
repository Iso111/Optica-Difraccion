# Simuladores de Difracción Óptica — UNAL Óptica (3010225)

Tres programas interactivos para simular **patrones de difracción óptica** en régimen de Fraunhofer (campo lejano) y Fresnel (campo cercano), usando Python con interfaz gráfica en tkinter.

## Proyectos

### Código 20: Fraunhofer Analítico 2D ✅
- **Ubicación:** `Codigo_1/fraunhofer_analitico.py`
- **Objetivo:** Graficar el patrón 2D de intensidad en Fraunhofer usando la expresión analítica
- **GUI:** tkinter con 5 pestañas (`ttk.Notebook`), cada una un ejercicio independiente. Ejecutar:
  `python Codigo_1/fraunhofer_analitico.py`
- **Régimen explícito:** cada pestaña muestra el número de Fresnel `N_F` y declara si el cálculo
  Fraunhofer es válido (verde) o si estás en campo cercano → usar Código 22 (rojo)

**Pestaña "Parcial 4 — Punto 1":**
- **Abertura:** marco rectangular (`a×b − c×d`) + círculo (`R`), separados `D`
- **Expresión:** `I = A_marco² + A_círc² + 2·A_marco·A_círc·cos(2πD·x'/λz)` (franjas de interferencia)
- **Casos degenerados** para validar: rectángulo `sinc²` (`c=d=R=0`), disco de Airy (`a=b=c=d=0`)

**Pestañas de taller** (ejercicios adicionales del curso):
- **Paralelogramo** (a=10µm, b=5µm, θ=60°): fórmula genérica del `Contexto_códigos.md`
  (`I ∝ sinc²(a·x'/λz)·sinc²(b·(x'cosθ+y'senθ)/λz)`); validado contra el rectángulo cuando θ=90°.
- **Cruz** (brazos de ancho `a`, largo total `L`): unión de dos barras vía inclusión-exclusión de
  conjuntos (`A_cruz = A_h + A_v − A_solape`); validado contra un cuadrado sólido cuando a=L.
- **Círculo con muesca** (D=2mm, corte de 1.414mm≈R√2, cuerda a 45°): calcula la **irradiancia
  axial cerrada** `I(0,0)=I₀·(Área/λz)²` pedida por el enunciado (área vía geometría de segmento
  circular) y además visualiza el **patrón 2D numérico** (FFT de la abertura rasterizada,
  explícitamente marcado como método distinto al resto de pestañas); ambos se validan cruzados
  entre sí (<1% de diferencia).
- **Doble cuadrado** (lados `a` y `3a`, separados `2a` borde a borde → `D=4a`): misma física de dos
  aberturas + interferencia que la Pestaña 0, con envolventes cuadradas.

### Código 21: Fraunhofer vía Transformada de Fourier (FFT 2D)
- **Ubicación:** `Codigo_2/`
- **Objetivo:** Demostrar que difracción de Fraunhofer = TF espacial 2D de la abertura
- **Método:** `numpy.fft.fft2` + `fftshift` con mapeo correcto de coordenadas (`dx' = λz/Lx`)
- **Casos:** Rectángulo (valida contra Código 20) y círculo (Disco de Airy → Bessel J₁)

### Código 22: Difracción de Fresnel (Campo Cercano)
- **Ubicación:** `Codigo_3/`
- **Objetivo:** Calcular difracción Fresnel usando integrales de Fresnel (espiral de Cornu)
- **Método:** `scipy.special.fresnel` vectorizado con `meshgrid`
- **Resultado clave:** Evolución del patrón en función del número de Fresnel, transición hacia Fraunhofer

## Requisitos

```bash
pip install numpy scipy matplotlib
```

Para GUI interactivo: tkinter (incluido con Python estándar).

## Estructura

```
Difracción/
├── README.md                  # Este archivo
├── .gitignore
├── Contexto_códigos.md        # Especificación matemática completa
├── Parcial4.pdf               # Examen (evaluación final)
├── Codigo_1/                  # Fraunhofer analítico
│   └── fraunhofer_analitico.py
├── Codigo_2/                  # Fraunhofer por FFT
│   └── fraunhofer_fft.py
├── Codigo_3/                  # Fresnel
│   └── fresnel.py
└── [PDFs de referencia]
```

## Guía de Uso

Cada código se ejecuta independientemente:

```bash
python Codigo_1/fraunhofer_analitico.py
python Codigo_2/fraunhofer_fft.py
python Codigo_3/fresnel.py
```

Cada programa abre una **interfaz gráfica interactiva** con deslizadores para variar parámetros en caliente (longitud de onda, distancia de observación, dimensiones de abertura, etc.).

## Referencias Teóricas

- **Fraunhofer + FFT:** `Fraunhofer_Transfomada_Fourier_espacial.pdf`
- **Fresnel + Integrales:** `Difraccion_Fresnel_Clotoide.pdf`

## Evaluación (Parcial 4)

El examen pide:
1. **Punto 1:** Fraunhofer de **dos aberturas separadas** (factor de interferencia) → amplía Código 20
2. **Punto 2:** Fraunhofer de **cualquier abertura** vía FFT, estilo PhET → Código 21
3. **Punto 3:** Fresnel con **evolución del número de Fresnel** en una sola figura → Código 22

---

**Autor:** Isaac David  
**Curso:** UNAL Óptica (3010225)  
**Fecha:** 2025
