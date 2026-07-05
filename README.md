# Simuladores de Difracción Óptica — UNAL Óptica (3010225)

Tres programas interactivos para simular **patrones de difracción óptica** en régimen de Fraunhofer (campo lejano) y Fresnel (campo cercano), usando Python con interfaz gráfica en tkinter.

## Proyectos

### Código 20: Fraunhofer Analítico 2D
- **Ubicación:** `Codigo_1/`
- **Objetivo:** Graficar el patrón 2D de intensidad en Fraunhofer usando fórmulas analíticas (`sinc²`)
- **Aberturas:** Paralelogramo/rectángulo parametrizables
- **Validación:** Verifica condición de campo lejano: `z ≥ 2D²/λ`

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
