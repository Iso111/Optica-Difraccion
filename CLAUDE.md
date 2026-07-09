# Coding Guidelines (project-specific)

Los cuatro principios de `~/.claude/CLAUDE.md` aplican a todo el código de este proyecto.
Refuerzos específicos de este proyecto:

- **Piensa primero:** antes de tocar cualquier expresión física (fórmula analítica, mapeo de
  coordenadas FFT, límites de integración de Fresnel), indica qué ecuación del
  `Contexto_códigos.md` o del PDF de referencia estás implementando y qué caso de validación
  usarás para comprobarla (p. ej. "disco de Airy: primer cero en 1.22λz/2R"). Si hay más de una
  interpretación física razonable (p. ej. cómo definir `D_char` para el criterio de campo lejano),
  presenta las opciones — no elijas en silencio.
- **Simplicidad:** cada código (20, 21, 22) es una app autónoma de un solo archivo. No introduzcas
  abstracciones compartidas entre `Codigo_1/2/3` (clases base, módulos comunes) a menos que se pida
  explícitamente — la duplicación puntual de un slider o un patch de matplotlib es preferible a una
  librería interna prematura.
- **Cambios quirúrgicos:** no toques el núcleo físico de un código (funciones de amplitud/intensidad,
  condición de campo lejano, mapeo `dx' = λz/L`, límites de Fresnel `u,v`) al hacer cambios de GUI o
  estilo. Un signo o factor mal movido invalida la validación numérica ya hecha.
- **Dirigido por objetivos:** toda tarea de física debe convertirse en un criterio verificable antes
  de escribir código — "agrega la abertura X" → "¿qué caso límite la valida? (¿coincide con sinc²
  cuando c=d=0? ¿con Airy cuando solo hay círculo?)". Declara el plan de verificación (qué input →
  qué output esperado) antes de implementar.
- **Marcar vs. tocar:** si detectas un error real en una fórmula ya validada de otro Código (p. ej.
  un signo en el mapeo de frecuencias del Código 21), no lo corrijas de paso — detente, repórtalo
  con el fix propuesto y aplícalo solo tras aprobación.
- **No dupliques trabajo entre códigos:** el Código 20 (analítico) y el 22 (Fresnel/Cornu) resuelven
  la MISMA abertura en regímenes distintos — no repliques ahí un motor de Fresnel completo si ya
  existe en el otro código; en su lugar, referencia o remite al código correspondiente (como hace el
  indicador de régimen del Código 20).

## Convenciones de unidades y estilo (aprendidas en Código 20)

- **SI internamente, GUI en unidades prácticas:** los sliders trabajan en µm/nm/mm (naturales para
  óptica), pero el núcleo físico (`intensidad`, `amplitud_*`, `regimen`) recibe y devuelve todo en
  metros. La conversión ocurre solo en `_leer()`/dibujo, nunca dentro de las funciones físicas puras.
- **Núcleo físico sin GUI:** separa siempre funciones puras (numpy in/out, sin tkinter) de la clase
  GUI, para poder testear headless con `matplotlib.use("Agg")` sin abrir ventanas.
- **`np.sinc` es normalizada:** `sinc(u) = sin(πu)/(πu)`. No multipliques por π extra al pasar
  argumentos — es un error común al traducir fórmulas de papers/PDFs a numpy.
- **Vectorizado, sin bucles `for`:** toda evaluación 2D usa `np.meshgrid` + operaciones matriciales.
- **GUI tkinter:** sigue el patrón de `Parcial 3/Óptica - Interferómetro de Michelson/interferometro_gui.py`
  — `_slider()` reutilizable (ttk.Scale + Entry editable con commit en Enter/FocusOut), figura
  matplotlib embebida vía `FigureCanvasTkAgg` + `NavigationToolbar2Tk`, recálculo en `recompute()`.

## Toma de decisiones: usa preguntas estructuradas

Antes de actuar sobre decisiones ambiguas o con varias opciones válidas, **pregunta con una encuesta
estructurada** (AskUserQuestion) en lugar de elegir en silencio. Esto aplica a:

- **Alcance de un código** ("¿el Código 20 debe cubrir solo el paralelogramo del examen o también el
  caso genérico con ángulo θ del `Contexto_códigos.md`?")
- **Elecciones físicas con más de una convención razonable** (definición de `D_char` para campo
  lejano, qué unidad de referencia usar para el número de Fresnel, cómo tratar `R=0` o aberturas
  degeneradas)
- **Decisiones de arquitectura de GUI** (una app por código vs. una app con pestañas; qué controles
  exponer como slider vs. fijos)
- **Operaciones riesgosas** (comandos git destructivos, sobrescribir un código ya validado)

**Cuándo NO preguntar:**
- Bugs con una solución clara y única.
- Typos, refactors triviales, agregar tests/validaciones numéricas.
- Operaciones ya aprobadas explícitamente en un plan aceptado por el usuario.

**Formato de encuesta:** opciones múltiples (2–4), mutuamente excluyentes, con descripciones claras
de la implicación física/técnica de cada una. Deja que el usuario elija una vez, no vuelvas a
preguntar lo mismo.

## Flujo de trabajo

- Cada código nuevo pasa por **plan mode** primero: leer la teoría relevante (`Contexto_códigos.md`
  + el PDF correspondiente + la parte del `Parcial4.pdf` que aplique), producir la expresión
  analítica/numérica, y solo entonces escribir el plan de implementación.
- Todo código se valida **headless** (sin abrir la ventana) contra al menos un caso límite conocido
  antes de darlo por terminado, y luego se hace un smoke test de la GUI completa.
- **Gate de regresión:** antes de dar por terminado cualquier cambio a un `.py` del núcleo físico
  (Código 20/21/22), correr `python validacion_fisica.py` y exigir exit 0 (40/40 checks). Si un
  cambio rompe un check, es una señal de regresión física — no relajar el check para que pase.
- **Doc sincronizada:** si un cambio modifica una fórmula o expresión física ya documentada en
  `Documentacion_Fisica.tex` (no un cambio de GUI/estilo), actualizar ese archivo y recompilar el
  PDF en el mismo commit.
- Los commits a git/GitHub se hacen solo cuando el usuario lo pide o lo aprueba explícitamente.
