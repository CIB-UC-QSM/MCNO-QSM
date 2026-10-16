# MCNO-QSM

Prototipo de inversión QSM con una CNN multikernel y MLP complejos independientes
por segmento dipolelet. El experimento ajusta un único volumen Sim2 completo;
sus métricas miden ajuste al sujeto de entrenamiento, no generalización.

## Ejecutar

Requiere Python 3.13 y `uv`. Desde la raíz del proyecto:

```bash
uv sync --locked
uv run pytest -q
uv run mcno-qsm overfit --config configs/sim2.toml
```

La configuración predeterminada usa CUDA si está disponible, 2000 actualizaciones,
Adam con tasa `0.001` y semilla 42. Para comprobar el pipeline completo:

```bash
uv run mcno-qsm overfit --steps 2 --device cuda --output runs/smoke
```

Cada ejecución necesita una carpeta de salida nueva. Todos los parámetros se
configuran en TOML; `--steps`, `--device` y `--output` permiten sobrescribirlos.
La CPU sirve para pruebas pequeñas; entrenar el volumen nativo requiere bastante
memoria y tiempo. El procesamiento por bloques y la recomputación de activaciones
reducen la memoria sin descartar frecuencias ni cambiar sus gradientes.

## Arquitectura

La entrada tiene forma `[B,1,X,Y,Z]`. Tres convoluciones **paralelas** de tamaño
`3×3×3`, `5×5×5` y `7×7×7`, con dos filtros cada una y GELU, producen seis canales
concatenados. Cada convolución conserva las dimensiones mediante padding cero.

La transformada aplica tres detalles starlet B3, con filtro separable
`[1,4,6,4,1]/16`, dilataciones `1,2,4` y borde espejo. Cada detalle se transforma a
Fourier y se divide en `8,4,2` bandas de `|D(k)|`, respectivamente. La aproximación
gruesa aporta un segmento adicional sin división angular: quince en total.

`create_dipolelet_windows` construye discos
`L_t(k) = sigmoid(100·(t−|D(k)|))`. Las ventanas son `L_0`, las diferencias
consecutivas y `1−L_último`, normalizadas para sumar uno. Los umbrales son
logarítmicos entre `0.03` y `1/3`. Son ventanas suaves **solapadas**: cada segmento
conserva las `N=X·Y·Z` frecuencias, sin selección por umbral de amplitud.

Cada segmento se ordena como `(N,6)` por muestra y pasa por su propio MLP complejo
de diez capas lineales: `6→16`, ocho `16→16` y `16→1`. Las nueve activaciones son
GELU separadas sobre real e imaginario, escaladas por `1.5` para compensar la
atenuación en profundidad. No hay sesgos en los MLP. Los pesos se comparten entre
frecuencias del mismo segmento; no se comparten entre segmentos.

Se suman los quince espectros de salida, se aplica la proyección hermítica
`(S(k)+conj(S(−k)))/2`, y se vuelve al espacio de imagen con FFT inversa. Esta
proyección garantiza una salida real incluso con pesos complejos arbitrarios.
No se multiplican de nuevo las ventanas durante la síntesis. Todas las FFT usan
normalización ortonormal y orden nativo, sin `fftshift`.

## Datos y física

Los archivos originales se leen sin modificarlos:

- `Sim2.mat`: variable `phase`, recortada desde `[0,0,0]` a `164×205×205`.
- `Sim2ChiGT.nii.gz`: susceptibilidad GT, affine y espaciado físico.
- `mask_final.mat`: variable `mask_final>0`.

`sim2_w.mat` se conserva, pero no se utiliza. Se asume explícitamente que el campo
está en ppm, escala `1`, y B0 apunta a z en los ejes de imagen. La configuración
permite cambiar la escala y B0. La entrada se divide por su desviación estándar
dentro de la máscara, sin centrarla; la salida y el GT se mantienen en ppm.

El forward usa `D(k)=1/3−(k·b0)²/|k|²`, con `D(0)=0`, frecuencias en unidades
físicas y 16 vóxeles de padding cero por lado. La susceptibilidad se anula fuera
de la máscara **antes** de simular el campo; el resultado se recorta al tamaño
original y las pérdidas se evalúan dentro de la máscara. Para B0 oblicuo se
simetrizan los pares de frecuencias de Nyquist para preservar una convolución real.

El campo objetivo de la pérdida física se simula desde el GT, una sola vez,
con el mismo forward usado para la predicción. La entrada de la red sigue siendo
el campo **medido** de Sim2.

```text
b_gt = A(mask * chi_gt)
L_physics = mean_mask((A(mask * chi_pred) - b_gt)^2) / mean_mask(b_gt^2)
L_L1 = mean_mask(abs(chi_pred - chi_gt)) / mean_mask(abs(chi_gt))
L = L_physics + L_L1
NRMSE = norm((chi_pred - chi_gt)[mask]) / norm(chi_gt[mask])
```

Los dos pesos de pérdida son configurables. No se ajusta escala, media ni offset
contra el GT al calcular el NRMSE. Su valor se registra como fracción, no porcentaje.

## Resultados

Cada fila de `history.csv` contiene pérdida total, física, L1 y NRMSE. La fila cero
es la inicialización; cada fila posterior evalúa el estado **después** de esa
actualización del optimizador. La consola muestra cada fila y `nrmse.png` contiene
la curva completa. Los artefactos visuales se actualizan cada cien pasos y al final.

Se guardan `best.pt`, metadatos y hashes SHA-256 de las entradas. `best`, `latest`
y `final` incluyen mapas NIfTI e imágenes de cortes centrales con NRMSE en el
título y contraste `[-0.1,0.1]` ppm. Sus diferencias firmadas `predicción−GT` se
muestran dentro de máscara con contraste `[-0.03,0.03]` ppm. Los NIfTI mantienen
el affine del GT. También se guarda el campo simulado usado por la pérdida.

`summary.json` informa si se alcanzó la meta orientativa NRMSE ≤ `0.10`. Esta meta
no detiene anticipadamente la ejecución ni garantiza que la arquitectura la
alcance. Se guarda el mejor estado aunque la pérdida y el NRMSE no sean monótonos.

## Referencia y adaptaciones

Basado en [ignacontreras/Dipolelets, revisión cf98d579](https://github.com/ignacontreras/Dipolelets/tree/cf98d579bbd5c100c8698a81ccd630f4055568a1/Codes),
especialmente `Starlet.py`, `Dipolelet.py` y `FANSI_tools.py`. El filtro B3 y los
bordes espejo se verifican contra SciPy; las ventanas se verifican contra la
fórmula sigmoide de esa revisión. Esta implementación usa PyTorch diferenciable,
devuelve coeficientes complejos en Fourier y fija un orden numérico de bandas.
La CNN y los MLP son el modelo de aprendizaje añadido a esa representación.

Las pruebas incluyen partición de unidad, reconstrucción exacta sin MLP, tamaños
pares e impares, señales de forward conocidas, gradientes con recomputación,
aislamiento de segmentos, NRMSE exacto, geometría NIfTI, imágenes y descenso de
pérdida sobre un pequeño fantoma sintético.

## Validación ejecutada

Las 15 pruebas pasan con Python 3.13.3 y PyTorch 2.14.1. Se ejecutaron 100 pasos
sobre Sim2 completo en una RTX PRO 6000 Blackwell, usando la configuración
predeterminada y sobrescribiendo únicamente pasos, dispositivo y salida:

```bash
uv run mcno-qsm overfit --config configs/sim2.toml --steps 100 --device cuda --output runs/sim2-100
```

| Paso | NRMSE enmascarado |
| --- | --- |
| 0 | 6.672893 |
| 10 | 1.102417 |
| 25 | 0.937481 |
| 50 | 0.819577 |
| 75 | 0.784580 |
| 100 | 0.762371 |

La pérdida física final fue `0.357378` y la L1 `0.715252`. La ejecución tardó
462 segundos y su pico de memoria asignada en GPU fue 8.18 GiB. Los resultados
están en `runs/sim2-100/`, incluidos `history.csv`, `nrmse.png`, `best.png`,
`best_difference.png` y `summary.json`. El NRMSE de los títulos puede variar en
las últimas cifras por las reducciones float32 de CPU frente a CUDA.

Esta prueba verifica entrenamiento y exportación a resolución nativa, pero **no
alcanzó la meta de sobreajuste NRMSE ≤ 0.10**. La corrida predeterminada de 2000
pasos está preparada y no se ha ejecutado.

## Comparación de profundidad

El modelo completo es una capa que recibe y devuelve un volumen de un canal.
`--depth` encadena capas con pesos independientes. Los volúmenes intermedios
pasan directamente a la siguiente capa; la normalización se aplica únicamente
al campo medido inicial y la máscara y las pérdidas a la salida final. Una capa
mantiene exactamente la arquitectura y el formato de checkpoint originales.

```bash
uv run mcno-qsm overfit --depth 3 --steps 100 --device cuda --output runs/sim2-depth3
uv run python -m mcno_qsm.compare run --steps 100 --target-gb 60 --output runs/depth-comparison
```

La comparación perfila evaluación y dos actualizaciones completas de Adam en
procesos CUDA independientes. Mide tanto memoria asignada como reservada y elige
la profundidad cuyo pico **reservado** es más cercano a 60 GB decimales
(`60 × 10⁹` bytes). No se añaden tensores artificiales ni se cambian tamaño de
volumen, anchura, bloques de frecuencias o recomputación entre profundidades.
La granularidad de una capa y el comportamiento del asignador impiden garantizar
exactamente 60 GB. Esta medida pertenece al proceso PyTorch; no incluye la memoria
de otros programas ni toda la sobrecarga del controlador CUDA.

Después entrena 1, 2, 3 capas y la profundidad seleccionada con el mismo número de
pasos, semilla, entrada, pérdidas y optimizador. Los perfiles y la selección quedan
en `profiles/` y `selection.json`. `experiment.json` registra configuración, hashes
de datos y versión de PyTorch para impedir reutilizar resultados incompatibles.
Puede reutilizarse una corrida completa y equivalente de una capa con
`--baseline runs/sim2-100`; su configuración, entradas y presupuesto se verifican.

Los archivos `nrmse_comparison.png`, `reconstructions.png`, `differences.png` y
`comparison.csv` comparan las curvas y las salidas del **mismo paso final**. También
se guardan cortes individuales por profundidad y los mejores checkpoints en cada
subcarpeta. Los cortes usan `gray`, los contrastes permanecen en `[-0.1,0.1]` y
`[-0.03,0.03]` ppm y todas las curvas tienen límites verticales `[0,1]`.

### Resultados: 100 pasos por profundidad

Se completaron las cuatro corridas con semilla 42. La referencia de una capa se
reutilizó desde `runs/sim2-100` tras verificar su equivalencia. Se comparan las
salidas del paso 100:

| Capas | NRMSE final | Pico de VRAM asignada (GB) |
| --- | --- | --- |
| 1 | 0.762371 | 8.78 |
| 2 | 0.650660 | 15.93 |
| 3 | **0.610543** | 23.07 |
| 8 | 0.982911 | 58.78 |

Ocho capas fueron la profundidad más cercana a 60 GB: el perfil midió 62.795 GB
reservados frente a 54.715 GB para siete capas. Durante los 100 pasos, el pico
reservado de ocho capas fue 62.799 GB y el asignado 58.776 GB. Su mejor NRMSE fue
0.963106 en el paso 99; las imágenes comparativas usan el paso 100 para conservar
el mismo criterio en todos los casos.

Tres capas obtuvieron el menor error en este experimento. Ocho capas mostraron
un ajuste mucho peor con el mismo presupuesto de pasos. Ninguna corrida alcanzó
NRMSE ≤ 0.10; los resultados corresponden a un solo sujeto y una sola semilla.
Los tiempos registrados incluyen carga compartida de GPU y no constituyen una
medición de velocidad con la GPU aislada.

Las curvas y cuadrículas están en `runs/depth-comparison/nrmse_comparison.png`,
`reconstructions.png` y `differences.png`. Las curvas conservan `[0,1]` y explicitan
que los valores superiores a uno quedan fuera del rango visible. Los CSV
conservan todos los valores originales. Para regenerar las figuras sin entrenar:

```bash
uv run python -m mcno_qsm.compare report --output runs/depth-comparison
```
