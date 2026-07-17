# Simulador de bases de datos

Genera bases de datos tabulares sinteticas (CSV) a partir de una especificacion
declarada por el usuario: variables continuas, variables categoricas, una
matriz de correlaciones opcional entre las continuas, y relaciones opcionales
de una categorica sobre una continua. La interfaz web se rellena subiendo un
Excel (pensado para poder definir muchas variables comodamente); tambien se
puede usar por linea de comandos con un fichero YAML/JSON equivalente.

Los valores generados **aproximan** los parametros indicados (media, mediana,
min, max, proporciones, correlaciones); no se garantiza una coincidencia
exacta, solo una aproximacion razonable, tal y como permite el enunciado del
problema.

## Instalacion

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Interfaz web (recomendado)

```bash
streamlit run app.py
```

El flujo es:

1. **Descarga la plantilla Excel** desde la propia app (boton "⬇️ Descargar
   plantilla Excel"). Trae una hoja de instrucciones, ejemplos ya rellenos y
   listas desplegables para `tipo`/`distribucion`.
2. **Rellena la plantilla fuera de la app** (en Excel, Google Sheets, etc.):
   borra las filas de ejemplo y anade tus variables. Al ser una tabla normal
   puedes copiar/pegar o arrastrar formulas para definir muchas variables
   rapidamente (pensado para decenas o cientos de filas, no solo unas pocas).
3. **Sube el Excel ya rellenado**. La app muestra una vista previa de lo
   leido (variables, matriz de correlaciones y relaciones categorica →
   continua si las hay) para que puedas revisarlo antes de generar nada.
4. **Genera la simulacion**. Se muestra la tabla resultante, un boton de
   descarga CSV, un informe objetivo-vs-obtenido (con el desglose por
   categoria si hay relaciones) y un histograma por cada variable continua.

## Formato del Excel

Un ejemplo ya relleno esta en `examples/plantilla_ejemplo.xlsx` (es el mismo
fichero que genera el boton "Descargar plantilla" de la app). Tiene 4 hojas:

### Hoja `instrucciones`

Texto explicando el resto de hojas, para no depender de este README al
rellenarla.

### Hoja `variables` (obligatoria)

Una fila por variable, columnas:

| nombre | tipo | distribucion | media | mediana | min | max | sd | categorias |
|---|---|---|---|---|---|---|---|---|
| edad | continuous | normal | 40 | 38 | 18 | 85 | | |
| ingresos | continuous | lognormal | 30000 | 27000 | 5000 | 250000 | | |
| sexo | categorical | | | | | | | Hombre:50;Mujer:50 |

- `distribucion`, `media`, `mediana`, `min`, `max`, `sd` solo aplican si
  `tipo = continuous`. `distribucion` es una de: `normal`, `lognormal`,
  `uniform`, `triangular`, `exponential`, `beta`. `sd` es opcional (dejala
  vacia si no quieres fijar la desviacion tipica).
- `categorias` solo aplica si `tipo = categorical`, formato
  `Nivel:proporcion;Nivel:proporcion` (ej. `Hombre:48;Mujer:50;Otro:2`); no
  hace falta que las proporciones sumen 100, se normalizan solas.

### Hoja `correlaciones` (opcional)

Matriz de correlaciones entre variables **continuas**: la primera columna y
la primera fila deben tener los mismos nombres de variable (en el mismo
orden), la diagonal debe ser 1 y la matriz debe ser simetrica.

|  | edad | satisfaccion |
|---|---|---|
| edad | 1 | 0.2 |
| satisfaccion | 0.2 | 1 |

### Hoja `efectos` (opcional)

Efecto de una categorica sobre la media/mediana de una continua, en formato
largo (una fila por variable continua + variable categorica + nivel):

| variable_continua | variable_categorica | nivel | delta_media | delta_mediana |
|---|---|---|---|---|
| ingresos | sexo | Mujer | 5000 | 4000 |

Deja `delta_media`/`delta_mediana` vacios para los niveles sin efecto (no
hace falta listarlos). Varias filas con la misma variable pueden usar
categoricas distintas (por ejemplo que `ingresos` dependa a la vez de `sexo`
y de `region`); los desplazamientos se suman.

**Restriccion:** una variable que aparece en `efectos` no puede aparecer
tambien en `correlaciones`. Forzar una correlacion reordena las filas de esa
variable, lo que deshace el efecto por categoria; la app (y la libreria) lo
rechazan con un error explicito en vez de generar un resultado enganoso.

## Uso por linea de comandos

```bash
python -m db_simulator.cli examples/example_spec.yaml -o output.csv --report
```

- `-o/--output`: ruta del CSV generado (por defecto `output.csv`).
- `-n/--rows`: sobrescribe el numero de filas de la especificacion.
- `-s/--seed`: sobrescribe la semilla aleatoria (para reproducibilidad).
- `--report`: imprime un informe comparando los objetivos con los valores
  realmente obtenidos (media, mediana, min, max, proporciones y correlaciones).

También se puede usar como libreria de Python:

```python
from db_simulator import Simulator

sim = Simulator.from_file("examples/example_spec.yaml")
df, report = sim.generate_with_report()
print(report.to_text())
df.to_csv("output.csv", index=False)
```

## Formato de la especificacion (YAML/JSON, para la CLI)

Es el mismo modelo que el Excel (variables, correlation_matrix, effects), en
formato YAML/JSON en vez de hojas de calculo:

```yaml
n_rows: 2000        # numero de filas a generar
seed: 42            # semilla aleatoria (opcional, para reproducibilidad)

variables:
  - name: edad
    type: continuous
    distribution: normal   # normal | lognormal | uniform | triangular | exponential | beta
    mean: 42
    median: 40
    min: 18
    max: 85

  - name: genero
    type: categorical
    categories:
      Hombre: 0.48
      Mujer: 0.50
      Otro: 0.02

correlation_matrix:            # opcional; solo entre variables continuas
  variables: [edad, ingresos]
  matrix:
    - [1.0, 0.6]
    - [0.6, 1.0]
```

### Variables continuas

Para cada variable se indica `distribution`, `mean`, `median`, `min` y `max`, y
opcionalmente `sd` (desviacion tipica objetivo). El programa ajusta
numericamente los parametros de la distribucion elegida para que su media,
mediana (y sd si se indica), calculadas dentro del rango `[min, max]`, se
acerquen a los valores objetivo, y despues genera los datos respetando
siempre el rango indicado. Distribuciones soportadas:

- `normal`: distribucion normal truncada al rango. Usa `sd` si se indica.
- `lognormal`: distribucion log-normal truncada al rango. Usa `sd` si se indica.
- `uniform`: uniforme entre `min` y `max` (por definicion no admite ajustar
  media/mediana/sd de forma independiente del rango).
- `triangular`: distribucion triangular con moda ajustada para aproximar la
  media indicada (un unico grado de libertad: no usa `sd`).
- `exponential`: exponencial (con desplazamiento) truncada al rango. Usa `sd`
  si se indica.
- `beta`: distribucion beta reescalada al rango `[min, max]`. Usa `sd` si se
  indica.

Nota: si los parametros pedidos son matematicamente incompatibles entre si
(por ejemplo, una media muy alta con una distribucion demasiado simetrica y
un rango estrecho), el ajuste numerico busca el valor mas cercano posible
dentro de lo factible.

### Variables categoricas

Se definen mediante un diccionario `categoria: proporcion`. Las proporciones
no necesitan sumar exactamente 1: se renormalizan automaticamente.

### Matriz de correlaciones

Se especifica una lista de variables continuas y su matriz de correlaciones
(simetrica, con 1.0 en la diagonal). Internamente se usa el metodo de
Iman-Conover: se generan las muestras de forma independiente respetando sus
distribuciones marginales y despues se reordenan (sin alterar los valores)
para que su estructura de rangos siga la correlacion deseada. Si la matriz
proporcionada no es valida (no semidefinida positiva), se proyecta
automaticamente a la matriz de correlacion valida mas cercana.

### Efecto de una categorica sobre una continua

Una variable continua puede depender de una categorica mediante `effects`:
para cada nivel de esa categorica se indica cuanto se suma (`mean_delta`,
opcionalmente tambien `median_delta` y `sd_delta`) a la media/mediana/sd
*base* de la continua, solo para las filas de ese nivel (los niveles no
listados no cambian). Por ejemplo, para que las mujeres tengan de media mas
ingresos que el resto:

```yaml
variables:
  - name: ingresos
    type: continuous
    distribution: lognormal
    mean: 30000
    median: 27000
    min: 5000
    max: 250000
    effects:
      - by: genero
        mean_delta: {Mujer: 5000}
        median_delta: {Mujer: 4000}

  - name: genero
    type: categorical
    categories: {Hombre: 0.49, Mujer: 0.49, Otro: 0.02}
```

Se pueden anadir varias entradas en `effects` (por ejemplo que ingresos
dependa a la vez de genero y de region); sus desplazamientos se suman.

**Restriccion:** una variable con `effects` no puede aparecer ademas en
`correlation_matrix`. Forzar una correlacion reordena las filas de esa
variable (para ajustar su relacion con otra), lo que deshace el efecto por
categoria. Si necesitas ambas cosas sobre el mismo dato, aplica el efecto por
categoria y correlaciona otra variable distinta.

## Tests

```bash
pip install -r requirements.txt pytest
python -m pytest
```

Los tests comprueban que las muestras generadas respetan los rangos exactos
y se aproximan a la media/mediana/proporciones/correlaciones objetivo.
