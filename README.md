# Simulador de bases de datos

Genera bases de datos tabulares sinteticas (CSV) a partir de una especificacion
declarada por el usuario en YAML o JSON: variables continuas, variables
categoricas y una matriz de correlaciones opcional entre las variables
continuas.

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

Abre una app en el navegador donde puedes anadir o quitar variables con un
boton, escribir el nombre y desplegar cada una para editar sus caracteristicas
(tipo, distribucion, media/mediana/min/max/sd o categorias y proporciones),
definir la matriz de correlaciones entre las variables continuas en una tabla
editable, y generar y descargar el CSV con un informe objetivo-vs-obtenido y
un histograma por cada variable continua, sin tocar ningun fichero YAML a mano.

Todos los campos de variables y de la matriz de correlaciones viven dentro de
un unico formulario: los valores que escribes se guardan al pulsar cualquier
boton de esa seccion (➕ Anadir variable, 🗑️ Eliminar, 💾 Aplicar cambios o
🚀 Generar simulacion), para que nunca se pierda lo que ya habias introducido
en otra variable.

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

## Formato de la especificacion

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

## Tests

```bash
pip install -r requirements.txt pytest
python -m pytest
```

Los tests comprueban que las muestras generadas respetan los rangos exactos
y se aproximan a la media/mediana/proporciones/correlaciones objetivo.
