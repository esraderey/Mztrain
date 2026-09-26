# Resultado: ahorro local de tiempo con ElasticShape

Fecha: 2026-09-07. Las seis corridas terminaron sin errores. La regla de decision
fijada antes de entrenar se cumple: tres cirugias aceptadas, tres objetivos
alcanzados, ahorro en las tres parejas y ratio geometrico menor que 0.90.

**Ahorro agregado observado: 30.15%.** Ratio geometrico de tiempos M/R=0.698526.
Tiempo medio hasta el objetivo: 55.886 s con ElasticShape y 79.917 s desde cero.
Los tiempos incluyen construir el modelo, entrenar, evaluar y efectuar la
cirugia con sus sondas y migracion. No son solo tiempos de GEMM o de train.

## Comparacion primaria preespecificada

Cada semilla tiene su propio objetivo Q: el BPC de su referencia al paso 4000.
Se busca el primer punto observado que iguala o mejora Q. En ElasticShape solo
cuentan puntos que ya tienen la arquitectura final. BPC menor es mejor.

| Semilla | Objetivo Q (BPC) | Desde cero (s) | ElasticShape (s) | Ahorro | BPC observado de ElasticShape al cruzar |
|---|---:|---:|---:|---:|---:|
| 17 | 3.064610 | 80.143 | 59.671 | 25.54% | 3.004433 |
| 29 | 3.079263 | 79.512 | 54.050 | 32.02% | 3.068794 |
| 43 | 3.119238 | 80.095 | 53.939 | 32.66% | 3.055244 |

Los cruces de M se observaron en los pasos 3750,3500,3500; los de R, en 4000.
El modelo de M antes del crecimiento tiene 553,728 parametros. Ambas rutas
terminan con 1,403,904 parametros y la misma estructura de parametros, verificada
nombre por nombre y forma por forma en las tres semillas.

La cadencia de evaluacion es de 250 pasos, aproximadamente 5 s con el modelo final.
No se interpolaron cruces. Usando el punto anterior de R y el cruce observado de
M, las diferencias locales conservadoras son 20.37%,27.63%,27.66%. Estas ventanas
describen resolucion temporal, NO intervalos de confianza ni una garantia contra
oscilaciones no observadas entre evaluaciones.

Separando evaluaciones y setup, los ratios de (train de M + cirugia)/train de R
son 0.7410,0.6689,0.6686: la ventaja no depende solamente de evaluar mas barato.

## Calidad dentro del presupuesto de referencia: secundario descriptivo

Esta comparacion se calculo DESPUES de las corridas y NO forma parte de la regla
de decision preespecificada. Se toma el ultimo punto de M que no excede el
tiempo objetivo de R; no se interpola ni se usa un punto posterior al presupuesto.

| Semilla | Presupuesto R (s) | Tiempo usado por M (s) | BPC R | BPC M |
|---|---:|---:|---:|---:|
| 17 | 80.143 | 78.972 | 3.064610 | 2.796452 |
| 29 | 79.512 | 78.584 | 3.079263 | 2.780076 |
| 43 | 80.095 | 78.384 | 3.119238 | 2.795127 |

Con presupuesto no mayor, se observo menor perdida de validacion en 3/3 casos.
La media es 3.087704 BPC en R frente a 2.790552 BPC en M. Esto es calidad a un
presupuesto concreto, NO mejor calidad final despues de converger.

## Cirugias y configuracion

| Semilla | Cirugia (s) | Incremento de CE en sonda (nats/token) | KL en sonda | Aceptada |
|---|---:|---:|---:|---|
| 17 | 1.066845 | 0.005772 | 0.003798 | Si |
| 29 | 0.169179 | 0.002405 | 0.001551 | Si |
| 43 | 0.153187 | 0.003908 | 0.002763 | Si |

- RTX 4060, PyTorch 2.11.0+cu128, fp32, TF32 desactivado, sin compilacion.
- WikiText-2 raw real: 10,884,797 caracteres de entrenamiento despues de reservar
  la sonda; 1,142,150 caracteres en validacion. Vocabulario de train+UNK: 1014.
  Evaluacion sobre ventanas no solapadas del split completo, descartando solo la
  cola que no completa una ventana de 128. El test no se uso.
- Batch 16, secuencia 128, 4 bloques y 4 cabezas. R: ancho 192/rango 96 desde cero.
  M: denso ancho 96 durante 2000 pasos; SVD completa y ensanchado a 192/rango 96.
- AdamW LR 0.0003, WD 0.01; mismos lotes por indice de paso en cada pareja.
  Sin warmup inicial; M recibe el warmup postcirugia preespecificado de 200 pasos.
- Noise 0.001; limites de CE y KL 0.05 nats/token, fijados antes de entrenar;
  sonda separada del train y de la validacion. No hubo reintentos ni ajustes.
- Orden R/M para 17, M/R para 29 y R/M para 43. Las seis corridas duraron en total
  549.963 s medidos, aproximadamente 9.17 min, mas carga/calentamiento comun.

## Alcance y limites

El resultado respalda un ahorro en ESTA escala, datos, hardware y configuracion.
No demuestra superioridad universal ni significacion poblacional con tres
semillas. No se barrieron tasas de aprendizaje ni inicializaciones para encontrar
la mejor referencia posible. Tampoco aísla el efecto de cada cambio de seguridad:
compara la ruta completa de crecimiento contra entrenar el modelo final desde cero.

Las curvas siguen mejorando: en R, 0.084-0.114 BPC por cada 500 pasos al final; en
M, 0.048-0.061. Por tanto, no se declara convergencia ni mejor calidad asintotica.
No se compara el final de M al paso 6000 contra R al 4000 como si costaran lo mismo.
No se excluyeron semillas; las tres referencias pasaron el control preespecificado.

Esta escala reducida NO reproduce exactamente S1/T8-T12. No se reemplazan sus
numeros ni se renueva el sello firmado del proyecto.

## Reproduccion y trazabilidad

- [Protocolo previo](PROTOCOL.md).
- [Datos completos y hashes](results.json).
- [Benchmark ejecutado](benchmark.py).
- [Auditoria numerica reproducible](analyze.py).

Desde D:/mztrain: `.venv/Scripts/python.exe .tmp/elasticshape-evidence-20260907/analyze.py`.
La auditoria recalcula los cruces, verifica igualdad de estructuras y comprueba
los SHA256 del protocolo, harness y tres archivos fuente contra los registrados.
Todos coinciden. El benchmark protege results.json contra sobreescritura; para
una repeticion, usar otra carpeta y conservar esta corrida.

SHA256 del protocolo: E27D12F38A2A179D994D0DF2EE7AE03FBCE22F04D78CCDE41B1575BB5F9CEEA8.
