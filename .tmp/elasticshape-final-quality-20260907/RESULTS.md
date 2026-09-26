# Calidad de un modelo pequeno: resultados y limite de la conclusion

Fecha: 2026-09-07. Se completaron seis entrenamientos sobre WikiText-2, con tres
semillas y la misma arquitectura final de **139,520 parametros**.

**A 40,000 pasos, ElasticShape obtuvo una perdida de prueba ligeramente mayor
que entrenar desde cero, en las tres semillas. No se confirmo calidad final
convergida:** todas las corridas terminaron por el tope de pasos, sin cumplir
la regla completa de meseta/sobreajuste. El veredicto preespecificado de calidad
final operativa es, por tanto, inconcluso; no se cambia la regla despues de ver
el resultado.

## Calidad medida sobre texto no visto

El test oficial se abrio solo despues de terminar los seis entrenamientos y
fijar todos los checkpoints con validacion. Se evaluo una vez cada checkpoint,
sobre 1,285,504 caracteres objetivo en ventanas no solapadas de 128.
En los seis casos, el mejor checkpoint de validacion resulto ser el del paso
40,000. No se uso el test para elegir modelos, LR ni puntos de parada.

| Semilla | Test BPC desde cero | Test BPC ElasticShape | Delta M-R | Acierto desde cero | Acierto ElasticShape |
|---|---:|---:|---:|---:|---:|
| 17 | 2.596676 | 2.624658 | +0.027982 | 48.176% | 47.411% |
| 29 | 2.568720 | 2.639360 | +0.070640 | 48.542% | 47.165% |
| 43 | 2.595957 | 2.621288 | +0.025331 | 47.832% | 47.252% |
| Media | **2.587118** | **2.628435** | **+0.041318** | **48.183%** | **47.276%** |

BPC es entropia cruzada en bits por caracter: menor es mejor. El acierto es
prediccion del siguiente caracter, no respuestas correctas de un chatbot.
La perplejidad media POR CARACTER es 6.0092 desde cero frente a 6.1836 con
ElasticShape: aproximadamente 2.9% mayor. La diferencia de acierto es de
0.91 puntos porcentuales a favor de entrenar desde cero.

Las tres diferencias de test exceden la tolerancia descriptiva de 0.02 BPC
fijada en el protocolo. Eso descarta describir estas mediciones como identicas
bajo dicha tolerancia, pero no constituye una prueba estadistica poblacional
ni permite extrapolar a calidad despues de converger.

## Por que no es todavia calidad final convergida

La regla exigia, ademas de una pendiente pequena, agotar la reduccion de LR,
entrenar al menos 3,000 pasos a LR minima y acumular al menos 6,000 pasos sin
mejora significativa. Ningun brazo cumplio todas las condiciones antes del tope.

| Semilla | Validacion R al tope | Validacion M al tope | Mejora reciente R (BPC/500 pasos) | Mejora reciente M |
|---|---:|---:|---:|---:|
| 17 | 2.570444 | 2.603851 | 0.004023 | 0.003138 |
| 29 | 2.544397 | 2.615939 | 0.005372 | 0.001410 |
| 43 | 2.574577 | 2.602385 | 0.005287 | 0.003409 |

Las pendientes se ajustan a los ultimos cinco puntos regulares de validacion.
M/29 activo su primer descenso de LR de 0.0003 a 0.00015 justo al paso 40,000;
no llego a entrenar con esa LR menor. Los otros cinco brazos mantuvieron la LR
inicial. En particular, una pendiente pequena por si sola no satisface la regla
preespecificada de parada ni demuestra haber agotado la mejora posible.

El patron observado fue una ventaja temprana de ElasticShape que se perdio
despues de prolongar el entrenamiento. Por ejemplo, en la semilla 17, al paso
18,000 M tenia 2.839 BPC de validacion y R 2.966; al paso 40,000, M tenia 2.604
y R 2.570. No debe confundirse velocidad inicial con mejor calidad al final
de un presupuesto largo.

## Configuracion y controles

- R: GPT factorizado, ancho 64, rango 32, 2 bloques y 4 cabezas, desde cero.
- M: GPT denso ancho 32 y 61,440 parametros por 2,000 pasos; conversion por
  SVD completa y ensanchado a la MISMA arquitectura final que R.
- Batch 16, contexto 128; AdamW LR inicial 0.0003, WD 0.01, foreach=True.
  Warmup postcirugia de 200 pasos solo en M. La misma politica adaptativa de
  LR por meseta para ambos brazos, activada desde el paso 6,000.
- RTX 4060, fp32, TF32 desactivado. Mismos lotes por indice de paso en cada
  pareja. Semillas 17,29,43; orden R/M, M/R, R/M.
- Noise 0.001; guardas de CE y KL en 0.05 nats/token sobre una sonda separada
  de train y validacion. Las tres cirugias fueron aceptadas, sin reintentos.
- Vocabulario de train+UNK: 1014. Entrenamiento: 10,884,797 caracteres despues
  de reservar 8193 para sondas. Validacion: split oficial completo en ventanas.
- Seleccion del checkpoint por menor BPC de validacion con arquitectura final.
  El test no intervino en entrenamiento, seleccion ni decisiones de protocolo.
- No se excluyo ninguna semilla, no hubo fallos numericos y las estructuras
  finales coinciden nombre por nombre y forma por forma. No hubo cambios de
  codigo de produccion durante esta prueba.

## Interpretacion practica

En este modelo de aproximadamente 140 mil parametros, **no hay evidencia de
que ElasticShape mejore la calidad al terminar 40,000 pasos**: el control desde
cero salio ligeramente mejor en validacion y test. Tampoco se puede afirmar
que esa diferencia sea definitiva: no se completo el criterio de convergencia.

Esto no invalida automaticamente el ahorro de tiempo del experimento anterior:
su modelo final tenia 1.4 millones de parametros y su objetivo era tiempo hasta
una calidad concreta. Cambian tanto la escala como la pregunta. No se atribuye
este resultado a un defecto matematico especifico sin una ablacion adicional.

Para estudiar calidad tras estabilizacion haria falta una extension separada
desde los estados guardados, permitiendo completar el descenso de LR y la
parada fijada. Esa extension no debe ajustar sus decisiones al test ya observado;
una confirmacion independiente de ajustes requeriria datos de prueba nuevos.

## Archivos y verificacion

- [Protocolo escrito antes de entrenar](PROTOCOL.md).
- [Resultados completos, curvas y hashes](results.json).
- [Benchmark ejecutado](benchmark.py).
- [Auditoria reproducible de seleccion y resultados](analyze.py).
- `best_R_17.pt`, `best_M_17.pt`, etc.: pesos seleccionados con validacion.
- `final_R_17.pt`, `final_M_17.pt`, etc.: pesos, optimizador, scheduler, RNG
  y lotes para una extension posterior. No se sobrescribieron corridas previas.

Desde D:/mztrain:
`.venv/Scripts/python.exe .tmp/elasticshape-final-quality-20260907/analyze.py`.
La auditoria verifica hashes de protocolo, harness, dependencia de evaluacion,
codigo de produccion y checkpoints; recalcula la seleccion por validacion y
las medias de test. Todos los controles pasaron.
