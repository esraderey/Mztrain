# Calidad final operativa de un modelo pequeno

Escrito el 2026-09-07 antes de entrenar. Este experimento nuevo no modifica las
corridas anteriores de velocidad. Pregunta: tras dejar de mejorar de forma
sostenida en validacion, se conserva la calidad de prediccion con ElasticShape?

## Configuracion fija

- WikiText-2 raw local, caracteres, vocabulario de train+UNK. Mismos splits y
  metrica de evaluacion completa que el banco anterior: primeros 8193 caracteres
  de train reservados para sondas, validacion oficial completa. El test oficial
  solo se carga/evalua DESPUES de terminar los seis entrenamientos y seleccionar
  sus checkpoints con validacion; nunca se usa para LR, parada o seleccion.
- RTX4060; fp32, TF32 off; batch16, secuencia128. AdamW: LR inicial0.0003,
  weight_decay0.01, betas(0.9,0.999), eps1e-8, foreach=True.
- Modelo final: ancho64, rango32, 2 bloques, 4 cabezas, embeddings atados.
  Esperado:139,520 parametros con vocabulario1014.
- R: modelo final factorizado desde cero. M: denso ancho32 por2000pasos;
  convertir por SVD completa y ensanchar a64. Noise0.001, max_loss_increase0.05
  y max_kl0.05 nats/token en sonda separada. Rechazo no se reintenta.
  M recibe warmup postcirugia200pasos, floor0.1. Sin warmup inicial en ambos.
- Semillas17,29,43; orden R/M, M/R, R/M. Mismos lotes por indice de paso en
  cada pareja. No se excluyen semillas ni se barren hiperparametros.

## Regla de entrenamiento y de parada

Evaluar validacion cada1000pasos y antes/despues del crecimiento. Solo se
seleccionan checkpoints con la arquitectura FINAL. Guardar el menor BPC de
validacion observado, aunque luego empeore. La seleccion no tiene acceso al test.

Desde el paso6000, ambos brazos usan ReduceLROnPlateau: factor0.5, patience2,
threshold0.005 BPC absoluto, min_lr0.00003. La politica es igual; los momentos
exactos de descenso dependen de la validacion de cada entrenamiento.

Parar por estancamiento solo si se cumplen TODAS:

1. Al menos12000pasos de entrenamiento.
2. LR minima y al menos3000pasos entrenados a esa LR.
3. Al menos6000pasos sin mejora acumulada mayor que0.005 BPC frente a un ancla
   que se actualiza al obtener una mejora significativa.
4. Meseta: pendiente absoluta ajustada a los ultimos5puntos de validacion menor
   que0.005 BPC por500pasos; o sobreajuste: perdida actual al menos0.02 BPC peor
   que el mejor checkpoint, sin mejora significativa segun la condicion3.

Limite:40000pasos por brazo y45minutos globales. Si se llega al limite sin
cumplir la parada, marcar `budget_cap`, no declarar convergencia. Guardar el
mejor checkpoint y el estado final para poder hacer una extension separada.
Una meseta de validacion es convergencia PRACTICA bajo esta politica, no prueba
de optimo global ni de convergencia matematica del optimizador.

## Evaluacion final y decision

Evaluar una sola vez cada checkpoint seleccionado sobre TODO el test oficial
en ventanas no solapadas de128. Registrar BPC, perplejidad por caracter(2**BPC)
y exactitud del siguiente caracter. No es perplejidad por palabra ni una prueba
de capacidades de chatbot o razonamiento.

Delta principal: test_BPC(M)-test_BPC(R), por semilla y media. Menor es mejor.
Solo emitir comparacion de calidad final operativa si los6brazos terminan por
meseta/sobreajuste, los3crecimientos se aceptan y las estructuras finales coinciden.

- Mejor M de forma consistente en esta muestra: delta negativo en3/3 y
  delta medio<=-0.02BPC.
- Peor M de forma consistente: delta positivo en3/3 y delta medio>=0.02BPC.
- Calidad similar observada: abs(delta)<=0.02BPC en las3parejas. Esto es una
  tolerancia descriptiva, NO una prueba estadistica formal de equivalencia.
- Resto: mixto/inconcluso. Con limites de presupuesto, describir lo medido
  sin llamarlo calidad final convergida. No se retocan limites para forzar veredicto.

Tres semillas y una configuracion no permiten generalizar a todo modelo/dato/LR.
No se modifica codigo de produccion, el sello firmado ni resultados anteriores.
