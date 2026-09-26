# Comparacion local preespecificada de ElasticShape

Fecha: 2026-09-07. Este protocolo se escribe antes de entrenar los brazos.
Pregunta: con la implementacion de seguridad actual, puede el crecimiento
alcanzar la calidad de validacion de entrenar el MISMO modelo final desde cero
en menos tiempo real? No se presupone un resultado positivo.

## Condiciones fijadas

- Texto real: WikiText-2 raw, copia Arrow local. Tokenizacion por caracteres,
  vocabulario del train con un token UNK para caracteres desconocidos.
- Primeros 8193 caracteres del train reservados para la sonda de la cirugia;
  excluidos del entrenamiento de ambos brazos. La validacion es su split
  oficial COMPLETO, ventanas de 128 sin solapamiento. No se usa el test.
- GPU local RTX 4060, parametros/forward fp32, TF32 desactivado, sin compilacion.
  AdamW LR=0.0003, WD=0.01, betas=(0.9,0.999), eps=1e-8, foreach=True.
  Batch 16, secuencia 128. Mismos lotes por indice de paso en cada pareja.
- Tres semillas nuevas: 17,29,43. Orden de brazos R/M, M/R, R/M respectivamente.
- R: GPT factorizado, ancho192, rango96 en las cuatro proyecciones, 4 bloques,
  4 cabezas, 4000 pasos. Sin warmup inicial.
- M: GPT denso ancho96, mismos bloques/cabezas, 2000 pasos; convertir por SVD
  completa y ensanchar a192; hasta4000 pasos mas. Mismo modelo final que R.
- Crecimiento unico: noise_scale=0.001; guardas fijadas max_loss_increase=0.05
  nats/token y max_kl=0.05 nats/token sobre la sonda separada. No se ajustan
  despues de ver resultados. Rechazo cuenta como fallo de M; no se reintenta.
  Warmup solo tras crecimiento:200 pasos, suelo0.1, como politica del metodo.
- Evaluar cada250 pasos, antes y despues del crecimiento. Ningun brazo recibe
  ajustes de hiperparametros ni seleccion de semillas basada en su resultado.

## Medicion y decision

Reloj primario: tiempo de pared acumulado desde construir el modelo de cada
brazo. Incluye inicializacion/SVD, mover modelo a GPU, entrenamiento, validacion,
cirugia, sondas, migracion y sincronizaciones. Excluye carga COMUN del corpus y
un calentamiento COMUN previo, sin conservar los pesos de calentamiento.
Registrar tambien entrenamiento, evaluacion, setup y cirugia por separado.

Para cada semilla, Q es el BPC de R al paso4000. T_R es el primer punto medido
de R con BPC<=Q. T_M es el primer punto de M, YA con la arquitectura final,
con BPC<=Q. No se interpolan cruces ni se cuenta la calidad del modelo pequeno.
Una semilla que no alcanza Q queda como no alcanzada, no se elimina del promedio.

Evidencia favorable LOCAL de ahorro:3/3 cirugias aceptadas,3/3 M alcanzan Q,
todos los ratios T_M/T_R<1 y media geometrica de ratios<=0.90. Si falla alguna
condicion, no se declara demostrado el ahorro bajo este protocolo. Reportar
tambien el rango de tiempos delimitado por la evaluacion anterior para revelar
la incertidumbre de la cadencia250. No se reclama significacion estadistica
poblacional con solo tres semillas ni superioridad sobre toda configuracion.

M y R deben tener igual estructura de parametros al final. Un R que no mejora
al menos1BPC desde el inicio, no es finito, o termina>3.8BPC se etiqueta referencia
insuficiente: no se celebra ventaja contra ese R y no se excluye en silencio.

Todos los runs completos se conservan, incluso desfavorables. Limite operativo
global:45 minutos de ejecucion; si se agota, resultado incompleto, sin cambiar
el protocolo. Un error de infraestructura permite reparar el harness y repetir
antes de iniciar las corridas validas, sin retocar el algoritmo de produccion.

No es una comparacion de calidad asintotica: comparar puntos de curvas aun
descendentes solo mide velocidad hasta un objetivo. Tampoco replica la escala
S1 de T8-T12: esta es una escala reducida nueva. No se modifica el sello firmado.
