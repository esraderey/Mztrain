# PREREGISTRO — T11: ¿protege la fase densa o protege el warmup?

**Fecha:** 2026-08-22, ANTES de correr ningún run de T11.
**Precondición:** T10 midió 4/6 degeneraciones en la referencia desde cero contra 0/6 en el morph
(Fisher exacto unilateral p = 0.0303) y declaró explícitamente que **el mecanismo no está
identificado**. Este experimento existe para atacar esa laguna antes de que la propiedad se
cuente como propiedad del morph.

## El problema de confusión

El morph tiene **dos** cosas que la referencia no tiene:

1. una **fase densa** de 2000 pasos antes de entrar a la parametrización factorizada, y
2. una **rampa de LR** (200 pasos, suelo 0.1) inmediatamente después de la cirugía.

T10 no puede separarlas. Si la rampa es la que protege, entonces el hallazgo no es sobre
ElasticShape: sería «una rampa de LR estabiliza al factorizado a LR alto», que es una receta más
barata y sin cirugía, y habría que decirlo así.

## Diseño: completar el 2×2

Dos celdas nuevas, 6 semillas cada una, a LR 1.2e-3, S1, endpoint `F_192@384`, resto del
protocolo idéntico a T10 (mismo banco, mismas semillas 0-5, mismo umbral de degeneración
BPC > 3.20 o NaN).

| | sin rampa de LR | con rampa de LR |
|---|---|---|
| **Desde cero** | 4/6 degeneran *(T10, ya medido)* | **R_warm** ← nueva |
| **Morph** | **M_nowarm** ← nueva | 0/6 degeneran *(T10, ya medido)* |

- **R_warm:** `F_192@384` desde cero con `LrWarmup(200, suelo 0.1)` aplicada **al inicio** del
  entrenamiento. Es el mismo tratamiento de LR que recibe el morph, en el único punto donde tiene
  sentido para un modelo recién inicializado.
- **M_nowarm:** el morph de T10 exactamente igual, pero **sin** rampa tras la cirugía.

Las dos celdas viejas se reutilizan de `t10_results.json` tal cual, sin volver a correr: mismo
banco, mismas semillas, mismo protocolo, misma sesión de hardware. Se declara aquí que son datos
previos y no nuevos.

## Reglas de decisión (fijadas aquí)

Sean `p` las tasas de degeneración sobre 6 semillas. Ya conocidas: `p(R, sin) = 4/6`,
`p(M, con) = 0/6`.

- **EL MÉRITO ES DE LA FASE DENSA:** `p(M_nowarm) ≤ 1/6` **y** `p(R_warm) ≥ 3/6`. Es decir,
  quitarle la rampa al morph no lo rompe, y dársela a la referencia no la salva.
- **EL MÉRITO ES DE LA RAMPA:** `p(R_warm) ≤ 1/6` **y** `p(M_nowarm) ≥ 3/6`. La protección
  viaja con la rampa, no con la fase densa. En este desenlace, **el claim de estabilidad de T10
  se reescribe**: no es una propiedad de ElasticShape.
- **AMBOS CONTRIBUYEN:** `p(R_warm) ≤ 1/6` **y** `p(M_nowarm) ≤ 1/6`. Cada mecanismo basta por
  separado; la protección está sobredeterminada.
- **NINGUNO BASTA SOLO:** `p(R_warm) ≥ 3/6` **y** `p(M_nowarm) ≥ 3/6`. La protección exige la
  combinación, y hay interacción que este diseño no resuelve.
- **INCONCLUSO:** cualquier patrón con tasas intermedias (2/6) que no encaje en los anteriores.

Se reporta además, de forma descriptiva, el **modo** de cada fallo (meseta no rota frente a
colapso post-ruptura, los dos modos que T10 identificó al inspeccionar las curvas) y el `T_M` de
los morphs sin rampa, para saber si la rampa afectaba a la velocidad aunque no a la estabilidad.

## Predicción del autor

**Creo que el mérito es de la fase densa**, con `p(M_nowarm)` ∈ {0, 1/6} y `p(R_warm)` ∈
{3/6, 4/6}. Confianza **media**, y esta vez la predicción se apoya en evidencia y no en intuición:
las curvas de T10 muestran que la degeneración **no se decide en la ventana temprana** —todas las
referencias, sanas y degeneradas, siguen en ~3.7 en el paso 1000-2000, y las sanas rompen entre
el 2000 y el 3000—, mientras que una rampa de 200 pasos actúa mucho antes de ese punto.

**Mecanismo que propongo** (y que este experimento no prueba, solo hace más o menos plausible):
lo que mata a la referencia es no romper la meseta unigrama/bigrama dentro del presupuesto. El
morph **entra a la forma factorizada ya rota** —BPC ~3.03-3.09 tras la fase densa, muy por
delante de la meseta—, así que su fase factorizada nunca se enfrenta a la parte difícil. Si esto
es correcto, la protección no viene de «suavizar el transitorio» sino de **saltarse la
bifurcación**, y debería sobrevivir a quitar la rampa.

Aviso sobre mi propio historial: en T10 predije mal las dos preguntas. Esta predicción está mejor
fundada que aquellas, lo cual no la hace cierta.

## Límites declarados

Una escala, una tarea, un LR, 6 semillas por celda, una sola configuración de rampa (200 pasos,
suelo 0.1 — la de todo el arco, sin barrido). El diseño distingue **cuál de los dos factores
lleva el crédito**; no mide la magnitud del efecto de ninguno, ni descarta un tercer factor no
contemplado. Reutiliza dos celdas de T10 en vez de volver a correrlas, con el ahorro y el riesgo
que eso implica (mismo hardware y misma sesión, pero no la misma ejecución).

## Presupuesto

R_warm 6 × ~280 s + M_nowarm 6 × ~370 s ≈ **65 min**. Si el reloj llega a 1.5 h se reporta con
las semillas completadas (mínimo 4 por celda para emitir veredicto). Parada por tiempo, nunca por
resultado.

**Artefacto:** `t11_results.json` (volcado atómico incremental, reanudable).
