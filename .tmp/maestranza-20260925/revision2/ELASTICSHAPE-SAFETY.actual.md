# ElasticShape: cirugia reversible y estado AdamW

Revision de desarrollo: 2026-09-07. El cambio aborda la atomicidad de la cirugia,
la magnitud funcional del ruido y la conservacion del estado del optimizador.
No sustituye la evidencia experimental de la version 1.3.2 original.

## Aplicar y aceptar un crecimiento

Usar `apply_event` en un limite entre pasos de entrenamiento, sin un backward o
una acumulacion de gradientes pendientes. El modelo soportado es el GPT de
referencia de `mztrain.elastic_shape`, con sus capas lineales, embeddings sin
`max_norm` in-place y LayerNorm con peso y bias. No es un adaptador de DDP,
modelos arbitrarios, hooks que modifiquen pesos ni optimizadores custom.

```python
from mztrain import GrowthEvent, LrWarmup, apply_event

evento = GrowthEvent(step=2000, factorize=True, new_d=384)
opt, reporte = apply_event(
    model,
    opt,
    evento,
    probe_x=probe_tokens,
    probe_targets=probe_next_tokens,
    max_loss_increase=0.02,
    max_kl=0.01,
)
if reporte["accepted"]:
    warmup = LrWarmup(opt, steps=200)
else:
    print(reporte["rejection_reason"])
    # El modelo y opt siguen siendo los anteriores al intento.
    # El caller decide si/como reprogramar el evento rechazado.
```

Los limites del ejemplo son ilustrativos, no umbrales validados para todas las
tareas. Fijarlos antes de operar a partir de un conjunto representativo de
calibracion. No usar el test final para ajustar los limites.

`max_loss_increase` limita el incremento absoluto de entropia cruzada media en
nats/token. Necesita `probe_x` y `probe_targets`, ambos de la misma forma; los
targets pueden usar el ignore_index=-100 habitual de cross_entropy. Un conjunto
sin targets validos produce una metrica no finita y se rechaza. `max_kl` limita
la media por token de KL(p_antes || p_despues), tambien en nats; necesita solo
`probe_x`. KL mide todos los tokens de la sonda, incluso los que el caller haya
ignorado para calcular la perdida. Se puede activar uno o ambos limites.

La sonda fija su precision independientemente del autocast del caller: lo
desactiva para parametros fp32/fp64 y usa autocast del dtype del embedding para
fp16/bf16, admitiendo S fp32 con factores bf16. Se calculan log-softmax, KL y
diferencias de loss en fp64 (fp32 en MPS), y se restauran los
modos training/eval individuales. Se reportan `logits_drift_rel`, `kl_div` y,
cuando hay targets, `loss_before`, `loss_after`, `loss_delta`. La deriva relativa
de logits no es por si sola una garantia de continuidad de perdida.

La API anterior sigue siendo valida: sin limites, una sonda solo informa, y sin
sonda no se evalua calidad. `guarded` indica si se configuraron limites. En todos
los casos se verifican parametros y estado del optimizer finitos tras la cirugia.
Si se proporciona una sonda, tambien se rechazan logits/metricas no finitos.

## Rollback y manejo de fallos

Al exceder un limite o encontrar un candidato no finito, el resultado contiene
`accepted=False`, `rolled_back=True`, `rejection_reason`, y el optimizer devuelto
es el ORIGINAL. Las metricas disponibles describen el candidato rechazado. Las
entradas previas de `optimizer` en el reporte describen acciones intentadas;
la ultima entrada indica el rollback.

Una excepcion de validacion/construccion/migracion se propaga despues de restaurar
el estado. Se preservan las identidades originales de modulos/parametros, sus
gradientes existentes, las opciones y buffers del optimizer original, modos y RNG
CPU/CUDA/generador explicito. No se copian todos los pesos para el rollback:
las primitivas construyen reemplazos y la transaccion conserva referencias a los
originales hasta aceptar. Esto mantiene ambos conjuntos en memoria durante el
intento; no elimina el pico transitorio de memoria del crecimiento.

Las primitivas `factorize_gpt`, `widen_gpt` y `deepen_gpt` tambien restauran su
topologia si fallan. Para optimizador y calidad de principio a fin usar
`apply_event`. El constructor y el forward personalizados que escriben en pesos
originales quedan fuera del contrato. Se exige un punto de entrenamiento sin
mutaciones concurrentes.

`widen_gpt` y `deepen_gpt` llamadas directamente (fuera de `apply_event`) dejan
el modelo en estado de "migracion pendiente" (`_mzshape_pending_recs` o
`_mzshape_new_param_sources`) hasta que `migrate_optimizer` migra el optimizer
y limpia ambos atributos. Mientras haya migracion pendiente, un `forward` con
gradiente habilitado emite `RuntimeWarning` (no falla): el optimizer anterior
ya no referencia esos parametros y seguir entrenando con el pierde momentum en
silencio. Inferencia y sondas bajo `torch.no_grad()` (incluida `rel_drift`,
funcion publica legado, y la sonda interna de `apply_event`) no avisan: el
aviso solo importa quien va a hacer backward/step con el optimizer viejo.
`apply_event` no se ve afectado por el aviso porque su sonda posterior a la
cirugia corre despues de `migrate_optimizer`. Ademas:

(a) `LrWarmup` guarda en cada grupo su base persistente (`_mzshape_base_lr`) y
el ultimo LR que escribio (`_mzshape_warmup_lr`); ambas claves se liberan al
llegar a `done`. Regla "la escritura externa gana": un warmup nuevo hereda la
base de uno sin terminar SOLO si el LR del grupo sigue siendo el que escribio
ese warmup (caso G4-B: reconstruccion del optimizer a mitad de warmup); si
alguien escribio el LR despues (scheduler, decaimiento manual, reanudacion desde
un checkpoint con otro LR), la base es ese LR vigente. Las claves viven en
`param_groups` y viajan con `state_dict`, asi que la regla vale tambien tras
`load_state_dict`. Coincidencia benigna: una escritura externa que deja
exactamente el mismo valor que escribio el warmup se trata como no tocada. Un
scheduler externo que escriba `lr` en cada paso compite con un warmup activo.
(b) `_require_adamw` exige exactamente `torch.optim.AdamW`, no subclases.
(c) `widen_embedding` con una tabla de norma cero produce columnas nuevas cero.
(d) La sonda en bf16 tiene un piso de KL de aproximadamente 3e-5 nats por la
cuantizacion de los logits.
(e) `dense_to_factorized` en bf16/fp16 deja U/V en ese dtype, conserva S en
fp32 y avisa con `RuntimeWarning`.

`ShapeSchedule.pop_due` consume los eventos al entregarlos, como antes: un
rechazo no los reinserta automaticamente. El caller debe decidir si vuelve a
intentarlo, con otro ruido o con un salto de ancho menor. Reducir ruido no
elimina la deriva de LayerNorm presente incluso con ruido cero.

## Nuevo contrato del ruido

Con W=U diag(S*gate) V, las filas nuevas delta_U se calibran para satisfacer,
salvo tolerancia del dtype:

```
||delta_U diag(S*gate) V||_F = noise_scale * ||U diag(S*gate) V||_F
```

Se calcula QR de (diag(S*gate)V)^T. La parte triangular permite medir las dos
normas mediante matrices de ancho rank: nunca se reconstruye W completa. Se
usa precision alta para reducir cancelaciones por reescalados de factores.
La magnitud queda calibrada en el espacio funcional; la direccion aleatoria
puede seguir dependiendo de la representacion. Un mapa efectivo nulo recibe
ruido nulo. Valores negativos/no finitos de `noise_scale` son invalidos.

Esto es una cota de norma Frobenius por mapa lineal. No garantiza error relativo
pequeno en cada entrada, ni en logits, ni en la red completa. La posterior
correccion de Q escala solo las filas q viejas, asi que el presupuesto de qkv se
cumple tal cual; LayerNorm si puede cambiar la magnitud del efecto; por ello la
sonda comprueba el candidato completo. Si el cast al dtype de U aplasta TODO el
ruido calibrado a cero (fp16 con factores minusculos), `widen_factorized_linear`
lanza `NoiseError` y `apply_event` lo convierte en rechazo (`accepted=False`).
Con `variance_compensation` el LayerNorm ensanchado tambien reescala `eps` a
`eps*d/d'` (cancela ese termino de forma exacta). `eps` no viaja en el
`state_dict` sino en la configuracion: `model.ln_eps` lo registra (y la
transaccion lo restaura en rollback), asi que para reconstruir un modelo
crecido se usa `GPT(..., ln_eps=model.ln_eps)` antes de `load_state_dict`.
El QR y las comprobaciones de calidad agregan coste al evento, que debe
incluirse en futuros benchmarks.

En embeddings se calibra la norma Frobenius de las columnas nuevas respecto a
la tabla original. El padding recibe ruido cero. Con ruido cero se conserva el
zero-padding anterior. Con ruido positivo cambian las inicializaciones respecto
a v1.3.2: no esperar curvas ni semillas equivalentes a las publicadas.

## Optimizador y precision

Se admite `torch.optim.AdamW` estandar. Se preservan sus grupos y opciones
(LR, weight decay, betas, eps, AMSGrad, foreach, fused, capturable, etc.), defaults
y metadatos de grupo, incluida la base de warmup. Los estados copiados no tienen
alias con los originales. AMSGrad migra tambien `max_exp_avg_sq`; la correccion
de Q escala este maximo y los momentos de U/bias correspondientes.

`lr=None` y `weight_decay=None` heredan valores de CADA grupo; un override
explicito afecta a todos, tambien cuando el evento no produce cirugia (factorize
redundante, `new_d == d`): en ese caso se devuelve un optimizer nuevo con el
estado copiado y el reporte lo indica. Los parametros de bloques nuevos heredan el grupo del
parametro homologo en el primer bloque. Si ese parametro estaba excluido del
optimizer, el nuevo tambien lo queda. `param_names` se actualiza al layout nuevo.

La conversion densa completa mantiene la politica historica de reiniciar los
momentos de todos los parametros; conserva los grupos/opciones. La migracion
entre parametrizaciones densa y factorizada no se presenta como exacta. Widen y
deepen transplantan los momentos existentes y dan estado inicial a parametros
enteramente nuevos. Dentro de un tensor ensanchado, las coordenadas nuevas
heredan el contador del tensor: no son equivalentes a un Adam recien iniciado.

La copia del estado no garantiza conservar la trayectoria de Adam. No se cambia
aqui el reescalado de momentos de gamma de LayerNorm, la politica de weight
decay ni el warmup; requieren experimentos separados. Los 3.16*LR documentados
para coordenadas nuevas son un limite del primer paso en un regimen concreto,
no de todo el transitorio ni una prueba de estabilidad.

La profundizacion conserva rango por capa, bias, dtype, modos y trainability
de la plantilla. El ensanchado conserva S y gates fp32 si U/V son bf16, y
`dense_to_factorized` conserva S en fp32 tambien en bf16/fp16; el forward de
`ZFactorizedLinear` admite ese dtype mixto con y sin autocast. La ruta
empirica recomendada sigue siendo parametros fp32 con autocast bf16 en train.

`reconstruct_weight` devuelve la W efectiva, U diag(S*wake_gate) V (la misma
funcion que el forward, en el dtype de U): la exportacion a denso es exacta
tambien durante un mini-warmup de ElasticRank con gates parciales.

## Verificacion y publicacion

Las regresiones nuevas estan en `tests/test_elastic_shape_safety.py`: presupuesto
funcional con reescalado de factores de 1e6, rango uno, estado AMSGrad, grupos,
exclusiones, bias, errores inyectados, rechazos por perdida/KL, no finitos,
restauracion de RNG y continuacion de entrenamiento. Incluyen CUDA con AdamW
capturable/fused y autocast bf16 cuando hay hardware disponible.

Verificacion local del 2026-09-07: 381 pruebas pasan, incluidas 34 regresiones
nuevas y las variantes CPU/CUDA de factores bf16 con S fp32. Ruff, Black y
`git diff --check` pasan para los cambios. El comando completo
`python -m pytest -q tests --disable-warnings --tb=short` termina con codigo 1
por cobertura global de 81.71%, inferior al umbral configurado de 85%; no por
pruebas fallidas. Cobertura de elastic_shape: 93%; shape_ops: 94%.

Los resultados T8-T14 describen la implementacion original. Para atribuir calidad
o ahorro de reloj a esta revision hay que repetir una comparacion controlada e
incluir el coste de las sondas. Tampoco se declara una nueva doble revision ciega.
El sello firmado anterior no cubre estas modificaciones: antes de publicar se
necesita un nuevo sello firmado por un titular autorizado; no se modifica la
lista de confianza ni se reemplaza el sello durante esta implementacion.
