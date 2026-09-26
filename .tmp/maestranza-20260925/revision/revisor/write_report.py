import json

R = "D:/mztrain/.tmp/maestranza-20260925/revision/reporte_REVISOR.json"
rep = {
    "schema_version": 1,
    "veredicto": "aprobar_con_cambios",
    "independencia": "contextual",
    "dimensiones": [
        {
            "dimension": "correccion",
            "juicio": "hallazgos",
            "evidencia": (
                "Cada arreglo corrige la causa raiz del hallazgo prescrito: pop de _mzshape_base_lr al terminar "
                "(elastic_shape.py:722-728); rama de overrides sin cirugia (667-673) correcta en 7 eventos x 3 kwargs "
                "(revisor/probe1.py: factorize con n_conv>0, con o sin new_d==d, no duplica el rebuild; redundante, "
                "new_d==d y evento vacio migran con estado 20/20 y el opt original intacto); el guard PER-LOG-005 "
                "(437-438) deja pasar recs=[] sin pendiente; SDPA exacta con noise=0, bias False/True, fp64 "
                "(2.8e-16/4.4e-16, dims nuevas 0) y coherente con _rescale_q_state (q_rows = omap[:d_old]); "
                "eps'=eps*d/d' es exacto: LN([x,0]) con gamma*sqrt(d/d') = gamma*x/sqrt(sigma^2+eps'*d'/d). "
                "Residuos menores: un warmup abandonado conserva la base caduca; falso positivo de NoiseError con "
                "ns<=1e-22; la whitelist de dtypes de probe_targets es demasiado ancha."
            ),
        },
        {
            "dimension": "regresiones",
            "juicio": "hallazgos",
            "evidencia": (
                "96/96 tests verdes (test_shape_ops, test_elastic_shape sin tocar, test_elastic_shape_safety); "
                "G4-B test_warmup_base_survives_optimizer_rebuild y test_lr_warmup_stacking_preserves_base verdes; "
                "apply_event factorize+widen+deepen con sonda bajo warnings 'error' y bajo autocast bf16 acepta sin "
                "aviso. Regresiones menores nuevas: el eps reescalado no viaja en el state_dict (un GPT reconstruido "
                "desde config difiere 1.8e-5 rel en logits; antes era exacto); noise_scale np.float32 o tensor 0-d, "
                "antes aceptados, ahora dan ValueError en widen_gpt/apply_event."
            ),
        },
        {
            "dimension": "errores",
            "juicio": "hallazgos",
            "evidencia": (
                "Los errores son ahora ValueError/NoiseError explicitos (anclas). probe_targets int32/int16 pasan la "
                "validacion nueva, pero F.cross_entropy los rechaza dentro de la transaccion (RuntimeError tras la "
                "cirugia, rollback OK): identico al pre-lote, validacion incompleta."
            ),
        },
        {
            "dimension": "seguridad",
            "juicio": "no_aplica",
            "evidencia": "Sin sinks, red, deserializacion ni entrada no confiable en el diff.",
        },
        {
            "dimension": "concurrencia",
            "juicio": "no_aplica",
            "evidencia": (
                "Sin hilos ni estado compartido; el guard de forward no usa hooks (dict compartido que la "
                "transaccion no restaura), acorde al contrato."
            ),
        },
        {
            "dimension": "eficiencia",
            "juicio": "ok",
            "evidencia": (
                "Validaciones O(n) sobre mapas/targets; una sincronizacion CUDA extra por .all()/float(norm) "
                "por evento/capa, despreciable frente a la cirugia."
            ),
        },
        {
            "dimension": "mantenibilidad",
            "juicio": "hallazgos",
            "evidencia": (
                "SAFETY.md (a) sobreafirma (solo libera la base si el warmup se avanza hasta done) y no documenta el "
                "eps reescalado, el rechazo por colapso del ruido ni que el override sin cirugia devuelve un optimizer "
                "NUEVO. shape_ops.py paso de LF a CRLF en el lote (se alinea con HEAD: el diff vs HEAD queda limpio); "
                "elastic_shape.py sigue en LF con indice CRLF (preexistente). Ruff E9,F,B,PLE limpio; las lineas "
                ">120 son preexistentes."
            ),
        },
        {
            "dimension": "oraculo",
            "juicio": "ok",
            "evidencia": (
                "Las 22 anclas nuevas FALLAN contra el arbol pre-lote (PYTHONPATH=scratchpad/lab/src: 22 failed) y "
                "pasan contra el actual; hay controles negativos; ningun test/assert previo se borro ni se relajo "
                "(los diffs de tests solo anaden; tests/test_elastic_shape.py mtime 2026-08-14, identico al lab)."
            ),
        },
    ],
    "objecion_mas_fuerte": {
        "descripcion": (
            "PER-LOG-001 solo queda arreglado si el warmup se avanza hasta done. Un LrWarmup abandonado o sin "
            "terminar conserva la base (tambien un checkpoint reanudado a mitad de warmup: _mzshape_base_lr vive en "
            "param_groups y se serializa en opt.state_dict(), pero LrWarmup.t no). Si luego el LR baja por fuera, "
            "el siguiente warmup tras un growth rampea hasta el pico caduco. Medido: warmup de 10 pasos abandonado "
            "tras 3, LR bajado a 1e-5, apply_event + LrWarmup(2): LR final 1e-3 (100x). SAFETY (a) dice que "
            "'respeta cambios externos del LR entre growths' sin esa condicion."
        ),
        "valoracion": (
            "No bloquea: es exactamente la semantica G4-B que el peritaje y la orden mandan preservar (un warmup sin "
            "terminar debe conservar su objetivo, y el codigo no puede distinguir 'sin terminar' de 'abandonado'). La "
            "precondicion del hallazgo original (warmup terminado + cambio externo) queda cubierta y anclada, y la "
            "ruta documentada (apply_event -> LrWarmup avanzado hasta done) es correcta. Hay que precisar SAFETY (a) "
            "y mencionar la persistencia en el state_dict; opcionalmente, exponer un reset explicito de la base."
        ),
    },
    "hallazgos": [
        {
            "severidad": "menor",
            "descripcion": (
                "Residuo de PER-LOG-001: un warmup sin terminar o abandonado (o reanudado de checkpoint con la clave "
                "en param_groups) mantiene la base caduca; tras un cambio externo de LR, el siguiente warmup termina "
                "100x por encima (1e-5 -> 1e-3, revisor/probe1.py). SAFETY (a) sobreafirma."
            ),
            "evidencia": "src/mztrain/elastic_shape.py:706-708,722-728; docs/ELASTICSHAPE-SAFETY.md:95-97",
        },
        {
            "severidad": "menor",
            "descripcion": (
                "Falso positivo de PER-MAT-002: float(noise.norm()) se calcula en el dtype de U (fp32/bf16 acumulan "
                "en fp32); con elementos < ~1e-23 los cuadrados subdesbordan y la norma da 0 aunque los 64/64 "
                "elementos sean no nulos -> NoiseError (y rechazo en apply_event) con noise_scale<=1e-22. Antes del "
                "lote ese ruido no nulo se aceptaba. Irrelevante en la practica, pero el criterio no mide 'colapso a "
                "cero': usar la norma en acc (fp64) o count_nonzero. El colapso parcial en fp16 conserva la "
                "calibracion (ratio 0.99-1.04)."
            ),
            "evidencia": "src/mztrain/shape_ops.py:107-109 (revisor/probe2.py, seccion (vi))",
        },
        {
            "severidad": "menor",
            "descripcion": (
                "La whitelist de dtypes de PER-LOG-004 incluye int32/int16/int8, que F.cross_entropy rechaza "
                "('expected scalar type Long'): ese error se sigue descubriendo DESPUES de la cirugia (el rollback "
                "es correcto), igual que antes. uint8 e int64 funcionan; -100 se acepta; los targets en otro device "
                "se validan bien (CPU<->CUDA probado)."
            ),
            "evidencia": "src/mztrain/elastic_shape.py:601-607",
        },
        {
            "severidad": "menor",
            "descripcion": (
                "Guard de PER-LOG-006: stacklevel=2 atribuye el RuntimeWarning a torch/nn/modules/module.py:1790 (no "
                "al codigo del usuario) y, con el filtro 'default', se muestra una sola vez por proceso: 2 episodios "
                "x 3 forwards pendientes -> 1 aviso visible. No se dispara dentro de apply_event (verificado con "
                "simplefilter('error'), incluidos fact+widen+deepen y autocast bf16)."
            ),
            "evidencia": "src/mztrain/elastic_shape.py:130-137",
        },
        {
            "severidad": "menor",
            "descripcion": (
                "PER-MAT-004: un widen con variance_compensation=True deja LN.eps = eps*d0/d (2.5e-6 tras 8->16->32). "
                "eps no esta en el state_dict, asi que reconstruir GPT(vocab,seq,d,...) + load_state_dict da "
                "eps=1e-5 y logits distintos (1.8e-5 rel); antes era exacto. No esta documentado en SAFETY/SPEC."
            ),
            "evidencia": "src/mztrain/shape_ops.py:255-256; src/mztrain/elastic_shape.py:276,285,351",
        },
        {
            "severidad": "menor",
            "descripcion": (
                "PER-API-010: el chequeo isinstance(noise_scale,(int,float)) en widen_gpt/apply_event rechaza "
                "np.float32 y tensores 0-d, que antes funcionaban y que las primitivas (_validate_noise) siguen "
                "aceptando: API inconsistente. _require_int acepta ints de numpy (numbers.Integral) y rechaza "
                "tensores 0-d (widen_factorized_linear(new_in=tensor) antes funcionaba). No afecta a ningun uso "
                "interno ni a los scripts de evidencia."
            ),
            "evidencia": "src/mztrain/elastic_shape.py:238-244,608-614; src/mztrain/shape_ops.py:34-37,77-79",
        },
        {
            "severidad": "menor",
            "descripcion": (
                "Documentacion incompleta del lote: SAFETY no menciona el rechazo por ruido que colapsa a cero "
                "(NoiseError -> accepted=False), el reescalado de eps ni que un evento sin cirugia con "
                "lr/weight_decay devuelve ahora un optimizer NUEVO (antes era el mismo objeto). La frase 'la "
                "posterior correccion de Q ... puede cambiar la magnitud' quedo parcialmente obsoleta tras PER-MAT-001."
            ),
            "evidencia": "docs/ELASTICSHAPE-SAFETY.md:83-104,126-130,145-148",
        },
    ],
    "verificaciones_ejecutadas": [
        {
            "comando": (
                'D:/mztrain/.venv/Scripts/python.exe -m pytest -q -p no:cacheprovider -o addopts="" '
                "tests/test_shape_ops.py tests/test_elastic_shape.py tests/test_elastic_shape_safety.py -W default"
            ),
            "resultado": "pass",
        },
        {
            "comando": (
                'PYTHONPATH=<scratchpad/lab/src pre-lote> pytest ... -k "ancla or migracion_pendiente" '
                "(discriminacion de anclas: se esperaba rojo, 22 failed)"
            ),
            "resultado": "pass",
        },
        {
            "comando": (
                "python revisor/probe1.py (warmup abandonado, G4-B, 21 combinaciones de override, "
                "migrate_optimizer([]) sin pendiente)"
            ),
            "resultado": "pass",
        },
        {
            "comando": (
                "python revisor/probe2.py post|pre (tipos int/float, dtypes de probe_targets, CPU<->CUDA, "
                "NoiseError por dtype y ns)"
            ),
            "resultado": "pass",
        },
        {
            "comando": (
                "python revisor/probe3.py (dedupe/atribucion del aviso, apply_event bajo warnings error y autocast "
                "bf16, exactitud SDPA con bias True/False en fp64, eps y reconstruccion desde config)"
            ),
            "resultado": "pass",
        },
        {"comando": "python -m ruff check --no-cache --select E9,F,B,PLE (4 archivos del lote)", "resultado": "pass"},
        {
            "comando": "diff/cmp del arbol actual vs scratchpad/lab (pre-lote); git status / diff --stat; mtimes",
            "resultado": "pass",
        },
    ],
}
with open(R, "w", encoding="utf-8") as fh:
    json.dump(rep, fh, ensure_ascii=False, indent=1)
with open(R, encoding="utf-8") as fh:
    json.load(fh)
print("ok")
