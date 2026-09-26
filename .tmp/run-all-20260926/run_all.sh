#!/usr/bin/env bash
# "corre todo mztrain" — todo lo ejecutable del repo, en secuencia, con cwd en esta carpeta para que las
# salidas relativas (checkpoints, json, html) NO pisen los artefactos del repo. Logs: <nombre>.log.
set -u
W=/d/mztrain
OUT=/d/mztrain/.tmp/run-all-20260926
PY=D:/mztrain/.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
cd "$OUT"
run() { local name=$1; shift; echo "### $name :: inicio $(date -u +%H:%M:%S) :: $*" | tee -a STATUS.log; local t0=$SECONDS; "$@" > "$OUT/$name.log" 2>&1; local rc=$?; echo "### $name :: exit=$rc :: $((SECONDS-t0))s" | tee -a STATUS.log; }
: > STATUS.log
# --- Fase A: ejemplos, scripts utilitarios, generacion, smokes grandes
run example_basic            $PY "$W/examples/example_basic.py"
run example_memory_estim     $PY "$W/examples/example_memory_estimation.py"
run example_transformer      $PY "$W/examples/example_transformer.py"
run example_zcodebert        $PY "$W/examples/example_zcodebert.py"
run calc_gpu_params_equiv    $PY "$W/scripts/calc_gpu_params_equiv.py"
( cd "$W" && run seal_verify $PY scripts/seal.py verify )
run generate_game_all        $PY "$W/generate_game.py" --all --checkpoint "$W/zcodebert_htmlgames.pt" --seed 7
run train_zcoder_1b_smoke    $PY "$W/scripts/train_zcoder_1b.py" --corpus-root "$W/src" --corpus-root "$W/tests" --skip-save
( cd "$W" && run test_zcoder_410m_full $PY scripts/test_zcoder_410m_full.py )
# --- Fase B: entrenamientos largos (checkpoints via hardlink en esta carpeta; guardan aqui, no en el repo)
[ -e zcodebert_trained.pt ] || cmd //c mklink //H zcodebert_trained.pt "D:\\mztrain\\zcodebert_trained.pt" >/dev/null
run train_html_games_20min   $PY "$W/train_html_games.py"
run analysis_full_1h         $PY "$W/analysis_full.py"
echo "### FIN :: $(date -u +%H:%M:%S)" | tee -a STATUS.log
