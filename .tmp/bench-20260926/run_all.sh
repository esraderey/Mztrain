#!/usr/bin/env bash
# Corredor secuencial de benchmarks tras las reparaciones (2026-09-26). Logs en esta carpeta.
set -u
cd /d/mztrain
PY=D:/mztrain/.venv/Scripts/python.exe
OUT=/d/mztrain/.tmp/bench-20260926
export PYTHONIOENCODING=utf-8
run() { local name=$1; shift; echo "### $name :: $(date -u +%H:%M:%S) :: $*" | tee -a "$OUT/STATUS.log"; local t0=$SECONDS; "$@" > "$OUT/$name.log" 2>&1; local rc=$?; echo "### $name :: exit=$rc :: $((SECONDS-t0))s" | tee -a "$OUT/STATUS.log"; }
: > "$OUT/STATUS.log"
run bench_run        $PY -m bench.run --label post_peritaje_20260926
run elastic_bench    $PY -m bench.elastic_bench
run elastic_guard    $PY -m bench.elastic_bench_guard
run elastic_real     $PY -m bench.elastic_bench_real
run governor_gpu     $PY -m bench.governor_gpu_check
ARROW="C:/Users/Raul/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3"
if [ -d "$ARROW" ] && $PY -c "import datasets" 2>/dev/null; then
  cd /d/mztrain/.tmp/elasticshape-evidence-20260907 && run elasticshape_bench $PY benchmark.py
else
  echo "### elasticshape_bench :: OMITIDO (sin Arrow local o sin 'datasets')" | tee -a "$OUT/STATUS.log"
fi
echo "### FIN :: $(date -u +%H:%M:%S)" | tee -a "$OUT/STATUS.log"
