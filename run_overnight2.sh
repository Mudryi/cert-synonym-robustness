#!/bin/bash
# Night 2 (2026-10-13): validity-aware worst-case search (S15) and its evaluation.
# Detach: tmux new -d -s night2 ./run_overnight2.sh ; progress: experiments/out/logs/night2.log
cd "$(dirname "$0")"
UV=${UV:-uv}
PY="$UV run --frozen python -m"
T2=experiments/out/tier2
LOG=experiments/out/logs/night2.log
export EXP_LLM_BUDGET_USD=9.5   # cumulative OpenAI cap (spent so far ~$2.73)
stage() {  # stage <name> <timeout minutes> <command...>
  local name=$1 mins=$2; shift 2
  echo "[$(date +%T)] START $name" >> $LOG
  timeout $((mins * 60)) "$@" > experiments/out/logs/night2_$name.log 2>&1
  echo "[$(date +%T)] END $name rc=$?" >> $LOG
}
stage worstv_search 240 $PY experiments.s15_worstcase_valid --out $T2 --name variants_worstv
stage worstv_judge 60 $PY experiments.s3_judge --out $T2 --name variants_worstv
stage worstv_retrieve 60 $PY experiments.s4_retrieve --out $T2 --name variants_worstv --models bm25,e5_base,bge_m3,hybrid,e5_large,m3_all
stage worstv_s14 30 $PY experiments.s14_worstcase_verified --out $T2 --name variants_worstv
for s in 13 14 15 16 17; do
  stage ft_s$s 60 $PY experiments.s9_finetune --out $T2 --models e5_base,bge_m3 --folds 5 --epochs 2 --seed $s \
      --eval-names variants,variants_worst,variants_worstv
done
stage worstv_s14_ft 30 $PY experiments.s14_worstcase_verified --out $T2 --name variants_worstv
stage results 10 $PY experiments.s11_results
echo "[$(date +%T)] ALL DONE" >> $LOG
