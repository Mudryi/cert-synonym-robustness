#!/usr/bin/env bash
# Overnight runner: tiers 1–7, detached-safe (launch inside tmux), no prompts.
#   tmux new -d -s certua './run_overnight.sh'       # full run
#   SMOKE=1 ./run_overnight.sh                        # tiny end-to-end check
# Stop early:  touch experiments/out/STOP   (checked between stages)
# Progress:    experiments/out/STATUS.md, logs in experiments/out/logs/
set -u
cd "$(dirname "$0")"
UV="$HOME/.local/bin/uv"
PY="$UV run --frozen python -m"
TF_PY=../textfooler_ukr/textfooler_env/bin/python
BA_PY=../bert_attack_uk/bert_attack_env/bin/python
OUT=experiments/out
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error
export EXP_LLM_BUDGET_USD="${EXP_LLM_BUDGET_USD:-8}"
HOURS="${HOURS:-10}"
DEADLINE=$(( $(date +%s) + HOURS * 3600 ))

if [[ "${SMOKE:-0}" == 1 ]]; then
  T1=$OUT/smoke_run/tier1; T2=$OUT/smoke_run/tier2
  T1_ARGS="--n-title 4 --n-section 4"; T2_ARGS="--n-title 6 --n-section 6 --per-article 2"
  LOOP_MAX=1; FT_FOLDS=2; FT_EPOCHS=1
else
  T1=$OUT/tier1; T2=$OUT/tier2
  T1_ARGS="--n-title 50 --n-section 50"
  T2_ARGS="--n-title 1000 --n-section 600 --per-article 3 --min-chars 300 --max-ioc 0.08 --max-latin 0.3"
  LOOP_MAX=100; FT_FOLDS=5; FT_EPOCHS=2
fi
MAIN_MODELS=bm25,bm25_titled,e5_base,bge_m3,hybrid
EXTRA_MODELS=e5_large,ukr_mpnet,para_mpnet,bm25_surface,hybrid_e5,m3_sparse,m3_colbert,m3_all
ABLATION="naive:dt:1,naive:dt:3,naive:dt:5,naive:mlm:3,val:dt:1:0.8,val:dt:3:0.7,val:dt:3:0.8,val:dt:3:0.9,val:dt:5:0.8,val:mlm:3:0.8,val:all:3:0.8"
mkdir -p $OUT/logs
STATUS=$OUT/STATUS.md
[[ -f $STATUS ]] || printf "# Overnight status\n\nstarted %s, deadline %s\n\n| time | tier | stage | rc | minutes | last log line |\n|---|---|---|---|---|---|\n" \
  "$(date '+%F %T')" "$(date -d @$DEADLINE '+%F %T')" > $STATUS

# GPU monitor every 10 min
( while true; do echo "$(date '+%F %T') $(nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader)"; sleep 600; done ) >> $OUT/logs/gpu.log 2>&1 &
GPU_MON=$!
trap 'kill $GPU_MON 2>/dev/null' EXIT

stopped() { [[ -f $OUT/STOP ]] || (( $(date +%s) > DEADLINE )); }

# stage <tier> <name> <timeout-min> <cmd...>   — never aborts the run
stage() {
  local tier=$1 name=$2 tmo=$3; shift 3
  if stopped; then echo "skip $name (stop/deadline)"; return 1; fi
  local log=$OUT/logs/${tier}_${name}.log t0=$(date +%s)
  echo "=== $(date '+%T') [$tier] $name: $*" | tee -a $OUT/logs/run.log
  timeout --kill-after=60 $(( tmo * 60 ))m "$@" >> "$log" 2>&1
  local rc=$?
  local last; last=$(grep -v -i "warn\|it/s\]" "$log" | tail -1 | tr '|' '/' | cut -c1-140)
  printf "| %s | %s | %s | %s | %s | %s |\n" "$(date '+%T')" "$tier" "$name" "$rc" "$(( ($(date +%s) - t0) / 60 ))" "$last" >> $STATUS
  echo "    rc=$rc  $last" | tee -a $OUT/logs/run.log
  return $rc
}

# full benchmark pipeline in a tier dir: queries → pools → variants → judge → pools(variants) → retrieval → methods → report
pipeline() {
  local tier=$1 dir=$2 qargs=$3
  stage $tier s1_queries 60 $PY experiments.s1_queries --out $dir $qargs
  stage $tier pool_input 10 $PY experiments.s2_compose --out $dir --write-pool-input
  [[ -f $dir/pool_dict_queries.jsonl ]] || stage $tier pool_dict_q 60 $TF_PY experiments/s2_pool_dict.py --input $dir/pool_input_queries.jsonl --output $dir/pool_dict_queries.jsonl
  [[ -f $dir/pool_mlm_queries.jsonl ]] || stage $tier pool_mlm_q 60 $BA_PY experiments/s2_pool_mlm.py --input $dir/pool_input_queries.jsonl --output $dir/pool_mlm_queries.jsonl
  stage $tier s2_compose 60 $PY experiments.s2_compose --out $dir
  stage $tier s3_judge 120 $PY experiments.s3_judge --out $dir
  [[ -f $dir/pool_dict_variants.jsonl ]] || stage $tier pool_dict_v 120 $TF_PY experiments/s2_pool_dict.py --input $dir/pool_input_variants.jsonl --output $dir/pool_dict_variants.jsonl
  [[ -f $dir/pool_mlm_variants.jsonl ]] || stage $tier pool_mlm_v 120 $BA_PY experiments/s2_pool_mlm.py --input $dir/pool_input_variants.jsonl --output $dir/pool_mlm_variants.jsonl
  stage $tier s4_retrieve 120 $PY experiments.s4_retrieve --out $dir --models $MAIN_MODELS
  stage $tier s5_methods 180 $PY experiments.s5_methods --out $dir --configs naive:dt:3,val:dt:3:0.8 --tag main
  stage $tier s6_report 30 $PY experiments.s6_report --out $dir --title "$tier: robustness of Ukrainian retrieval to synonym substitution"
}

echo "##### start $(date)  deadline $(date -d @$DEADLINE)" | tee -a $OUT/logs/run.log
stage tier0 corpus 10 $PY experiments.s0_corpus

# ---------------- Tier 1: pilot (go / no-go)
pipeline tier1 $T1 "$T1_ARGS"
cp -f $T1/report_variants.md $OUT/report_tier1.md 2>/dev/null

# ---------------- Tier 2: full benchmark
pipeline tier2 $T2 "$T2_ARGS"
cp -f $T2/report_variants.md $OUT/report_tier2.md 2>/dev/null

# ---------------- Tier 3: more retrievers + method ablations
stage tier3 s4_extra 240 $PY experiments.s4_retrieve --out $T2 --models $EXTRA_MODELS
stage tier3 s5_ablation 240 $PY experiments.s5_methods --out $T2 --models bm25,e5_base,bge_m3,hybrid --configs "$ABLATION" --tag ablation
stage tier3 s6_report 30 $PY experiments.s6_report --out $T2 --title "Full benchmark + extra retrievers + ablations"
cp -f $T2/report_variants.md $OUT/report_tier3.md 2>/dev/null

# ---------------- Tier 4: stress analyses
stage tier4 dose_compose 60 $PY experiments.s2_compose --out $T2 --types dict,mlm --schedule 1,2,3,4 --name variants_dose
stage tier4 dose_judge 120 $PY experiments.s3_judge --out $T2 --name variants_dose
stage tier4 dose_retrieve 60 $PY experiments.s4_retrieve --out $T2 --name variants_dose --models bm25,e5_base,bge_m3,hybrid
stage tier4 dose_report 30 $PY experiments.s6_report --out $T2 --name variants_dose --title "Dose-response: number of substituted words"
stage tier4 worst_search 240 $PY experiments.s7_worstcase --out $T2
stage tier4 worst_judge 120 $PY experiments.s3_judge --out $T2 --name variants_worst
stage tier4 worst_retrieve 60 $PY experiments.s4_retrieve --out $T2 --name variants_worst --models bm25,e5_base,bge_m3,hybrid,e5_large,m3_all
stage tier4 worst_report 30 $PY experiments.s6_report --out $T2 --name variants_worst --title "Worst-case valid substitutions and transfer"
cp -f $T2/report_variants_dose.md $OUT/report_tier4_dose.md 2>/dev/null
cp -f $T2/report_variants_worst.md $OUT/report_tier4_worst.md 2>/dev/null
stage tier4 figures 20 $PY experiments.s8_final --out $T2

# ---------------- Tier 5: second (local, zero-cost) judge — agreement + both-judges robustness
for n in variants variants_dose variants_worst; do
  stage tier5 local_judge_$n 120 $PY experiments.local_judge --out $T2 --name $n
  stage tier5 report_$n 30 $PY experiments.s6_report --out $T2 --name $n --title "$n (with second judge)"
done
cp -f $T2/report_variants.md $OUT/report_tier5.md 2>/dev/null

# ---------------- Tier 6: training-time defence (article-level K-fold fine-tuning)
stage tier6 finetune 360 $PY experiments.s9_finetune --out $T2 --models e5_base,bge_m3 --folds $FT_FOLDS --epochs $FT_EPOCHS --seed 13
stage tier6 s6_report 30 $PY experiments.s6_report --out $T2 --title "Full benchmark incl. fine-tuned encoders"
cp -f $T2/report_variants.md $OUT/report_tier6.md 2>/dev/null
stage tier6 figures 20 $PY experiments.s8_final --out $T2

# ---------------- Tier 7: keep the GPU busy until deadline / STOP (zero API cost)
#   (a) new random perturbation samples judged by the local judge → stability of conclusions across samples
#   (b) fine-tuning with more seeds → variance of the training-time defence
# local-judge threshold calibrated in tier 5 against the API judge (fallback 0.5)
THR=$(python3 -c "import json;print(round(json.load(open('$T2/summary_variants.json'))['local_judge']['best_threshold'],2))" 2>/dev/null || echo 0.5)
echo "tier7 local-judge threshold: $THR" | tee -a $OUT/logs/run.log
for s in $(seq 1 $LOOP_MAX); do
  stopped && break
  n=variants_seed$s
  stage tier7 ${n}_compose 30 $PY experiments.s2_compose --out $T2 --types dict,mlm,tech --seed $((100 + s)) --name $n
  stage tier7 ${n}_judge 60 $PY experiments.s3_judge --out $T2 --name $n --backend local --local-threshold $THR
  stage tier7 ${n}_retrieve 60 $PY experiments.s4_retrieve --out $T2 --name $n --models bm25,e5_base,bge_m3,hybrid,e5_large
  stage tier7 ${n}_report 30 $PY experiments.s6_report --out $T2 --name $n --title "Extra perturbation sample $s (local judge only)"
  stage tier7 ft_seed$s 360 $PY experiments.s9_finetune --out $T2 --models e5_base,bge_m3 --folds $FT_FOLDS --epochs $FT_EPOCHS --seed $((13 + s))
done
stage final s6_report 30 $PY experiments.s6_report --out $T2 --title "Full benchmark (final)"
stage final figures 20 $PY experiments.s8_final --out $T2
echo "##### done $(date)" | tee -a $OUT/logs/run.log
printf "\nfinished %s\n" "$(date '+%F %T')" >> $STATUS
