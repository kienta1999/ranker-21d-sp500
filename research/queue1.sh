#!/bin/bash
# batch 1: speed check + model/label/feature basics
cd "$(dirname "$0")/.."
run() { name=$1; shift; uv run python research/run_wf.py --name "$name" "$@" > research/logs/$name.log 2>&1; tail -4 research/logs/$name.log | head -3; }
run wf_base_st3 --stride 3
run wf_rank_st3 --stride 3 --label y_rank
run wf_t300_st3 --stride 3 --params '{"n_estimators":300}'
run wf_lr02_t300_st3 --stride 3 --params '{"n_estimators":300,"learning_rate":0.02}'
run lgb_reg_st3 --stride 3 --model lgb --label y_rank
run lgb_rank_st3 --stride 3 --model lgb --label y_relevance --params '{"objective":"lambdarank","lambdarank_truncation_level":50}'
run wf_extra_st3 --stride 3 --features extra
echo QUEUE1_DONE
