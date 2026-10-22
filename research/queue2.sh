#!/bin/bash
cd "$(dirname "$0")/.."
run() { name=$1; shift; uv run python research/run_wf.py --name "$name" "$@" > research/logs/$name.log 2>&1; grep -E "^(dev|hold)" research/logs/$name.log || tail -3 research/logs/$name.log; }
run rankall_st3 --stride 3 --rank-all
run rankall_extra_st3 --stride 3 --rank-all --features extra
run lgbr_rankall_extra_st3 --stride 3 --rank-all --features extra --model lgb --label y_relevance --params '{"objective":"lambdarank","lambdarank_truncation_level":50}'
run wf_y63_st3 --stride 3 --label y63
run lgbr_y63_st3 --stride 3 --model lgb --label y63_relevance --params '{"objective":"lambdarank","lambdarank_truncation_level":50}'
run wf_ysec_st3 --stride 3 --label y_sec
run wf_top10_st3 --stride 3 --label y_top_decile --params '{"objective":"binary:logistic"}'
echo QUEUE2_DONE
