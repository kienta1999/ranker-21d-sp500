#!/bin/bash
# LGB lambdarank tuning on rank-all + extra features (select on DEV only)
cd "$(dirname "$0")/.."
B='"objective":"lambdarank","lambdarank_truncation_level":50'
run() { name=$1; shift; uv run python research/run_wf.py --name "$name" --stride 3 --rank-all --features extra --model lgb --label y_relevance "$@" > research/logs/$name.log 2>&1; grep -E "^(dev|hold)" research/logs/$name.log || tail -3 research/logs/$name.log; }
run lr_r150 --rounds 150 --params "{$B}"
run lr_r600 --rounds 600 --params "{$B}"
run lr_nl15 --params "{$B,\"num_leaves\":15}"
run lr_nl63 --params "{$B,\"num_leaves\":63}"
run lr_md500 --params "{$B,\"min_data_in_leaf\":500}"
run lr_md8000 --params "{$B,\"min_data_in_leaf\":8000}"
run lr_tr20 --params '{"objective":"lambdarank","lambdarank_truncation_level":20}'
run lr_tr150 --params '{"objective":"lambdarank","lambdarank_truncation_level":150}'
run lr_xendcg --params '{"objective":"rank_xendcg"}'
run lr_ff03 --params "{$B,\"feature_fraction\":0.3}"
run lr_lr005_r1000 --rounds 1000 --params "{$B,\"learning_rate\":0.005}"
echo QUEUE3_DONE
