#!/bin/bash
# feature-group ablation + seeds + stride on the default LGB lambdarank config
cd "$(dirname "$0")/.."
P='{"objective":"lambdarank","lambdarank_truncation_level":50}'
run() { name=$1; shift; uv run python research/run_wf.py --name "$name" --rank-all --features extra --model lgb --label y_relevance --params "$P" "$@" > research/logs/$name.log 2>&1; grep -E "^(dev|hold)" research/logs/$name.log || tail -3 research/logs/$name.log; }
run ab_nomom --stride 3 --drop mom_12_1,mom_12_1_rank,ret_126d,ret_126d_rank,info_discrete,info_discrete_rank,mom_12_1_secrank
run ab_novol --stride 3 --drop idio_vol_60d,idio_vol_60d_rank,max_ret_21d,max_ret_21d_rank,skew_60d,skew_60d_rank,idio_vol_60d_secrank
run ab_noliq --stride 3 --drop amihud_21d,amihud_21d_rank,log_dollar_vol_63d,log_dollar_vol_63d_rank,dvol_trend,dvol_trend_rank
run ab_noevent --stride 3 --drop seasonal_ret,seasonal_ret_rank,ear_3d,ear_3d_rank,dist_52w_low,ret_5d,ret_5d_rank
run ab_nosector --stride 3 --drop sector_ret_63d,sector_ret_21d
run ab_noregime --stride 3 --drop spy_ret_21d,spy_rsi_14,vix_level,term_spread_10y3m,ig_credit_spread,inflation_5y5y
run lr_s5 --stride 3 --seeds 1,2,3,4,5
run lr_st1 --stride 1
echo QUEUE4_DONE
