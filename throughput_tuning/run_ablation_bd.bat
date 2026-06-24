@echo off
setlocal

cd /d "%~dp0\.."
set KMP_DUPLICATE_LIB_OK=TRUE

set CONFIG=configs/default.yaml
set MODEL=models\congestion_simvp_pibt120_mixed_e4.pt
set SEEDS=42-46
set MAX_TIME=900
set NUM_AGENTS=120
set DEVICE=cuda
set GAP=0.35

echo [1/3] Running B: Rule-only
python -m throughput_tuning.sweep_tuned --seeds %SEEDS% --config %CONFIG% --model %MODEL% --max-time %MAX_TIME% --num-agents %NUM_AGENTS% --device %DEVICE% --kinodynamic --continuous-safe-gap %GAP% --ai-cost-mode tiebreak --ai-cost-weight 0 --skip-ai-fraction 0.5 --throughput-profile balanced --out reports\throughput_tuning\ablation_B_rule_only
if errorlevel 1 exit /b %errorlevel%

echo [2/3] Running D: AI + Rule
python -m throughput_tuning.sweep_tuned --seeds %SEEDS% --config %CONFIG% --model %MODEL% --max-time %MAX_TIME% --num-agents %NUM_AGENTS% --device %DEVICE% --kinodynamic --continuous-safe-gap %GAP% --ai-cost-mode tiebreak --skip-ai-fraction 0.5 --throughput-profile balanced --out reports\throughput_tuning\final_gap035_tiebreak_balanced_900s_seed42_46
if errorlevel 1 exit /b %errorlevel%

echo [3/3] Comparing B vs D
python throughput_tuning\compare_ablation_bd.py --rule-only reports\throughput_tuning\ablation_B_rule_only --ai-rule reports\throughput_tuning\final_gap035_tiebreak_balanced_900s_seed42_46 --out reports\throughput_tuning\compare_B_vs_D
if errorlevel 1 exit /b %errorlevel%

echo Done.
