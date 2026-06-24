# Throughput Tuning: Rule-only vs AI + Rule Ablation

이 폴더는 PIBT 기반 AMR 시뮬레이션에서 **Rule-only(B)** 와 **AI + Rule(D)** 를 같은 조건으로 비교하기 위한 재현 코드입니다.

핵심 질문은 다음입니다.

> rule 기반 throughput 개선만으로 좋아진 것인가, 아니면 AI 혼잡도 예측이 추가로 기여했는가?

## 비교 정의

| 구분 | 의미 | 핵심 옵션 |
|---|---|---|
| B: Rule-only | 목표 근접 보너스, 대기 탈출 priority 등 rule은 사용하지만 AI 예측 혼잡도 cost는 사용하지 않음 | `--ai-cost-weight 0` |
| D: AI + Rule | 같은 rule에 더해 AI가 예측한 future congestion heatmap을 후보 선택에 반영 | 기본 AI cost 사용 |

두 실험은 아래 조건을 동일하게 둡니다.

- 같은 map
- 같은 AMR 수
- 같은 seed
- 같은 pickup/delivery random assignment
- 같은 PIBT engine
- 같은 vertex/swap conflict 방지
- 같은 kinodynamic 옵션
- 같은 safety gap

차이는 **AI predicted congestion을 candidate ordering에 쓰는지 여부**입니다.

## 실행 전 준비

Anaconda Prompt에서 실행합니다.

```bat
cd /d D:\lap\MACPF_clean
activate macpf
set KMP_DUPLICATE_LIB_OK=TRUE
```

`KMP_DUPLICATE_LIB_OK=TRUE`는 Windows에서 torch/OpenMP DLL 충돌이 나는 경우를 피하기 위한 임시 설정입니다.

## B: Rule-only 실행

```bat
python -m throughput_tuning.sweep_tuned --seeds 42-46 --config configs/default.yaml --model models\congestion_simvp_pibt120_mixed_e4.pt --max-time 900 --num-agents 120 --device cuda --kinodynamic --continuous-safe-gap 0.35 --ai-cost-mode tiebreak --ai-cost-weight 0 --skip-ai-fraction 0.5 --throughput-profile balanced --out reports\throughput_tuning\ablation_B_rule_only
```

결과 저장 위치:

```text
reports\throughput_tuning\ablation_B_rule_only
```

## D: AI + Rule 실행

```bat
python -m throughput_tuning.sweep_tuned --seeds 42-46 --config configs/default.yaml --model models\congestion_simvp_pibt120_mixed_e4.pt --max-time 900 --num-agents 120 --device cuda --kinodynamic --continuous-safe-gap 0.35 --ai-cost-mode tiebreak --skip-ai-fraction 0.5 --throughput-profile balanced --out reports\throughput_tuning\final_gap035_tiebreak_balanced_900s_seed42_46
```

결과 저장 위치:

```text
reports\throughput_tuning\final_gap035_tiebreak_balanced_900s_seed42_46
```

## B와 D 비교

두 실험이 끝난 뒤 아래 명령어를 실행합니다.

```bat
python throughput_tuning\compare_ablation_bd.py --rule-only reports\throughput_tuning\ablation_B_rule_only --ai-rule reports\throughput_tuning\final_gap035_tiebreak_balanced_900s_seed42_46 --out reports\throughput_tuning\compare_B_vs_D
```

생성 파일:

```text
reports\throughput_tuning\compare_B_vs_D\seed_comparison.csv
reports\throughput_tuning\compare_B_vs_D\summary.json
reports\throughput_tuning\compare_B_vs_D\summary.md
```

## 한 번에 실행

위 과정을 한 번에 돌리고 싶으면:

```bat
throughput_tuning\run_ablation_bd.bat
```

주의: 120대, 900초, seed 5개 기준이라 시간이 오래 걸릴 수 있습니다.

## 해석 방법

가장 중요한 값은 `D - B`입니다.

| 지표 | 좋은 방향 | 의미 |
|---|:---:|---|
| `ai_total_completed_deliveries` | 증가 | 배송 완료 수 증가 |
| `ai_total_completed_targets` | 증가 | pickup/delivery target 도달 증가 |
| `ai_average_deliveries_per_agent` | 증가 | AMR 1대당 처리량 증가 |
| `ai_total_waiting_time` | 감소 | 전체 대기시간 감소 |
| `ai_waiting_ratio` | 감소 | 시뮬레이션 중 대기 비율 감소 |
| `ai_pibt_candidate_reject_vertex` | 감소 | 같은 칸 진입 충돌 압력 감소 |
| `ai_pibt_candidate_reject_swap` | 감소 | 서로 자리 바꾸기 충돌 압력 감소 |
| `ai_congestion_overlap_cell_count` | 감소 | 혼잡 구역 겹침 감소 |
| `ai_collision_count` | 0 유지 | 충돌 없음 |
| `ai_interpolated_safe_gap_violation_count` | 0 유지 | 연속 안전거리 위반 없음 |

발표용 문장 예시는 다음과 같습니다.

> Rule-only 대비 AI + Rule은 동일한 rule 구조 위에서 예측 혼잡도 정보를 추가로 사용하여 배송 완료 수를 증가시키고, 총 대기시간과 vertex/swap conflict pressure를 감소시켰다. 따라서 성능 향상이 rule만의 효과가 아니라 AI 기반 future congestion prediction의 추가 기여를 포함한다고 해석할 수 있다.

## GitHub에 올리는 방법

현재 폴더만 올리려면:

```bat
cd /d D:\lap\MACPF_clean
git status
git add throughput_tuning
git commit -m "Add throughput tuning ablation scripts"
git push origin dongjin
```

이미 다른 수정 파일들이 많다면 `git add .`는 쓰지 않는 것을 권장합니다. 이번 재현 파일만 올리려면 `git add throughput_tuning`만 사용하세요.
