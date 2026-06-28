# CALM — Congestion-Aware Lookahead Mobility

다중 AMR 경로계획(MAPF)을 **lifelong PIBT** 솔버로 풀고, 그 위에 학습된 **혼잡(congestion) 히트맵 예측**을 결합하는 프로젝트입니다.

기존 MACPF(우선순위 계획, prioritized planning)에서 **솔버를 lifelong PIBT로 교체**해 새로 시작했습니다. 우선순위 계획은 밀도가 오르면 계획시간이 폭발했지만(200대·60초 ≈ 641s), PIBT는 거의 선형이라 같은 조건을 **~0.36s**에 풉니다.

> **브랜치:** `main` = 이 CALM 프로젝트 · `legacy/macpf` = 초기 MACPF (보존용, 개발 안 함)

## 구조

```
configs/default.yaml          # 시뮬레이션 파라미터 (단일 원본)
environment.yml               # conda 환경 (congestion_prediction / OpenSTL, CUDA)
calm/
  PiBT/                       # ★ lifelong PIBT MAPF 엔진
    pibt.py                       lifelong PIBT 솔버
    scenario.py                   출발/목표 선택 (staging / distributed)
    grid.py / distance.py         walkability·이웃 / BFS 거리장
    metrics.py                    occupancy·additive congestion·collision
    factory_map_generator.py      50x80 공장 맵
    config.py / types.py / viz.py
  generate_heatmap/           # 혼잡 히트맵 데이터셋
    generate.py                   데이터 생성 (PIBT 연결)
    render_heatmap.py             데이터셋 -> MP4 시각화
    __main__.py                   python -m calm.generate_heatmap
  congestion_prediction/      # 혼잡 예측 모델 (OpenSTL / SimVP)
    train_openstl_congestion.py   학습
    predict.py                    추론 래퍼 (best.ckpt -> torch.nn.Module)
    visualize.py                  학습 결과 시각화
    OpenSTL/                      서드파티 클론 (레포 미포함)
  evaluation/                 # PIBT 혼잡회피 평가·비교 (예측기를 import)
    grid_eval.py                  혼잡적용 PIBT vs vanilla 비교 그리드 (+ 영상)
    ab_eval.py                    A/B 평가 (콘솔)
    make_analysis_summary.py      metrics.csv -> analysis_summary.md (런마다 자동, 단독 재생성 가능)
```

## 셋업 & 실행

### 1) PIBT 엔진 · 데이터 생성 (가벼움 — numpy, pyyaml)
레포 루트가 import 경로에 있어야 합니다(루트에서 실행하거나 `PYTHONPATH=<레포 루트>`).

```bash
python -m calm.generate_heatmap                                   # 데이터셋 생성 (기본 sweep)
python -m calm.generate_heatmap --rounds 4 --seconds 3600 --num_of_process 8
python -m calm.generate_heatmap.render_heatmap --dataset data/heatmap_dataset/<타임스탬프>
```

### 2) 혼잡 예측 (OpenSTL, GPU/CUDA)

```bash
conda env create -f environment.yml
conda activate OpenSTL
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # GPU 빌드 확인

# OpenSTL 클론 (서드파티 — 레포에 미포함)
cd calm/congestion_prediction
git clone -b OpenSTL-Lightning https://github.com/chengtan9907/OpenSTL.git
# 필수 패치: OpenSTL/openstl/utils/main_utils.py 의 gcc/head 환경탐지부를
#            try/except 로 감싸기 (Windows엔 gcc가 없어 그대로면 크래시)
```

실행 (이 폴더에서):
```bash
python train_openstl_congestion.py     # 학습 -> work_dirs/
python predict.py                       # 추론 검증
python visualize.py                     # 학습 결과 그림
```

### 3) PIBT 혼잡회피 평가·비교 (`calm/evaluation` — 예측기를 import해서 사용)
```bash
cd calm/evaluation
python grid_eval.py                     # 혼잡적용 vs vanilla 비교 그리드 (+ 영상)
python grid_eval.py --video-only --from-run reports/CALM_comparison/<타임스탬프>   # 영상만 재인코딩(재시뮬 없음)
python ab_eval.py                       # A/B 평가 (콘솔)
python make_analysis_summary.py reports/CALM_comparison/<타임스탬프>   # csv -> analysis_summary.md 재생성
```

## 출력(`reports/`) 분류

`reports/`는 여러 코드가 공유하므로, **어떤 코드가 / 언제 만들었는지** 구분되도록 코드별·실행시각별로 저장합니다.

```
reports/
├── congestion_prediction/<yymmdd_hhmm>/   # visualize · predict 산출물 (실행 단위)
└── CALM_comparison/<yymmdd_hhmm>/          # grid_eval 비교 실행 (metrics.csv·표·영상)
```

## 레포에 포함하지 않는 것 (재생성 / 외부)

- `data/` (히트맵 데이터셋), `models/`·`work_dirs/` (체크포인트·로그), `reports/` (영상·그림) — 용량이 커서 제외, 고정 시드로 재생성 가능
- `OpenSTL/` — 서드파티(자체 `.git`). 위 클론 + 패치로 받기

## 주요 인자 (`generate_heatmap`)

- `--num_of_process` 병렬 프로세스 수
- `--base-seed` 시드 시작값(에피소드마다 +1, 스폰 위치 결정)
- `--seconds` 에피소드 길이(초) · `--rounds` 대수 sweep 왕복 횟수
- `--min-agents` / `--max-agents` 대수 sweep 범위
- `--distributed-start-frac` 맵 전체 분산 출발 비율(0~1)
- `--center-value` / `--step-value` 혼잡 라벨 정의

데이터는 `data/heatmap_dataset/<타임스탬프>/episode_*.npz` + `metadata.json`로 저장됩니다.
