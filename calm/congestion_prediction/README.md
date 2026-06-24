깃허브에 올릴 것과 올리지 말 것을 정리해 드립니다.

    ✅ 올려야 할 것 (핵심은 2개)

    1. train_openstl_congestion.py — 학습 스크립트 (메인)
    2. environment.yml — 환경 재현용 (cu128 torch, timm≥0.9 등 수정된 버전)
    3. README.md — 셋업/실행 방법 (직접 작성 권장, 아래 내용 포함)
    4. .gitignore — 데이터·산출물 제외용

    ❌ 올리지 말 것

    - 260614_0043/ (학습 데이터) — npz 100개 × 130MB ≈ 13GB. 깃허브 일반 업로드
    불가(파일 100MB 제한). → 구글드라이브/외부 스토리지 링크 또는 Git LFS로, README에
    받는 법만 기재
    - OpenSTL/ (클론한 외부 레포) — 자체 .git이 있는 서드파티 레포라 통째로 올리지
    마세요. README에 "원본 클론 + 패치" 방식으로 안내
    - work_dirs/, lightning_logs/, __pycache__/ — 학습 산출물/캐시(자동 생성)

    ⚠️ "다른 곳에서 쓰려면" 반드시 손봐야 할 부분

    지금 코드엔 이 PC 전용 절대경로가 박혀 있어서 그대로는 다른 PC에서 안 됩니다:

    - 스크립트 안 OPENSTL_ROOT = r"C:\Robot\OpenSTL" → 상대경로나 환경변수로 변경 필요
    - DATA_DIR = r"C:\Robot\260614_0043" → 마찬가지로 인자/환경변수로 변경 권장
    - OpenSTL 한 줄 패치 — openstl/utils/main_utils.py의 gcc 호출을 try/except로 감싼
    수정. 이건 라이브러리 쪽이라, README에 "이 줄을 이렇게 고쳐라" 또는 .patch 파일로
    첨부

    README에 들어가면 좋은 내용

    1. conda env create -f environment.yml && conda activate OpenSTL
    2. git clone <OpenSTL repo>  (그리고 main_utils.py gcc 부분 패치)
    3. 데이터(260614_0043) 내려받아 지정 경로에 배치
    4. python train_openstl_congestion.py

    정리하면 올릴 핵심은 train_openstl_congestion.py + environment.yml + README.md +
    .gitignore 이고, 데이터와 OpenSTL 클론은 올리지 말고 README로 안내하는 게
    정석입니다. 추가로 절대경로 3곳을 상대경로/환경변수로 바꿔두면 바로 이식
    가능합니다.