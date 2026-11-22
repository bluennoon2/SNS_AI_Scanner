# 음성 분석 모듈 (`audio/`)

[SNS AI Scanner](https://github.com/bluennoon2/SNS_AI_Scanner)의 음성 분석을 담당합니다. 사용자가 유튜브·인스타그램 같은 SNS 영상 링크를 붙여넣으면, 영상에서 음성을 추출해 한국어 AI 합성 음성이 의심되는지 분석하는 것이 목표입니다. 분석 결과에는 파일 전체의 판정과 AI 음성 의심 구간의 시작·끝 시각이 포함됩니다.

현재는 파일을 업로드해 CPU에서 분석할 수 있습니다. SNS 링크로 영상을 가져와 오디오를 추출하고 기존 분석기에 연결하는 기능은 개발 예정입니다.

- **전체 검사**는 NII-Small이 4초 단위로 계산한 점수를 평균해 파일 전체의 AI 음성 의심 여부를 판단합니다.
- **부분 검사**는 한국어 편집 데이터로 재학습한 BAM이 편집 경계 후보를 찾고, NII가 각 구간을 따로 검사해 AI 음성 의심 구간을 표시합니다.

## 폴더 구성

아래 경로와 명령은 팀 저장소의 `audio/`를 기준으로 합니다. 팀 저장소 최상위에서 시작한다면 먼저 `cd audio`를 실행하세요.

| 폴더 | 내용 |
|---|---|
| `service/` | 공개 서비스 v1 (CPU). FastAPI + Caddy + Docker Compose, 전체 검사, 동의·철회 관리, 재학습 프레임워크(`donations/mlops/`, 학습 실행은 꺼 둠), 테스트 |
| `demo/` | 부분 검사 서버 (CPU). BAM 경계 → 구간별 NII → TRAIN 기반 확률 보정, 웹 페이지, AWS 켜기/끄기 스크립트 |
| `demo/code/` | BAM 로더(`bam_cfprf/`)와 BAM 실행 환경 패키지 목록 |
| `research/bam_finetune/` | BAM 한국어 재학습·평가 코드와 결과 보고서 (학습 데이터는 포함하지 않음) |
| `docs/` | 최종 발표 PPT, 발표 대본 PDF, 시연 영상 |

데이터셋(AIHub 기반 음성, 합성 음성), 접속 토큰, AWS 계정 정보는 포함하지 않았습니다.

## 모델 가중치 (GitHub Release)

가중치는 음성 파트의 [기존 모델 릴리스](https://github.com/gsw0408/korean-ai-speech-detection/releases/tag/weights-v1)에서 받습니다. 아래 명령은 각 파일을 모델이 읽는 위치에 저장합니다.

| 파일 | 크기 | 넣을 위치 |
|---|---:|---|
| `wav2vec2-small-antideepfake.onnx` | 378 MB | `service/models/` (전체 검사, 공개 서비스) |
| `nii_frames_dyn.onnx` | 378 MB | 데모 서버 `assets/` (길이가 바뀌는 구간 판정용 NII) |
| `bam_ft_final.pt` | 1.46 GB | 데모 서버 `out/` (한국어 재학습 BAM 전체 가중치) |

```bash
gh release download weights-v1 -R gsw0408/korean-ai-speech-detection -p 'wav2vec2-small-antideepfake.onnx' -D service/models
gh release download weights-v1 -R gsw0408/korean-ai-speech-detection -p 'nii_frames_dyn.onnx' -D assets
gh release download weights-v1 -R gsw0408/korean-ai-speech-detection -p 'bam_ft_final.pt' -D out
```

BAM을 만들려면 공식 BAM 코드와 공식 가중치도 필요합니다(라이선스가 명시되지 않아 이 저장소에는 넣지 않았습니다).

- 공식 코드는 `git clone https://github.com/media-sec-lab/BAM.git code/BAM`으로 받은 뒤 `git -C code/BAM checkout 55f3fb9e3b4dd6281597b86d7712fb23454179f6`으로 버전을 맞춥니다.
- 가중치: BAM README 안내에 따라 160 ms BAM 체크포인트(`bam_checkpoint.zip`)와 `WavLM-Large.pt`를 받습니다. 모델 구조를 만드는 데 쓰이고, 실제 가중치는 `bam_ft_final.pt`로 덮어씁니다.

## 실행

### 공개 서비스 v1 (CPU)
전체 검사 서버는 `service/`에서 `docker compose up`으로 실행합니다. 모델 해시는 `service/bundles/nii-small-v1/manifest.json`과 일치해야 합니다. [서비스 실행 안내](service/README.md)

### 부분 검사 서버 (CPU, Ubuntu x86-64, Python 3.10)
`demo/demo_core.py`는 `audio/` 아래에서 다음 파일을 찾습니다.

```
demo/code/bam_cfprf/  ← 음성 모듈에 포함된 BAM 로더
code/BAM/             ← 공식 BAM (위 커밋)
assets/               ← bam_checkpoint.zip, WavLM-Large.pt, nii_frames_dyn.onnx
out/                  ← bam_ft_final.pt
```

```bash
python3.10 -m venv /opt/bam
sed 's/+cu113$/+cpu/' demo/code/bam_requirements.txt > /opt/bam/bam_requirements_cpu.txt
/opt/bam/bin/pip install --no-deps -r /opt/bam/bam_requirements_cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu
/opt/bam/bin/pip install fastapi uvicorn==0.29.0 python-multipart onnxruntime scipy soundfile
sudo apt-get install -y ffmpeg libsndfile1
cd demo && DEMO_TOKEN='원하는-접속-토큰으로-교체' /opt/bam/bin/python -m uvicorn app:app --host 0.0.0.0 --port 8080 --workers 1
```

브라우저에서 `http://<서버 주소>:8080/?k=<접속 토큰>`으로 엽니다. 업로드한 음성은 분석 후 삭제되고 저장하지 않습니다.

AWS에서 켜고 끌 때는 `demo/start_demo.ps1`, `demo/stop_demo.ps1`을 씁니다. 환경 변수 `DEMO_INSTANCE_ID`, `DEMO_SECURITY_GROUP_ID`를 설정하고, `demo/DEMO_ACCESS.local.txt`에 `DEMO_TOKEN=<토큰>` 한 줄을 만들어 두세요.

## 주요 결과 (발표 자료 기준)

- 공개 모델 6종 비교 후 SSL-AASIST를 한국어로 추가 학습 → 처음 보는 생성기만 있는 FINAL에서 최고 AUC 0.678. NII는 0.982여서 NII로 변경
- 1·2초 AI 삽입 구분 AUC: 고정 4초 창 + NII 0.595 / 0.736 → 재학습 BAM + NII **0.728 / 0.887**, ±0.3초 경계 재현율 95.8%
- 한계: 처음 보는 음성 변환(Seed-VC) 1초 삽입은 10개 중 4개만 표시, 실제 음성 오표시 존재. 통제된 합성 편집 실험 결과이며 독립 평가가 더 필요합니다.

자세한 수치와 조건은 `research/bam_finetune/REPORT.md`, `docs/`를 참고하세요.

## 라이선스와 출처

- NII-Small(`nii-yamagishilab/wav2vec-small-anti-deepfake`): CC BY-NC-SA 4.0, 비상업적 이용만 가능합니다. `service/MODEL_ATTRIBUTION.md` 참고.
- BAM(`media-sec-lab/BAM`): 공식 저장소에 라이선스 표기가 없습니다. 재학습 가중치(`bam_ft_final.pt`)는 연구·학습 목적으로만 사용하세요.
- 이 음성 모듈의 코드는 수업 프로젝트 결과물입니다.
