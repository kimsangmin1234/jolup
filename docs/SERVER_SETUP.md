# 서버에서 파이프라인 돌리기

클라우드 세션은 유휴 시 컨테이너가 회수되어 장시간 작업이 끊긴다.
감성·임베딩 생성과 발행 시각 크롤링은 각각 수 시간이 걸리므로 24시간
꺼지지 않는 서버에서 돌리는 편이 확실하다.

## 필요 사양

작업이 API 호출과 네트워크 대기 위주여서 요구 사양이 낮다.

| 항목 | 필요량 |
|---|---|
| CPU | 1~2코어 |
| RAM | 2GB |
| 디스크 | 5GB |
| GPU | 불필요 |

학습 모델도 파라미터 3.5M 규모라 CPU로 충분하다.

---

## Oracle Cloud Always Free

영구 무료 등급이 ARM 4코어 / 24GB RAM 을 제공한다. 이 작업에는 과할
정도지만 비용이 들지 않는다. 카드 등록이 필요하나 Always Free 자원은
과금되지 않는다.

### 인스턴스 생성

1. https://cloud.oracle.com 가입 (지역은 `South Korea Central (Chuncheon)` 권장)
2. **Compute → Instances → Create instance**
3. 설정
   - Image: **Canonical Ubuntu 22.04**
   - Shape: **Ampere** → `VM.Standard.A1.Flex`
   - OCPU `2`, Memory `12GB` (무료 한도 내)
4. **SSH keys** → *Generate a key pair for me* → 개인키 내려받기
5. **Create**

### 접속

```bash
chmod 400 ssh-key-*.key
ssh -i ssh-key-*.key ubuntu@<인스턴스 공인 IP>
```

접속이 안 되면 보안 목록에서 22번 포트가 열려 있는지 확인한다.

### 주의: ARM 아키텍처

Ampere 인스턴스는 `aarch64` 다. PyTorch 는 ARM 리눅스 휠을 제공하므로
`pip install torch` 가 그대로 동작하지만, 설치가 x86 보다 오래 걸린다.

---

## 네이버클라우드 / NHN클라우드

국내 리전이라 지연이 짧고 결제가 편하다. 최소 사양(1vCPU/2GB)이면
월 1만원 안팎이며, 며칠만 쓰고 삭제하면 비용이 얼마 되지 않는다.

### 네이버클라우드

1. https://www.ncloud.com 가입 → 결제 수단 등록
2. **Services → Compute → Server → 서버 생성**
3. 설정
   - 이미지: **Ubuntu 22.04**
   - 서버 타입: **Micro (1vCPU, 2GB)** 이상
4. 인증키(pem) 생성·저장
5. **공인 IP 신청** (별도 메뉴에서 할당해야 외부 접속이 된다)
6. **ACG**(방화벽)에서 22번 포트 허용

```bash
chmod 400 <키>.pem
ssh -i <키>.pem root@<공인 IP>
```

### NHN클라우드

절차가 거의 같다. **Compute → Instance → 인스턴스 생성** 에서 Ubuntu 를
고르고, **플로팅 IP** 를 연결한 뒤 보안 그룹에서 22번을 연다.

---

## 공통 설치 절차

```bash
sudo apt update && sudo apt install -y python3-pip git unzip curl
git clone https://github.com/kimsangmin1234/jolup.git
cd jolup
```

### 깃 푸시 자격증명 (진행분을 저장소에 올리려면)

서버에서 푸시하려면 GitHub 개인 액세스 토큰이 필요하다.
https://github.com/settings/tokens 에서 `repo` 권한 토큰을 발급한 뒤:

```bash
git remote set-url origin https://<토큰>@github.com/kimsangmin1234/jolup.git
```

푸시가 필요 없으면 `PUSH=0` 으로 실행한다. 결과는 서버에만 남는다.

### 실행

```bash
export OPENAI_API_KEY=sk-...
nohup bash run_on_server.sh > run.log 2>&1 &
```

진행 확인:

```bash
tail -f run.log
```

SSH 를 끊어도 `nohup` 덕분에 계속 돈다. 중간에 멈추면 같은 명령을 다시
실행하면 된다. 모든 단계가 재실행 안전하다.

---

## 실행 단계

| 단계 | 내용 | 예상 소요 |
|---|---|---|
| 0 | 본문 레코드 복원 (깃 조각 합치기) | 1분 |
| 1 | 요약 + 감성 (Batch API) | 4~6시간 |
| 2 | 의미 임베딩 (Batch API) | 1~2시간 |
| 3 | 발행 시각 크롤링 (1단계와 병렬) | 약 12시간 |
| 4 | 발행 시각 기반 재라벨링 | 5분 |
| 5 | 학습 | 30분~1시간 |

1·3단계가 병렬로 돌아 전체 12~14시간이면 끝난다.

## 환경 변수

| 변수 | 용도 |
|---|---|
| `OPENAI_API_KEY` | 필수 |
| `TAG` | 실험 이름 (기본 `fnspid-server`) |
| `SKIP_CRAWL=1` | 발행 시각 크롤링 건너뛰기 |
| `SKIP_TRAIN=1` | 학습 건너뛰기 |
| `PUSH=0` | 깃 푸시 안 함 |

크롤링을 건너뛰면 `next_day` 라벨로 학습한다. 뉴스 시점에 아직 실현되지
않은 값이라 시각 정보 없이도 안전하다.
