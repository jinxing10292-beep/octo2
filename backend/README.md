# Octo AI 자동 훈련 백엔드

한국어 문장을 자동 생성/크롤링하여 브라우저의 LLM 자동 훈련을 지원하는 Flask 백엔드

## 🚀 설치 및 실행

### 1. 요구 사항
- Python 3.8+
- pip

### 2. 설치
```bash
cd backend
pip install -r requirements.txt
```

### 3. 실행
```bash
python app.py
```

서버가 `http://localhost:5000`에서 시작됩니다.

## 📚 API 엔드포인트

### 1. 헬스 체크
```bash
GET /api/health
```
응답:
```json
{
  "status": "ok",
  "timestamp": "2024-10-04T10:30:00.000000",
  "version": "1.0.0"
}
```

### 2. 문장 반환 (메인 API)
```bash
GET /api/sentences?count=10&source=builtin
```

파라미터:
- `count`: 반환할 문장 수 (기본값: 5, 최대: 100)
- `source`: 데이터 출처
  - `builtin`: 내장 한국어 문장 데이터베이스 (기본값, 가장 빠름)
  - `naver_news`: 네이버 뉴스 크롤링
  - `wiki`: 위키피디아 크롤링

응답:
```json
{
  "count": 10,
  "sentences": [
    "안녕하세요. 반갑습니다.",
    "오늘 날씨가 정말 좋네요.",
    ...
  ],
  "source": "builtin",
  "timestamp": "2024-10-04T10:30:00.000000"
}
```

### 3. 배치 데이터 (훈련용)
```bash
GET /api/batch?batch_size=20
```
응답:
```json
{
  "batch_size": 20,
  "sentences": [...],
  "timestamp": "2024-10-04T10:30:00.000000"
}
```

### 4. 단일 문장 (실시간 훈련)
```bash
GET /api/sentence
```
응답:
```json
{
  "sentence": "안녕하세요. 반갑습니다.",
  "length": 10,
  "timestamp": "2024-10-04T10:30:00.000000"
}
```

### 5. 통계
```bash
GET /api/stats
```
응답:
```json
{
  "total_sentences": 50,
  "avg_length": 15.5,
  "max_length": 35,
  "min_length": 5,
  "timestamp": "2024-10-04T10:30:00.000000"
}
```

## 🛠️ 기능

### 데이터 소스
1. **내장 데이터 (Builtin)** - ⚡ 가장 빠름
   - 50개의 일반적인 한국어 문장
   - 인터넷 필요 없음
   - 즉시 응답

2. **웹 크롤링** - 🌐 더 많은 데이터
   - 네이버 뉴스
   - 위키피디아
   - 실시간 데이터

### CORS 활성화
브라우저에서 직접 API 호출 가능 (localhost 또는 배포 환경)

## 🔧 커스터마이징

### 문장 데이터 추가
`app.py`의 `KOREAN_SENTENCES` 리스트에 문장 추가:
```python
KOREAN_SENTENCES = [
    "기존 문장들...",
    "새로운 한국어 문장을 추가합니다.",
    "원하는 만큼 추가할 수 있습니다.",
]
```

### 크롤링 소스 확장
`crawl_korean_sentences()` 함수 수정:
```python
def crawl_korean_sentences(source="your_source"):
    # 새로운 크롤링 로직 추가
    ...
```

## ⚠️ 주의사항

1. **크롤링 정책**
   - 각 사이트의 `robots.txt` 준수
   - 이용약관 확인
   - 과도한 요청 금지

2. **성능**
   - 크롤링은 네트워크 지연 발생
   - 내장 데이터 권장

3. **보안**
   - 로컬 개발: `localhost:5000` 사용
   - 배포 시: 프로덕션 서버(Gunicorn 등) 사용

## 📊 백엔드와 브라우저 통합

브라우저의 자동 훈련 UI에서:
```javascript
// API 호출
const response = await fetch('http://localhost:5000/api/sentence');
const data = await response.json();
const sentence = data.sentence;

// 훈련
trainStep(model, sentence);
```

## 🚀 배포

### 로컬 개발
```bash
python app.py  # 자동 재로드 활성화
```

### 프로덕션
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Docker
```dockerfile
FROM python:3.9
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY app.py .
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

## 📝 라이선스
MIT

## 🤝 기여
이슈 및 PR 환영합니다!
