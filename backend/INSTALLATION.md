# 🔧 설치 및 설정 가이드

## Windows에서 설치

### 1. Python 설치 확인
```powershell
python --version
pip --version
```

**Python이 없으면:**
- https://www.python.org/downloads/ 에서 Python 3.9+ 설치
- 설치 시 "Add Python to PATH" 체크

### 2. 백엔드 폴더로 이동
```powershell
cd backend
```

### 3. 가상 환경 생성 (권장)
```powershell
python -m venv venv
.\venv\Scripts\activate
```

**가상 환경이 활성화되면:**
```
(venv) C:\path\to\backend>
```

### 4. 패키지 설치
```powershell
pip install -r requirements.txt
```

**설치 진행 상황:**
```
Collecting Flask==2.3.3
  Downloading Flask-2.3.3-py3-none-any.whl (101 kB)
     |████████████████████████████████| 101 kB ...
...
Successfully installed Flask-2.3.3 Flask-CORS-4.0.0 ...
```

### 5. 서버 실행
```powershell
python app.py
```

**성공 시:**
```
🚀 Octo AI 자동 훈련 백엔드 시작
📍 API 주소: http://localhost:5000
...
 * Running on http://localhost:5000
 * Press CTRL+C to quit
```


---

## ⚠️ 일반적인 오류 및 해결책

### "Python이 설치되지 않음"
```
'python' is not recognized as an internal or external command
```
**해결:**
1. Python 설치
2. 경로 설정 확인: `py --version`
3. 또는 `python3 app.py` 사용

### "포트 5000이 이미 사용 중"
```
Address already in use
```
**해결:**
```bash
# 포트 사용 확인
lsof -i :5000  (macOS/Linux)
netstat -ano | findstr :5000  (Windows)

# 다른 포트에서 실행
python app.py  # app.py 수정: port=5001
```

### "모듈을 찾을 수 없음"
```
ModuleNotFoundError: No module named 'flask'
```
**해결:**
```bash
# 가상 환경 활성화 확인
pip install -r requirements.txt  # 다시 실행
```

### "CORS 오류"
```
Access to XMLHttpRequest blocked by CORS policy
```
**해결:**
- 백엔드가 실행 중인지 확인
- `http://localhost:5000` 접근 테스트
- 브라우저 콘솔에서 오류 확인

---

## 🚀 프로덕션 배포

### 1. Gunicorn 설치
```bash
pip install gunicorn
```

### 2. 서버 실행
```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```



## 📦 Docker 배포 (선택)

### Dockerfile
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app.py .

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

### 빌드 및 실행
```bash
docker build -t octo-backend .
docker run -p 5000:5000 octo-backend
```

---

## ✅ 설치 완료 체크리스트

- [ ] Python 3.8+ 설치됨
- [ ] 가상 환경 생성됨 (권장)
- [ ] requirements.txt 설치됨
- [ ] `python app.py` 실행됨
- [ ] `http://localhost:5000/api/health` 응답함
- [ ] 브라우저에서 `index.html` 열림
- [ ] 자동 훈련 UI 보임
- [ ] 백엔드 상태: ✅ 연결됨

**모두 완료하면 자동 훈련 시작 준비 완료! 🎉**
