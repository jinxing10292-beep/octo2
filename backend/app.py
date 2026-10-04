"""
Octo AI - 자동 훈련 백엔드
한국어 문장을 생성하고 크롤링하여 제공
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import requests
from bs4 import BeautifulSoup
import random
import re
from datetime import datetime

app = Flask(__name__)
CORS(app)

# ===== 한국어 문장 데이터 =====

# 1. 기본 한국어 문장 데이터베이스 (크롤링 없이 작동)
KOREAN_SENTENCES = [
    # 일상 회화
    "안녕하세요. 반갑습니다.",
    "오늘 날씨가 정말 좋네요.",
    "어제는 비가 많이 내렸어요.",
    "커피 한 잔 마실래요?",
    "지금 몇 시예요?",
    "어디 가세요?",
    "무엇을 도와드릴까요?",
    "다시 만나요.",
    "잘 가세요.",
    "감사합니다.",
    
    # 기술 관련
    "인공지능은 미래 기술입니다.",
    "기계학습 모델을 훈련합니다.",
    "신경망은 복잡한 패턴을 학습합니다.",
    "데이터는 매우 중요합니다.",
    "알고리즘을 최적화합니다.",
    
    # 일상 활동
    "아침에 일어나서 준비합니다.",
    "회사에 출근했어요.",
    "점심시간이 되었습니다.",
    "저녁 시간에 산책을 합니다.",
    "밤에 공부를 합니다.",
    
    # 감정 표현
    "정말 기뻐요.",
    "조금 슬픕니다.",
    "화가 났어요.",
    "걱정이 됩니다.",
    "행복한 날이에요.",
    
    # 질문과 답변
    "뭘 하고 있어요?",
    "어떻게 지내세요?",
    "뭐 하고 싶어요?",
    "언제 만날 거예요?",
    "왜 그래요?",
    
    # 계절과 날씨
    "봄이 왔습니다.",
    "여름은 덥습니다.",
    "가을은 아름답습니다.",
    "겨울에는 눈이 옵니다.",
    "날씨가 구름이 많네요.",
    
    # 학습과 교육
    "한국어를 배우고 있습니다.",
    "새로운 기술을 공부합니다.",
    "책을 읽으면서 배웁니다.",
    "강의를 들으면서 공부합니다.",
    "친구와 함께 공부합니다.",
    
    # 음식과 식사
    "아침 밥을 먹었어요.",
    "맛있는 음식을 좋아합니다.",
    "한식을 자주 먹습니다.",
    "식당에서 저녁을 먹습니다.",
    "물을 마시고 싶어요.",
    
    # 여행과 관광
    "여행을 가고 싶습니다.",
    "서울에 가본 적이 있어요.",
    "경주는 아름다운 도시입니다.",
    "제주도에 가고 싶어요.",
    "부산은 해변이 좋습니다.",
    
    # 일과 직업
    "회사원입니다.",
    "프로그래머로 일합니다.",
    "학생이에요.",
    "선생님입니다.",
    "의사가 되고 싶어요.",
]

# 2. 온라인 뉴스/블로그 크롤링 함수
def crawl_korean_sentences(source="naver_news"):
    """
    한국어 웹에서 문장을 크롤링합니다.
    경고: 각 사이트의 robots.txt와 이용약관을 준수하세요.
    """
    sentences = []
    
    if source == "naver_news":
        try:
            # Naver 뉴스 제목 크롤링 (데모용)
            url = "https://news.naver.com/"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            response = requests.get(url, headers=headers, timeout=5)
            response.encoding = 'utf-8'
            
            soup = BeautifulSoup(response.text, 'lxml')
            
            # 뉴스 제목 추출 (셀렉터는 사이트 구조 변경 시 업데이트 필요)
            for item in soup.select('a.nclicks')[:10]:
                title = item.get_text(strip=True)
                if title and len(title) > 5:
                    sentences.append(title)
        except Exception as e:
            print(f"Naver 크롤링 오류: {e}")
    
    elif source == "wiki":
        try:
            # 위키피디아 한국어 랜덤 문장
            url = "https://ko.wikipedia.org/wiki/특수:임의_페이지"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            response = requests.get(url, headers=headers, timeout=5)
            response.encoding = 'utf-8'
            
            soup = BeautifulSoup(response.text, 'lxml')
            
            # 첫 번째 문단의 문장들 추출
            paragraphs = soup.select('p')
            for p in paragraphs[:3]:
                text = p.get_text(strip=True)
                # 문장 분리 (마침표, 물음표, 느낌표 기준)
                for sentence in re.split(r'[.!?\n]+', text):
                    sentence = sentence.strip()
                    if sentence and len(sentence) > 5 and len(sentence) < 200:
                        sentences.append(sentence)
        except Exception as e:
            print(f"Wiki 크롤링 오류: {e}")
    
    return sentences


# ===== REST API =====

@app.route('/api/health', methods=['GET'])
def health():
    """헬스 체크"""
    return jsonify({
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    })


@app.route('/api/sentences', methods=['GET'])
def get_sentences():
    """
    한국어 문장을 반환합니다.
    쿼리 파라미터:
    - count: 반환할 문장 수 (기본값: 5, 최대: 100)
    - source: 'builtin' (기본값), 'naver_news', 'wiki'
    """
    count = min(int(request.args.get('count', 5)), 100)
    source = request.args.get('source', 'builtin')
    
    sentences = []
    
    # 기본 문장 데이터베이스에서 선택
    if source == 'builtin' or source == 'all':
        sentences.extend(random.sample(KOREAN_SENTENCES, min(count, len(KOREAN_SENTENCES))))
    
    # 웹 크롤링 추가 (선택)
    if source != 'builtin':
        crawled = crawl_korean_sentences(source)
        if crawled:
            sentences.extend(crawled[:count - len(sentences)])
    
    return jsonify({
        "count": len(sentences),
        "sentences": sentences,
        "source": source,
        "timestamp": datetime.now().isoformat()
    })


@app.route('/api/batch', methods=['GET'])
def get_batch():
    """
    훈련용 배치 데이터를 반환합니다.
    자동 훈련에 최적화된 형식
    """
    batch_size = min(int(request.args.get('batch_size', 10)), 50)
    
    batch_sentences = random.sample(KOREAN_SENTENCES, min(batch_size, len(KOREAN_SENTENCES)))
    
    return jsonify({
        "batch_size": len(batch_sentences),
        "sentences": batch_sentences,
        "timestamp": datetime.now().isoformat()
    })


@app.route('/api/sentence', methods=['GET'])
def get_single_sentence():
    """단일 한국어 문장을 반환합니다. (실시간 훈련용)"""
    sentence = random.choice(KOREAN_SENTENCES)
    
    return jsonify({
        "sentence": sentence,
        "length": len(sentence),
        "timestamp": datetime.now().isoformat()
    })


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """백엔드 통계"""
    return jsonify({
        "total_sentences": len(KOREAN_SENTENCES),
        "avg_length": sum(len(s) for s in KOREAN_SENTENCES) / len(KOREAN_SENTENCES),
        "max_length": max(len(s) for s in KOREAN_SENTENCES),
        "min_length": min(len(s) for s in KOREAN_SENTENCES),
        "timestamp": datetime.now().isoformat()
    })


# ===== 에러 핸들링 =====

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Not found"}), 404


@app.errorhandler(500)
def server_error(error):
    return jsonify({"error": "Server error"}), 500


if __name__ == '__main__':
    print("🚀 Octo AI 자동 훈련 백엔드 시작")
    print("📍 API 주소: http://localhost:5000")
    print("📚 문장 API: http://localhost:5000/api/sentences")
    print("📊 통계: http://localhost:5000/api/stats")
    print("\n⚠️  CORS 활성화: 브라우저에서 접근 가능")
    app.run(debug=True, host='localhost', port=5000)
