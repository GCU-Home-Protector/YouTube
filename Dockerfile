# Python 3.9 기반 이미지 사용
FROM python:3.9-slim

# 작업 디렉토리 설정
WORKDIR /app

# 의존성 파일들을 컨테이너에 복사
COPY requirements.txt .

# 의존성 설치
RUN pip install --no-cache-dir -r requirements.txt

# Flask 앱 코드를 컨테이너에 복사
COPY . /app

# Flask가 사용할 포트 설정
EXPOSE 5000

# 앱 실행 명령어 설정
ENTRYPOINT ["python", "./main/app.py"]