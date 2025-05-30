# Python 학습 및 프로젝트 저장소

이 저장소는 Python 프로그래밍 언어를 학습하고 다양한 프로젝트를 개발하는 작업 공간입니다.

## 프로젝트 개요

이 저장소는 다음과 같은 내용을 포함하고 있습니다:

1. **Python 기초 학습**: 기본 자료구조(리스트, 튜플, 딕셔너리, 집합)와 프로그래밍 개념(함수, 클래스, 반복, 제너레이터, 예외 처리)
2. **고급 Python 주제**: 동시성과 병렬 처리, 파일 시스템 조작
3. **실전 프로젝트**: Steam 스토어 크롤러 등의 실용적인 애플리케이션
4. **일일 연습 코드**: 날짜별로 정리된 연습 코드

## 주요 기능

### Steam 스토어 크롤러 (python.py)

- Steam 스토어에서 할인 중인 게임 정보를 자동으로 수집
- 게임 제목, 가격, 할인율 등의 정보 추출
- 수집된 정보를 JSON 형식으로 저장
- 가격 순으로 정렬된 게임 목록 제공

## 설치 방법

1. 저장소 클론:
   ```
   git clone <저장소 URL>
   ```

2. 필요한 패키지 설치:
   ```
   pip install cloudscraper beautifulsoup4
   ```

## 사용 방법

### Steam 스토어 크롤러 실행:
```
python python.py
```
실행 후 `steam_games.json` 파일에 결과가 저장됩니다.

## 프로젝트 구조

```
Python/
├── .idea/                  # IDE 설정 파일
├── .venv/                  # 가상 환경
├── 과제/                   # 학습 과제
│   ├── __pycache__/
│   ├── log/                # 로그 파일
│   ├── sample/             # 샘플 파일
│   ├── upload/             # 업로드 파일
│   ├── uploads/            # 업로드 파일
│   ├── calculate_module.py # 계산 모듈
│   ├── class.py            # 클래스 예제
│   ├── concurrency_parallelism.py # 동시성 및 병렬 처리
│   ├── dictionary.py       # 딕셔너리 예제
│   ├── exception.py        # 예외 처리
│   ├── fileSystem.py       # 파일 시스템 조작
│   ├── function.py         # 함수 예제
│   ├── generator.py        # 제너레이터 예제
│   ├── iteration.py        # 반복 예제
│   ├── list.py             # 리스트 예제
│   ├── set.py              # 집합 예제
│   ├── students.csv        # 샘플 CSV 파일
│   └── tuple.py            # 튜플 예제
├── myenv/                  # 가상 환경
├── practice/               # 연습 코드
│   ├── __pycache__/
│   ├── backups/            # 백업 파일
│   ├── datas/              # 데이터 파일
│   ├── 20250519.py         # 날짜별 연습 코드
│   ├── 20250520.py
│   └── ...
├── .gitignore              # Git 무시 파일 목록
├── page_source.html        # 크롤링된 페이지 소스
├── python.py               # Steam 스토어 크롤러
├── README.md               # 이 파일
└── steam_games.json        # 크롤링 결과 저장 파일
```

## 의존성

- Python 3.x
- cloudscraper: 웹 크롤링을 위한 라이브러리
- BeautifulSoup4: HTML 파싱을 위한 라이브러리
- 기타 표준 라이브러리: json, datetime 등

## 학습 내용

이 저장소는 다음과 같은 Python 학습 주제를 다룹니다:

1. **기본 자료구조**
   - 리스트, 튜플, 딕셔너리, 집합의 사용법

2. **프로그래밍 개념**
   - 함수, 클래스, 반복, 제너레이터, 예외 처리

3. **고급 주제**
   - 동시성과 병렬 처리
   - 파일 시스템 조작

4. **실전 응용**
   - 웹 크롤링
   - 데이터 처리 및 저장

## 라이선스

이 프로젝트는 개인 학습 목적으로 작성되었습니다.