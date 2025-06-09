# 한글 표시 문제 해결 가이드

## 문제 설명

WSL(Windows Subsystem for Linux)에서 Python 파일을 실행할 때 한글 문자가 `\uc9c0`와 같은 유니코드 이스케이프 시퀀스로 표시되는 문제가 발생할 수 있습니다. 이는 Windows와 Linux 간의 파일 인코딩 처리 방식 차이로 인해 발생합니다.

## 해결 방법

### 1. 파일 인코딩 선언 추가

모든 한글이 포함된 Python 파일에 다음과 같은 인코딩 선언을 파일 맨 위에 추가합니다:

```python
# -*- coding: utf-8 -*-
```

이 선언은 Python 인터프리터에게 이 파일이 UTF-8 인코딩으로 작성되었음을 알려줍니다.

### 2. 환경 변수 설정

WSL에서 Python을 실행할 때 다음 환경 변수를 설정하여 입출력 인코딩을 UTF-8로 지정합니다:

```bash
export PYTHONIOENCODING=utf-8
```

이 설정을 영구적으로 적용하려면 `.bashrc` 또는 `.zshrc` 파일에 추가하세요:

```bash
echo 'export PYTHONIOENCODING=utf-8' >> ~/.bashrc
source ~/.bashrc
```

### 3. 테스트 스크립트 실행

제공된 테스트 스크립트를 실행하여 한글 표시가 제대로 되는지 확인합니다:

```bash
python test_korean_display.py
```

이 스크립트는 다양한 한글 문자를 출력하고, 환경 변수 설정과 Python 인코딩 정보를 표시합니다.

## 수정된 파일 목록

다음 파일들에 인코딩 선언이 추가되었습니다:

- `minesweeper.py` (기존에 추가됨)
- `minesweeper_ai.py` (기존에 추가됨)
- `run_minesweeper_ai.py` (기존에 추가됨)
- `20250605.py` (새로 추가됨)

## 추가 권장사항

1. 모든 새로운 Python 파일을 생성할 때 UTF-8 인코딩을 사용하고, 파일 상단에 인코딩 선언을 추가하세요.

2. 에디터 설정:
   - PyCharm: File > Settings > Editor > File Encodings에서 기본 인코딩을 UTF-8로 설정
   - VS Code: 하단 상태 표시줄에서 인코딩을 UTF-8로 설정

3. WSL에서 한글 표시를 위한 로케일 설정:
   ```bash
   sudo apt-get update
   sudo apt-get install -y locales
   sudo locale-gen ko_KR.UTF-8
   echo 'export LANG=ko_KR.UTF-8' >> ~/.bashrc
   source ~/.bashrc
   ```

## 문제 해결

한글 표시 문제가 계속 발생하는 경우:

1. 터미널 자체가 UTF-8을 지원하는지 확인하세요.
2. `locale` 명령어로 현재 로케일 설정을 확인하세요.
3. Python 버전이 3.x인지 확인하세요 (Python 2는 기본적으로 ASCII를 사용).
4. 파일이 실제로 UTF-8로 저장되었는지 확인하세요 (BOM 없이).

## 참고 자료

- [PEP 263 – Defining Python Source Code Encodings](https://peps.python.org/pep-0263/)
- [Python 공식 문서: 소스 파일 인코딩](https://docs.python.org/3/tutorial/interpreter.html#source-code-encoding)
- [WSL 문서: 로케일 설정](https://docs.microsoft.com/windows/wsl/setup/environment#set-up-your-linux-username-and-password)