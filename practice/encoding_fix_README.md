# 한글 인코딩 문제 해결 가이드

## 문제 설명

WSL(Windows Subsystem for Linux)에서 Python 파일을 실행할 때 한글 문자가 `\uc9c0`와 같은 유니코드 이스케이프 시퀀스로 표시되는 문제가 발생했습니다. 이는 Windows와 Linux 간의 파일 인코딩 처리 방식 차이로 인해 발생합니다.

## 원인

- Windows에서는 Python 파일이 UTF-8 with BOM 또는 CP949(한국어 Windows 인코딩)로 저장될 수 있습니다.
- Linux(WSL)에서는 기본적으로 UTF-8 without BOM을 사용합니다.
- 명시적인 인코딩 선언이 없으면, Python 인터프리터는 시스템 기본 인코딩을 사용하여 파일을 해석합니다.
- WSL에서 Windows 파일 시스템에 접근할 때 인코딩 변환 과정에서 문제가 발생할 수 있습니다.

## 해결 방법

다음과 같은 방법으로 문제를 해결했습니다:

1. 모든 Python 파일에 명시적인 인코딩 선언 추가:
   ```python
   # -*- coding: utf-8 -*-
   ```
   
   이 선언은 파일 맨 위에 추가되었으며, Python 인터프리터에게 이 파일이 UTF-8 인코딩으로 작성되었음을 알려줍니다.

2. 수정된 파일 목록:
   - `minesweeper.py`
   - `minesweeper_ai.py`
   - `run_minesweeper_ai.py`

## 추가 권장사항

1. 모든 새로운 Python 파일을 생성할 때 UTF-8 인코딩을 사용하고, 파일 상단에 인코딩 선언을 추가하세요.

2. 에디터 설정:
   - PyCharm: File > Settings > Editor > File Encodings에서 기본 인코딩을 UTF-8로 설정
   - VS Code: 하단 상태 표시줄에서 인코딩을 UTF-8로 설정

3. WSL에서 Python을 실행할 때 환경 변수 설정:
   ```bash
   export PYTHONIOENCODING=utf-8
   ```
   이 설정을 `.bashrc` 또는 `.zshrc` 파일에 추가하면 영구적으로 적용됩니다.

## 참고 자료

- [PEP 263 – Defining Python Source Code Encodings](https://peps.python.org/pep-0263/)
- [Python 공식 문서: 소스 파일 인코딩](https://docs.python.org/3/tutorial/interpreter.html#source-code-encoding)