# -*- coding: utf-8 -*-
"""
이 스크립트는 한글 표시가 제대로 되는지 테스트합니다.
WSL에서 실행할 때 다음과 같이 환경 변수를 설정하세요:
export PYTHONIOENCODING=utf-8
"""

print("=" * 50)
print("한글 표시 테스트")
print("=" * 50)
print("다음 한글 문자들이 제대로 표시되어야 합니다:")
print("가나다라마바사아자차카타파하")
print("안녕하세요, 반갑습니다!")
print("지뢰찾기 게임을 즐겨보세요.")
print("=" * 50)

# 파일 인코딩 테스트
with open("korean_test.txt", "w", encoding="utf-8") as f:
    f.write("한글 파일 쓰기 테스트\n")
    f.write("이 파일이 UTF-8로 저장되었습니다.\n")

print("korean_test.txt 파일이 생성되었습니다.")
print("파일 내용을 확인해보세요.")

# 환경 변수 확인
import os
print("\n환경 변수 정보:")
print(f"PYTHONIOENCODING: {os.environ.get('PYTHONIOENCODING', '설정되지 않음')}")
print(f"LANG: {os.environ.get('LANG', '설정되지 않음')}")
print(f"LC_ALL: {os.environ.get('LC_ALL', '설정되지 않음')}")

# Python 인코딩 정보
import sys
print("\nPython 인코딩 정보:")
print(f"기본 인코딩: {sys.getdefaultencoding()}")
print(f"파일 시스템 인코딩: {sys.getfilesystemencoding()}")
print(f"표준 출력 인코딩: {sys.stdout.encoding}")