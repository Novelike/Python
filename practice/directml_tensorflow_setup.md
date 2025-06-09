# DirectML TensorFlow 설정 가이드

이 가이드는 Intel 내장 GPU와 CPU를 함께 사용하여 딥러닝을 수행할 수 있도록 DirectML TensorFlow를 설정하는 방법을 설명합니다.

## DirectML TensorFlow란?

DirectML TensorFlow는 Microsoft의 DirectML 라이브러리를 사용하여 TensorFlow를 실행하는 특별한 버전입니다. 이를 통해 NVIDIA GPU뿐만 아니라 Intel, AMD 등의 다양한 GPU에서도 TensorFlow를 가속화할 수 있습니다.

주요 장점:
- Intel 내장 GPU에서 딥러닝 가속화 가능
- CPU와 GPU를 함께 사용하여 성능 향상
- CUDA가 필요 없음 (NVIDIA GPU가 아닌 시스템에서 유용)

## Conda 가상 환경 설정

### 1. Conda 가상 환경 생성

명령 프롬프트 또는 Anaconda Prompt를 관리자 권한으로 실행한 후 다음 명령어를 입력합니다:

```bash
conda create -n dmltf python=3.10 -y
conda activate dmltf
```

### 2. DirectML TensorFlow 설치

DirectML TensorFlow와 필요한 패키지를 설치합니다:

```bash
pip install tensorflow-directml
pip install numpy matplotlib
```

### 3. 추가 필요 패키지 설치

지뢰찾기 AI 프로젝트에 필요한 추가 패키지를 설치합니다:

```bash
pip install tk
```

## PyCharm에서 Conda 인터프리터 설정

### 1. PyCharm에서 인터프리터 추가

1. `File` > `Settings` > `Project` > `Python Interpreter` 메뉴로 이동합니다.
2. 톱니바퀴 아이콘을 클릭하고 `Add...`를 선택합니다.
3. 왼쪽 패널에서 `Conda Environment`를 선택합니다.
4. `Existing environment`를 선택하고 방금 생성한 `dmltf` 환경을 선택합니다.
   - 일반적인 경로: `C:\Users\<사용자명>\miniconda3\envs\dmltf\python.exe`
5. `OK`를 클릭하여 인터프리터를 추가합니다.

### 2. 프로젝트 인터프리터 설정

1. 추가한 `dmltf` 인터프리터를 선택합니다.
2. `Apply`를 클릭하여 변경사항을 적용합니다.

## 코드 실행

코드는 이미 DirectML TensorFlow를 사용하도록 수정되었습니다. 다음과 같은 변경사항이 적용되었습니다:

1. DirectML 디바이스 감지 및 설정
2. 자동 디바이스 선택 (Intel GPU와 CPU 모두 활용)
3. 오류 처리 개선 및 CPU 폴백 메커니즘 추가

PyCharm에서 `dmltf` 인터프리터를 사용하여 코드를 실행하면 자동으로 Intel 내장 GPU와 CPU를 함께 사용하여 딥러닝을 수행합니다.

## 성능 모니터링

DirectML TensorFlow가 제대로 작동하는지 확인하려면:

1. 작업 관리자를 열고 `성능` 탭으로 이동합니다.
2. CPU와 GPU 사용량을 모니터링합니다.
3. 코드 실행 시 GPU 사용량이 증가하면 DirectML이 제대로 작동하는 것입니다.

## 문제 해결

### DirectML이 GPU를 감지하지 못하는 경우

1. 그래픽 드라이버가 최신 버전인지 확인합니다.
2. Windows 업데이트를 실행하여 DirectX가 최신 버전인지 확인합니다.
3. 코드에 추가된 디버그 정보를 확인합니다 (`TF_DIRECTML_VERBOSE=1` 설정).

### 메모리 오류가 발생하는 경우

Intel 내장 GPU는 메모리가 제한적일 수 있습니다. 다음과 같이 조정해 보세요:

1. 배치 크기를 줄입니다 (예: `batch_size=32` 또는 더 작게).
2. 모델 복잡도를 줄입니다.
3. 혼합 정밀도 훈련을 비활성화합니다 (메모리 사용량 감소).

### CPU 부하가 여전히 높은 경우

1. `update_frequency` 값을 늘려 학습 빈도를 줄입니다.
2. 더 작은 신경망 아키텍처를 사용합니다 (예: ResNet 대신 CNN).
3. 자가 대전 메커니즘의 빈도를 줄입니다.

## 추가 리소스

- [DirectML TensorFlow GitHub](https://github.com/microsoft/tensorflow-directml)
- [DirectML 문서](https://docs.microsoft.com/windows/ai/directml/dml)
- [TensorFlow 문서](https://www.tensorflow.org/guide)