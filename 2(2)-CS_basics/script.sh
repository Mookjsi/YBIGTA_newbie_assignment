#!/bin/bash

# miniconda가 존재하지 않을 경우 설치
if ! command -v conda &>/dev/null; then
    echo "Miniconda가 설치되어 있지 않습니다. 설치를 진행합니다..."
    curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda.sh
    bash miniconda.sh -b -p $HOME/miniconda
    export PATH="$HOME/miniconda/bin:$PATH"
    rm -f miniconda.sh
    echo "Miniconda 설치 완료."
else
    echo "Miniconda가 이미 설치되어 있습니다."
fi

# Conda 환경 생성 및 활성화
ENV_NAME="myenv"
if ! conda env list | grep -q "$ENV_NAME"; then
    echo "Conda 가상환경이 없습니다. $ENV_NAME 가상환경을 생성합니다..."
    conda create -y -n $ENV_NAME python=3.8
else
    echo "$ENV_NAME 가상환경이 이미 존재합니다."
fi

# 가상환경 활성화
echo "가상환경 활성화 중..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

## 건드리지 마세요! ##
python_env=$(python -c "import sys; print(sys.prefix)")
if [[ "$python_env" == *"/envs/myenv"* ]]; then
    echo "가상환경 활성화: 성공"
else
    echo "가상환경 활성화: 실패"
    exit 1
fi

# 필요한 패키지 설치
echo "필요한 패키지를 설치합니다..."
pip install -q mypy

# Submission 폴더 파일 실행
echo "Submission 디렉토리에서 Python 파일 실행 중..."
cd submission || { echo "submission 디렉토리로 이동 실패"; exit 1; }

for file in *.py; do
    # 문제 번호 추출
    problem_number=$(basename "$file" .py)
    
    # 입력 파일 경로 및 출력 파일 경로 정의
    input_file="../input/${problem_number}_input"
    output_file="../output/${problem_number}_output"
    
    # Python 파일 실행
    if [[ -f "$input_file" ]]; then
        echo "Executing $file with input $input_file..."
        python "$file" < "$input_file" > "$output_file"
        echo "Output saved to $output_file."
    else
        echo "Input file $input_file not found. Skipping $file."
    fi
done

# mypy 테스트 실행
echo "mypy 테스트 실행 중..."
mypy *.py

# 가상환경 비활성화
echo "가상환경 비활성화..."
conda deactivate

echo "스크립트 실행 완료!"
