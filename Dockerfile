FROM rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0

WORKDIR /app

COPY app/requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir \
    pymupdf \
    openpyxl \
    python-docx \
    pillow \
    accelerate \
    huggingface-hub \
    packaging \
    pyyaml \
    regex \
    requests \
    safetensors \
    tokenizers \
    tqdm

RUN pip install --no-cache-dir --no-deps qwen-vl-utils -U transformers

RUN python3 -c "import torch, torchvision; print('Torch:', torch.__version__, 'Torchvision:', torchvision.__version__); assert 'rocm' in torch.__version__, 'CUDA torch detected!'"

RUN python3 -c "from huggingface_hub import snapshot_download; \
    snapshot_download(repo_id='Qwen/Qwen2.5-VL-7B-Instruct', local_dir='/models/Qwen2.5-VL-7B-Instruct')"

COPY app/ /app/

RUN chmod +x /app/app.py /app/server.py && \
    rm -rf /app/.ipynb_checkpoints && \
    mkdir -p /app/corpus /app/output /app/index

CMD ["python3", "/app/server.py"]
