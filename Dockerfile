FROM rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0

WORKDIR /app

COPY app/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --no-deps qwen-vl-utils || true
RUN pip install --no-cache-dir -r /app/requirements.txt

RUN python3 -c "from huggingface_hub import snapshot_download; \
    snapshot_download(repo_id='Qwen/Qwen2.5-VL-7B-Instruct', local_dir='/models/Qwen2.5-VL-7B-Instruct')"

COPY app/ /app/

RUN mkdir -p /app/corpus /app/output /app/index

CMD ["sleep", "infinity"]
