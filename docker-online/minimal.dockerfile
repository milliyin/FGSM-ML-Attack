FROM public.ecr.aws/lambda/python:3.10

# Install basic packages first
RUN pip install --no-cache-dir --target ${LAMBDA_TASK_ROOT} \
    fastapi==0.116.2 \
    mangum==0.17.0 \
    pillow==10.2.0 \
    numpy==1.26.4 \
    python-multipart==0.0.20

# Install PyTorch CPU-only using direct wheel URLs
RUN pip install --no-cache-dir --target ${LAMBDA_TASK_ROOT} \
    https://download.pytorch.org/whl/cpu/torch-2.6.0%2Bcpu-cp310-cp310-linux_x86_64.whl

# Install torchvision CPU-only
RUN pip install --no-cache-dir --target ${LAMBDA_TASK_ROOT} \
    https://download.pytorch.org/whl/cpu/torchvision-0.21.0%2Bcpu-cp310-cp310-linux_x86_64.whl

# Copy your application files
COPY main.py ${LAMBDA_TASK_ROOT}/
COPY fgsm.py ${LAMBDA_TASK_ROOT}/
COPY mnist_model.pt ${LAMBDA_TASK_ROOT}/

# Final cleanup
RUN find ${LAMBDA_TASK_ROOT} -name "*.pyc" -delete
RUN find ${LAMBDA_TASK_ROOT} -name "__pycache__" -type d -exec rm -rf {} + || true

# Verify PyTorch installation
RUN python3 -c "import sys; sys.path.insert(0, '/var/task'); import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

CMD ["main.lambda_handler"]