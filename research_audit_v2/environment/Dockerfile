FROM python:3.12-slim
WORKDIR /work
COPY research_audit_v2/environment/requirements-lock.txt /tmp/requirements-lock.txt
RUN pip install --no-cache-dir -r /tmp/requirements-lock.txt
COPY research_audit_v2 /work/research_audit_v2
CMD ["python", "-m", "research_audit_v2.src.run_all", "--config", "research_audit_v2/configs/development.yaml"]
