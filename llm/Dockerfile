FROM ghcr.io/ggml-org/llama.cpp:server

COPY Modelfile.gguf /models/plano-orchestrator-4b.gguf

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
