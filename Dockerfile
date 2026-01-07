FROM us-docker.pkg.dev/llm-exp-405305/batch-images/base:latest

# Install your dependencies
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

# Copy your source code
COPY src/ ./src/
COPY README.md ./
RUN uv sync --frozen