# Deploy this TensorFold checkout to spark0. Override targets with environment variables.
set shell := ["bash", "-euo", "pipefail", "-c"]
head := env("GLM53_HEAD", "colum@100.95.49.21")
worker := env("GLM53_WORKER", "colum@10.50.0.11")
remote := env("GLM53_REMOTE_DIR", "glm53-tensorfold")
port := env("GLM53_PORT", "8000")
ssh := "ssh -F /dev/null -o BatchMode=yes -o ConnectTimeout=20"

default:
    @just --list --unsorted

# Copy the checkout; preserve remote configuration, credentials, and logs.
sync:
    rsync -a --delete --exclude .git --exclude .agents --exclude .codex --exclude .aws --exclude .env --exclude scripts/local.sh --exclude logs --exclude __pycache__ --exclude .pytest_cache ./ {{head}}:{{remote}}/

# Initialize the deployment profile once, without overwriting local changes.
setup: sync
    {{ssh}} {{head}} 'cd ~/{{remote}} && if [ ! -e .env ]; then cp .env.spark.example .env; echo "Created .env; configure HF_TOKEN before launch."; else echo "Keeping existing .env and scripts/local.sh."; fi'

# Start from spark0; the upstream launcher prepares both nodes and waits for health.
start: setup
    {{ssh}} -t {{head}} 'cd ~/{{remote}} && ./start.sh'

restart: setup
    {{ssh}} -t {{head}} 'cd ~/{{remote}} && ./start.sh restart'

stop:
    {{ssh}} {{head}} 'cd ~/{{remote}} && ./stop.sh'

status:
    {{ssh}} {{head}} 'docker ps --filter name=glm53-flash-tf; curl -fsS -m 3 http://127.0.0.1:{{port}}/health'
    {{ssh}} {{head}} 'ssh -o BatchMode=yes {{worker}} docker ps --filter name=glm53-flash-tf'

logs:
    {{ssh}} {{head}} 'docker logs --tail 100 glm53-flash-tf 2>&1'

# Run functional inference, prefix reuse, and four concurrent requests on spark0.
verify:
    {{ssh}} {{head}} 'cd ~/{{remote}} && python3 tools/verify_deployment.py --base http://127.0.0.1:{{port}}'
