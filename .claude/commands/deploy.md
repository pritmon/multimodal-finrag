Build and push the Docker image to ECR, then roll out to EKS.

Steps:
1. Build the Docker image
2. Tag and push to ECR
3. Restart the EKS deployment to pull the latest image
4. Wait for rollout to complete
5. Run a health check

```bash
cd "/Users/pritammondal/CLAUDE CODE/multimodal-finrag"

echo "=== Building Docker image ==="
docker build -t finrag-api .

echo "=== Tagging ==="
docker tag finrag-api:latest 020262236277.dkr.ecr.us-east-1.amazonaws.com/finrag-api:latest

echo "=== Logging into ECR ==="
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 020262236277.dkr.ecr.us-east-1.amazonaws.com

echo "=== Pushing to ECR ==="
docker push 020262236277.dkr.ecr.us-east-1.amazonaws.com/finrag-api:latest

echo "=== Rolling out on EKS ==="
kubectl rollout restart deployment/finrag-api -n finrag
kubectl rollout status deployment/finrag-api -n finrag --timeout=300s

echo "=== Health check ==="
sleep 10
curl -s http://finrag.44.206.217.242.nip.io/health | python3 -m json.tool
```

Report success or failure at each step.
