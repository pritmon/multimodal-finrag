Check the health of all live deployments.

```bash
echo "=== EKS (Primary) ==="
curl -s --max-time 10 http://finrag.44.206.217.242.nip.io/health | python3 -m json.tool

echo ""
echo "=== ECS ==="
curl -s --max-time 10 http://13.222.137.204:8000/health | python3 -m json.tool
```

Report the status of each deployment — is it up? Is the index loaded?
