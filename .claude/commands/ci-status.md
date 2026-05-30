Check the latest GitHub Actions CI run status for this repo.

```bash
curl -s "https://api.github.com/repos/pritmon/multimodal-finrag/actions/runs?per_page=5" \
  | python3 -c "
import sys, json
data = json.load(sys.stdin)
for run in data.get('workflow_runs', []):
    status = run['conclusion'] or run['status']
    symbol = '✅' if status == 'success' else ('❌' if status == 'failure' else '🔄')
    print(f\"{symbol} [{status.upper()}] {run['name']} — {run['head_commit']['message'][:60]}\")
    print(f\"   Branch: {run['head_branch']} | {run['created_at']}\")
    print(f\"   URL: {run['html_url']}\")
    print()
"
```

Tell me if CI is passing or failing and which commit caused any failure.
