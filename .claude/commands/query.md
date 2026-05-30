Send a test query to the live EKS deployment and show the answer.

Arguments: $ARGUMENTS (the question to ask — defaults to "What is this document about?")

```bash
QUESTION="${ARGUMENTS:-What is this document about?}"
echo "Asking: $QUESTION"
echo ""
curl -s --max-time 30 -X POST http://finrag.44.206.217.242.nip.io/query \
  -H "Content-Type: application/json" \
  -d "{\"question\": \"$QUESTION\", \"top_k\": 5}" | python3 -m json.tool
```

Show the answer and the source chunks that were retrieved.
