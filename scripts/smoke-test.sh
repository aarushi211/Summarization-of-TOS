#!/usr/bin/env bash
set -euo pipefail

SERVICE_NAME="$1"
REGION="$2"
PROJECT_ID="$3"

REVISION_URL=$(gcloud run services describe "$SERVICE_NAME" \
  --region="$REGION" \
  --project="$PROJECT_ID" \
  --format='json' \
  | python3 -c "
import sys, json
svc = json.load(sys.stdin)
for t in svc.get('status', {}).get('traffic', []):
    if t.get('tag') == 'candidate':
        print(t.get('url', ''))
        break
")

if [ -z "$REVISION_URL" ]; then
  echo "ERROR: Could not resolve tagged revision URL for 'candidate'."
  exit 1
fi

echo "Smoke testing: $REVISION_URL/health"

for i in $(seq 1 12); do
  ID_TOKEN=$(gcloud auth print-identity-token --audiences="$REVISION_URL")
  STATUS=$(curl -s -o /dev/null -w "%{http_code}" \
    -H "Authorization: Bearer $ID_TOKEN" \
    "$REVISION_URL/health" 2>/dev/null || echo "000")
  echo "Attempt $i — HTTP $STATUS"
  if [ "$STATUS" = "200" ]; then
    echo "Health check passed."
    exit 0
  fi
  sleep 15
done

echo "Health check failed after all retries — aborting traffic migration."
exit 1