#!/usr/bin/env bash
#
# LitGENE health check
# ---------------------
# Checks that BOTH the backend (genellmweb docker container serving HTTPS on
# :5000) and the public frontend (https://litgene.cs.unm.edu) are up, then
# emails a health-status report to the team.
#
# Run by cron every 12 hours. See crontab for the schedule.
#
# Topology (for reference):
#   public  https://litgene.cs.unm.edu  -> UNM CS Apache proxy (64.106.39.210)
#                                       -> this box's HTTPS backend (64.106.39.56:5000)
#   backend = docker container `genellmweb`, entrypoint python3 app.py -> gunicorn TLS

set -u

# --- configuration -----------------------------------------------------------
RECIPIENTS="asahu@salud.unm.edu kvirupakshappa@salud.unm.edu OMacaulay@salud.unm.edu"
FROM="litgene-health@sahu.cs.unm.edu"
CONTAINER="genellmweb"
BACKEND_URL="https://localhost:5000/"
INFERENCE_URL="https://localhost:5000/submit_prompt"
FRONTEND_URL="https://litgene.cs.unm.edu/"
# Known-good prompt (>3 words, required by /submit_prompt) used to exercise the
# full inference path: GPU embedding, cosine similarity, lazy CSV loads,
# g:Profiler enrichment. A healthy run renders result.html ("LitGENE Predicted
# Results"); any failure renders error.html ("Invalid Input") but still HTTP 200,
# so we must inspect the body, not the status code.
PROBE_PROMPT="breast cancer tumor suppressor gene"
SUCCESS_MARKER="LitGENE Predicted Results"
CURL_TIMEOUT=25
INFERENCE_TIMEOUT=90
LOG_FILE="/data/GENELLM_WEBAPP/scripts/health_check.log"
HOSTNAME_FQDN="$(hostname -f 2>/dev/null || hostname)"

# --- checks ------------------------------------------------------------------
overall_ok=1

# 1. Backend container is running
if docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null | grep -q true; then
    container_state="RUNNING"
    container_status="$(docker ps --filter "name=^${CONTAINER}$" --format '{{.Status}}' 2>/dev/null)"
else
    container_state="NOT RUNNING"
    container_status="$(docker inspect -f '{{.State.Status}}' "$CONTAINER" 2>/dev/null || echo 'container not found')"
    overall_ok=0
fi

# 2. Backend HTTPS responds 200
backend_code="$(curl -sk -o /dev/null -w '%{http_code}' --max-time "$CURL_TIMEOUT" "$BACKEND_URL" 2>/dev/null)"
if [ "$backend_code" = "200" ]; then
    backend_state="OK"
else
    backend_state="FAIL"
    overall_ok=0
fi

# 3. End-to-end inference probe: POST a known prompt and verify the result page
#    comes back (not the error page). This exercises the model, embeddings, the
#    lazily-loaded CSVs and the enrichment call — the only check that proves the
#    backend actually works rather than merely serving a static page.
inference_error=""
probe_body="$(curl -sk --max-time "$INFERENCE_TIMEOUT" --data-urlencode "text=${PROBE_PROMPT}" "$INFERENCE_URL" 2>/dev/null)"
probe_rc=$?
if [ "$probe_rc" -ne 0 ]; then
    inference_state="FAIL"
    inference_error="request failed (curl exit $probe_rc — timeout/no response after ${INFERENCE_TIMEOUT}s)"
    overall_ok=0
elif printf '%s' "$probe_body" | grep -qF "$SUCCESS_MARKER"; then
    inference_state="OK"
else
    inference_state="FAIL"
    overall_ok=0
    # error.html renders the message inside <p class="mb-4">...</p>; extract it
    inference_error="$(printf '%s' "$probe_body" \
        | tr '\n' ' ' \
        | grep -oE '<p class="mb-4">[^<]*</p>' \
        | sed -E 's/<[^>]+>//g' \
        | head -1)"
    [ -z "$inference_error" ] && inference_error="unexpected response (no '$SUCCESS_MARKER' marker; not an error page either)"
fi

# 4. Public frontend responds 200
frontend_code="$(curl -sk -o /dev/null -w '%{http_code}' --max-time "$CURL_TIMEOUT" "$FRONTEND_URL" 2>/dev/null)"
if [ "$frontend_code" = "200" ]; then
    frontend_state="OK"
else
    frontend_state="FAIL"
    overall_ok=0
fi

# --- report ------------------------------------------------------------------
if [ "$overall_ok" -eq 1 ]; then
    subject="[LitGENE] HEALTHY - backend & frontend OK ($HOSTNAME_FQDN)"
    headline="ALL SYSTEMS HEALTHY"
else
    subject="[LitGENE] PROBLEM - service down ($HOSTNAME_FQDN)"
    headline="*** PROBLEM DETECTED - one or more services are down ***"
fi

timestamp="$(date '+%Y-%m-%d %H:%M:%S %Z')"

body="$(cat <<EOF
$headline

Time:  $timestamp
Host:  $HOSTNAME_FQDN

BACKEND (docker container '$CONTAINER', HTTPS :5000)
  container : $container_state ($container_status)
  https     : $backend_state (HTTP ${backend_code:-no-response}) $BACKEND_URL
  inference : $inference_state${inference_error:+ - $inference_error}
              (POST $INFERENCE_URL, prompt: "$PROBE_PROMPT")

FRONTEND (public, via UNM proxy)
  https     : $frontend_state (HTTP ${frontend_code:-no-response}) $FRONTEND_URL

----------------------------------------------------------------------
Quick recovery if backend is down:
  docker restart $CONTAINER      # model load takes ~30-60s before it serves 200

This is an automated message from the LitGENE health-check cron
($0) running every 12 hours on $HOSTNAME_FQDN.
EOF
)"

# --- send + log --------------------------------------------------------------
echo "$body" | mail -s "$subject" -r "$FROM" $RECIPIENTS

echo "[$timestamp] overall_ok=$overall_ok container=$container_state backend=$backend_state($backend_code) inference=$inference_state frontend=$frontend_state($frontend_code)${inference_error:+ inference_error=\"$inference_error\"}" >> "$LOG_FILE"

# exit non-zero on problem so cron/monitoring can also react
[ "$overall_ok" -eq 1 ]
