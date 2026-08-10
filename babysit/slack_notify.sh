#!/bin/bash
# babysit -> Slack. Posts "$*" (Slack mrkdwn) to the destination in notify.txt via chat.postMessage.
#
#   bash slack_notify.sh "message text"
#
# Token from $BABYSIT_SLACK_TOKEN (a Slack bot token with chat:write, plus im:write to DM you). It is
# never printed. A U... destination is resolved to its DM channel via conversations.open. Prints only
# "slack ok"/"slack error"; exit 0 on success. Works from unattended ScheduleWakeup ticks (no
# interactive OAuth). Alternatives: the Slack MCP chat.postMessage, or a slack-api helper.
set -uo pipefail
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MSG="$*"
[ -z "$MSG" ] && { echo "slack_notify: empty message"; exit 2; }
CHAN="$(grep -vE '^[[:space:]]*#|^[[:space:]]*$' "$D/notify.txt" 2>/dev/null | head -1 | tr -d '[:space:]')"
[ -z "$CHAN" ] && { echo "slack_notify: no destination in notify.txt (copy notify.txt.example)"; exit 3; }
TOKEN="${BABYSIT_SLACK_TOKEN:-}"
[ -z "$TOKEN" ] && { echo "slack_notify: set BABYSIT_SLACK_TOKEN to a Slack bot token"; exit 3; }
BABYSIT_TOKEN="$TOKEN" BABYSIT_CHAN="$CHAN" BABYSIT_MSG="$MSG" python3 - <<'PY'
import json, os, sys, urllib.request
tok, chan, msg = os.environ["BABYSIT_TOKEN"], os.environ["BABYSIT_CHAN"], os.environ["BABYSIT_MSG"]
def api(method, payload):
    r = urllib.request.Request("https://slack.com/api/" + method, data=json.dumps(payload).encode(),
        headers={"Authorization": "Bearer " + tok, "Content-type": "application/json; charset=utf-8"})
    return json.load(urllib.request.urlopen(r, timeout=30))
if chan[:1] in "Uu":                       # DM: resolve the member id to its IM channel
    o = api("conversations.open", {"users": chan})
    if not o.get("ok"):
        print("slack error:", o.get("error")); sys.exit(1)
    chan = o["channel"]["id"]
m = api("chat.postMessage", {"channel": chan, "text": msg, "unfurl_links": False})
print("slack ok ->", chan) if m.get("ok") else (print("slack error:", m.get("error")) or sys.exit(1))
PY
