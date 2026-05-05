#!/usr/bin/env python3
"""
ARIES — Wazuh Custom Integration Script (Python logic).

Wazuh's integratord daemon calls the bash wrapper (custom-aries),
which invokes this script using Wazuh's embedded Python.

The alert JSON arrives as a temp file whose path is passed as the
first CLI argument.

This script:
  1. Reads the Wazuh alert JSON from the file.
  2. Wraps it in the ARIES ingestion envelope.
  3. POSTs to the ARIES ingestion endpoint (POST /ingest/siem?vendor=wazuh).

Configuration is via environment variables or the constants below.

Exit codes:
  0 — success
  1 — transient failure (Wazuh will retry)
"""

import json
import os
import sys
import urllib.request
import urllib.error

# ── Configuration ────────────────────────────────────────────────────
# When running inside Docker, the FastAPI service is reachable by container name.
# Override with env var ARIES_INGESTION_URL if needed.
ARIES_INGESTION_URL = os.environ.get(
    "ARIES_INGESTION_URL",
    "http://aries_fastapi:8000/ingest/siem?vendor=wazuh",
)
ARIES_TENANT_ID = os.environ.get("ARIES_TENANT_ID", "default")

# Wazuh passes the alert file path as the first CLI argument
# and optionally an API key as the second argument.
ALERT_FILE = sys.argv[1] if len(sys.argv) > 1 else None


def read_alert() -> dict:
    """Read the Wazuh alert JSON.

    Wazuh 4.x writes the alert to a temporary file and passes
    the path as the first argument to the integration script.
    """
    if ALERT_FILE and os.path.isfile(ALERT_FILE):
        with open(ALERT_FILE, "r") as f:
            return json.load(f)

    # Fallback: try reading from stdin (older Wazuh / testing)
    raw = sys.stdin.read()
    if raw.strip():
        return json.loads(raw)

    sys.stderr.write("custom-aries: no alert data received\n")
    sys.exit(1)


def forward_to_aries(alert: dict) -> None:
    """POST the alert to the ARIES ingestion endpoint."""
    payload = json.dumps({
        "vendor": "wazuh",
        "tenant_id": ARIES_TENANT_ID,
        "raw": alert,
    }).encode("utf-8")

    req = urllib.request.Request(
        ARIES_INGESTION_URL,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "X-Tenant-ID": ARIES_TENANT_ID,
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = resp.read().decode("utf-8")
            result = json.loads(body)
            alert_id = result.get("alert_id", "unknown")
            sys.stdout.write(
                f"custom-aries: alert forwarded successfully "
                f"(alert_id={alert_id}, status={resp.status})\n"
            )
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8", errors="replace")
        sys.stderr.write(
            f"custom-aries: HTTP {e.code} from ARIES — {error_body}\n"
        )
        sys.exit(1)
    except urllib.error.URLError as e:
        sys.stderr.write(
            f"custom-aries: connection error — {e.reason}\n"
        )
        sys.exit(1)
    except Exception as e:
        sys.stderr.write(f"custom-aries: unexpected error — {e}\n")
        sys.exit(1)


def main() -> None:
    alert = read_alert()

    # Wazuh wraps alerts in a top-level key depending on version
    if "alert" in alert:
        alert = alert["alert"]

    forward_to_aries(alert)


if __name__ == "__main__":
    main()
