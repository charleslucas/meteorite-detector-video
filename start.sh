#!/usr/bin/env bash
# macOS/Linux launcher — equivalent of start.bat
# Opens the UI in Google Chrome (Safari's HTTPS-Only mode blocks http://localhost).
set -e
cd "$(dirname "$0")"

# Use the local virtual environment if present, otherwise fall back to system python3.
if [ -d ".venv" ]; then
  source .venv/bin/activate
fi

command -v python3 >/dev/null 2>&1 || { echo "Python 3 not found."; exit 1; }
python3 -c "import flask" 2>/dev/null || pip install flask

PORT=7860
URL="http://127.0.0.1:${PORT}"

# Open Chrome once the server has had a moment to start (server startup is foreground).
if [ -d "/Applications/Google Chrome.app" ]; then
  ( sleep 1.5; open -a "Google Chrome" "$URL" ) &
else
  echo "Google Chrome not found — open $URL manually."
  ( sleep 1.5; open "$URL" ) &
fi

# --no-browser so app.py doesn't also launch the default (Safari) browser.
python3 ui/app.py --no-browser --port "$PORT" "$@"
