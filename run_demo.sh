#!/usr/bin/env bash
cd "$(dirname "$0")"
[ -d .venv ] && source .venv/bin/activate
python -m aria_enforce.app
