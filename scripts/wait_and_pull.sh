#!/bin/sh
# Wait for the GDELT block to end, then pull. One test request each 10 minutes.
cd "$HOME/projects/perceptual-bias" || exit 1
i=0
while [ $i -lt 18 ]; do
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 45 \
    -A 'perceptual-bias-research/1.0' \
    'https://api.gdeltproject.org/api/v2/doc/doc?query=inflation%20sourcelang:english&mode=artlist&maxrecords=10&format=CSV&startdatetime=20240812000000&enddatetime=20240812235959')
  echo "$(date '+%H:%M:%S') probe -> HTTP $code"
  if [ "$code" = "200" ]; then
    echo "block cleared; starting pull"
    exec .venv/bin/python scripts/pull_gdelt.py 2017-01 2026-08
  fi
  i=$((i+1))
  sleep 600
done
echo "still blocked after 3 hours"
