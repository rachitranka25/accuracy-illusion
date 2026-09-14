#!/bin/bash
# Unattended OFI collector: keeps the order-flow feed alive for the whole session.
# The order book is only available live (NSE gives no free history), so a dropped
# connection means those minutes are lost forever — this restarts on any crash or
# network blip until the market closes.
cd "$(dirname "$0")"
LOG=/tmp/ofi_daemon.log
echo "=== daemon started $(TZ=Asia/Kolkata date '+%F %H:%M IST') ===" >> "$LOG"

while true; do
  IST_HM=$(TZ=Asia/Kolkata date '+%H%M')
  IST_DOW=$(TZ=Asia/Kolkata date '+%u')          # 1=Mon .. 7=Sun

  # stop after the close, and don't run on weekends
  if [ "$IST_DOW" -ge 6 ] || [ "$IST_HM" -gt 1535 ]; then
    echo "$(TZ=Asia/Kolkata date '+%F %H:%M') market closed — daemon exiting" >> "$LOG"
    exit 0
  fi

  # before the open: idle until 9:15
  if [ "$IST_HM" -lt 0913 ]; then
    sleep 60; continue
  fi

  echo "$(TZ=Asia/Kolkata date '+%F %H:%M') starting collector…" >> "$LOG"
  python3 ofi_collector.py >> "$LOG" 2>&1
  echo "$(TZ=Asia/Kolkata date '+%F %H:%M') collector exited — restarting in 20s" >> "$LOG"
  sleep 20
done
