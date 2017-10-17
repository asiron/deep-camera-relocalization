#!/bin/bash

COMMAND="bash train.sh"
CHECK=$(ps aux | grep "${COMMAND}" | wc -l)

if [[ $CHECK -ne 2 ]]; then
  echo "$(date) Training stopped. Restarting training..."
  $COMMAND
else
  echo "$(date) Training already running."
fi