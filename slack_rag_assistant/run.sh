#!/bin/bash

# Load env variables
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi

# Start the Slack app via Socket Mode
echo "Starting Slack RAG Assistant..."
python main.py
