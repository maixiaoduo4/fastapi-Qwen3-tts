#!/bin/bash
# Qwen3-TTS FastAPI Server Startup Script
# Usage: ./start_server.sh [OPTIONS]
#
# Options:
#   --host HOST       Host address (default: 0.0.0.0)
#   --port PORT       Port number (default: 8009)
#   --model MODEL     Model path (default: /ubuntu-22.04/Qwen3-tts/Qwen3-TTS-12Hz-1.7B-VoiceDesign)
#   --workers N       Number of workers (default: 1)
#   --gpu GPU_ID      GPU device ID to use (e.g., 0, 1, 2, or 0,1 for multi-GPU)
#   --reload          Enable auto-reload for development
#   -h, --help        Show this help message

set -e

# Default values
HOST="0.0.0.0"
PORT="8009"
MODEL_PATH="/ubuntu-22.04/Qwen3-tts/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
WORKERS=1
RELOAD=""
GPU_ID=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --workers)
            WORKERS="$2"
            shift 2
            ;;
        --reload)
            RELOAD="--reload"
            shift
            ;;
        --gpu)
            GPU_ID="$2"
            shift 2
            ;;
        -h|--help)
            echo "Qwen3-TTS FastAPI Server Startup Script"
            echo ""
            echo "Usage: ./start_server.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --host HOST       Host address (default: 0.0.0.0)"
            echo "  --port PORT       Port number (default: 8009)"
            echo "  --model MODEL     Model path (default: /ubuntu-22.04/Qwen3-tts/Qwen3-TTS-12Hz-1.7B-VoiceDesign)"
            echo "  --workers N       Number of workers (default: 1)"
            echo "  --gpu GPU_ID      GPU device ID to use (e.g., 0, 1, 2, or 0,1 for multi-GPU)"
            echo "  --reload          Enable auto-reload for development"
            echo "  -h, --help        Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Export model path as environment variable
export QWEN3_TTS_MODEL_PATH="$MODEL_PATH"

# Set CUDA_VISIBLE_DEVICES if GPU_ID is specified
if [[ -n "$GPU_ID" ]]; then
    export CUDA_VISIBLE_DEVICES="$GPU_ID"
fi

echo "============================================"
echo "Qwen3-TTS FastAPI Server"
echo "============================================"
echo "Host:       $HOST"
echo "Port:       $PORT"
echo "Model:      $MODEL_PATH"
echo "Workers:    $WORKERS"
echo "GPU:        ${GPU_ID:-auto}"
echo "Script Dir: $SCRIPT_DIR"
echo "============================================"
echo ""

# Start the server
cd "$SCRIPT_DIR"
uvicorn api_server:app \
    --host "$HOST" \
    --port "$PORT" \
    --workers "$WORKERS" \
    $RELOAD
