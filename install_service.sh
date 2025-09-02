#!/bin/bash

# Vietnamese ASR Transcription Service Installer
# This script installs and manages the systemd service

SERVICE_NAME="vietnamese-asr"
SERVICE_FILE="vietnamese-asr.service"
SERVICE_PATH="/etc/systemd/system/$SERVICE_FILE"

echo "Vietnamese ASR Transcription Service Installer"
echo "=============================================="

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "Error: This script must be run as root (use sudo)"
    exit 1
fi

# Check if service file exists
if [ ! -f "$SERVICE_FILE" ]; then
    echo "Error: Service file $SERVICE_FILE not found in current directory"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "/home/thinh/venv" ]; then
    echo "Error: Virtual environment not found at /home/thinh/venv"
    echo "Please activate your virtual environment first"
    exit 1
fi

# Check if transcribe_mic.py exists
if [ ! -f "/home/thinh/nemo/vietnamese_asr/example/transcribe_mic.py" ]; then
    echo "Error: transcribe_mic.py not found at /home/thinh/nemo/vietnamese_asr/example/transcribe_mic.py"
    exit 1
fi

echo "Installing service..."

# Copy service file to systemd directory
cp "$SERVICE_FILE" "$SERVICE_PATH"

# Set proper permissions
chmod 644 "$SERVICE_PATH"

# Reload systemd daemon
systemctl daemon-reload

# Enable service to start on boot
systemctl enable "$SERVICE_NAME"

echo "Service installed successfully!"
echo ""
echo "Service management commands:"
echo "  Start service:   sudo systemctl start $SERVICE_NAME"
echo "  Stop service:    sudo systemctl stop $SERVICE_NAME"
echo "  Restart service: sudo systemctl restart $SERVICE_NAME"
echo "  Check status:    sudo systemctl status $SERVICE_NAME"
echo "  View logs:       sudo journalctl -u $SERVICE_NAME -f"
echo "  Disable service: sudo systemctl disable $SERVICE_NAME"
echo ""
echo "To start the service now, run: sudo systemctl start $SERVICE_NAME"
