#!/bin/bash
# Deployment script for ipazia server

echo "🚀 Deploying Hugging Face API Server to ipazia..."

# Create directory for the API server
mkdir -p ~/hf_api_server
cd ~/hf_api_server

# Copy files from local machine (run this from your local machine)
echo "📁 Files to transfer:"
echo "  - hf_api_server.py"
echo "  - requirements_api_server.txt"

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements_api_server.txt

# Set environment variables
export HF_MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
export API_HOST="0.0.0.0"
export API_PORT="9000"

# Create systemd service file
echo "🔧 Creating systemd service..."
sudo tee /etc/systemd/system/hf-api.service > /dev/null <<EOF
[Unit]
Description=Hugging Face API Server
After=network.target

[Service]
Type=simple
User=castangia
WorkingDirectory=/home/castangia/hf_api_server
Environment=HF_MODEL_NAME=Qwen/Qwen2.5-7B-Instruct
Environment=API_HOST=0.0.0.0
Environment=API_PORT=9000
ExecStart=/home/castangia/miniconda3/envs/luxia/bin/python hf_api_server.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd and start service
echo "🔄 Starting service..."
sudo systemctl daemon-reload
sudo systemctl enable hf-api
sudo systemctl start hf-api

# Check status
echo "📊 Service status:"
sudo systemctl status hf-api --no-pager

echo "✅ Deployment complete!"
echo "🌐 API available at: http://ipazia126.polito.it:9000"
echo "📖 API docs at: http://ipazia126.polito.it:9000/docs"
