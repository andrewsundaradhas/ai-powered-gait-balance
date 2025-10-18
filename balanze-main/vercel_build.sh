#!/bin/bash

# Install Python dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir -p ~/.streamlit/

# Create Streamlit config
cat << EOF > ~/.streamlit/config.toml
[server]
port = 3000
enableCORS = false
enableXsrfProtection = false

[browser]
serverAddress = "0.0.0.0"
serverPort = 3000

[theme]
base = "dark"

[runner]
magicEnabled = false

[logger]
level = "debug"

[client]
showErrorDetails = true

[deprecation]
showPyplotGlobalUse = false

[sessionManager]
enableCORS = false

[server]
enableCORS = false

EOF

echo "Build completed successfully!"
