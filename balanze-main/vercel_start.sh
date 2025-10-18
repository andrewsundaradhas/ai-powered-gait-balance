#!/bin/bash

# Start FastAPI in the background
uvicorn app.main:app --host 0.0.0.0 --port 8000 &

# Start Streamlit in the foreground
streamlit run app/streamlit_dashboard.py --server.port 3000 --server.address 0.0.0.0 --server.enableCORS false --server.enableXsrfProtection false
