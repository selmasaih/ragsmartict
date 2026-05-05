#!/bin/bash
echo "Starting Streamlit App on Azure App Service..."
python -m pip install -r requirements.txt
python -m streamlit run src/app.py --server.port 8000 --server.address 0.0.0.0
