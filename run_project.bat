@echo off
python -m pip install -r requirements.txt
python src\train.py
streamlit run app.py
