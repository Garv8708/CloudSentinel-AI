@echo off
echo CloudSentinel AI - Windows setup helper
echo Make sure Python 3.10 or 3.11 is installed and added to PATH.
python -m venv .venv
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
echo.
echo Next: put your labeled CSV at data\network_traffic.csv
echo Then run: python -m src.train_model --data data/network_traffic.csv
echo Finally run: streamlit run dashboard\app.py
pause
