# scripts/bootstrap.sh
set -e
python -m pip install -r requirements.txt
python -m agents.rl_train
python -m evaluation.evaluate
streamlit run ui/app.py