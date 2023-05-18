sudo apt update
sudo apt install -y git git-lfs python3.10 python3.10-venv python3.10-dev

python3.10 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install torch transformers accelerate
pip install git+https://github.com/tju01/evals.git#egg=evals
pip install --upgrade sacrebleu
