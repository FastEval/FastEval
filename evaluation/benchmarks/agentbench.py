import subprocess
import os

def install():
    os.makedirs('.tmp', exist_ok=True)

    subprocess.run(['git', 'clone', 'https://github.com/THUDM/AgentBench.git'], cwd='.tmp')
    subprocess.run(['pip', 'install', '--upgrade', 'pip'], cwd='.tmp/AgentBench')
    subprocess.run(['pip', 'install', '-r', 'requirements.txt'], cwd='.tmp/AgentBench')

def evaluate_model(model_type, model_name, model_args, evaluation_id):
    install()
