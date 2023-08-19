import subprocess
import os
import http.server
import threading

from evaluation.models.models import create_model

def install():
    os.makedirs('.tmp', exist_ok=True)

    subprocess.run(['git', 'clone', 'https://github.com/THUDM/AgentBench.git'], cwd='.tmp')
    subprocess.run(['pip', 'install', '--upgrade', 'pip'], cwd='.tmp/AgentBench')
    subprocess.run(['pip', 'install', '-r', 'requirements.txt'], cwd='.tmp/AgentBench')

class InferenceRequestHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        print(self.path)

def run_inference_server(model_type, model_name, model_args):
    model = create_model(model_type, model_name, model_args)

    server = http.server.HTTPServer(('127.0.0.1', 35812), InferenceRequestHandler)
    server.serve_forever()

def run_inference_server_in_separate_thread(model_type, model_name, model_args):
    threading.Thread(target=run_inference_server, args=(model_type, model_name, model_args)).start()

def evaluate_model(model_type, model_name, model_args, evaluation_id):
    install()
    run_inference_server_in_separate_thread(model_type, model_name, model_args)
