import subprocess
import os
import http.server
import threading
import json
import functools

from evaluation.models.models import create_model

class InferenceRequestHandler(http.server.BaseHTTPRequestHandler):
    def __init__(self, model, *args, **kwargs):
        self.model = model
        super().__init__(*args, **kwargs)

    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers.get('Content-Length'))).decode('utf-8'))
        conversation = [(item['role'], item['content']) for item in request['messages']]
        reply = self.model.reply(conversation) # TODO: Temperature? max num tokens?
        self.send_response(200)
        self.end_headers()
        self.wfile.write(json.dumps(reply).encode('utf-8'))

def evaluate_model(model_type, model_name, model_args, evaluation_id):
    os.makedirs('.tmp', exist_ok=True)

    # Clone repository, setup virtual environment

    subprocess.run(['git', 'clone', 'https://github.com/THUDM/AgentBench.git'], cwd='.tmp')
    subprocess.run(['python3', '-m', 'venv', 'venv'], cwd='.tmp/AgentBench')

    new_environment = os.environ.copy()
    new_environment['PATH'] = os.path.join(os.getcwd(), '.tmp/AgentBench/venv/bin') + ':' + os.environ['PATH']

    subprocess.run(['pip', 'install', '--upgrade', 'pip'], env=new_environment)
    subprocess.run(['pip', 'install', '-r', 'requirements.txt'], cwd='.tmp/AgentBench', env=new_environment)

    # Build docker containers for tasks that require them

    subprocess.run(['bash', 'scripts/build_docker.sh'], cwd='.tmp/AgentBench')

    # Task: Operating System

    subprocess.run(['pip', 'install', '-r', 'src/tasks/os_interaction/requirements.txt'], cwd='.tmp/AgentBench', env=new_environment)

    subprocess.run(['python', 'src/tasks/os_interaction/images.py', 'build',
        '-c', 'configs/tasks/os_interaction/dev.yaml', '-r', '.'], cwd='.tmp/AgentBench', env=new_environment)

    # Task: DataBase

    subprocess.run(['docker', 'pull', 'mysql'])

    subprocess.run(['pip', 'install', '-r', 'src/tasks/dbbench/requirements.txt'], cwd='.tmp/AgentBench', env=new_environment)

    # Task: Knowledge Graph

    # This setup is way too complicated.
    # I'm going to skip this task for now.

    # Run the inference

    model = create_model(model_type, model_name, model_args)

    server = http.server.ThreadingHTTPServer(('127.0.0.1', 35812), functools.partial(InferenceRequestHandler, model))

    threading.Thread(target=server.serve_forever).start()

    agent_file = os.path.join(os.getcwd(), 'evaluation', 'benchmarks', 'agentbench_agent.yaml')

    subprocess.run(['python', 'eval.py', '--task', 'configs/tasks/example.yaml',
        '--agent', agent_file], env=new_environment, cwd='.tmp/AgentBench')

    server.shutdown()
