import subprocess
import os
import http.server
import threading
import json
import functools
import uuid

from evaluation.models.models import create_model

class InferenceRequestHandler(http.server.BaseHTTPRequestHandler):
    def __init__(self, model, stop_event, *args, **kwargs):
        self.model = model
        self.stop_event = stop_event
        super().__init__(*args, **kwargs)

    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers.get('Content-Length'))).decode('utf-8'))
        conversation = [(item['role'], item['content']) for item in request['messages']]
        reply = self.model.reply(conversation, stop_event=self.stop_event) # TODO: Temperature? max num tokens?
        self.send_response(200)
        self.end_headers()
        self.wfile.write(json.dumps(reply).encode('utf-8'))

def evaluate_model(model_type, model_name, model_args, evaluation_id):
    os.makedirs('.tmp', exist_ok=True)

    # Clone repository, setup virtual environment

    subprocess.run(['git', 'clone', 'https://github.com/FastEval/AgentBench.git'], cwd='.tmp')
    subprocess.run(['python3', '-m', 'venv', 'venv'], cwd='.tmp/AgentBench')

    new_environment = os.environ.copy()
    new_environment['PATH'] = os.path.join(os.getcwd(), '.tmp/AgentBench/venv/bin') + ':' + os.environ['PATH']
    new_environment['OPENAI_API_BASE'] = 'https://api.openai.com/v1' # For lateral thinking puzzle

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

    # Task: Lateral Thinking Puzzle

    subprocess.run(['pip', 'install', '-r', 'src/tasks/lateralthinkingpuzzle/requirements.txt'], cwd='.tmp/AgentBench', env=new_environment)
    subprocess.run(['pip', 'install', 'openpyxl'], env=new_environment) # Missing dependency

    # Task: AlfWorld

    subprocess.run(['pip', 'install', 'textworld'], env=new_environment) # Missing dependency

    # Task: Mind2Web

    subprocess.run(['pip', 'install', 'addict'], env=new_environment) # Missing dependency
    subprocess.run(['pip', 'install', 'lxml'], env=new_environment) # Missing dependency
    subprocess.run(['pip', 'install', 'datasets'], env=new_environment) # Missing dependency
    subprocess.run(['pip', 'install', 'torch'], env=new_environment) # Missing dependency

    # Task: Knowledge Graph

    # This setup is way too complicated.
    # I'm going to skip this task for now.

    # Run the inference

    model = create_model(model_type, model_name, model_args)

    stop_event = threading.Event()

    server = http.server.ThreadingHTTPServer(('127.0.0.1', 35812), functools.partial(InferenceRequestHandler, model, stop_event))

    threading.Thread(target=server.serve_forever).start()

    agent_file = os.path.join(os.getcwd(), 'evaluation', 'benchmarks', 'agentbench_agent.yaml')

    tasks = {
        # 'os_interaction': 'configs/tasks/os_interaction/dev.yaml', # Mostly runs, but sometimes stuck in the end. Also completely different scores from paper
        # 'dbbench': 'configs/tasks/dbbench/dev.yaml', # MySQL connection error towards the end
        # 'lateral-thinking-puzzle': 'configs/tasks/lateralthinkingpuzzle/dev.yaml', # ZeroDivisionError
        # 'lateral-thinking-puzzle-zh': 'configs/tasks/lateralthinkingpuzzle_zh/dev.yaml', # todo
        # 'knowledge-graph': 'configs/tasks/knowledgegraph/dev.yaml', # Excluded for now
        # 'alfworld': 'configs/tasks/alfworld/dev.yaml', # Doesn't work
        # 'mind2web': 'configs/tasks/mind2web/dev.yaml', # todo
        # 'card-game': 'configs/tasks/card_game/dev.yaml', # Progress bar stuck at 0
    }

    scores = {}
    for task_name, task_config_file in tasks.items():
        output_directory = os.path.abspath(os.path.join('.tmp', 'agentbench', str(uuid.uuid4())))

        subprocess.run([
            'python', 'eval.py',
            '--task', task_config_file,
            '--agent', agent_file,
            '--workers', str(os.cpu_count()),
            '--output', output_directory,
        ], env=new_environment, cwd='.tmp/AgentBench')

        results_file = os.path.join(output_directory, 'results.json')
        with open(results_file) as f:
            results = json.load(f)

        score = results['score']['overall']['acc']
        scores[task_name] = score

    print(scores)

    server.shutdown()
