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

        reply = self.model.reply(
            conversation,
            temperature=0,
            max_new_tokens=128,
            stop_event=self.stop_event,
        )

        self.send_response(200)
        self.end_headers()
        self.wfile.write(json.dumps(reply).encode('utf-8'))

def install():
    new_environment = os.environ.copy()
    new_environment['OPENAI_API_BASE'] = 'https://api.openai.com/v1' # For lateral thinking puzzle

    if os.path.exists('.tmp/AgentBench/installation-is-done'):
        new_environment['PATH'] = os.path.join(os.getcwd(), '.tmp/AgentBench/venv/bin') + ':' + os.environ['PATH']
        return new_environment

    os.makedirs('.tmp', exist_ok=True)

    # Clone repository, setup virtual environment

    subprocess.run(['git', 'clone', 'https://github.com/THUDM/AgentBench.git'], cwd='.tmp')
    subprocess.run(['python3', '-m', 'venv', 'venv'], cwd='.tmp/AgentBench')

    new_environment['PATH'] = os.path.join(os.getcwd(), '.tmp/AgentBench/venv/bin') + ':' + os.environ['PATH']

    subprocess.run(['pip', 'install', '--upgrade', 'pip'], env=new_environment)
    subprocess.run(['pip', 'install', '-r', 'requirements.txt'], cwd='.tmp/AgentBench', env=new_environment)
    subprocess.run(['pip', 'install', 'multiprocess'], cwd='.tmp/AgentBench', env=new_environment)

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

    # Task: Card game

    subprocess.run(['pip', 'install', 'websockets'], env=new_environment) # Missing dependency

    os.close(os.open('.tmp/AgentBench/installation-is-done', os.O_CREAT))

    # Task: Knowledge Graph
    # This setup is way too complicated. I'm going to skip this task.

    return new_environment

def evaluate_model(model_type, model_name, model_args, evaluation_id):
    new_environment = install()

    model = create_model(model_type, model_name, model_args)

    stop_event = threading.Event()

    server = http.server.ThreadingHTTPServer(('127.0.0.1', 35812), functools.partial(InferenceRequestHandler, model, stop_event))

    threading.Thread(target=server.serve_forever).start()

    agent_file = os.path.join(os.getcwd(), 'evaluation', 'benchmarks', 'agentbench_agent.yaml')

    tasks = {
        'os_interaction': { # Runs
            'config_file': 'configs/tasks/os_interaction/dev.yaml',
            'num_workers': 26,
            'path_to_score': ['score', 'overall', 'acc'],
        },
        'dbbench': { # MySQL connection error and lots of "not available" stuff before.
            'config_file': 'configs/tasks/dbbench/dev.yaml',
            'num_workers': 15,
        },
        'lateral-thinking-puzzle': { # Runs (after adding retrying functionality for RateLimit etc. to the OpenAI client in AgentBench)
            'config_file': 'configs/tasks/lateralthinkingpuzzle/dev.yaml',
            'num_workers': 50,
            'path_to_score': ['main'],
        },
        'lateral-thinking-puzzle-zh': { # Runs
            'config_file': 'configs/tasks/lateralthinkingpuzzle_zh/dev.yaml',
            'num_workers': 50,
            'path_to_score': ['main'],
        },
        'knowledge-graph': { # Excluded due to complexity of setup
            'config_file': 'configs/tasks/knowledgegraph/dev.yaml',
            'num_workers': 50,
        },
        'alfworld': { # Doesn't work
            'config_file': 'configs/tasks/alfworld/dev.yaml',
            'num_workers': 50,
        },
        'mind2web': { # TODO Requires dataset
            'config_file': 'configs/tasks/mind2web/dev.yaml',
            'num_workers': 50,
        },
        'card-game': { # Runs. But always zero score?
            'config_file': 'configs/tasks/card_game/dev.yaml',
            'num_workers': 16,
            'path_to_score': ['score', 'win_rate'],
        },
    }

    included_tasks = ['alfworld']

    scores = {}
    for task_name, task_information in tasks.items():
        if task_name not in included_tasks:
            continue

        scores[task_name] = []
        for i in range(1):
            output_directory = os.path.abspath(os.path.join('.tmp', 'agentbench', str(uuid.uuid4())))

            subprocess.run([
                'python3', 'eval.py',
                '--task', task_information['config_file'],
                '--agent', agent_file,
                '--workers', str(task_information['num_workers']),
                '--output', output_directory,
            ], env=new_environment, cwd='.tmp/AgentBench')

            results_file = os.path.join(output_directory, 'results.json')
            with open(results_file) as f:
                results = json.load(f)

            score = results
            for path_item in task_information['path_to_score']:
                score = score[path_item]

            scores[task_name].append(score)

    print(scores)

    server.shutdown()
