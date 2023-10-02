"""
⭐ == very important code that should be read first

1. project contain dockerfile ?
    yes -> NotImplemented
    no  -> 2.

2. need GPU ?
    yes -> 3
    no  -> NotImplemented

3. generate dockerfile with GPU

4. build dockerfile interactively
"""

from void_terminal.toolbox import CatchException, update_ui, get_log_folder, gen_time_str
from void_terminal.toolbox import ProxyNetworkActivate
from void_terminal.crazy_functions.crazy_utils import request_gpt_model_in_new_thread_with_ui_alive as ask_gpt_alive
from void_terminal.request_llm.bridge_all import predict_no_ui_long_connection as ask_gpt
import os


DockerfileFixPrompt = """
Please fix a Dockerfile that returns build error, 
only answer with the corrected Dockerfile, 
wrap the fixed Dockerfile with markdown codeblock.

The error is as follows:
```
{err}
```

The current Dockerfile is as follows:
```
{dockerfile}
```
"""


def wait_user_input(reason, next_state, chatbot, history, system_prompt, web_port):
    """
    chatbot         Chat display box handle, Displayed to the user
    history         Chat history, Context summary
    system_prompt   Silent reminder to GPT
    request         Current port number and ip
    """
    chatbot._cookies['lock_plugin'] = 'github_runner->github_runner'      # Assign plugin lock, lock plugin callback path, When the next user submits, will directly jump to the function
    chatbot._cookies['plugin_state'] = next_state            # Assign plugin status
    chatbot.append((reason, reason))
    yield from update_ui(chatbot=chatbot, history=history)

def clear_state(chatbot):
    chatbot._cookies['lock_plugin'] = None     # Unlock plugin, Avoid forgetting to cause deadlock
    chatbot._cookies['plugin_state'] = None    # Release plugin status, Avoid forgetting to cause deadlock

def fetch_github_resp_to_path(url, directory, chatbot=None, history=None):
    """
    fetch github resp to path
    """
    with ProxyNetworkActivate("Download_Resp"):
        print('downloading')
        import git
        git.Repo.clone_from(url, directory)
        print('download complete')



def github_runner__(txt, llm_kwargs, plugin_kwargs, chatbot, history, system_prompt, web_port):
    """
    txt             Text entered by the user in the input field, For example, a paragraph that needs to be translated, For example, a file path that contains files to be processed
    llm_kwargs      GPT model parameters, Such as temperature and top_p, Generally pass it on as is
    plugin_kwargs   Plugin model parameters, Such as temperature and top_p, Generally pass it on as is
    chatbot         Chat display box handle, Displayed to the user
    history         Chat history, Context summary
    system_prompt   Silent reminder to GPT
    request         Current port number and ip
    """
    # Initializing history
    history = []

    # Initializing plugin status
    state = chatbot._cookies.get('plugin_state', None)

    if state is None:
        # First time to enter the plugin
        clear_state(chatbot)
        yield from wait_user_input("Please enter the github resp url: ", 'GetGithubUrl', chatbot, history, system_prompt, web_port)
        return
    
    elif state == 'GetGithubUrl':
        # Get github url
        clear_state(chatbot)
        url = txt
        directory = os.path.join(get_log_folder(plugin_name='github_runner'), gen_time_str())
        yield from fetch_github_resp_to_path(chatbot, history, url, directory)

    else:
        chatbot._cookies['lock_plugin'] = None     # Unlock plugin, Avoid forgetting to cause deadlock
        chatbot._cookies['plugin_state'] = None    # Release plugin status, Avoid forgetting to cause deadlock
        chatbot.append((f"Task complete", "Task complete"))
        yield from update_ui(chatbot=chatbot, history=history) # Refresh the page
        return


def simple_qa(q, readme, llm_kwargs, plugin_kwargs, chatbot, history, system_prompt, short_answer=True):
    """
    Ask question according to readme
    """
    if short_answer:
        inputs = q + \
            "\n\nAnswer this question with a short answer according to:\n\n" + readme
    else:
        inputs = q + \
            "\n\nAnswer this question according to:\n\n" + readme
    print(inputs)
    # ⭐ ⭐ ⭐
    res = ask_gpt(inputs=inputs, llm_kwargs=llm_kwargs, history=[], sys_prompt=r"You are a programmer!", observe_window=[])
    return res.replace('\n', ' ')

def get_nvidia_driver_version():
    import subprocess
    try:
        command = ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader']

        # Run the command and capture the output
        output = subprocess.check_output(command).decode().strip()

        # Split the output into individual lines and extract the driver version
        lines = output.split('\n')
        version = lines[0].strip()

        return version

    except (subprocess.CalledProcessError, OSError):
        # Handle exceptions if the command fails or NVIDIA driver is not installed
        return "NVIDIA driver not found"
    
def get_code_block(reply):
    import re
    pattern = r"```([\s\S]*?)```" # regex pattern to match code blocks
    matches = re.findall(pattern, reply) # find all code blocks in text
    def finalize_result(s):
        dockerfile = "\n".join(s.split("\n")[1:])
        assert "FROM" in dockerfile
        return dockerfile
    if len(matches) == 1: 
        return finalize_result(matches[0])
    for match in matches:
        if 'FROM ' in match:
            return finalize_result(match)
    raise RuntimeError("GPT is not generating proper code.")


def generate_docker_file_from_readme(url, project_dir, llm_kwargs, plugin_kwargs, chatbot, history, system_prompt):
    """
    generate docker file from readme
    """
    # file read readme
    readme_path = os.path.join(project_dir, 'README.md')
    if not os.path.exists(readme_path):
        readme_path = os.path.join(project_dir, 'readme.md')
        if not os.path.exists(readme_path):
            raise FileNotFoundError('readme not found')

    # file read dockerfile
    with open(readme_path, 'r') as f:
        readme = f.read()
        readme = """\n```\n""" + readme + """\n```"""

    # ask questions about readme
    qa = lambda q, short_answer: simple_qa(q, readme, llm_kwargs, plugin_kwargs, chatbot, history, system_prompt, short_answer)

    # ⭐
    # manual chain of thought
    tips = []
    tips.append('Extact information about project installation. Be faithful with the original readme material.')
    tips.append(qa(tips[-1], False))
    tips.append('On what version of ubuntu system can this project run?')
    tips.append(qa(tips[-1], True))
    tips.append('Are GPUs used in this project? (you should answer YES the document mentions anything such as CUDA, pytorch, tensorflow, nerual network in README)')
    tips.append(qa(tips[-1], True))

    # ⭐
    # choose base image
    qa_from_tip = lambda q, short_answer: simple_qa(q, "\n".join(tips), llm_kwargs, plugin_kwargs, chatbot, history, system_prompt, short_answer)
    tips.append(f"""
        What docker image do you select as base image? 
                
            My nvidia driver version is {get_nvidia_driver_version()},

            If cuda is mentioned in this project, 
            you should choose a docker image with nvidia driver installed,
            such as `FROM nvidia/cuda:11.3.1-runtime-ubuntu20.04`.

            otherwise,
            if this is a python project,
            you should choose python image such as `FROM python:3.11`,
        """
    )
    tips.append(qa_from_tip(tips[-1], False))

    # ⭐ now we write the dockerfile
    inputs = "Write a Dockerfile for the following project, wrap the output Dockerfile with markdown codeblock:\n\n" + "".join(tips)
    inputs_show_user = f"Write a Dockerfile for the following project, wrap the output Dockerfile with markdown codeblock:\n\n{url}\n\n{project_dir}"
    gpt_say = yield from ask_gpt_alive(
        inputs=inputs, inputs_show_user=inputs_show_user, 
        llm_kwargs=llm_kwargs, chatbot=chatbot, history=history, 
        sys_prompt= r"You are a Dockerfile programmer!"
    )

    dockerfile_path = os.path.join(project_dir, 'Dockerfile')
    with open(dockerfile_path, 'w') as f:
        f.write(get_code_block(gpt_say))
    return dockerfile_path



def build_and_debug_dockerfile(dockerfile_path, llm_kwargs, plugin_kwargs, chatbot, history, system_prompt):
    class DockerfileDebugger():
        def __init__(self, dockerfile_path) -> None:
            self.workdir = os.path.dirname(dockerfile_path)
            self.cur_dockerfile = dockerfile_path

        def build_with_timeout(self, timeout=3600):
            import subprocess
            command = ['docker build -t debug_dockerfile .']
            process = subprocess.Popen(command, shell=True, stderr=subprocess.PIPE, cwd=self.workdir)
            try:
                stdout, stderr = process.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
                return stdout, stderr.decode()
            return stdout, stderr.decode()
        
        def build_dockerfile(self):
            _, err = self.build_with_timeout()
            if len(err) == 0:
                # successfully build
                return True, ""
            else:
                return False, err
        
        def fix_dockerfile(self, err, ntry):
            # read old dockerfile
            with open(self.cur_dockerfile, 'r') as f:
                dockerfile = f.read()
            dockerfile_bk = "This dockerfile suffers from error:\n\n{err}\n\n\n\n" + dockerfile
            # backup dockerfile
            with open(self.cur_dockerfile+f'.bak_{ntry}', 'w') as f:
                f.write(dockerfile_bk)
            # ⭐ fix dockerfile
            inputs = DockerfileFixPrompt.format(err=err, dockerfile=dockerfile)
            gpt_say = ask_gpt(inputs=inputs, llm_kwargs=llm_kwargs, history=[], 
                              sys_prompt=r"You are a great Dockerfile programmer!", observe_window=[])

            # ⭐ rewrite dockerfile
            with open(self.cur_dockerfile, 'w') as f:
                f.write(get_code_block(gpt_say))
            return
        
        def build_err_retry_build_err_retry(self):
            max_retry = 10
            for i in range(max_retry):
                success, err = self.build_dockerfile()
                if success:
                    return True, self.cur_dockerfile
                else:
                    self.fix_dockerfile(err, i)
            return False, self.cur_dockerfile
        
    dd = DockerfileDebugger(dockerfile_path)
    success, dockerfile_fp = dd.build_err_retry_build_err_retry()
    chatbot.append((success, dockerfile_fp))
    yield from update_ui(chatbot=chatbot, history=history)
    return success, dockerfile_fp

@CatchException
def github_runner(txt, llm_kwargs, plugin_kwargs, chatbot, history, system_prompt, web_port):
    # directory = "gpt_log/default/github_runner/2023-10-02-21-55-05"
    # url='https://github.com/dreamgaussian/dreamgaussian'
    # yield from generate_docker_file_from_readme(url, directory, llm_kwargs, plugin_kwargs, chatbot, history, system_prompt)
    dockerfile_path = 'gpt_log/default/github_runner/2023-10-02-21-55-05/Dockerfile'
    yield from build_and_debug_dockerfile(dockerfile_path, llm_kwargs, plugin_kwargs, chatbot, history, system_prompt)

