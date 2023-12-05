import os
import io
import yaml
import argparse

from langchain.llms import OpenAI
from multiprocessing import Pool
from pathlib import Path

# #########################
# Arguments
# #########################
parser = argparse.ArgumentParser()

# Arguments
parser.add_argument('-mn', '--model_name', type=str,
                    default='meta-llama/Llama-2-7b-chat-hf')


# Parse the arguments
p = parser.parse_args()

# Set the argument
for key, value in vars(p).items():
    globals()[key] = value

# Set the path
dir_path = Path(os.path.dirname(os.path.realpath(__file__)))
print(dir_path)

# Load the config
config = yaml.safe_load(open(dir_path / 'config.yaml'))

# Step 0 - Server
# Starting the server for open-source model
if 'gpt' not in model_name:
    os.system(f"python -u -m vllm.entrypoints.openai.api_server \
                    --model {model_name} \
                    --tensor-parallel-size {config['SERVER']['tensor_parallel_size']} \
                    --port {config['SERVER']['port']} &")

# Load the chat model with openai format
chat_model = ChatOpenAI(**model_config[model_name])


# We initiate a langchain chat object by reusing the server from plan.


# Step 1 - Gather data
# We load the premise.json to obtain title and premise.
# title = ''
# premise = ''

# We load the plan.json to sort a dictionary which formatted as:
# entities =   [{'name': '', 'description': ''}, ...]
# plots    =   [{'text': '', 'scene': '', 'entities': []}, ...]


# Step 2 - Generate the style prompt
# Based on the f'Title: {title}; Premise: {premise}', we ask the
# model the generate the image/movie style keywords for the story
# style_prompt = ''
# This will be used as the prompt_2 in the pipeline


# Step 3 - Generate the character prompt
# Based on the description of entities, generate the visual description for them.
# entities =   [{'name': '', 'description': '', 'visual': ''}, ...]


# Step 4 - Generate the text-to-image prompt
# This step should be done in parallel.
# Based on the text and scene description, generate the corresponding t2i prompt.
# Based on the text, use fuzzy match to check if keys in entities appear in the text.
# If so, extract the visual prompt and append to the t2i prompt.
# Append the style prompt to the t2i prompt as well.
# t2i_prompts = ['', ...]


# Step 5 - Generate the text-to-video prompt
# Based on each of the t2i prompt, generate the corresponding visual effects, camero info.
# Append to the t2i_prompts.
# with Pool() as pool:
#     pool.map(run, some_list)
