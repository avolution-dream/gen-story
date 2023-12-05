import os
import io
import json
import yaml
import argparse

from langchain.llms import OpenAI
from langchain.schema import HumanMessage
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from pathlib import Path

from storygen.premise.premise import Premise
from storygen.plan.plan import Plan
from storygen.plan.entity import Entity, EntityList


# #########################
# Arguments
# #########################
parser = argparse.ArgumentParser()

# Arguments
parser.add_argument('-mn', '--model_name', type=str,
                    default='gpt-3.5-turbo')



# #########################
# Helper Functions
# #########################
# ====================================
# Style Related Function
# ====================================
def create_style(title: str, premise: str, chat_model: None, style_prompt: str):
    """
    Generate the style prompt for the t2i prompt.

    plan: storygen.plan.plan.Plan
    style_prompt: str
    """
    style = chat_model.predict(style_prompt.format(title=title,
                                                   premise=premise))
    return style.strip()


# ====================================
# Entity Related Function
# ====================================
class VisualEntity(Entity):
    def __init__(self, name: str, description: str, visual: str):
        super().__init__(name, description)
        self.visual = visual


class VisualEntityList(EntityList):
    def __str__(self):
        # Return numbered list of visual entity names with descriptions and visual representations
        return '\n\n'.join([f'{i+1}. {entity.name}\n{entity.description}\n{entity.visual}'
                            for i, entity in enumerate(self.entities)])


def create_visual_entity(entity, visual_prompt: str):
    """
    Add visual description to the entity class attribute.

    entity: __main__.VisualEntity
    visual_prompt: str

    Notice that chat_model is a global instance for threadpooling.
    """

    visual = chat_model.predict(
        visual_prompt.format(description=entity.description)).strip()

    return VisualEntity(entity.name,
                        entity.description,
                        visual)


def create_visual_entity_list(entity_list, visual_prompt: str):
    """
    Parallel call LLM to generate visual descriptions for characters.

    Notice that chat_model is a global instance for threadpooling.
    """

    # Parallel call LLM to add visual description
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(lambda e: create_visual_entity(e, visual_prompt),
                                   entity)
                   for entity in entity_list]

        # Get the entity list
        visual_entity_list = [future.result() for future in futures]

    return visual_entity_list


# ====================================
# Plot Related Function
# ====================================
class Plot:
    """
    A plot object.
    """
    def __init__(self, text: str=None, scene: str=None, entities: list=None,
                 t2i_prompt: str=None, t2v_prompt: str=None):
        """
        text: (str) the plot content, also serves as the voiceover text.
        scene: (str) the scene content.
        entity_meta: (str) a sentence containing all the visual descriptions
            for the entities appear in the text.
        t2i_prompt: (str) the text-to-image prompt.
        t2v_prompt: (str) based on t2i_prompt, adding visual effects, camera info,
            and audio effects.
        """
        # Default attributes
        self.text = text
        self.scene = scene
        self.entities = entities

        self.t2i_prompt = t2i_prompt
        self.t2v_prompt = t2v_prompt
        self.entity_meta = ''


class PlotList:
    """
    A collection of plot objects.
    """
    def __init__(self, plots=None):
        self.plots = plots if plots is not None else []

    def __len__(self):
        return len(self.plots)

    def __str__(self):
        return '\n\n'.join([f'{i+1}. {plot.text}' for i, plot in enumerate(self.plots)])

    def __iter__(self):
        return iter(self.plots)

    def __getitem__(self, index):
        return self.plots[index]


def get_plots(outline: None):
    """
    Generate an instance of plot list.
    We extract only the plot from the innermost node.
    """

    def process_node(node, plot_list):
        """
        Helper function to get innermost plots.
        """
        # Check if the node has children
        if not node.children:
            # This is an innermost node, process it here
            plot = Plot()
            plot.text = node.text
            plot.scene = node.scene
            plot.entities = node.entities
            plot_list.append(plot)  # Append the plot to the list
        else:
            # If the node has children, recursively process each child
            for child in node.children:
                process_node(child, plot_list)

    # Initialize an empty list
    plots = []

    # Fill in the list with innermost plots
    process_node(plan.outline, plots)

    return PlotList(plots)


def keyword_in(keyword, target, threshold=0.8):
    """
    Check if the keyword is similar to any word in the text.

    keyword: The keyword to check.
    target: The corresponding target.
    threshold: The similarity threshold, above which we consider a match.
    """
    assert type(target) in (list, str)

    if type(target) == str:
        target = target.split()

    for word in target:
        if Levenshtein.ratio(keyword, word) >= threshold:
            return True
    return False


def update_entity_meta(plots, plan):
    """
    Add the entity meta string for each plot.
    """
    # Make the visual dict for entities
    entity_dict = {}

    for entity in plan.entity_list:
        entity_dict[entity.name] = entity.visual

    # Iterate over plots
    for plot in plots:
        for entity_name in plot.entities:

            # Double check if it the text contains it
            if keyword_in(entity_name, plot.text):

                # Double check if it is in the entity_dict
                if keyword_in(entity_name, list(entity_dict.keys())):
                    plot.entity_meta += entity_dict[entity_name] + '\n'

    return plots


# #########################
# Main
# #########################
# Parse the arguments
p = parser.parse_args('')

# Set the argument
for key, value in vars(p).items():
    globals()[key] = value

# Set the path
current_dir = Path(os.getcwd())
parent_dir = Path(os.path.dirname(current_dir))

print(parent_dir)

# Load the config and prompts
config = yaml.safe_load(open(current_dir / 'config.yaml'))
prompts = json.load(open(current_dir / 'prompts.json'))

# ===============================
# Step 0 - Server
# ===============================
# Starting the server for open-source model
if 'gpt' not in model_name:
    os.system(f"python -u -m vllm.entrypoints.openai.api_server \
                --model {model_name} \
                --tensor-parallel-size {config['SERVER']['tensor_parallel_size']} \
                --port {config['SERVER']['port']} &")

# Load the chat model with openai format
chat_model = ChatOpenAI(**config['MODEL'][model_name])


# ===============================
# Step 1 - Gather data
# ===============================
# Load the premise related
premise = Premise.load(parent_dir / config['PATH']['premise_path'])
title = premise.title
premise = premise.premise

# Load the plan related
plan = Plan.load(parent_dir / config['PATH']['plan_path'])


# Step 2 - Generate the style prompt
style_keywords = create_style(plan.premise.title,
                              plan.premise.premise,
                              chat_model,
                              prompts['style'])


# Step 3 - Generate the character prompt
# # Update the visual description to each entity
# # Each item in entity list is an Entity object
# # which contains attribute of name, description, and visual
plan.entity_list = VisualEntityList(
    create_visual_entity_list(plan.entity_list, prompts['visual'])
)


# Step 4 - Generate the text-to-image prompt
# # Get all the plots from the innermost node of the plan
plots = get_plots(plan.outline)

# # Update the entity visual prompts for each plots
plots = update_entity_meta(plots, plan)