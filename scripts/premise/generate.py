# Copyright (c) Meta Platforms, Inc. and affiliates.

import argparse
import os

from pathlib import Path

from storygen.common.llm.llm import *
from storygen.common.llm.prompt import load_prompts
from storygen.premise.premise import Premise
from storygen.premise.premise_writer import *
from storygen.common.config import Config
from storygen.common.util import *


# ================================================
# Helper functions for user generated contents
# ================================================
def generate_title_user(premise_object, title_str):
    """
    Adding the title attribute to the premise.

    title_str: (str) a short title for the story.
    """
    premise_object.title = title_str
    return premise_object


def generate_premise_user(premise_object, premise_str):
    """
    Adding the premise attribute to the premise.

    premise: (str) the premise sentence(s) for the story.
    """
    premise_object.premise = premise_str
    return premise_object


# ================================================
# The main function
# ================================================
if __name__=='__main__':

    # =======================
    # Preparation
    # =======================
    # Set the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', nargs='+', default=['defaults'])
    parser.add_argument('-u', '--user_gen', action='store_true',
                        help='Flag if user provide info by themselves.')
    parser.add_argument('-t', '--title_str', type=str, default='A Happy Day')
    parser.add_argument('-p', '--premise_str', type=str, default='',
                        help='If empty, the model will generate based on the title.')
    args = parser.parse_args()

    # Set the path
    dir_path = os.path.dirname(os.path.realpath(__file__))
    config = Config.load(Path(dir_path), args.configs)
    init_logging(config.logging_level)

    # Load the prompts
    prompts = load_prompts(Path(dir_path))

    # Set the LLM Client
    llm_client = LLMClient()

    # =======================
    # Get premise
    # =======================
    # Set the class for premise
    premise = Premise()

    # Generate title and premise
    if args.user_gen and args.title_str:
        logging.info(f'Using user provided title.')
        generate_title_user(premise,
                            args.title_str)
    else:
        generate_title(premise,
                       prompts['title'],
                       config['model']['title'],
                       llm_client)
    logging.info(f'Generated title: {premise.title}')

    if args.user_gen and args.premise_str:
        logging.info(f'Using user provided premise.')
        generate_premise_user(premise,
                              args.premise_str)
    else:
        generate_premise(premise,
                         prompts['premise'],
                         config['model']['premise'],
                         llm_client)
    logging.info(f'Generated premise: {premise.premise}')

    # Save the results
    os.makedirs(os.path.dirname(config['output_path']), exist_ok=True)
    premise.save(config['output_path'])
