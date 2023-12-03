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


if __name__=='__main__':

    # =======================
    # Preparation
    # =======================
    # Set the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', nargs='+', default=['defaults'])
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
    generate_title(premise,
                   prompts['title'],
                   config['model']['title'],
                   llm_client)
    logging.info(f'Generated title: {premise.title}')

    generate_premise(premise,
                     prompts['premise'],
                     config['model']['premise'],
                     llm_client)
    logging.info(f'Generated premise: {premise.premise}')

    # Save the results
    os.makedirs(os.path.dirname(config['output_path']), exist_ok=True)
    premise.save(config['output_path'])
