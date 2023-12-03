# Copyright (c) Meta Platforms, Inc. and affiliates.

import argparse
import os

from pathlib import Path

from storygen.common.llm.llm import LLMClient
from storygen.common.llm.prompt import load_prompts
from storygen.premise.premise import Premise
from storygen.plan.plan import Plan
from storygen.plan.plan_writer import *
from storygen.common.config import Config
from storygen.common.util import *
from storygen.plan.setting import Setting
from storygen.plan.entity import *

# ================================================
# Helper functions for user generated contents
# ================================================
def generate_setting_user(plan_object, setting_str):
    """
    Adding the setting attribute to the plan object.

    setting_str: (str) a setting for the story.
    """
    plan_object.setting = Setting(setting_str)
    logging.debug(f'Setting: {plan.setting.setting}')
    return plan_object


def generate_entity_user(plan_object,
                         entity_name_list,
                         entity_description_list):
    """
    Set the entity_list for the plan object.

    entity_name_list: (list) a list of entity names.
    entity_description_list: (list) a list of entity descriptions.
    """

    # Make the assertion
    assert (
        len(entity_name_list) == len(entity_description_list)
    ), 'Entity lists are not of equal length.'

    # Initialize the entity list
    plan_object.entity_list = EntityList()

    # Make the entity
    for name, desc in zip(entity_name_list, entity_description_list):
        # Preprocess for strings
        name = name.strip(string.whitespace + string.punctuation)
        desc = desc.rstrip()

        # Add the attribute
        plan_object.entity_list.entities.append(Entity(name, desc))

    return plan_object


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
    parser.add_argument('-s', '--setting_str', type=str,
                        default='The story is set in 80s China where everyone has a hope.')
    parser.add_argument('-char_name', '--entity_name_list', action='append',
                        help='Add one entity name.')
    parser.add_argument('-char_desc', '--entity_description_list', action='append',
                        help='Add one entity description.')
    args = parser.parse_args()

    # Set the path
    dir_path = os.path.dirname(os.path.realpath(__file__))
    config = Config.load(Path(dir_path), args.configs)
    init_logging(config['logging_level'])

    # Load the prompts & premise
    premise = Premise.load(config['premise_path'])
    prompts = load_prompts(Path(dir_path))

    # Set the LLM Client
    client = LLMClient()

    # =======================
    # Get the plan
    # =======================
    # Initialize the plan class based on the premise
    plan = Plan(premise)

    # 1. Generate the setting
    if args.user_gen and args.setting_str:
        logging.info(f'Using user provided setting {args.setting_str}.')
        generate_setting_user(plan,
                              args.setting_str)
    else:
        generate_setting(plan,
                         client,
                         prompts['setting'],
                         config['model']['setting'])

    logging.info(f'Generated setting: {plan.setting}')

    # 2. Generate the entities
    if args.user_gen and args.entity_name_list:
        logging.info(f'Using user provided entities.')
        generate_entity_user(plan,
                             args.entity_name_list,
                             args.entity_description_list)
    else:
        success = False
        for i in range(config['model']['entity']['max_attempts']):
            try:
                generate_entities(plan,
                                  client,
                                  prompts['entity'],
                                  config['model']['entity'])
                success = True
                break
            except:
                logging.warning(f'Failed to generate entities, retrying ({i+1}/{config["model"]["entity"]["max_attempts"]})')
        if not success:
            raise Exception('Failed to generate entities')
    logging.info(f'Generated entities: {plan.entity_list}')

    # 3. Generate the outlines
    success = False
    for i in range(config['model']['outline']['max_attempts']):
        # TODO retry mechanism could be more sophisticated if needed, e.g. beam search or MCTS, similar to how we do it in generate_story
        try:
            generate_outline(plan,
                             client,
                             prompts['outline'],
                             config['model']['outline'])
            success = True
            break
        except:
            logging.warning(f'Failed to generate outline, retrying ({i+1}/{config["model"]["outline"]["max_attempts"]})')
    if not success:
        raise Exception('Failed to generate outline')

    logging.info(f'Generated plan: {plan}')

    # Save the results
    os.makedirs(os.path.dirname(config['output_path']), exist_ok=True)
    plan.save(config['output_path'])
