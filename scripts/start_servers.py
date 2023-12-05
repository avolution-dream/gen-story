# Copyright (c) Meta Platforms, Inc. and affiliates.

import argparse
import logging
import os

from pathlib import Path

from storygen.common.config import *
from storygen.common.server import *
from storygen.common.util import *
from storygen.common.llm.prompt import load_prompts


def recursive_start_servers(config, prompts, started_server_configs):

    server_config = ServerConfig.from_config(config)

    if server_config not in started_server_configs:
        start_server(server_config)
        started_server_configs.add(server_config)

    for key in prompts:
        if type(prompts[key]) is dict:
            if key in config and type(config[key]) is Config:
                recursive_start_servers(config[key], prompts[key], started_server_configs)


if __name__=='__main__':
    # Set the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', type=str,
                        choices=['premise', 'plan', 'story'],
                        required=True)
    parser.add_argument('--configs', nargs='+', default=['defaults'])
    args = parser.parse_args()

    # Set the directory path
    dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                            args.step)

    # Load the prompts
    prompts = load_prompts(Path(dir_path))

    # Load the config and
    config = Config.load(Path(dir_path), args.configs)
    init_logging(config['logging_level'])

    # Start the server
    logging.info('Starting model server(s)...')
    recursive_start_servers(config['model'], prompts, set())
