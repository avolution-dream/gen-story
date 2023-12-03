# Auto Storyboard Generation

**This codebase is adapted from Meta's [DOC Story Generation V2](https://github.com/facebookresearch/doc-storygen-v2/tree/main/storygen).**

The high-level purpose is to generate stories in a hierarchical way to maintain story coherence. Simply put, we first generate a high-level plan consists of four nodes (*e.g.*, intro, development, turn, conclusion), then expanding at each node with plots.

<!-- This repository contains code for automatically generating stories of a few thousand words in length using LLMs, based on the same main ideas and overall structure as https://github.com/yangkevin2/doc-story-generation, but substantially modified to work with newer open-source and chat-based LLMs. The main goal of this rewrite was simplicity, to make it easier to modify / build upon the codebase. -->

## Installation

```bash
conda create --name story python=3.10
conda activate story

pip install -r requirements.txt
pip install -e .
```

<!-- By default we use VLLM to serve models.
You'll need to make a one-line change to the VLLM package to get their API server to work with logprobs requests that are used for reranking.
In your install of VLLM (you can find it using e.g., `pip show vllm`), find the line at https://github.com/vllm-project/vllm/blob/acbed3ef40f015fcf64460e629813922fab90380/vllm/entrypoints/openai/api_server.py#L177 (your exact line number might vary slightly depending on VLLM version) and change the `p` at the end to e.g., `max(p, -1e8)`. This will avoid an error related to passing jsons back from the server, due to json not handling inf values. -->

## Running the Pipeline

### Getting Started
We divide the storyboard generation procedure into 3 steps: Premise, Plan (with expansions), and Storyboard.

Everything will be run from the `scripts` directory:

```bash
cd scripts
```

Everything will read the information from the `defaults` configuration in `config.yaml` unless specified otherwise using the `--configs` flag.

See the corresponding `config.yaml` for details on options for each step of the pipeline. You'll have to fill in the particular model you want to use (marked TODO in each `config.yaml`). This system was mainly tested with LLaMA2-7B-Chat and ChatGPT, with the default options given; several other options are supported but not as heavily tested. When changing the model, make sure you also change `server_type` and `prompt_format` as needed. You can also add new options directly to the config as needed; you can also see the main prompts in `prompts.json`.

By default we use VLLM to serve models. Start the server(s) for the models you're using (this will start them in the background).

```bash
python start_servers.py --step premise &
python start_servers.py --step plan &
```

### Get the Premise
This part is to generate the title and a premise for the given title. If you would like the model to take over the job:
```bash
python premise/generate.py
```

You may specify the title and premise string by yourself with the following command:
```bash
python premise/generate.py --user_gen \
    --title_str 'A Happy Day' \
    --premise_str 'A pleasure-seeking, work-eschewing twenty-something realizes he has worked no days during the first few months of the pandemic despite loving his employment after receiving pandemic package payment fortnightly.'
```

By default, files are written to the `./script/output/` folder. Premise and Plan are formatted as jsons which can be edited for human interaction.

### Get the Plan and Storyboard
If you would like the model to take over the job:
```bash
python plan/generate.py
```
You may specify the settingby yourself:
```bash
python plan/generate.py --user_gen \
    --setting_str 'The story is set in 80s China where everyone has a hope.' \
```

TODO.

### Close the Server
After you're done with the above, close your servers (this command also runs in the background).

```bash
python close_servers.py
```

Note that `start_servers.py` relies on `close_servers.py` to delete the `server_configs.txt` file; just delete it manually before starting servers next time if you close servers in a different way. Alternatively, if memory allows, you can just keep all the servers alive simultaneously before closing them at the end, or reuse the servers between steps if you're using the same model by setting them to use the same `port` in `config.yaml` (it's fine if the sampling params differ).


## Some Known Issues / Potential Improvements

<!-- - When start multiple model servers for different models, we should allocate them to different GPUs or load on multi-GPU as needed. -->
- During plan generation, the model likes to overgenerate characters when determining which characters appear in a given plot point.
- Diversity of premise / plan ideas is kind of bad when using chat models, since they like to generate the same ideas over and over. Can try to increase temperature, or other ways to increase diversity, ideally without sacrificing quality.
- We should implement vaguest-first expansion (using a model to predict which node is the most vague) rather than breadth-first expansion when creating the outline during plan generation.
- Some model types and some more obscure options in the code aren't well-tested. Please let us know if you run into any issues.


## License

This repo is licensed under the Apache 2.0 License. See the LICENSE file for details.


## TODO
- Adding text-to-image prompt in each generated plots
- Intead of automating the whole process, allow user to
    - Provide premise
    - Provide characters
- Add the gradio interface for this one
