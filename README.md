# Large Language Models Empowered Personalized Web Agents

## 🔍 Overview

LLM-based Web agents overlook the importance of personalized data (e.g., user profiles and historical Web behaviors) in assisting the understanding of users' personalized instructions and executing customized actions. **PersonalWAB** (Personalized Web Agent Benchmark) serves as the first comprehensive benchmark designed to evaluate Web agents on tasks such as personalized search, recommendation, and review generation. The benchmark includes a set of personalized user data, Web functions, and evaluation paradigms that facilitate the development of more personalized Web agents.
**PUMA** (Personalized User Memory-enhanced Alignment) is a framework developed to adapt LLMs to the personalized Web agent task. By leveraging a memory bank and task-specific retrieval strategies, PUMA filters relevant historical Web behaviors, enabling fine-tuned and optimized personalized action execution.

This repository now also includes an additive **KG-PUMA** path that replaces flat memory conditioning with temporal graph retrieval, graph evidence serialization, and graph-aware DPO rewards while keeping the underlying LoRA-tuned causal LM training stack intact.


> For more details, refer to our paper accepted to **WWW 2025**: [Large Language Models Empowered Personalized Web Agents](https://arxiv.org/abs/2410.17236).

## ⚙️ Installation

### Requirements

- Python 3.11
- PyTorch 2.4.1
- CUDA 12.5
- openjdk

Original paper experiments used the stack above. The current local development environment used for the KG-PUMA changes is:

- conda env: `puma`
- Python `3.14`
- PyTorch `2.11`
- CUDA `13.0`

To install the required dependencies, run:
```bash
pip install -r requirements.txt
```


## 📊 PersonalWAB Benchmark

![](https://hongrucai.github.io/images/personalwab.png)

The **PersonalWAB** benchmark includes:

- **Personalized User Data**: 1,000 diverse user profiles and 40,000+ web behaviors, originated from real-world data.
- **User Instructions**: 9,000+ highly personalized natural language instructions tailored to each user's profile.
- **User Simulatior**: Simulates interactions aligned with user profiles and historical behaviors.
- **Evaluation Paradigms**:  Single-turn track tests for isolated tasks and multi-turn for more complex interactions.

The dataset is available in "PersonalWAB/envs/pwab/data". Or you can download [here](https://hongrucai.github.io/PersonalWAB/download).

### Task Description

**Personalized Search**: Personalized product search using user instructions and behavioral history.  
**Personalized Recommendation**: Recommend items based on implicit preferences.  
**Personalized Review Generation**: Generate reviews aligned with user preferences.

### Running Experiments 

To run experiments on the **PersonalWAB** benchmark, use the following command:

```bash
source scripts/set_api_key.sh  # Set your OpenRouter API key
bash scripts/run_singleturn.sh  # Single-turn track
bash scripts/run_multiturn.sh   # Multi-turn track
```

The default OpenAI SDK client in this repo is routed through OpenRouter's `https://openrouter.ai/api/v1` endpoint. Bare OpenAI model names such as `gpt-4o-mini` are normalized automatically when requests are sent.

You can modify agent strategies, memory mechanisms, and parameters in the scripts to explore various configurations.

For experiments using task-specific memory, use the PUMA framework to generate function selection results. For InteRecAgent, you need to provide the memory file generated from history behaviors before running in training or test set.

### Benchmark Evaluation

The **PersonalWAB** benchmark supports two evaluation tracks: single-turn and multi-turn interactions. The key metrics for evaluation include:

- **Function Accuracy**: The accuracy of selecting appropriate web functions.
- **Result Accuracy**: The relevance of returned results to user preferences.
- **Avg. Steps**: The average number of actions executed to complete a user instruction.

### Leaderboard Submission

When running `run.py`, the script will automatically save the detailed task execution results. This includes the agent's prompts, actions, function accuracy (Function Acc), result accuracy (Res Acc), and other relevant information.

- After testing, the program will generate a detailed result file with all the execution data.
- Upload this complete result file to an online storage service such as Google Drive or OneDrive.
- Once uploaded, please submit the download link through the provided [Google Form](https://forms.gle/UQdxUG8f28xbRd5Z8).
- Ensure that the result file is accessible via the shared link, and that it contains all relevant information for evaluation and ranking on the leaderboard.


## 🧠 PUMA Framework

The **PUMA** framework adapts LLMs for personalized Web agent tasks by utilizing:

- **Long-term Memory Bank**: A retrieval mechanism that filters relevant historical web behaviors.
- **Fine-tuning**: LLM fine-tuning with heuristically generated pseudo-labels.
- **Direct Preference Optimization**: Aligns parameter generation with user preferences through DPO.

For more detailed information, refer to our paper.

### KG-PUMA Graph Path

The graph-conditioned redesign is implemented as an additive path on top of the original PUMA codebase. The main graph modules are:

- `PUMA/graph/schema.py`
- `PUMA/graph/builder.py`
- `PUMA/graph/retriever.py`
- `PUMA/graph/serializer.py`
- `PUMA/graph/reward.py`
- `PersonalWAB/agents/kg_puma_agent.py`

The intended milestone-1 pipeline is:

1. Build a temporal user graph from pre-task history only.
2. Retrieve a task-conditioned subgraph instead of flat top-K memories.
3. Serialize compact graph evidence into the SFT prompt.
4. Generate graph-conditioned tool and parameter supervision.
5. Build DPO pairs with composite graph-aware rewards.

The graph path is retrieval- and supervision-centric. It does not add a GNN decoder or change the LoRA fine-tuning backbone in version 1.

### Training PUMA

The original flat-memory PUMA workflow is still supported.

**STEP 1: Prepare the SFT dataset.**

You may need to copy the 'user_interactions.json' and 'user_history.json' files from the PersonalWAB benchmark.    
```bash
cd PUMA
bash scripts/pre_sft_func_data.sh
bash scripts/pre_sft_param_data.sh
```
**STEP 2: Train the LLaMA model with SFT**  

There are three options for training: function, parameter, or both, which means SFT on function data, parameter data, or both function and parameter. You can specify in the script.
```bash
bash scripts/finetune_function_param.sh
```
**STEP 3: Generate function results and diverse parameter predictions for DPO** 

This step generates the function predictions and diverse parameter predictions for DPO training. The function results will be used to select the appropriate Web functions, while the parameter results are parameters for the selected functions. You may need to generate on training and test set separately by specifying the split.
```bash
bash scripts/generate_function.sh
bash scripts/generate_param_dpo.sh
```
**STEP 4: Evaluate the parameter results in PersonalWAB**  

This step evaluates the parameter results in the PersonalWAB benchmark. It will score the parameter results in the PersonalWAB benchmark, which will be used in the DPO training.
```bash
cd ..
bash scripts/fast_test_dpo.sh
```
**STEP 5: Prepare the DPO dataset**  

This step prepares the DPO dataset by choosing the highest and lowest parameter results from the previous step. 
```bash
cd PUMA
bash scripts/pre_dpo_data.sh
```
**STEP 6: Train with DPO**  

This step trains the model with DPO using the prepared DPO dataset. You need to first merge the LoRA adapter saved in SFT with base model, and then train the DPO model with the merged model.
```bash
bash scripts/merge.sh
bash scripts/dpo_llama.sh
```
**STEP 7: Evaluate the DPO model in PersonalWAB**  
This step evaluates the final DPO model in the PersonalWAB benchmark. But you can also use the model to generate the final function and parameter results (as in step 3)and use scripts/fast_test.sh to see the performance.
```bash
cd ..
bash scripts/run_singleturn_puma.sh
```

### Training KG-PUMA

The graph-conditioned path uses the same SFT and DPO training entrypoints, but swaps in graph-conditioned data builders and a graph-aware runtime agent.

Convenience wrappers matching the original PUMA layout are provided here:

- `PUMA/scripts/pre_graph_sft_func_data.sh`
- `PUMA/scripts/pre_graph_sft_param_data.sh`
- `PUMA/scripts/pre_graph_dpo_data.sh`
- `PUMA/scripts/finetune_graph_function_param.sh`
- `PUMA/scripts/dpo_graph_llama.sh`
- `PUMA/scripts/generate_graph_function.sh`
- `PUMA/scripts/generate_graph_param_dpo.sh`
- `scripts/fast_test_graph.sh`
- `scripts/fast_test_graph_dpo.sh`
- `scripts/run_singleturn_kg_puma.sh`

These commands assume the same merged benchmark files used by the original PUMA pipeline, for example:

- `data/user_instructions.json`
- `data/user_history.json`
- `data/user_profiles.json`
- `data/all_products.json`

**STEP 1: Prepare graph-conditioned SFT datasets.**

```bash
cd PUMA
bash scripts/pre_graph_sft_func_data.sh
bash scripts/pre_graph_sft_param_data.sh
```

**STEP 2: Run graph-conditioned SFT.**

```bash
bash scripts/finetune_graph_function_param.sh
```

`PUMA/scripts/finetune_graph_function_param.sh` defaults to graph datasets and recommendation up-weighting. Override paths or hyperparameters with environment variables before calling the script.

**STEP 3: Optionally generate graph-conditioned predictions.**

```bash
bash scripts/generate_graph_function.sh
bash scripts/generate_graph_param_dpo.sh
```

These wrappers write graph-mode outputs to `output/res/graph_function_res.json` and `output/res/graph_param_res.json` by default.

**STEP 4: Build graph-aware DPO pairs.**

`prepare_graph_dpo_data.py` constructs chosen/rejected pairs from composite rewards with task, faithfulness, validity, multi-hop, and length components.

```bash
bash scripts/pre_graph_dpo_data.sh
```

**STEP 5: Run graph-aware DPO.**

```bash
bash scripts/dpo_graph_llama.sh
```

**STEP 6: Evaluate graph-conditioned predictions offline.**

```bash
cd ..
bash scripts/fast_test_graph.sh
bash scripts/fast_test_graph_dpo.sh
```

**STEP 7: Evaluate with graph retrieval at runtime.**

Use `kg_puma` to assemble graph evidence online during inference.

```bash
cd ..
bash scripts/run_singleturn_kg_puma.sh
```

### Graph-Specific Notes

- Non-review graph retrieval is built from pre-task history only; target product metadata is only injected for review tasks.
- `test.py --graph_mode` logs decomposed graph rewards: task score, faithfulness, validity, hop bonus, length penalty, and final composite reward.
- `run.py --agent_strategy kg_puma` is the online graph-conditioned inference path.
- The flat-memory `puma` path remains unchanged for baseline comparison.

## 📚 Citation

If you use source code or dataset in your research, please cite our paper:
```bibtex
@inproceedings{cai2024personalwab,
      title={Large Language Models Empowered Personalized Web Agents}, 
      author={Hongru Cai, Yongqi Li, Wenjie Wang, Fengbin Zhu, Xiaoyu Shen, Wenjie Li, Tat-Seng Chua},
      year={2025},
      booktitle={Proceedings of the ACM Web Conference 2025},
      series={WWW'25}
}
```

## 📄 License

This project is licensed under the [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) License.

The benchmark implementation in this project is based on the [tau-bench](https://github.com/sierra-research/tau-bench), with significant modifications and enhancements made to suit the needs of this project. The tau-bench is originally licensed under the [MIT License](https://github.com/sierra-research/tau-bench?tab=MIT-1-ov-file), and we continue to honor and adhere to its licensing terms for the portions derived from it.

## 📬 Contact

For inquiries, feel free to reach out to Hongru Cai at [henry.hongrucai@gmail.com](mailto:henry.hongrucai@gmail.com).
