# Data Analyst Agent for Population-Level Statistical Prediction

## Engineering Specification

### Project Overview

This project builds a system that uses an LLM agent as an autonomous data analyst to generate synthetic social science data with population-level statistical realism. The system is evaluated against SSDataBench, enabling direct comparison with 15 LLMs tested in the original paper.

The core idea: instead of prompting an LLM to generate individuals one by one (the SSDataBench paradigm), we give an LLM agent access to real data and let it explore, model, and generate through code execution. The agent functions as a data scientist, not a survey respondent.

### Technical Constraints

- **Language**: Python throughout (3.10+)
- **Initial scope**: Start with a single dataset provided by the user. The system must work end-to-end on one dataset before expanding.
- **LLM provider flexibility**: The system must support both OpenAI-compatible and Anthropic-compatible API formats, with configurable base URL, API key, and model name. This allows testing with different providers and self-hosted endpoints.

### Repository Dependencies

- **SSDataBench**: https://github.com/lszshu/SSDataBench
  - `evaluation/`: evaluation logic (statistical tests, bootstrap resampling)
  - `real_data/`: processed survey datasets (CSV)
  - `simulated_data/`: generated synthetic datasets (CSV)
  - `evaluation_results/`: output from evaluation runs
  - `scripts/evaluation/`: per-dataset evaluation entry points
  - Entry point pattern: `python scripts/evaluation/<dataset>.py --sim-root <path> --output-base <path>`

### System Architecture

```
project-root/
  .env                      # LLM_API_KEY and other secrets (git-ignored)
  ssdatabench/              # cloned SSDataBench repo (submodule or copy)
  src/
    config/                 # experiment configurations
      datasets.yaml         # dataset registry (paths, variable schemas, context)
      experiments.yaml      # experiment matrix definition
      llm.yaml              # LLM provider config (base_url, model, temperature)
    data/                   # data loading and preprocessing
      loader.py             # load real data from SSDataBench
      schema.py             # variable schema definitions (background vs target, types)
      splitter.py           # train/eval split logic
    agent/                  # the data analyst agent
      orchestrator.py       # main agent loop (explore -> model -> validate -> iterate)
      sandbox.py            # code execution environment (subprocess + temp dir)
      prompt_templates.py   # prompt construction for each agent stage
      context.py            # manages what the agent can see (data, variable descriptions)
      llm_client.py         # unified LLM client (OpenAI + Anthropic compatible)
    generation/             # output formatting
      formatter.py          # convert agent output to SSDataBench-compatible CSV
    evaluation/             # evaluation bridge
      runner.py             # call SSDataBench evaluation scripts
      comparator.py         # compare results across conditions
    experiments/            # experiment orchestration
      runner.py             # run full experiment matrix
      conditions.py         # define experimental conditions (full, no-semantic, etc.)
  tests/                    # TDD test files, mirroring src/ structure
  results/                  # experiment outputs
  scripts/                  # CLI entry points
```

### Experimental Design

#### Conditions

Four experimental conditions, each varying what information the agent receives:

1. **Baseline (Direct LLM Generation)**: The SSDataBench paradigm. Already evaluated in the paper. We re-run with the same model we use for the agent (configured via `llm.yaml`) for fair comparison using their simulation code.

2. **Full Agent**: The agent receives:
   - The real dataset (background + target variables for the training split)
   - Variable descriptions from survey instruments
   - Historical and regional context
   - The prediction task specification
   
   The agent writes and executes code to explore data, build models, and generate synthetic data.

3. **Agent Without Semantics**: The agent receives:
   - The real dataset (same as Full Agent)
   - Variable names only (no survey question text, no descriptions, no context)
   - The prediction task specification
   
   This isolates the contribution of semantic/world knowledge.

4. **Agent Without Data (LLM-as-Modeler)**: The agent receives:
   - Variable descriptions from survey instruments
   - Historical and regional context
   - No real data values
   - The prediction task specification
   
   The agent must write a generative model based purely on its knowledge of social phenomena. This isolates the contribution of data-driven calibration.

#### Unseen Variable Experiment

For the Full Agent condition, additionally test with a subset of target variables hidden:
- The agent sees all background variables and a subset of target variables during exploration
- At generation time, the agent must also produce values for variables it never saw in the data
- The agent receives the variable descriptions for unseen variables but no data
- This tests the agent's ability to leverage semantic knowledge for zero-shot distributional prediction

#### Datasets (Start Small)

The user provides a single dataset initially. All phases (0 through 5) are built and validated on this one dataset before expanding.

Phase 1 (pilot): One cross-sectional dataset provided by the user.
Phase 2 (expand): Add remaining cross-sectional datasets from SSDataBench.
Phase 3 (full): Add longitudinal datasets (NLSY, CFPS, Add Health, Understanding Society).

#### Evaluation

Use SSDataBench evaluation code directly. For each condition and dataset:
- Format generated data as CSV matching SSDataBench's expected schema
- Run the corresponding evaluation script
- Collect pass rates across all five statistical pattern types
- Compare across conditions

---

## Implementation Phases

### Phase 0: Project Setup and SSDataBench Integration

**Goal**: Establish project structure, clone SSDataBench, verify the evaluation pipeline works, and confirm LLM API connectivity.

**Tasks**:
1. Initialize project repository with the directory structure above
2. Clone SSDataBench as a submodule or vendored dependency
3. Install SSDataBench dependencies (`pip install -r ssdatabench/requirements.txt`)
4. Set up `.env` with LLM API credentials
5. Place the initial dataset (provided by the user) into the expected location
6. Run one of their evaluation scripts on their existing simulated data to verify the pipeline works:
   ```
   python ssdatabench/scripts/evaluation/gss_2018.py
   ```
7. Inspect the output format in `evaluation_results/` to understand what metrics are produced
8. Verify LLM API connectivity with a simple test call for both OpenAI and Anthropic formats

**Tests**:
- `test_ssdatabench_integration.py`:
  - `test_evaluation_script_runs()`: verify the evaluation script completes without error on existing simulated data
  - `test_evaluation_output_format()`: verify output contains expected pass rate structure (by type, by variable)
  - `test_real_data_loadable()`: verify real data CSVs load correctly and contain expected columns

- `test_llm_connectivity.py`:
  - `test_openai_api_reachable()`: send a simple prompt via OpenAI-compatible endpoint, verify response
  - `test_anthropic_api_reachable()`: send a simple prompt via Anthropic-compatible endpoint, verify response
  - `test_config_loads_from_env()`: verify `.env` values are picked up correctly
  - `test_config_loads_from_yaml()`: verify `llm.yaml` values are picked up correctly
  - `test_env_overrides_yaml()`: verify environment variables take precedence over yaml config

**Acceptance Criteria**: We can run SSDataBench evaluation on their existing simulated data, parse the output pass rates, and make successful LLM API calls with both provider formats.

---

### Phase 1: Data Layer

**Goal**: Build data loading, schema definition, and train/eval splitting that aligns with SSDataBench's data format.

**Tasks**:
1. **Schema definition** (`src/data/schema.py`):
   - Define a `DatasetSchema` class that captures: dataset name, background variable names, target variable names, variable types (categorical/numerical), variable descriptions (from survey instruments), historical and regional context string, allowed values per variable
   - Populate schemas for the pilot dataset (GSS 2018 or CPS-ASEC) by reading SSDataBench's config files or prompt templates

2. **Data loader** (`src/data/loader.py`):
   - Load real data CSV from `ssdatabench/real_data/`
   - Load the 1000 sampled individuals that SSDataBench uses (check their sampling logic)
   - Return a pandas DataFrame with proper typing

3. **Splitter** (`src/data/splitter.py`):
   - Given the 1000 sampled individuals, split into train (visible to agent) and eval (for generation)
   - Support configurable split ratios
   - Ensure the split is reproducible (fixed random seed)

**Tests**:
- `test_schema.py`:
  - `test_schema_has_all_variables()`: verify schema includes all variables present in the real data CSV
  - `test_background_target_partition()`: verify every variable is classified as either background or target
  - `test_variable_descriptions_present()`: verify descriptions are non-empty for all target variables

- `test_loader.py`:
  - `test_load_real_data()`: verify data loads with correct shape and column names
  - `test_data_types_match_schema()`: verify categorical variables have values within allowed sets
  - `test_sample_size()`: verify we get the expected 1000 individuals

- `test_splitter.py`:
  - `test_split_sizes()`: verify train + eval = total
  - `test_split_reproducibility()`: verify same seed produces same split
  - `test_no_overlap()`: verify no individual appears in both train and eval

**Acceptance Criteria**: We can load any SSDataBench dataset, define its schema, and split it reproducibly.

---

### Phase 2: Agent Sandbox

**Goal**: Build a secure code execution environment where the agent can write and run Python code against data.

**Tasks**:
1. **Sandbox** (`src/agent/sandbox.py`):
   - Create an isolated execution environment (subprocess running Python in a temp directory)
   - The sandbox has access to: pandas, numpy, scipy, statsmodels, sklearn, matplotlib
   - The sandbox has access to the training data (written as a CSV to the temp directory)
   - Capture stdout, stderr, and any files written by the agent's code
   - Enforce a timeout per execution (e.g., 60 seconds)
   - Support multi-step execution (agent writes code, sees output, writes more code)

2. **Context manager** (`src/agent/context.py`):
   - Define what information the agent can see for each experimental condition
   - `FullContext`: data + variable descriptions + historical context
   - `NoSemanticContext`: data + variable names only
   - `NoDataContext`: variable descriptions + historical context, no data
   - `UnseenVariableContext`: data for subset of variables + descriptions for all variables
   - Each context generates a system prompt and provides data files accordingly

**Tests**:
- `test_sandbox.py`:
  - `test_execute_simple_code()`: run `print(1+1)`, verify output is `2`
  - `test_execute_pandas_code()`: load a CSV, compute mean, verify output
  - `test_timeout()`: run an infinite loop, verify it times out
  - `test_error_capture()`: run code with a NameError, verify stderr is captured
  - `test_file_output()`: write a CSV from code, verify it can be read back
  - `test_multi_step()`: execute two code blocks sequentially, verify second can reference first's output

- `test_context.py`:
  - `test_full_context_has_descriptions()`: verify variable descriptions are in the prompt
  - `test_no_semantic_context_lacks_descriptions()`: verify no descriptions, only variable names
  - `test_no_data_context_lacks_data()`: verify no data file is provided
  - `test_unseen_context_hides_variables()`: verify specified variables are absent from data but present in descriptions

**Acceptance Criteria**: Agent can execute arbitrary Python code in a sandbox, read data, produce outputs, and the context manager correctly controls information access per condition.

---

### Phase 3: Agent Orchestrator

**Goal**: Build the main agent loop that drives the data analysis workflow.

**Tasks**:
1. **Prompt templates** (`src/agent/prompt_templates.py`):
   - System prompt: establishes the agent's role as a data analyst tasked with building a generative model
   - Exploration prompt: instructs the agent to explore the data and report findings
   - Modeling prompt: instructs the agent to select and implement a generative model based on findings
   - Validation prompt: provides validation results and asks the agent to diagnose and improve
   - Generation prompt: instructs the agent to use its model to generate the required synthetic data

2. **Orchestrator** (`src/agent/orchestrator.py`):
   - Implements the multi-stage workflow:
     ```
     1. Initialize context (data + descriptions per condition)
     2. Exploration: agent writes EDA code, executes, sees results
     3. Modeling: agent writes model fitting code, executes
     4. Validation: agent writes validation code (comparing generated vs real on held-out)
     5. If validation fails threshold: loop back to step 3 (max N iterations)
     6. Generation: agent generates full synthetic dataset using final model
     7. Return generated data as DataFrame
     ```
   - Each step: construct prompt -> call LLM API -> extract code from response -> execute in sandbox -> capture output -> feed back to next prompt
   - Maintain conversation history so the agent has context of its previous analyses
   - Configurable max iterations for the validation loop (default: 3)

3. **LLM client** (`src/agent/llm_client.py`):
   - Unified client supporting both OpenAI-compatible and Anthropic-compatible API formats
   - Configuration via a provider config object:
     ```python
     @dataclass
     class LLMConfig:
         provider: str          # "openai" or "anthropic"
         base_url: str          # e.g., "https://api.openai.com/v1" or custom endpoint
         api_key: str           # loaded from env or config
         model: str             # e.g., "gpt-4.1", "claude-sonnet-4-20250514"
         temperature: float     # default 1.0
         max_tokens: int        # default 4096
     ```
   - For OpenAI-compatible providers: use the `openai` Python SDK with custom `base_url`
   - For Anthropic-compatible providers: use the `anthropic` Python SDK with custom `base_url`
   - Both providers expose the same interface:
     ```python
     class LLMClient:
         def __init__(self, config: LLMConfig): ...
         def chat(self, messages: list[dict], system: str = None) -> str: ...
     ```
   - Handles: API calls, response parsing, code extraction from markdown blocks
   - API key is loaded from environment variable (`LLM_API_KEY`) or passed in config
   - A `.env` file at project root stores credentials:
     ```
     LLM_PROVIDER=openai
     LLM_BASE_URL=https://api.openai.com/v1
     LLM_API_KEY=sk-...
     LLM_MODEL=gpt-4.1
     ```

**Tests**:
- `test_prompt_templates.py`:
  - `test_system_prompt_contains_role()`: verify the system prompt establishes the data analyst role
  - `test_exploration_prompt_references_data()`: verify data file path is mentioned
  - `test_modeling_prompt_includes_findings()`: verify previous exploration output is included

- `test_orchestrator.py`:
  - `test_exploration_produces_output()`: run exploration stage on pilot dataset, verify agent produces some statistical analysis
  - `test_modeling_produces_code()`: run through exploration + modeling, verify agent produces a model that can be executed
  - `test_generation_produces_dataframe()`: run full pipeline, verify output is a DataFrame with correct columns and row count
  - `test_validation_loop_iterates()`: mock a failing validation, verify the agent loops back
  - `test_full_pipeline_end_to_end()`: run complete pipeline on pilot dataset, verify output format matches SSDataBench expectations

- `test_llm_client.py`:
  - `test_openai_provider_init()`: verify client initializes with OpenAI config
  - `test_anthropic_provider_init()`: verify client initializes with Anthropic config
  - `test_custom_base_url()`: verify client uses custom base_url when provided
  - `test_api_call_openai()`: verify a simple prompt returns a response via OpenAI format
  - `test_api_call_anthropic()`: verify a simple prompt returns a response via Anthropic format
  - `test_code_extraction()`: verify code blocks are correctly extracted from markdown responses
  - `test_config_from_env()`: verify LLMConfig loads from environment variables

**Acceptance Criteria**: The full agent pipeline runs end-to-end on the pilot dataset, producing a synthetic dataset in the correct format.

---

### Phase 4: Output Formatting and Evaluation Bridge

**Goal**: Convert agent output to SSDataBench format and run evaluation.

**Tasks**:
1. **Formatter** (`src/generation/formatter.py`):
   - Take agent's generated DataFrame and format it to match SSDataBench's expected CSV schema
   - Handle variable name mapping, value encoding, column ordering
   - Write to `ssdatabench/simulated_data/<run_name>/` following their naming convention (`sim_profiles_*.csv` or similar)

2. **Evaluation runner** (`src/evaluation/runner.py`):
   - Call SSDataBench's evaluation scripts programmatically
   - Parse output pass rates from `evaluation_results/`
   - Return structured results (dict of type -> dataset -> pass_rate)

3. **Comparator** (`src/evaluation/comparator.py`):
   - Load pass rates from multiple conditions
   - Produce comparison tables (condition x type x dataset)
   - Compute summary statistics (mean pass rate per condition, per type)

**Tests**:
- `test_formatter.py`:
  - `test_output_columns_match()`: verify output CSV has same columns as SSDataBench simulated data
  - `test_output_values_in_range()`: verify all generated values are within allowed ranges defined in schema
  - `test_output_row_count()`: verify output has 1000 rows (matching SSDataBench sample size)

- `test_evaluation_runner.py`:
  - `test_run_evaluation_on_formatted_data()`: run SSDataBench evaluation on agent-generated data, verify it completes
  - `test_parse_pass_rates()`: verify pass rates are correctly extracted from evaluation output

- `test_comparator.py`:
  - `test_comparison_table_shape()`: verify table has correct dimensions (conditions x types)
  - `test_summary_statistics()`: verify mean computation is correct

**Acceptance Criteria**: Agent-generated data flows through formatting, evaluation, and comparison without errors. Pass rates are produced and comparable to paper results.

---

### Phase 5: Experiment Runner

**Goal**: Orchestrate the full experiment matrix across conditions and datasets.

**Tasks**:
1. **Conditions** (`src/experiments/conditions.py`):
   - Define each experimental condition as a configuration object:
     - `DirectGeneration`: use SSDataBench simulation code (baseline)
     - `FullAgent`: full context
     - `AgentNoSemantic`: no descriptions
     - `AgentNoData`: no data
   - Each condition specifies: context type, whether to run agent or direct generation, any condition-specific parameters

2. **Experiment runner** (`src/experiments/runner.py`):
   - Takes: list of conditions, list of datasets, experiment config
   - For each (condition, dataset) pair:
     1. Set up context per condition
     2. Run generation (agent or direct)
     3. Format output
     4. Run evaluation
     5. Store results
   - Support resuming (skip already-completed runs)
   - Log everything (prompts, agent code, execution output, timing)

3. **Logging** (`src/experiments/logger.py`):
   - For each agent run, save: all prompts sent, all LLM responses, all code executed, all execution outputs, final generated data, evaluation results
   - This is critical for reproducibility and for the paper (showing agent reasoning traces)

**Tests**:
- `test_conditions.py`:
  - `test_all_conditions_defined()`: verify all four conditions exist
  - `test_condition_context_types()`: verify each condition produces the correct context type

- `test_experiment_runner.py`:
  - `test_single_run()`: run one condition on one dataset, verify results are stored
  - `test_resume_skips_completed()`: verify completed runs are not re-executed
  - `test_logging_completeness()`: verify all expected log files are produced

**Acceptance Criteria**: Full experiment matrix can be run, resumed, and produces complete results with logging.

---

### Phase 6: Unseen Variable Experiment

**Goal**: Extend the system to support the unseen variable experimental condition.

**Tasks**:
1. **Unseen variable configuration** (extend `src/data/schema.py`):
   - Allow marking specific target variables as "unseen"
   - Define which variables to hide (select a meaningful subset per dataset)

2. **Unseen variable context** (extend `src/agent/context.py`):
   - When variables are marked unseen: exclude their columns from the training data, but include their descriptions in the prompt
   - At generation time, the agent must produce values for all variables including unseen ones

3. **Unseen variable evaluation** (extend `src/evaluation/runner.py`):
   - Run evaluation separately for seen and unseen variables
   - Report pass rates split by seen/unseen to show the differential

**Tests**:
- `test_unseen_variables.py`:
  - `test_unseen_vars_excluded_from_data()`: verify unseen variables are not in the training data
  - `test_unseen_vars_in_descriptions()`: verify unseen variable descriptions are provided to agent
  - `test_generation_includes_unseen()`: verify output includes columns for unseen variables
  - `test_evaluation_split_seen_unseen()`: verify pass rates are reported separately

**Acceptance Criteria**: The system can run the unseen variable experiment and report differential pass rates.

---

### Phase 7: Baseline (Direct LLM Generation)

**Goal**: Re-run the SSDataBench direct generation baseline with the same LLM provider and model used for the agent, ensuring fair comparison.

**Tasks**:
1. **Direct generation wrapper** (extend `src/experiments/conditions.py`):
   - Use SSDataBench's own simulation code (`ssdatabench/simulation/`)
   - Configure to use the same model, provider, and temperature as the agent condition (from `llm.yaml`)
   - Generate 1000 synthetic individuals per dataset

2. **Integration with experiment runner**:
   - The `DirectGeneration` condition calls SSDataBench's simulation instead of the agent
   - Output is formatted and evaluated through the same pipeline

**Tests**:
- `test_direct_generation.py`:
  - `test_direct_gen_produces_output()`: verify SSDataBench simulation produces a CSV
  - `test_direct_gen_format_compatible()`: verify output format matches evaluation expectations
  - `test_direct_gen_uses_same_model()`: verify model configuration matches agent condition

**Acceptance Criteria**: We have baseline pass rates from the same model used for agent conditions, enabling fair comparison.

---

## Configuration Files

### datasets.yaml (example for pilot)

```yaml
datasets:
  gss_2018:
    real_data_path: ssdatabench/real_data/gss_2018.csv
    type: cross-sectional
    context: "Adults aged 18 and older in U.S. households, from the 2018 General Social Survey"
    survey_year: 2018
    background_variables:
      - Gender
      - Race
      - Age
    target_variables:
      # populated from SSDataBench config
    evaluation_script: ssdatabench/scripts/evaluation/gss_2018.py
```

### experiments.yaml (example)

```yaml
experiment:
  name: pilot_gss_2018
  datasets:
    - gss_2018
  conditions:
    - full_agent
    - agent_no_semantic
    - agent_no_data
    - direct_generation
  agent:
    max_iterations: 3
    sandbox_timeout_seconds: 60
    train_eval_split: 0.5
  unseen_variables:
    enabled: false  # enable in Phase 6
```

### llm.yaml (example)

```yaml
# LLM provider configuration
# Can also be overridden by environment variables (LLM_PROVIDER, LLM_BASE_URL, LLM_API_KEY, LLM_MODEL)
provider: openai
base_url: https://api.openai.com/v1
model: gpt-4.1
temperature: 1.0
max_tokens: 4096
# api_key: loaded from LLM_API_KEY env var, never stored in yaml
```

---

## Python Dependencies

```
# Core
python >= 3.10
pandas
numpy
scipy
statsmodels
scikit-learn
matplotlib

# LLM clients
openai
anthropic

# Config and environment
pyyaml
python-dotenv

# Testing
pytest
pytest-mock

# SSDataBench dependencies (from their requirements.txt)
# installed separately via: pip install -r ssdatabench/requirements.txt
```

---

## Key Design Decisions

1. **The agent generates a model, not individual cases.** The agent's final output is executable code (a fitted statistical model or sampling procedure) that can generate any number of synthetic individuals. We then use this code to produce 1000 individuals for evaluation.

2. **The agent sees a train split, evaluation uses the eval split.** We split the 1000 sampled individuals. The agent builds its model on the train split. We generate 1000 synthetic individuals and evaluate them against the eval split's statistics. This prevents data leakage while giving the agent real data to learn from.

3. **Conversation history is preserved within a run.** The agent's multi-step workflow maintains full conversation context so later stages can reference earlier findings. This is essential for coherent model building.

4. **Everything is logged.** Every prompt, response, code execution, and output is saved. This enables both reproducibility and qualitative analysis of agent reasoning for the paper.

5. **SSDataBench evaluation code is used as-is.** We do not modify their evaluation logic. We only build a bridge to format our output and call their scripts. This ensures results are directly comparable.

---

## Success Criteria

The project succeeds if:

1. The Full Agent condition achieves meaningfully higher average pass rates than Direct LLM Generation across SSDataBench's five statistical pattern types.

2. Ablation results show that both semantic knowledge and data access contribute to performance (neither Agent-No-Semantic nor Agent-No-Data fully matches Full Agent).

3. The unseen variable experiment shows that the Full Agent can produce non-trivial distributional predictions for variables it has never observed in data.

4. All results are reproducible through logged agent traces and deterministic evaluation.
