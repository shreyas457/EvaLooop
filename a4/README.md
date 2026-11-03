# A4 — EvaLoop extension: the code ↔ pseudocode loop

**Fork:** https://github.com/shreyas457/EvaLooop (branch `a4-pseudocode-loop`)
**Upstream:** https://github.com/evalooop/EvaLooop @ `ce22fe2`
**Patch:** `a4-pseudocode-loop.patch` (= `git diff ce22fe2 HEAD`)

This adds a new loop type to EvaLoop (Option B, a new attack). The model's
inverse task is no longer "summarize your code in one sentence". It is "describe your code as
language-agnostic pseudocode". The pseudocode then becomes the next cycle's generation prompt.
The model's own output is still the only source of perturbation.

## The loop

| | Standard loop (`code_generation_summarization`) | Extension (`code_generation_pseudocode`) | Ablation (`code_generation_pseudocode_noassert`) |
|---|---|---|---|
| Forward task | code generation (identical prompt in all three) | same | same |
| Inverse task | one-sentence summary starting "write a python function to" | step-by-step pseudocode: function name and parameters, control flow, operations, return value, with pseudocode keywords instead of Python syntax | same as extension |
| Carried to the next cycle | `"""\n<summary>\n<original assert>\n"""` | `"""\n<pseudocode>\n<original assert>\n"""` | `"""\n<pseudocode>\n"""` |
| Signature stability | the assert line | the assert line, plus the function name inside the pseudocode | only the function name the model writes in its pseudocode |
| Failure condition | generated code fails MBPP+ base or plus tests (EvalPlus) | same | same |
| Budget | 10 cycles, greedy | same | same |

Prompts used, verbatim:

* Generation (all loops, unchanged from upstream `config.py`):
  `Generate Python code for the following task: \n{description}\nReturn only the code, without explanations or comments.`
  The first fenced block of the response is taken as the code.
* Pseudocode (`CodeToPseudocodeTask.DEFAULT_PROMPT_TEMPLATE`):
  ````
  Convert the following Python code into pseudocode:

  ```
  {code}
  ```

  Write language-agnostic pseudocode that describes the algorithm step by step: the function name and its parameters, the control flow, the operations performed and the value returned. Use plain pseudocode keywords (FUNCTION, IF, ELSE, FOR EACH, WHILE, RETURN) instead of Python syntax, and do not copy the Python code.
  Return only the pseudocode, without explanations.
  ````
  If the response contains a fenced block, its content is used. Otherwise the stripped response is used.

The pseudocode is not post-filtered: rewriting it would add a perturbation that doesn't come from
the model. If the "pseudocode" is close to Python, the loop degenerates into a copy task, so
`analyze_a4.py` measures that per run instead (`pseudocode_leakage.csv`: share of prompts that parse
as Python, verbatim line copies, token Jaccard with the source code).

## What I changed in the framework, and why

New files:

* `evaloop/tasks/code_pseudocode.py` contains `CodeToPseudocodeTask`, registered as `code_to_pseudocode`.
  It subclasses `CodeSummarizationTask` so that the assert-line re-attachment is the same code path
  as the standard loop. `retain_assert: False` drops the line for the ablation.
* `tests/test_pseudocode.py` has unit tests for the task, the loop configs, two mocked cycles, and task subsets.
* `a4/analyze_a4.py` produces summary.csv, the paired comparison, leakage diagnostics, failure pre-labels,
  the manual-inspection sample, the data-package cross-check, and survival curves.
* `a4/A4_pseudocode_loop_colab.ipynb` is the Colab runner for all experiments.

Modified files. The loop runner (`evaluation/cycle.py`) and the tester are unchanged.

* `evaloop/tasks/__init__.py` gets one import line to register the task, as CONTRIBUTING.md describes.
* `evaloop/core/config.py`:
  * new `loop_type` field with two new branches in `get_task_configs()`. CONTRIBUTING.md names this as
    the one place a new loop touches existing code. The generation config is shared, so it is
    provably identical across loops.
  * `task_ids` / `n_tasks` / `task_seed` plus `select_tasks()` for optional seeded subsets. Not used for
    the main runs, which cover all 378 tasks.
  * `use_chat_template` and `dtype` are passed through to vLLM (see below).
* `evaloop/core/evaluator.py`: `run_loop()` runs `config.loop_type`. Upstream hard-coded the standard loop.
  `run_code_generation_summarization()` is kept and delegates to it. The results file is named after
  the loop type and records `loop_type` and the full config.
* `evaloop/cli.py` adds the `--loop_type`, `--use_chat_template`, `--n_tasks`, `--task_seed`, `--task_ids`
  and `--dtype` flags. All defaults reproduce upstream behaviour.
* `evaloop/models/open_source_vllm.py` adds optional chat-template formatting. Upstream wraps every
  prompt as the plain text `System: …\n\nUser: …\n\nAssistant:`. Instruct models aren't trained on
  that format, and it lets them continue with invented `User:` turns. With `--use_chat_template` the
  tokenizer's own template is used, and templates that reject a system role (Gemma) get it folded
  into the user turn. It is off by default, so upstream behaviour is unchanged. All of my runs,
  baseline included, use it.

Things I found while reading the code, which matter when interpreting the data package:

* `prompts/task_prompts.md` in the data package documents the summarization prompt as "Summarize
  what the following code does in a detailed description". The released `config.py`, and the actual
  summaries in `loops/*.json`, use the one-sentence prompt ("write a python function to …"). A third
  variant is the unused default inside `CodeSummarizationTask`. My baseline uses the `config.py` prompt,
  because that matches the data.
* The assert line is taken as `cycle_input.splitlines()[-2]`, which only works for MBPP+'s prompt layout
  (assert on the second-to-last line, closing `"""` last). Every later prompt is built by the same
  wrapper, so the line survives every cycle.
* If the inverse step errors after a passing cycle, the loop ends without a failing cycle being recorded.
  `analyze_a4.py` only labels loops that have a failing cycle.

## How to run

### Colab (what I used)

1. The notebook clones the fork (`FORK_URL` in the config cell).
2. Upload the provided data package folder (`assignment2-data/`) to `MyDrive/A4_evaloop/` for the cross-check.
3. Open `a4/A4_pseudocode_loop_colab.ipynb` in Colab with a GPU runtime (L4 or A100. On a T4 use
   `DTYPE="half"` and the 1.5B/3B models listed there). Run all cells. Results go to
   `MyDrive/A4_evaloop/results/`, finished runs are skipped on re-run, and the last cell assembles
   `MyDrive/A4_evaloop/submission/`.

### Command line (any Linux + CUDA machine)

```bash
pip install -e . "vllm==0.11.0" "transformers>=4.56,<5" scipy
for model in Qwen/Qwen2.5-Coder-7B-Instruct ByteDance-Seed/Seed-Coder-8B-Instruct; do
  name=$(basename $model | tr 'A-Z.-' 'a-z__')
  for loop in code_generation_summarization code_generation_pseudocode code_generation_pseudocode_noassert; do
    evaloop evaluate --model $model --loop_type $loop --use_chat_template True \
      --max_cycles 10 --temperature 0.0 --max_new_tokens 1024 \
      --auto_batch_size False --preferred_batch_size 512 \
      --output_dir results/a4 --experiment_name $name
  done
done
python a4/analyze_a4.py --results_dir results/a4 --out_dir results/a4/analysis \
    --data_package path/to/assignment2-data
```

Unit tests, which need no GPU, network, or API key: `pip install -e ".[dev]" && pytest && ruff check evaloop tests a4`.

## Setup, dependencies, and where each number comes from

* Models under evaluation: `Qwen/Qwen2.5-Coder-7B-Instruct` and `ByteDance-Seed/Seed-Coder-8B-Instruct`
  (HuggingFace weights, served locally with vLLM 0.11.0, bf16, `max_model_len` 4096). No API models
  and no LLM judge are used. Hardware: one Colab A100-40GB. Cost is Colab GPU time only
  (about 1.5 A100-hours, ~90 min, for all six runs).
* Tools: I chose the extension, models and experimental protocol. Claude (Anthropic, via Claude Code)
  assisted with the implementation and the analysis scripts, and suggested initial failure categories.
  Manual inspection (`analysis/manual_labels.csv`): I reviewed the first 20 sampled loop failures in
  `analysis/manual_inspection.md` against their prompts and code and confirmed every label.
  `test_scaffold_propagation` was added to the label set. It marks failures where the pseudocode
  describes the model's own pasted test code (`assert` lines), which the next cycle re-implements
  as invalid Python such as `def assert(...)`.
* Decoding: greedy (temperature 0), max 1024 new tokens, max 10 cycles, one run per configuration,
  system prompt `You are a helpful programming assistant.`, each model's chat template.
* Tasks: all 378 MBPP+ tasks (EvalPlus `get_mbpp_plus()`, MBPP+ v0.2.0). No subsetting.
* Testing: EvalPlus `untrusted_check` against base and plus inputs (upstream tester, unchanged). A
  cycle passes only if both pass. Generated code is executed only inside the Colab VM.
* Baseline: the same models on the same 378 tasks under the standard loop, re-run with the same
  serving stack. The data-package records for the same models are a reproducibility cross-check
  only (`datapackage_check.csv`). They were produced with a different prompt format, so they are
  not the baseline.
* `summary.csv`: `success_rate` = pass@1 / T. `asl_score` = ASL_pow = Σᵢ i²·nᵢ / (T·M), the leaderboard's
  `aslScore` column. I report ASL without the semantic-similarity weighting because no judge was
  used. `analyze_a4.py` computes both with the framework's own `evaloop.analysis.metrics`, and
  `evaloop analyze --metrics ASL_pow,ASL_bias,pass@1,pass_drop` on each results file gives the same values.
* Every table in the report comes from one file in `analysis/`: summary/metrics (`summary.csv`,
  `metrics_full.csv`), paired differences with 95% paired-bootstrap CIs (10,000 resamples, seed 0) and
  Wilcoxon signed-rank p-values (`paired_comparison.csv`), rankings (`ranking.csv`), leakage
  (`pseudocode_leakage.csv`), automatic failure categories (`failure_label_counts.csv`), and manual
  labels (`manual_labels.csv`, the sample drawn with seed 0 from loops that failed at cycle ≥ 2).
* Per-cycle records are referenced as (model, configuration, task_id, cycle), with the configuration
  names from `summary.csv`.
