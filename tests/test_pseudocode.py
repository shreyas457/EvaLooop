"""Unit tests for the code <-> pseudocode loop (task, config wiring, mocked cycles)."""

import pytest

from evaloop.core.config import LOOP_TYPES, EvaluationConfig
from evaloop.evaluation.cycle import BatchEvaluationCycle
from evaloop.tasks import CodeToPseudocodeTask, TaskFactory

from conftest import ASSERT_LINE, MBPP_STYLE_PROMPT

MOCK_CODE = "def similar_elements(a, b):\n    return tuple(set(a) & set(b))"
MOCK_PSEUDOCODE = (
    "FUNCTION similar_elements(a, b)\n"
    "    RETURN the elements present in both a and b, as a tuple\n"
    "END FUNCTION"
)


class MockModel:
    """Generation prompts get code, pseudocode prompts get fenced pseudocode."""

    def __init__(self):
        self.prompts = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if prompt.startswith("Generate Python code"):
            return f"```python\n{MOCK_CODE}\n```"
        return f"```\n{MOCK_PSEUDOCODE}\n```"


def make_config(tmp_path, **kwargs):
    return EvaluationConfig(model="gpt-4", output_dir=str(tmp_path), **kwargs)


class TestCodeToPseudocodeTask:
    def test_registered(self):
        task = TaskFactory().create_task({"type": "code_to_pseudocode"})
        assert isinstance(task, CodeToPseudocodeTask)
        assert task.produces_testable_code is False

    def test_create_prompt_embeds_code(self):
        prompt = CodeToPseudocodeTask({}).create_prompt(MOCK_CODE)
        assert MOCK_CODE in prompt
        assert "pseudocode" in prompt.lower()
        assert "do not copy the Python code" in prompt

    def test_custom_template(self):
        task = CodeToPseudocodeTask({"prompt_template": "PSEUDO: {code}"})
        assert task.create_prompt("x") == "PSEUDO: x"

    @pytest.mark.parametrize("response", [
        f"```\n{MOCK_PSEUDOCODE}\n```",
        f"```pseudocode\n{MOCK_PSEUDOCODE}\n```",
        f"Here it is:\n```text\n{MOCK_PSEUDOCODE}\n```\nDone.",
        f"  {MOCK_PSEUDOCODE}  \n",
    ])
    def test_extract_output(self, response):
        assert CodeToPseudocodeTask({}).extract_output(response) == MOCK_PSEUDOCODE

    def test_extract_output_ignores_empty_fence(self):
        assert CodeToPseudocodeTask({}).extract_output("```\n```") == "```\n```"

    def test_postprocess_retains_assert_line_by_default(self):
        result = CodeToPseudocodeTask({}).postprocess(
            MOCK_PSEUDOCODE, {"cycle_input": MBPP_STYLE_PROMPT}
        )
        assert result == f'"""\n{MOCK_PSEUDOCODE}\n{ASSERT_LINE}\n"""\n'

    def test_postprocess_without_assert(self):
        result = CodeToPseudocodeTask({"retain_assert": False}).postprocess(
            MOCK_PSEUDOCODE, {"cycle_input": MBPP_STYLE_PROMPT}
        )
        assert result == f'"""\n{MOCK_PSEUDOCODE}\n"""\n'
        assert "assert" not in result


class TestLoopConfig:
    def test_default_loop_is_standard(self, tmp_path):
        config = make_config(tmp_path)
        types = [t["type"] for t in config.get_task_configs()]
        assert types == ["code_generation", "code_summarization"]

    @pytest.mark.parametrize("loop_type,retain", [
        ("code_generation_pseudocode", True),
        ("code_generation_pseudocode_noassert", False),
    ])
    def test_pseudocode_loops(self, tmp_path, loop_type, retain):
        config = make_config(tmp_path, loop_type=loop_type)
        generation, inverse = config.get_task_configs()
        assert inverse == {"type": "code_to_pseudocode", "retain_assert": retain}
        # The generation step must be identical to the standard loop's
        standard = make_config(tmp_path).get_task_configs()[0]
        assert generation == standard

    def test_every_loop_type_builds_tasks(self, tmp_path):
        factory = TaskFactory()
        for loop_type in LOOP_TYPES:
            configs = make_config(tmp_path, loop_type=loop_type).get_task_configs()
            tasks = [factory.create_task(c) for c in configs]
            assert [t.produces_testable_code for t in tasks] == [True, False]

    def test_unknown_loop_type_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="loop_type"):
            make_config(tmp_path, loop_type="code_generation_haiku")


class TestPseudocodeCycle:
    @pytest.mark.parametrize("retain", [True, False])
    def test_two_cycles_carry_pseudocode(self, tmp_path, retain):
        loop_type = "code_generation_pseudocode" + ("" if retain else "_noassert")
        factory = TaskFactory()
        tasks = [factory.create_task(c)
                 for c in make_config(tmp_path, loop_type=loop_type).get_task_configs()]
        model = MockModel()
        runner = BatchEvaluationCycle(model, tasks)

        first = runner.run_single_cycle([{"task_id": "Mbpp/2", "prompt": MBPP_STYLE_PROMPT}], 1)
        result = first["cycle_results"][0]
        assert result["task_1_output"] == MOCK_CODE
        next_prompt = result["task_2_output"]
        assert MOCK_PSEUDOCODE in next_prompt
        assert (ASSERT_LINE in next_prompt) is retain

        second = runner.run_single_cycle([{"task_id": "Mbpp/2", "prompt": next_prompt}], 2)
        # Cycle-2 generation prompt is built from the cycle-1 pseudocode
        assert MOCK_PSEUDOCODE in model.prompts[2]
        # With the assert retained, it keeps being carried from the cycle input
        assert (ASSERT_LINE in second["cycle_results"][0]["task_2_output"]) is retain


class TestTaskSubset:
    PROMPTS = [{"task_id": f"Mbpp/{i}", "prompt": "p"} for i in range(50)]

    def _select(self, tmp_path, **kwargs):
        return make_config(tmp_path, **kwargs).select_tasks(list(self.PROMPTS))

    def test_all_by_default(self, tmp_path):
        assert len(self._select(tmp_path)) == 50

    def test_seeded_sample_is_reproducible_and_ordered(self, tmp_path):
        a = self._select(tmp_path, n_tasks=10, task_seed=7)
        b = self._select(tmp_path, n_tasks=10, task_seed=7)
        assert a == b and len(a) == 10
        ids = [int(p["task_id"].split("/")[1]) for p in a]
        assert ids == sorted(ids)

    def test_explicit_ids(self, tmp_path):
        chosen = self._select(tmp_path, task_ids=["Mbpp/3", "Mbpp/1"])
        assert [p["task_id"] for p in chosen] == ["Mbpp/1", "Mbpp/3"]

    def test_unknown_ids_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown task ids"):
            self._select(tmp_path, task_ids=["Mbpp/999"])
