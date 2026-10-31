import re
from typing import Dict, Any

from .code_summarization import CodeSummarizationTask
from .factory import register_task


@register_task("code_to_pseudocode")
class CodeToPseudocodeTask(CodeSummarizationTask):
    """Inverse task of code generation: describe the code as pseudocode.

    Used by the code <-> pseudocode loop. Where the standard loop compresses the
    code into a one-sentence natural-language summary, this task asks for a
    language-agnostic, step-by-step algorithm description. The pseudocode then
    becomes the next cycle's generation input, so the model's own pseudocode is
    the only source of perturbation.

    Signature stability: by default the cycle's assert line is re-attached exactly
    like CodeSummarizationTask.postprocess does (the function name and call shape
    are pinned by one example). With ``retain_assert: False`` the pseudocode is
    wrapped without it, so the function name must survive through the pseudocode
    alone (ablation of the assert-line design decision).
    """

    DEFAULT_PROMPT_TEMPLATE = (
        "Convert the following Python code into pseudocode:\n\n"
        "```\n{code}\n```\n\n"
        "Write language-agnostic pseudocode that describes the algorithm step by step: "
        "the function name and its parameters, the control flow, the operations performed "
        "and the value returned. Use plain pseudocode keywords (FUNCTION, IF, ELSE, FOR EACH, "
        "WHILE, RETURN) instead of Python syntax, and do not copy the Python code.\n"
        "Return only the pseudocode, without explanations."
    )

    _FENCE_PATTERN = re.compile(r"```(?:[\w+-]+)?[ \t]*\n?([\s\S]*?)```")

    def __init__(self, task_config: Dict[str, Any]):
        """Initialize task with configuration."""
        super().__init__(task_config)
        self.retain_assert = task_config.get("retain_assert", True)

    def create_prompt(self, input_text: str) -> str:
        """Create a prompt asking for pseudocode of the given code."""
        prompt_template = self.task_config.get("prompt_template", self.DEFAULT_PROMPT_TEMPLATE)
        return prompt_template.format(code=input_text)

    def extract_output(self, raw_response: str) -> str:
        """Extract the pseudocode, unwrapping a fenced block if the model used one."""
        matches = self._FENCE_PATTERN.findall(raw_response)
        if matches and matches[0].strip():
            return matches[0].strip()
        return raw_response.strip()

    def postprocess(self, output: str, context: Dict[str, Any]) -> str:
        """Wrap the pseudocode as a docstring, with or without the cycle's assert line."""
        if self.retain_assert:
            return super().postprocess(output, context)
        return f"\"\"\"\n{output}\n\"\"\"\n"
