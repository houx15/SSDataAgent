from ssdataagent.agent.code_extraction import extract_python_block


def test_extracts_fenced_python():
    text = "Here is code:\n```python\nx = 1\nprint(x)\n```\nDone."
    assert extract_python_block(text) == "x = 1\nprint(x)"


def test_extracts_bare_fence():
    text = "```\nprint('hi')\n```"
    assert extract_python_block(text) == "print('hi')"


def test_extracts_py_alias():
    text = "```py\nprint('y')\n```"
    assert extract_python_block(text) == "print('y')"


def test_returns_first_block_when_multiple():
    text = "```python\nA\n```\nthen\n```python\nB\n```"
    assert extract_python_block(text) == "A"


def test_returns_none_when_no_block():
    assert extract_python_block("no code here") is None


def test_handles_empty_block():
    assert extract_python_block("```python\n\n```") == ""
