# Project goal

Build a beginner-friendly interdisciplinary tutoring and diagnostic
application for probability, statistics, and Python data analysis.

The user-facing language is Simplified Chinese.
Code identifiers and technical documentation may use English.

# Learner profile

The repository owner is a beginner in:

- Python
- Probability and statistics
- LLM application development
- Software engineering

Explain important changes in beginner-friendly language.

# Working process

- Work on only one independently testable task per turn.
- Before editing, state the goal and files that will change.
- Prefer minimal patches over large rewrites.
- After editing, list:
  - changed files;
  - important design decisions;
  - commands executed;
  - test results;
  - remaining risks.
- Run relevant tests after every functional change.
- Do not add a production dependency without explaining why.
- Never commit API keys, passwords, tokens, or personal learner data.

# Engineering rules

- Use Python 3.11 or later.
- Use type hints.
- Use Pydantic for important input and output schemas.
- Use pytest for tests.
- Keep business logic outside Streamlit page code.
- Do not use eval() or exec() on learner-submitted code.
- Do not run untrusted code in the main web application process.
- Keep the OpenAI model name configurable through an environment variable.

# Pedagogical rules

- Do not immediately reveal the final answer.
- Use progressive hints:
  1. conceptual cue;
  2. method or formula cue;
  3. partial worked step;
  4. complete explanation.
- Separate:
  - statistical concept understanding;
  - mathematical calculation;
  - Python implementation;
  - data interpretation.
- Deterministic graders decide numerical and code correctness.
- The LLM may explain results but must not invent scores.
- Every diagnosis must cite observable evidence from the learner response.

# Definition of done

A task is complete only when:

- the implementation is understandable;
- relevant tests pass;
- errors are handled;
- documentation is updated;
- the change can be demonstrated locally.
