# Design Doc: Glue Library Choice

## Criteria

- Easy to learn: should have less than one day of onboarding for the average student.
- Easy to replace: easy to scale up, or out, or toss it entirely.
- Substantially better than rolling our own.
- Must support C/C++/bash/Python interop.

## Choice

The libraries we used are:

- doit: if Make had better syntax and was in Python.
    - https://github.com/pydoit/doit
- plumbum: if Python could use bash constructs natively.
    - https://plumbum.readthedocs.io/en/latest/

### doit

`doit` is the simplest of the Python task runners that I could find.

It is close enough to regular bash that anybody should be able to read it, and it has some nice features:

- Task dependencies.
- Task documentation from docstring.
- Configurable logic for determining whether to re-execute a task.

Unfortunately, I don't think `doit` interacts well with `plumbum`. So we have one silly layer of just passing arguments
through, i.e., lines of code whose sole purpose is to feed parameters from X to Y. For now, this is fine, but fixes are
welcome.

Alternatives that were considered:

- [snakemake](https://snakemake.github.io/)
    - Popular in the bioinformatics community, made by academics.
    - Make-like syntax.
    - Target can be executed in different containers, conda envs, etc.
    - Save inputs/outputs of each step.
    - But quite verbose.
    - People probably are more familiar with Python than Make.
    - The extra features aren't something we need right now.
- [Metaflow](https://metaflow.org/)
    - Developed by Netflix.
    - Native Python code.
    - Extremely high powered reproducibility, save even variable values.
    - But very opinionated on how your workflow should be structured.
    - Learning curve is too high for our use case.

### plumbum

Conceptually, `plumbum` is a nice wrapper around `argparse` and `subprocess`
with additional remote execution capabilities.

- `argparse`
    - But passing arguments to/from argparse applications is verbose.
- `subprocess`
    - Capturing stdout/stderr properly is tricky, `plumbum` handles it.
    - `plumbum` also supports remote execution (with some caveats).
 