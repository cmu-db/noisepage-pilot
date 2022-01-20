# dodos

The `dodos` folder contains the various `dodo` files used by [doit](https://pydoit.org/).

Think of `doit` scripts as a custom DSL which can freely mix `bash` and `Python`.

TODO(WAN):

- More `doit` documentation.
- It is OK to hardcode and refer to paths here (paths embedded into the caller of an app) even though we ban it elsewhere (paths embedded into an app).
- Write up more guidelines. 
- Explain the slight callstack magic for artifact/build folder convention.
