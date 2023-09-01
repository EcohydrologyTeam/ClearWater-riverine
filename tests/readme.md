Tests
===

`tests` is for unit testing scripts (i.e. `test_basic.py`) designed for manual and automated testing. 

Example models that showcase the Clearwater-riverine model capabilities and/or used for tutorials do not belong in this folder, but may be place in the `examples/data` directory. Also, Jupyter Notebooks do not belong in the `tests` directory--please use the `examples/dev_sandbox/` directory. 

Example models that are needed for testing purposes may use the `tests/data` directory.

The pytest library is used for testing in this repo. The relative paths in the *.py modules in `tests` are referenced from the root directory of the repo, i.e. please run pytest only at the ./ClearWater-riverine directory level or pytest will give the "FileNotFoundError" message.

