# Visual_Intelligence_2023

To run the project execute the following command in the terminal:

```bash
python3 configurable_classification.py
```

Settings can be found in the files "parameters.yaml" and "scatter_parameters.yaml", but they will be overwritten at every execution.
The correct way to edit the settings is by running lib/scripts/make_settings.py

There is also a tool for automated execution, "gridsearch.py", which can be used to run the program with different settings. Warning: it is not an actual gridsearch.
Execution results are saved in the folder results and numbered ascendently.