#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = CHE1147_Final_Project
PYTHON_VERSION = 3.10
PYTHON_INTERPRETER = uv run

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Full initial setup
.PHONY: setup
setup: create_environment requirements


## Sync Python dependencies
.PHONY: requirements
requirements:
	uv sync
	



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format





## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	uv venv --python $(PYTHON_VERSION)
	@echo ">>> New uv virtual environment created. Activate with:"
	@echo ">>> Windows: .\\\\.venv\\\\Scripts\\\\activate"
	@echo ">>> Unix/macOS: source ./.venv/bin/activate"
	



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Run the entire pipeline: data -> features -> train -> evaluate
.PHONY: all
all: requirements data features train evaluate

## Process raw data to create cleaned interim data
.PHONY: data
data:
	$(PYTHON_INTERPRETER) src/dataset.py

## Generate features from interim data
.PHONY: features
features:
	$(PYTHON_INTERPRETER) src/features.py

## Train models on processed features
.PHONY: train
train:
	$(PYTHON_INTERPRETER) src/modeling/train.py

## Evaluate models on the test set
.PHONY: evaluate
evaluate:
	$(PYTHON_INTERPRETER) src/modeling/predict.py

## Generate plots after evaluation
.PHONY: plots
plots:
	$(PYTHON_INTERPRETER) src/plots.py


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
