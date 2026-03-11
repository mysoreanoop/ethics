# Makefile for Reddit Emotion Analysis

# Python interpreter
PYTHON ?= python
PIP ?= $(PYTHON) -m pip

# Directories
DATA_DIR = data
PLOT_DIR = plots
SRC_DIR = src

# Targets
.PHONY: all install fetch analyze plot nlu figures nlu-all clean help

all: install fetch analyze plot
	@echo "Pipeline completed successfully."

help:
	@echo "Available commands:"
	@echo "  make install   - Install Python dependencies"
	@echo "  make fetch     - Scrape data from Reddit"
	@echo "  make analyze   - Run emotion recognition on data"
	@echo "  make plot      - Generate graphs and conclusions"
	@echo "  make nlu       - Run BERTopic + GoEmotions + ABSA correlation pipeline"
	@echo "  make figures   - Re-generate figures from NLU analysis outputs"
	@echo "  make nlu-all   - Run NLU analysis and figures in sequence"
	@echo "  make clean     - Remove all generated data and plots"

$(DATA_DIR):
	mkdir -p $(DATA_DIR)

$(PLOT_DIR):
	mkdir -p $(PLOT_DIR)

install:
	$(PIP) install -r requirements.txt

fetch: $(DATA_DIR)
	@echo "Fetching data..."
	$(PYTHON) $(SRC_DIR)/fetch_data.py

analyze: fetch
	@echo "Analyzing emotions..."
	$(PYTHON) $(SRC_DIR)/analyze_emotions.py

plot: $(PLOT_DIR) analyze
	@echo "Visualizing results..."
	$(PYTHON) $(SRC_DIR)/visualize.py

clean:
	rm -rf $(DATA_DIR) $(PLOT_DIR)
	@echo "Cleaned up data and plots."

nlu:
	@echo "Running integrated NLU pipeline..."
	$(PYTHON) $(SRC_DIR)/analyze_topics_emotions.py

figures:
	@echo "Generating integrated NLU figures..."
	$(PYTHON) $(SRC_DIR)/plot_topics_emotions.py

nlu-all: nlu figures
	@echo "Integrated NLU analysis and figures completed."
