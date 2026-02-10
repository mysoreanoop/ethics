# Makefile for Reddit Emotion Analysis

# Python interpreter
PYTHON = python3
PIP = pip3

# Directories
DATA_DIR = data
PLOT_DIR = plots
SRC_DIR = src

# Targets
.PHONY: all install fetch analyze plot clean help

all: install fetch analyze plot
	@echo "Pipeline completed successfully."

help:
	@echo "Available commands:"
	@echo "  make install   - Install Python dependencies"
	@echo "  make fetch     - Scrape data from Reddit"
	@echo "  make analyze   - Run emotion recognition on data"
	@echo "  make plot      - Generate graphs and conclusions"
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
