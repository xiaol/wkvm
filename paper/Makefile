LATEXMK ?= latexmk
BUILD_DIR := build
MAIN := main

.PHONY: all pdf publish-pdf arxiv clean

all: pdf

pdf:
	mkdir -p $(BUILD_DIR)
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error -outdir=$(BUILD_DIR) $(MAIN).tex

publish-pdf: pdf
	cp $(BUILD_DIR)/$(MAIN).pdf wkvm-technical-report.pdf

arxiv: pdf
	tar -czf $(BUILD_DIR)/wkvm-arxiv-source.tar.gz $(MAIN).tex references.bib

clean:
	$(LATEXMK) -C -outdir=$(BUILD_DIR) $(MAIN).tex
	rm -f $(BUILD_DIR)/wkvm-arxiv-source.tar.gz
