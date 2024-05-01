PYTHON := python3
CURRENT_DIR := $(shell pwd)

.PHONY: build_products_app

build_products_app:
	poetry run streamlit run src/products_app/app.py