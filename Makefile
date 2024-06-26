PYTHON := python3
CURRENT_DIR := $(shell pwd)

.PHONY: build_products_app

build_products_app:
	poetry run streamlit run src/products_app/app.py --server.port 8000

build_songs_app:
	poetry run streamlit run src/songs_app/app.py --server.port 8001

build_chatbot_app:
	poetry run streamlit run src/chatbot_app/app.py