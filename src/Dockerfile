FROM python:3.10

WORKDIR /chatbot_app

COPY . /chatbot_app

RUN python -m pip install --upgrade pip

RUN pip install -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "chatbot_app.app:app", "--host", "0.0.0.0", "--port", "8000"]