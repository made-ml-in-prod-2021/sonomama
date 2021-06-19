FROM python:3.6-slim-stretch
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY model.pkl /model.pkl
COPY app.py /app.py
COPY features /features

WORKDIR .

ENV PATH_TO_MODEL="/model.pkl"

CMD ["python", "app.py"]
