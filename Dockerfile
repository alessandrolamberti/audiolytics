FROM python:3.8
WORKDIR /app
COPY requirements.txt .
RUN apt-get update -y && apt-get install -y --no-install-recommends build-essential gcc \
                                        libsndfile1 

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install python-multipart
RUN pip install Jinja2

COPY . .
CMD ["python", "main.py"]
