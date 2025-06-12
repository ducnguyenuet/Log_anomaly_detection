FROM python:3.8.6-slim-buster

WORKDIR /workdir/

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy scripts
COPY run/ .

CMD "run/run_drain.py"
