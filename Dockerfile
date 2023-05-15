FROM python:3.10.11-slim-buster
WORKDIR /wasteai-backend

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .
# CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]
CMD [ "waitress-serve", "--port" , "5000", "app:app"]