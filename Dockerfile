FROM python:3
WORKDIR /app
COPY . .
RUN python3 -m pip install -U -r requirements.txt
CMD python -u ./bot.py
