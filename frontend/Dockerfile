# The frontend uses a NGINX server to serve the static files and a Python server to process API commands
FROM python:3.10-alpine

RUN apk update && apk upgrade && apk add --no-cache curl nginx
WORKDIR /app
COPY ./requirements.txt /app/requirements.txt
RUN pip3 install -r requirements.txt

COPY public /app/public
COPY run.sh /app
COPY server.py /app
COPY nginx_default.conf /etc/nginx/http.d/default.conf

CMD ["sh", "run.sh"]