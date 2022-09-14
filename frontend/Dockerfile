FROM python:3.10-alpine
RUN apk update && apk upgrade && apk add --no-cache curl nginx
WORKDIR /app
COPY ./requirements.txt /app/requirements.txt
RUN pip3 install -r requirements.txt

EXPOSE 8000

COPY public /app/public
COPY run.sh /app
COPY server.py /app
COPY nginx_default.conf /etc/nginx/http.d/default.conf

# CMD ["python", "-u", "server.py"]
CMD ["sh", "run.sh"]