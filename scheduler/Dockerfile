FROM redis:7.0.2-alpine3.16
COPY redis.conf /usr/local/etc/redis/redis.conf
COPY redis.conf /data/db/dummy.txt
RUN chmod a+rwx -R /data/db && rm /data/db/dummy.txt
RUN apk add nodejs npm python3 py3-pip
RUN npm i redis http-server

WORKDIR /app
COPY ./requirements.txt /app/requirements.txt
RUN pip3 install -r requirements.txt

COPY run.sh /app
COPY scheduler.py /app

CMD ["sh", "run.sh"]