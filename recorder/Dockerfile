FROM alpine:3.24.2

RUN apk add --no-cache ffmpeg

WORKDIR /app

COPY record.sh .

CMD ["sh", "/app/record.sh"]
