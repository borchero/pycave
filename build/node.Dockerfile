FROM node:16.3.0

COPY package.json package-lock.json /app/

WORKDIR /app
RUN npm install
