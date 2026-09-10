FROM oven/bun:1
WORKDIR /app
COPY package.json bun.lock ./
RUN bun install --frozen-lockfile --production
COPY . .
ARG GIT_SHA=unknown
ENV GIT_SHA=$GIT_SHA
EXPOSE 8080
CMD ["bun", "run", "index.ts"]