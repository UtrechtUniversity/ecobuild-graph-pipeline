FROM oven/bun:1.1.18 AS builder 

WORKDIR /app

COPY package.json bun.lock ./
RUN bun install --frozen-lockfile

# Copy the rest of your frontend source code
COPY . .

RUN bun run build 

FROM nginx:alpine AS runtime
# remove default nginx start page
RUN rm /usr/share/nginx/html/index.html /usr/share/nginx/html/50x.html

COPY --from=builder /app/dist /usr/share/nginx/html
RUN mv /usr/share/nginx/html/index-*.html /usr/share/nginx/html/index.html || true

RUN rm /etc/nginx/conf.d/default.conf

COPY nginx.conf /etc/nginx/conf.d/your_app.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
