FROM nginx:alpine

# Copy custom Nginx config to proxy /api and /ws
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy frontend source files
COPY . /usr/share/nginx/html

EXPOSE 80
