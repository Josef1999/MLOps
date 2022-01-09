# 警告
istio部署的iris样例，nginx代理中需设置http协议版本（>1.0）
```bash
server {
    listen 8888;
    server_name iris;
    location /{
            proxy_pass http://192.168.49.2:32612;
            proxy_http_version 1.1;
            }
}
```