#拉取镜像
docker pull prom/node-exporter
docker pull prom/prometheus:v2.0.0
docker pull grafana/grafana:4.2.0

kubectl apply -f    node-exporter.yaml 
kubectl apply -f    rbac-setup.yaml
kubectl apply -f    configmap.yaml 
kubectl apply -f    prometheus.deploy.yaml 
kubectl apply -f    prometheus.service.yaml 