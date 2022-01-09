kubectl create namespace foo

kubectl label namespace foo istio-injection=enabled

kubectl apply -f istio-yaml/seldon-gateway.yaml