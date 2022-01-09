echo "[Deploy] Start"

echo "-----start docker-----"
service docker start
# the service can be started but the docker socket not ready, wait for ready
WAIT_N=0
while true; do
  # docker ps -q should only work if the daemon is ready
  docker ps -q >/dev/null 2>&1 && break
  if [ ${WAIT_N} -lt 5 ]; then
    WAIT_N=$((WAIT_N + 1))
    echo "[SETUP] Waiting for Docker to be ready, sleeping for ${WAIT_N} seconds ..."
    sleep ${WAIT_N}
  else
    echo "[SETUP] Reached maximum attempts, not waiting any longer ..."
    break
  fi
done


echo "-----docker build-----"
docker build . -f Dockerfile -t josefren/seldon-core-kdd-base:0.2

echo "-----s2i build-----"
s2i build . josefren/seldon-core-kdd-base:0.2 josefren/kdd-dagmm-classifier:0.1

echo "-----SeldonDeployment start-----"
kubectl apply -f kdd_clf.json

echo "[Deploy] Success"