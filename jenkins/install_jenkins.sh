#helm repo add jenkinsci https://charts.jenkins.io

#helm repo update

kubectl create namespace jenkins

kubectl apply -f jenkins-yaml/jenkins-volume.yaml 
kubectl apply -f jenkins-yaml/jenkins-sa.yaml

chart=jenkinsci/jenkins
while ((1))
do
    failure_info=$(helm install jenkins -n jenkins -f jenkins-yaml/jenkins-values.yaml $chart |grep FAILED)
    echo ${#failure_info} 
    echo ${failure_info}
    if [ ${#failure_info} == 0 ]
    then
        break
    fi
done
echo Install Done