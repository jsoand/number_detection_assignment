# Number detection from image
Project to detect number from image (model trained based on mnist database)

Includes apps to build docker containers and k8s.yml files for deplyment into minikube

## build and deploy to minikube
First install docker and minikube
- https://minikube.sigs.k8s.io/docs/start/
- https://docs.docker.com/engine/install/

Build applications docker containers into minikube

In each app folder run
`minikube image build -t app-name .`
where app-name is same as the folder name.

Run
`kubectl create -f k8s.yml`
in each app folder to deploy that app to minikube

Run
`minikube service webserver-src`
to get the webserver ip and open it in browser
