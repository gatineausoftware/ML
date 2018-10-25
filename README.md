


minikube service mysql --url
http://192.168.99.100:31280


dockerdocker build -t jup .

docker run -p 8000:8000 -p 8081:8080 --name jup --mount type=bind,source="$(pwd)",target=/app  -d jup:latest

