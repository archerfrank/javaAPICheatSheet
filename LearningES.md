# Learning ES
## Installation
```
docker pull elasticsearch:7.10.1

docker run --name elasticsearch -p 9200:9200 -p 9300:9300  -e "discovery.type=single-node" -e ES_JAVA_OPTS="-Xms64m -Xmx128m" -d elasticsearch:7.10.1

http://localhost:9200/
```