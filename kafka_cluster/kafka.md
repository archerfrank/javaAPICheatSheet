# Kafka


## Download the images
```
docker pull prom/prometheus:v2.25.0
docker pull grafana/grafana:7.4.3-ubuntu
docker pull kafkamanager/kafka-manager:3.0.0.4
docker pull sheepkiller/kafka-manager:1.3.1.8
```

Start the docker compose
```
docker-compose up -d
docker-compose down

cd E:\kafka\bin\windows
kafka-topics.bat --create --topic topic --partitions 4 --zookeeper 127.0.0.1:2181 --replication-factor 2
kafka-topics.bat --describe --topic topic --zookeeper 127.0.0.1:2181
kafka-console-producer.bat --topic=topic --broker-list=127.0.0.1:9093
kafka-console-consumer.bat --topic=topic --bootstrap-server=127.0.0.1:9093
kafka-topics.bat --describe --zookeeper 127.0.0.1:2181
```


## Grafana
http://localhost:3000/
Username : admin
PWD : admin

add datasource below.
http://prometheus:9090

Import dashbroad.json


## Verify kafka broker

```
cd E:\kafka\bin\windows
kafka-verifiable-producer.bat –broker-list 127.0.0.1:9093,127.0.0.1:9094 –topic topic –max-messages 64
kafka-verifiable-consumer.bat –broker-list 127.0.0.1:9093,127.0.0.1:9094 –topic topic –group-id testGroup

kafka-run-class.bat org.apache.kafka.tools.VerifiableProducer --broker-list 127.0.0.1:9093,127.0.0.1:9094 --topic topic --max-messages 64

kafka-run-class.bat org.apache.kafka.tools.VerifiableConsumer --broker-list 127.0.0.1:9093,127.0.0.1:9094 --topic topic --group-id testGroup


docker-compose stop kafka1
docker-compose start kafka1
```





