# Elastic Search Cluster in Docker Compose

The cluster includes 1 master node and 2 data node, one kibana. It is built in the Windows 10. Copy the es folder into user's home directory before you start the cluster. Only the content in the user home directory can be mounted to docker as volume in Windows.



Command to start the cluster

```shell
docker-compose -f docker-compose.yml up -d
```



Verify the cluster using below url.

http://localhost:9200/_nodes?pretty



Tear down the cluster

```shell
docker-compose down
```

