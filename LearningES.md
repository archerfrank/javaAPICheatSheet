# Learning ES
## Installation
```shell
docker pull elasticsearch:7.10.1
docker pull kibana:7.10.1

docker run --name elasticsearch -p 9200:9200 -p 9300:9300  -e "discovery.type=single-node" -e ES_JAVA_OPTS="-Xms256m -Xmx512m" -d elasticsearch:7.10.1

http://localhost:9200/
```

Please use docker compose version.



## Index

### Index aliases

The index aliases API allows you to create another name for an index or multiple indices and then use it as an alternative name in an index operation. The alias APIs give us flexibility in the following aspects:

- Re-indexing with zero downtime
- Grouping multiple indices
- Views on a subset of documents



### Reindexing with zero downtime                   

 It is recommended to use aliases instead of indices in production. Since your current index design may not be perfect, you'll need to reindex later. For example, some fields of the documents have changed. By using an alias, you can transparently switch your application from using the old index to the new index without downtime. 



# Analyzer

![](./imgs/c4999022-9032-4680-96f6-cf1b1af27815.png)



# Search API



# The scroll parameter

In addition to paging using the parameters, from and size, Elasticsearch also supports the scrollparameter, and works like a forward-only cursor. It keeps the search context active, just like a snapshot corresponding to a given timestamp.

Basically, if you need to process the returned results further and continue after the process, you need to keep such a snapshot. An identifier, _scroll_id, is provided in the result of the initial request so that you can use the identifier to get the next batch of results. If there are no more results, an empty array is returned in hits.You can use thesizeparameterto control the number of hits returned in the batch.

Let's look at an example where scrolling is executed three times to consume all documents. We will set the size to 313 and specify 10 minutes for the live time period.



If you complete the process, you need to clean up the context since it still consumes the computing resource before the timeout. As shown in the following screenshot, you can use the scroll_id parameter to specify one or more contexts in the DELETE API:

## SQL

```
docker exec -it es-master /bin/bash
bin/elasticsearch-sql-cli http://172.19.0.2:9200

show tables;
describe cf_etf;
describe cf_etf_hist_price;
describe cf_etf_split;

select database();


select symbol from cf_etf where symbol like 'AC%';

select * from cf_etf_hist_price where date='2019-01-31' limit 1;
select * from cf_etf_hist_price where date=CAST('2019-01-31' AS DATETIME) + INTERVAL 1 day limit 1;
select histogram(open, 0.5) AS open_price, count(*) AS count from cf_etf_hist_price where symbol='RFEM' group by open_price;
```


## Analyzer

```
bin/elasticsearch-plugin install analysis-icu
bin/elasticsearch-plugin install analysis-smartcn
bin/elasticsearch-plugin install https://github.com/medcl/elasticsearch-analysis-ik/releases/download/v7.10.1/elasticsearch-analysis-ik-7.10.1.zip


activate.bat
pip install -r requirements.txt
```