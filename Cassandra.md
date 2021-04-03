# Cassandra 

The ideal Cassandra application has the following characteristics:

- Writes exceed reads by a large margin.
- Data is rarely updated and when updates are made they are idempotent.
- Read Access is by a known primary key.
- Data can be partitioned via a key that allows the database to be spread evenly across multiple nodes.
- There is no need for joins or aggregates.

Some of my favorite examples of good use cases for Cassandra are:

- Transaction logging: Purchases, test scores, movies watched and movie latest location.
- Storing time series data (as long as you do your own aggregates).
- Tracking pretty much anything including order status, packages etc.
- Storing health tracker data.
- Weather service history.
- Internet of things status and event history.
- Telematics: IOT for cars and trucks.
- Email envelopes—not the contents.



https://blog.pythian.com/cassandra-use-cases/



## Use case

https://github.com/KillrVideo

https://github.com/jxblum/spring-data-cassandra-examples





## Awesome

https://github.com/jeffreyscarpenter/awesome-cassandra



## Read and Write

Write operations **are always sent to all replicas**, regardless of consistency level. The consistency level simply controls how many responses the coordinator waits for before responding to the client.

For read operations, the coordinator generally only issues read commands to enough replicas to satisfy the consistency level. The one exception to this is when speculative retry may issue a redundant read request to an extra replica if the original replicas have not responded within a specified time window.



## Replication Strategy

`NetworkTopologyStrategy` also attempts to choose replicas within a datacenter from different racks as specified by the [Snitch](https://cassandra.apache.org/doc/latest/operating/snitch.html#snitch). If the number of racks is greater than or equal to the replication factor for the datacenter, each replica is guaranteed to be chosen from a different rack. Otherwise, each rack will hold at least one replica, but some racks may hold more than one. Note that this rack-aware behavior has some potentially [surprising implications](https://issues.apache.org/jira/browse/CASSANDRA-3810). For example, if there are not an even number of nodes in each rack, the data load on the smallest rack may be much higher. Similarly, if a single node is bootstrapped into a brand new rack, it will be considered a replica for the entire ring. For this reason, many operators choose to configure all nodes in a single availability zone or similar failure domain as a single “rack”.



### Consistent Hashing using a Token Ring

 if we have an eight node cluster with evenly spaced tokens, and a replication factor (RF) of 3, then to find the owning nodes for a key we first hash that key to generate a token (which is just the hash of the key), and then we “walk” the ring in a clockwise fashion until we encounter three distinct nodes, at which point we have found all the replicas of that key. 



## Memtables

Memtables are in-memory structures where Cassandra buffers writes. In general, there is one active memtable per table.



## SSTables

```
Data.db
```

The actual data, i.e. the contents of rows.

```
Index.db
```

An index from partition keys to positions in the `Data.db` file. For wide partitions, this may also include an index to rows within a partition.

```
Summary.db
```

A sampling of (by default) every 128th entry in the `Index.db` file.

```
Filter.db
```

A Bloom Filter of the partition keys in the SSTable.



Within the `Data.db` file, rows are organized by partition. **These partitions are sorted in token order (i.e. by a hash of the partition key when the default partitioner, `Murmur3Partition`, is used).** Within a partition, rows are stored in the order of their clustering keys.



