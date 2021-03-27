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