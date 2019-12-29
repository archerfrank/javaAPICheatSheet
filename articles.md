## 分布式事务 Distributed transaction
https://www.cnblogs.com/savorboard/p/distributed-system-transaction-consistency.html

## 一致性哈希， consistent hash.


## Distributed Cache
To store more data in memory we partition data into shards. and put each shard on its own server. Every cache clients knows about all cache shards. Cache clients use consistent hashing algorithm to pick a shard for storing and retrieving a particular cache key. 

### Performance
Constant time to pick up key and values.
Connection is via TCP or UDP, which is fast.

### Scalability 
We could create more shards to store more data.

But shards could be hotspot, add more server, could create more shards, but not replicate the hot shard.

### Availability

Each server, each shard, not high availability.

### How to sovle availability and hot shards issue

Use data replication via master and slave for each cache server. put to master and get to both mast and slaves.

Configuration server to elect master or cache server elect the leader by themselves.

As you have seen cache client has many responsibilities: maintain a list of cache servers, pick a shard
to route a request to, handle a remote call and any potential failures, emit metrics.
Ideally, client software should be very simple, dumb if you want.

Another idea, is to make cache servers responsible for picking a shard.
Client sends request to a random cache server and cache server applies consistent hashing
(or some other partitioning algorithm) and redirects request to the shard that stores
the data.
This idea is utilized by Redis cluster.
Consistent hashing algorithm is great.

But it has two major flaws: so called domino effect and the fact that cache servers do not split the circle evenly.

### Other points 

Distributed Cache should use consistent hash.

LRU cache, hash map with double linked list.


## Long polling with Tomcat or Spring.

https://www.baeldung.com/spring-deferred-result
https://www.nurkiewicz.com/2013/03/deferredresult-asynchronous-processing.html

ListenableFuture could be used in long polling to unblock the http worker thread.
```java
ListeningExecutorService service = MoreExecutors.listeningDecorator(Executors.newFixedThreadPool(10));
ListenableFuture<Explosion> explosion = service.submit(
    new Callable<Explosion>() {
      public Explosion call() {
        return pushBigRedButton();
      }
    });
Futures.addCallback(
    explosion,
    new FutureCallback<Explosion>() {
      // we want this handler to run immediately after we push the big red button!
      public void onSuccess(Explosion explosion) {
        walkAwayFrom(explosion);
      }
      public void onFailure(Throwable thrown) {
        battleArchNemesis(); // escaped the explosion!
      }
    },
    service);
```

Server-sent events with RxJava and SseEmitter
https://www.nurkiewicz.com/2015/07/server-sent-events-with-rxjava-and.html
https://howtodoinjava.com/spring-boot2/rest/spring-async-controller-sseemitter/
https://www.baeldung.com/spring-mvc-sse-streams

Netty 

https://blog.csdn.net/zxhoo/article/details/17709765
https://netty.io/4.0/api/io/netty/handler/codec/ReplayingDecoder.html
https://netty.io/wiki/reference-counted-objects.html

Guava

https://github.com/google/guava/wiki/ImmutableCollectionsExplained