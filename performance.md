# A Java Performance Toolbox

## Operating System Tools and Analysis

On Unix-based systems, these are sar (System Accounting Report) and its constituent tools like **vmstat +iostat, +prstat**, and so on. On Windows, there are graphical resource monitors as well as command-line utilities like **typeperf**

## CPU USAGE

CPU usage is typically divided into two categories: user time, and system time (Windows refers to this as privileged time). User time is the percentage of time the CPU is executing application code, while system time is the percentage of time the CPU is executing kernel code. System time is related to the application; if the application performs I/O, for example, the kernel will execute the code to read the file from disk, or write the network buffer, and so on. Anything that uses an underlying system resource will cause the application to use more system time.

The goal in performance is to drive CPU usage as high aspossible for as short a time as possible.

The CPU can be idle for a number of reasons:

1. The application might be blocked on a synchronization primitive and unable to execute until that lock is released.
2. The application might be waiting for something, such as a response to come back from a call to the database.
3. The application might have nothing to do.

### Java and Single-CPU usage

Driving the CPU usage higher is always the goal for batch jobs, because it means the job will be completed faster. If the CPU is already at 100%, then you can of course still look for optimizations that allow the work to be completed faster (while trying also to keep the CPU at 100%).

### THE CPU RUN QUEUE

In general, then, you want the processor queue length to be 0 on Windows, and equal to (or less than) the number of CPUs on Unix systems. That isn’t a hard and fast rule; there are system processes and other things that will come along periodically and briefly raise that value without any significant performance impact. But if the run queue length is too high for any significant period of time, it is an indication that the machine is overloaded and you should look into reducing the amount of work the machine is doing.

### QUICK SUMMARY

1. CPU time is the first thing to examine when looking at performance of an application.
2. The goal in optimizing code is to drive the CPU usage up (for a shorter period of time), not down.
3. Understand why CPU usage is low before diving in and attempting to tune an application.

## DISK USAGE

Monitoring disk usage has two important goals. The first of these regardsthe application itself: if the application is doing a lot of disk I/O,then it is easy for that I/O to become a bottleneck.

The basic I/O monitors on some systems are better than on others. Here is some partial output of iostat on a Linux system: 

% iostat -xm 5
avg-cpu:  %user   %nice %system %iowait  %steal   %idle
          23.45    0.00   37.89    0.10    0.00   38.56

          Device:         rrqm/s   wrqm/s     r/s     w/s    rMB/s
          sda               0.00    11.60    0.60   24.20     0.02

          wMB/s avgrq-sz avgqu-sz   await r_await w_await  svctm  %util
          0.14    13.35     0.15    6.06    5.33    6.08   0.42   1.04

The application here is writing data to disk sda. At first glance, the disk statistics look good. The w_await—which is the time to service each I/O write—is fairly low (6.08 milliseconds), and the disk is only 1.04% utilized. But there is a clue here that something is wrong: the system is spending 37.89% of its time in the kernel. If the system is doing other I/O (in other programs), that’s one thing; if all that system time is from the application being tested, then something inefficient is happening.

The fact that the system is doing 24.2 writes per second is another clue here: that is a lot of writes when writing only 0.14 MB per second. I/O has become a bottleneck; we can now look into how the application is performing its writes

The other side of the coin comes if the disk cannot keep up with the I/O requests:
% iostat -xm 5
avg-cpu:  %user   %nice %system %iowait  %steal   %idle
          35.05    0.00    7.85   47.89    0.00    9.20

          Device:         rrqm/s   wrqm/s     r/s     w/s    rMB/s
          sda               0.00     0.20    1.00  163.40     0.00

          wMB/s avgrq-sz avgqu-sz   await r_await w_await  svctm  %util
          81.09  1010.19   142.74  866.47   97.60  871.17   6.08 100.00

The nice thing about Linux is that it tells us immediately that the disk is 100% utilized; it also tells us that processes are spending 47.89% of their time in iowait (that is, waiting for the disk).

Even on other systems where only raw data is available, that data will tell us something is amiss: the time to complete the I/O (w_await) is 871 milliseconds, the queue size is quite large, and the disk is writing 81 MB of data/second. This all points to disk I/O as a problem, and that the amount of I/O in the application (or, possibly, elsewhere in the system) must be reduced.


A second reason to monitor disk usage—even if the application is not expected to perform a significant amount of I/O—is to help monitor if the system is **swapping**.

### QUICK SUMMARY

1. Monitoring disk usage is important for all applications. For applications that don’t directly write to disk, system swapping can still affect their performance.

2. Applications that write to disk can be bottlenecked both because they are writing data inefficiently (too little throughput) or because they are writing too much data (too much throughput).


## NETWORK USAGE
On Unix systems, one popular command-line tool is nicstat (http://sourceforge.net/projects/nicstat), which presents a summary of the traffic on each interface, including the degree to which the interface is utilized:

% nicstat 5
Time      Int       rKB/s   wKB/s   rPk/s   wPk/s   rAvs    wAvs   %Util  Sat
17:05:17  e1000g1   225.7   176.2   905.0   922.5   255.4   195.6  0.33   0.00

The e1000g1 interface is a 1000 MB interface; it is not utilized very much (0.33%) in this example. The usefulness of this tool—and others like it—is that it calucates the utilization of the interface. In this output, there is 225.7 KB/sec of data being written and 176.2 KB/sec of data being read over the interface. Doing the division for a 1000 MB network yields the 0.33% utilization figure, and the nicstat tool was able to figure out the bandwidth of the interface automatically.

Be careful that the bandwidthis measured in bits per second, but tools generally report bytes per second.A 1000 megabit network yields 125 megabytes per second. In this example,0.22 megabytes per second are read and 0.16 megabytes per second are written;adding those and dividing by 125 yields a 0.33% utilization rate. 

Networks cannot sustain a 100% utilization rate. For local-areaEthernet networks, a sustained utilization rate over 40% indicates that theinterface is saturated.

## Java Monitoring Tools

jcmd
Print basic class, thread, and VM information of a Java process. This is suitable for use in scripts; it is executed like this:

% jcmd process_id command optional_arguments

jconsole
Provides a graphical view of JVM activities, including thread usage, class usage, and GC activities

jhat
Reads and helps analyze memory heap dumps. This is a post-processing utility.

jmap
Provides heap dumps and other information about JVM memory usage. Suitable for scripting, though the heap dumps must be used in a post-processing tool.

jinfo
Provides visibility into the system properties of the JVM, and allows some system properties to be set dynamically. Suitable for scripting.

jstack
Dumps the stacks of a Java process. Suitable for scripting.

jstat
Provides information about GC and class loading activities. Suitable for scripting.

jvisualvm
A GUI tool to monitor a JVM, profile a running application, and analyze JVM heap dumps (which is a post-processing activity, though jvisualvm can also take the heap dump from a live program).


It can be very useful to look at the stack of running threads to determine if they are blocked. The stacks can be obtained via jstack:

```sh
    % jstack process_id
    ... Lots of output showing each thread's stack ...
```

Stack information can also be obtained from jcmd:

```sh
    % jcmd process_id Thread.print
    ... Lots of output showing each thread's stack ...
```

### SAMPLING PROFILERS

Profiling happens in one of two modes: sampling mode or instrumented mode. Sampling mode is the basic mode of profiling and carries the least amount of overhead.

Unfortunately, sampling profilers can be subject to all sorts of errors. Sampling profilers work when a timer periodically fires; the profiler then looks at each thread and determines which method the thread is executing. That method is then charged with having been executed since the timer previously fired.

### INSTRUMENTED PROFILERS

Instrumented profilers are much more intrusive than sampling profilers, but they can also give more beneficial information about what’s happening inside the program.

The invocation count of an instrumented profile is certainly accurate, andthat additional information is often quite helpful in determining where thecode is actually spending more time and which things are more fruitful tooptimize. 

instrumented profilers work by altering the bytecode sequence of classes as they are loaded (inserting code to count the invocations and so on). They are much more likely to introduce performance differences into the application than are sampling profilers. 

For example, the JVM will inline small methods (see Chapter 4) so that no method invocation is needed when the small method code is executed. The compiler makes that decision based on the size of the code; depending on how the code is instrumented, it may no longer be eligible to be inlined. This may cause the instrumented profiler to overestimate the contribution of certain methods.

Sampling profilers in Java can only take the sample of a thread when the thread is at a **safepoint—essentially, whenever it is allocating memory**. 



