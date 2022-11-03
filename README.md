# Scalable-Computing-for-Data-Analytics

A: Implementation 

Implement 3 main classes:

Q-Model class: this will take as inputs a set of jobs (represented as a rate) and outputs a set of processed jobs (represented as a rate). The model must enable an M/M/1/K queue or an M/M/c/K queue, with equations to solve the necessary equations for throughput time and number of jobs in the system.
Job-Flow class: this class will take as inputs a collection of jobs (representing each job as a rate) and outputs a set of jobs (representing each as a rate). The methods will enable merging or splitting of flows.
Q-network class: this will take the first two objects and enable a user to build an arbitrary data center model, which is represented as a Jackson network. You can use command line or GUI methods to build a data centre.

B: Simulation 

We will have a number n=50 input flows of rate =20/sec to a data centre gateway. The gateway will merge these flows, and then distribute them to a collection of CPUs in the data centre. You must compare two different processor networks:

100 M/M/1 CPU processors with rate µ=25/sec
10 GPUs, each represented as an M/M/20 cluster where each processor has rate µ=20/sec
Compare the designs in terms of job mean throughput time. 

Run a simulation where you randomly fail up to 20% of the processors. Plot the results of mean throughput time versus failure %. 
