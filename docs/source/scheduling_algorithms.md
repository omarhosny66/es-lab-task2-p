# Scheduling Algorithms

## Overview

The scheduling algorithms covered and to be implemented are:

1. Earliest Deadline First (EDF) for Single-Node
2. Latest Deadline First (LDF) for Single-Node
3. Earliest Deadline First (EDF) for Multi-Node.
4. Latest Deadline First (LDF) for Multi-Node
5. Least Laxity (LL) for Multi-Node

## JSON Input

The input to the scheduling algorithms is a JSON object that describes the application and platform models. The application model includes tasks and messages, while the platform model includes nodes and links. The JSON input contains following objects and should conform to the [input JSON schema](README.md#api-input-schema-for-schedule-jobs).

- **Tasks**: Represent the tasks to be scheduled.
- **Messages**: Represent dependencies between tasks.
- **Nodes**: Represent the either a compute node where tasks can be executed or router in the network or a sensor or a actuators.
- **Links**: Represent the communication links between nodes.


### Example JSON Input

We will use the following JSON model describing the application and platform model as our example input for the scheduling algorithms.

```json
{
    "application": {
      "tasks": [
        {
            "id": 1,
            "wcet": 1,
            "mcet": 1,
            "deadline": 2
        },
        {
            "id": 2,
            "wcet": 1,
            "mcet": 1,
            "deadline": 5
        },
        {
            "id": 3,
            "wcet": 1,
            "mcet": 1,
            "deadline": 4
        },
        {
            "id": 4,
            "wcet": 1,
            "mcet": 1,
            "deadline": 3
        },
        {
            "id": 5,
            "wcet": 1,
            "mcet": 1,
            "deadline": 5
        },
        {
            "id": 6,
            "wcet": 1,
            "mcet": 1,
            "deadline": 6
        }
        ],
        "messages": [
        {
            "id": 1,
            "sender": 1,
            "receiver": 2,
            "size": 2
        },
        {
            "id": 2,
            "sender": 1,
            "receiver": 3,
            "size": 2
        },
        {
            "id": 3,
            "sender": 2,
            "receiver": 4,
            "size": 2
        },
        {
            "id": 4,
            "sender": 3,
            "receiver": 6,
            "size": 2
        },
        {
            "id": 5,
            "sender": 2,
            "receiver": 5,
            "size": 2
        }
      ]
    },
    "platform": {
        "nodes": [
        {
            "id": 1,
            "type": "compute"
        },
        {
            "id": 2,
            "type": "compute"
        },
        {
            "id": 3,
            "type": "router"
        }
      ]
    },
    "links": [
        {
            "id": 1,  
            "start_node": 1,
            "end_node": 3,
            "link_delay": 2,
            "bandwidth": 100,
            "type": "ethernet"
        },
        {
            "id": 2,  
            "start_node": 2,
            "end_node": 3,
            "link_delay": 2,
            "bandwidth": 100,
            "type": "ethernet"
        }
    ]
}
```
### Example Scheduling Algorithms Output
Given the example JSON input, an example output of a scheduling algorithms that conforms to [output schema](README.md#output-schema-for-schedule-jobs) would be as follows:

``` json
{
    "schedule": [
        {
            "task_id": 1,
            "node_id": 1,
            "start_time": 0,
            "end_time": 1,
            "deadline": 2
        },
        {
            "task_id": 2,
            "node_id": 1,
            "start_time": 1,
            "end_time": 2,
            "deadline": 5
        },
        {
            "task_id": 3,
            "node_id": 2,
            "start_time": 1,
            "end_time": 2,
            "deadline": 4
        },
        {
            "task_id": 4,
            "node_id": 1,
            "start_time": 2,
            "end_time": 3,
            "deadline": 3
        },
        {
            "task_id": 5,
            "node_id": 2,
            "start_time": 2,
            "end_time": 3,
            "deadline": 5
        },
        {
            "task_id": 6,
            "node_id": 1,
            "start_time": 3,
            "end_time": 4,
            "deadline": 6
        }

    ]
}
```

## EDF and LDF Scheduling Algorithms

The Earliest Deadline First (EDF) algorithm schedules jobs based on the earliest deadlines. It prioritizes jobs with the nearest deadlines to ensure that all deadlines are met as soon as possible. Given $n$ independent tasks with deadlines, $d_1,\ldots, d_n$, schedule them to minimize the maximum lateness, defined by the following equation,
                                    
$$L_{\text{max}} = \max_{1 \leq i \leq n} \{f_i - d_i\}$$


where $f_i$ is the finishing time of task $i$.  Note that the above equation is negative if all deadlines are  met. EDF is  widely used in systems where meeting  deadlines is crucial, such as in multimedia systems for audio/video processing to ensure smooth streaming without delays. 
The Latest Deadline First (LDF) algorithm schedules tasks based on the latest deadlines. It aims to delay task execution as much as possible while still meeting deadlines, to allow more urgent tasks to execute first. The algorithm to produce an LDF schedule proceeds in two stages: firstly, a precedence graph is constructed. Going from tail to head: among the tasks without successors or whose successors have been
all selected, LDF selects the tasks with the latest deadline to be scheduled last.
The Earliest Deadline First (EDF) and Latest Deadline First (LDF) algorithms for single-node systems, schedules jobs similarly to the multi-node version but considers only one compute node.

![alt text](./images/edf_and_ldf.PNG)

*Figure 1: The LDF scheduling strategy in single node, the task with the latest deadline to be scheduled last, and work backwards.  In EDF scheduling strategy in single node, the task with the closest deadline is scheduled first.*
### Earliest Deadline First for Single Node 

EDF scheduling for single node can be understood with an example as follows. Say, we want to schedule the application model described by the DAG in figure 1. Analyzing the order of tasks in the figure 1, task 1 must be completed first because it is the starting task with no dependencies. Following task 1, the next tasks to consider based on available dependencies are tasks 3 and 2. Task 3 ($d_3=4$) has an earlier deadline than task 2 ($d_2=5$). Task 2 is scheduled after task 3 as the next available task with the next closest deadline. Task 4 can only be scheduled after task 2 is complete due to the dependency, even though task 4 has an earlier deadline ($d_4=3$) than task 2, its start is contingent on task 2's completion. Tasks 5 and 6 follow based on their deadlines, $d_5=5$ and $d_6=6$, and dependencies.

### Usage

``` PYTHON
schedule = edf_single_node(application_model)
```

### Example

For the example JSON input, the output of the EDF single-node algorithm would be as follows, note that the task 4 has missed its deadline:

``` json
{
    "schedule": [
        {
            "task_id": 1,
            "node_id": 1,
            "start_time": 0,
            "end_time": 1,
            "deadline": 2
        },
        {
            "task_id": 3,
            "node_id": 1,
            "start_time": 1,
            "end_time": 2,
            "deadline": 4
        },
        {
            "task_id": 2,
            "node_id": 1,
            "start_time": 2,
            "end_time": 3,
            "deadline": 5
        },
        {
            "task_id": 4,
            "node_id": 1,
            "start_time": 3,
            "end_time": 4,
            "deadline": 3
        },
        {
            "task_id": 5,
            "node_id": 1,
            "start_time": 4,
            "end_time": 5,
            "deadline": 5
        },
        {
            "task_id": 6,
            "node_id": 1,
            "start_time": 5,
            "end_time": 6,
            "deadline": 6
        }

    ]
}
```
### Latest Deadline First for Single Node
The LDF scheduling for single node can be understood with an example as follows. From the figure 1, task 1 must be completed first because it is the starting task with no dependencies. Following task 1, the next tasks to consider based on available dependencies are tasks 3 and 2. Task 2 ($d_2=5$) has the latest deadline than task 3 ($d_3=4$). Task 4 can only be scheduled after task 2 is complete. As the deadline for task 3 is $d_3=4$ and task 4 is $d_4=3$. Tasks 5 and 6 follow based on their deadlines, $d_5=5$ and $d_6=6$, and dependencies.


### Usage

``` PYTHON
schedule = ldf_single_node(application_model, platform_model)
```
### Example

For the example JSON input, the output of the LDF single-node algorithm would be as follows, note that this produces a feasible schedule as no tasks miss their deadlines:

```  json
{
    "schedule": [
        {
            "task_id": 1,
            "node_id": 1,
            "start_time": 1,
            "end_time": 1,
            "deadline": 2
        },
        {
            "task_id": 2,
            "node_id": 1,
            "start_time": 1,
            "end_time": 2,
            "deadline": 2
        },
        {
            "task_id": 4,
            "node_id": 2,
            "start_time": 2,
            "end_time": 3,
            "deadline": 3
        },
        {
            "job_id": 3,
            "node_id": 1,
            "start_time": 3,
            "end_time": 4,
            "deadline": 4
        },
        {
            "job_id": 5,
            "node_id": 1,
            "start_time": 4,
            "end_time": 5,
            "deadline": 5
        },
        {
            "job_id": 6,
            "node_id": 1,
            "start_time": 5,
            "end_time": 6,
            "deadline": 6
        }

    ]
}
```

## Earliest Deadline First (EDF) for Multi-Node
EDF scheduling for multi node can be understood with the following example. For example, when we want to schedule the application graph shown in figure 1 on three compute nodes. Node 1 might start with task 1 (earliest starting task with no dependencies). As soon as task 1 completes, node 1 can take on task 3, and node 2 can begin task 2 simultaneously since these tasks are now available and have the next earliest deadlines. Task 4 must wait for task 2 to complete but could be started immediately on node 3 if task 2 finishes before task 3. Task 5 and 6 would then be allocated based on their completion of respective dependencies.

### Usage


``` PYTHON
schedule = edf_multinode(application_model, platform_model)
```


### Example


For the the example JSON model, the output of the EDF multi-node algorithm should be as follows, note that in this case EDF produces a feasible schedule as no tasks miss their deadlines, however, as discussed EDF is not guranteed to produce a feasible schedule even if one exists:

``` json
{
    "schedule": [
        {
            "task_id": 1,
            "node_id": 1,
            "start_time": 0,
            "end_time": 1,
            "deadline": 2
        },
        {
            "task_id": 3,
            "node_id": 1,
            "start_time": 1,
            "end_time": 2,
            "deadline": 4
        },
        {
            "task_id": 2,
            "node_id": 2,
            "start_time": 1,
            "end_time": 2,
            "deadline": 5
        },
        {
            "task_id": 4,
            "node_id": 2,
            "start_time": 2,
            "end_time": 3,
            "deadline": 3
        },
        {
            "task_id": 5,
            "node_id": 2,
            "start_time": 3,
            "end_time": 4,
            "deadline": 5
        },
        {
            "task_id": 6,
            "node_id": 1,
            "start_time": 2,
            "end_time": 3,
            "deadline": 6
        }

    ]
}
```

## Latest Deadline First (LDF) for Multi-Node
The LDF scheduling for multi node can be understood with an example as follows.Assume a system with 3 nodes, node 1 starts with task 1. Upon completion, if node 2 and node 3 are free, task 2 and task 3 can start simultaneously on different nodes, minimizing their laxity as they are now the most critical tasks available. Task 4 can start on any node that becomes free first, preferably the one that finishes task 2 to keep the flow of dependencies smooth. Task 5 and 6 would then be allocated based on their completion of dependencies.


### Usage


``` PYTHON
schedule = ldf_multinode(application_model, platform_model)
```

### Example

For the the example JSON application model, the output of the LDF multi-node algorithm should be as follows:

``` json
{
    "schedule": [
        {
            "task_id": 1,
            "node_id": 1,
            "start_time": 0,
            "end_time": 1,
            "deadline": 2
        },
        {
            "task_id": 2,
            "node_id": 1,
            "start_time": 1,
            "end_time": 2,
            "deadline": 5
        },
        {
            "task_id": 4,
            "node_id": 1,
            "start_time": 2,
            "end_time": 3,
            "deadline": 3
        },
        {
            "task_id": 3,
            "node_id": 2,
            "start_time": 1,
            "end_time": 2,
            "deadline":4 
        },
        {
            "task_id": 5,
            "node_id": 2,
            "start_time": 2,
            "end_time": 3,
            "deadline": 5
        },
        {
            "task_id": 6,
            "node_id": 1,
            "start_time": 3,
            "end_time": 4,
            "deadline": 6
        }

    ]
}
```

## Least Laxity

Laxity in scheduling algorithms is defined as the difference between a task’s deadline and the required CPU time to complete it.  This metric is crucial in dynamic preemptive scheduling where priorities are  adjusted dynamically based on current  task states and requirements. The task with the least laxity gets the highest  priority.  

In figure 1, after task 1, both task 2 and task 3 are ready for execution. At time 1, laxity of task  2 is $d_2 - e_2 = 5-1 = 3$. Laxity of task 3 is $d_3 - e_3 = 4-1 = 2$. Task 3 is selected as it has least laxity followed by task 2. Task 4, which depends on task 2, can now start. Laxity of task 4 at time 2 is $d_4 - e_4 = 3-1 = 2 $. With the lowest possible laxity, task 4 gets scheduled immediately after task 2, completing at time 3. Task 3 can now be executed, followed by tasks 5 and 6. Their scheduling is straightforward as there are no remaining dependencies that affect their execution.
### Usage

``` PYTHON
schedule = ll_multinode(application_model, platform_model)
```
### Example

For the example JSON application model, the output of Least Laxity multi-node algorithm should be as follows:

``` json
{
    "schedule": [
        {
            "task_id": 1,
            "node_id": 1,
            "start_time": 0,
            "end_time": 1,
            "deadline": 2
        },
        {
            "task_id": 3,
            "node_id": 1,
            "start_time": 1,
            "end_time": 2,
            "deadline": 4
        },
        {
            "task_id": 2,
            "node_id": 2,
            "start_time": 1,
            "end_time": 2,
            "deadline": 5
        },
        {
            "task_id": 4,
            "node_id": 1,
            "start_time": 2,
            "end_time": 3,
            "deadline": 3
        },
        {
            "task_id": 5,
            "node_id": 2,
            "start_time": 2,
            "end_time": 3,
            "deadline": 5
        },
        {
            "task_id": 6,
            "node_id": 1,
            "start_time": 3,
            "end_time": 4,
            "deadline": 6
        }

    ]
}
```
