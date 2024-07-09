"""
This module contains the scheduling algorithms used in the scheduling API.

It provides implementations for both Least Deadline First (LDF) and Earliest Deadline First (EDF) scheduling strategies, applicable in single-core and multi-core processor environments. Functions within are designed to be called with specific application and platform data structures.

Functions:
- ldf_singlecore: Schedules tasks on a single-core processor using LDF.
- edf_singlecore: Schedules tasks on a single-core processor using EDF.
- rms_singlecore: Schedules tasks on a single-core processor using RMS.
- ll_singlecore: Schedules tasks on a single-core processor using LL.
- ldf_multicore: Schedules tasks on multiple cores using LDF.
- edf_multicore: Schedules tasks on multiple cores using EDF.
"""

__author__ = "Priya Nagar"
__version__ = "1.0.0"


import networkx as nx

# just an eample for the structure of the schedule to be returned and to check the frontend and backend connection
example_schedule = [
    {
        "task_id": 3,
        "node_id": 0,
        "end_time": 20,
        "deadline": 256,
        "start_time": 0,
    },
    {
        "task_id": 2,
        "node_id": 0,
        "end_time": 40,
        "deadline": 300,
        "start_time": 20,
    },
    {
        "task_id": 1,
        "node_id": 0,
        "end_time": 60,
        "deadline": 250,
        "start_time": 40,
    },
    {
        "task_id": 0,
        "node_id": 0,
        "end_time": 80,
        "deadline": 250,
        "start_time": 60,
    },
]

def ldf_single_node(application_data):
    """
    Schedule jobs on a single node using the Latest Deadline First (LDF) strategy.

    This function schedules jobs based on their latest deadlines after sorting them and considering dependencies through a directed graph representation.

    .. todo:: Implement Latest Deadline First Scheduling (LDF) algorithm for single compute node.

    
    Args:
        application_data (dict): Contains jobs and messages that indicate dependencies among jobs.

    Returns:
        list of dict: Scheduling results with each job's details, including execution time, node assignment,
                      and start/end times relative to other jobs.
    """
    tasks = application_data.get("tasks")
    messages = application_data.get("messages")

    Graph = nx.DiGraph()  # Create directed graph

    # Create a dictionary holding main task parameters {id, deadline, worst_case_execution_time}
    tasks_dict = {}
    for task in tasks:
        tasks_dict[task['id']] = {'deadline' : task['deadline'], 'wcet' : task['wcet']}

    # Added nodes to graph
    for task in tasks:
        Graph.add_node(task['id'])

    # Added edges to nodes according to dependencies
    for message in messages:
        Graph.add_edge(message['sender'], message['receiver'])

    scheduler_queue = []
    leaf_nodes = []
    added_nodes = []

    num_tasks =  Graph.number_of_nodes()
    unprocessed_tasks = num_tasks
    while unprocessed_tasks != 0:    
        # Get leaves
        for task in Graph:
            # Check if task has no out edges and not added to the queue
            if (Graph.out_degree()[task] == 0 and not (added_nodes.count(task))):
                leaf_nodes.append((tasks_dict[task]['deadline'], task))
                added_nodes.append(task)

        leaf_nodes.sort()  # Sort according to the latest deadline
        latest = leaf_nodes.pop()[1]  # Get the latest deadline task id
    
        in_edges = Graph.in_edges(latest)
        Graph.remove_edges_from(list(in_edges))  # Remove the in edges for the latest task from graph
        Graph.remove_node(latest)  # Remove the latest task from the graph
        scheduler_queue.append(latest)  # Append task id
        unprocessed_tasks -= 1  # Decrement unprocessed tasks counter

    start_time = 0
    ldf_single_node_schedule = []
    missed_deadline = []

    for t in range(num_tasks):
        id = scheduler_queue.pop()
        end_time = start_time + tasks_dict[id]['wcet']
        # Check if the task meets its deadline
        if end_time > tasks_dict[id]['deadline']:
            missed_deadline.append(id)  # Append tasks that missed the deadline
            continue
        else:
            # Append to ldf single-node schedule
            ldf_single_node_schedule.append({
                "task_id": id,
                "node_id": 0,
                "end_time": end_time,
                "deadline": tasks_dict[id]['deadline'],
                "start_time": start_time,
            })
            start_time = end_time  # Update the next available start time for the node

    return {"schedule": ldf_single_node_schedule, "name": "LDF Single Node", "missed_deadline" : missed_deadline}


def edf_single_node(application_data):
    """
    Schedule jobs on single node using the Earliest Deadline First (EDF) strategy.

    This function processes application data to schedule jobs based on the earliest
    deadlines. It builds a dependency graph and schedules accordingly, ensuring that jobs with no predecessors are
    scheduled first, and subsequent jobs are scheduled based on the minimum deadline of available nodes.

    .. todo:: Implement Earliest Deadline First Scheduling (EDF) algorithm for single compute node.

    Args:
        application_data (dict): Job data including dependencies represented by messages between jobs.

    Returns:
        list of dict: Contains the scheduled job details, each entry detailing the node assigned, start and end times,
                      and the job's deadline.
    """
    tasks = application_data.get("tasks")
    messages = application_data.get("messages")

    Graph = nx.DiGraph()  # Create directed graph

    # Create a dictionary holding main task parameters {id, deadline, worst_case_execution_time}
    tasks_dict = {}
    for task in tasks:
        tasks_dict[task['id']] = {'deadline' : task['deadline'], 'wcet' : task['wcet']}

    # Added nodes to graph
    for task in tasks:
        Graph.add_node(task['id'])

    # Added edges to nodes according to dependencies
    for message in messages:
        Graph.add_edge(message['sender'], message['receiver'])

    scheduler_queue = []
    root_nodes = []
    added_nodes = []

    num_tasks =  Graph.number_of_nodes()
    unprocessed_tasks = num_tasks
    while unprocessed_tasks != 0:    
        # Get roots
        for task in Graph:
            # Check if task has no in edges and not added to the queue
            if (Graph.in_degree()[task] == 0 and not (added_nodes.count(task))):
                root_nodes.append((tasks_dict[task]['deadline'], task))
                added_nodes.append(task)

        root_nodes.sort(reverse=True)  # Sort according to the earliest deadline in reverse order
        earliest = root_nodes.pop()[1]  # Get the latest deadline task id
    
        outEdges = Graph.out_edges(earliest)    # Remove the out edges for the latest task from graph
        Graph.remove_edges_from(list(outEdges))
        Graph.remove_node(earliest)  # Remove the earliest task from the graph
        scheduler_queue.append(earliest)  # Append task id
        unprocessed_tasks -= 1    # Decrement unprocessed tasks counter

    scheduler_queue.reverse()  # reverse the order of the schedule to pop the earliest first

    start_time = 0
    edf_single_node_schedule = []
    missed_deadline = []

    for t in range(num_tasks):
        id = scheduler_queue.pop()
        end_time = start_time + tasks_dict[id]['wcet']
        # Check if the task meets its deadline
        if end_time > tasks_dict[id]['deadline']:
            missed_deadline.append(id)  # Append tasks that missed the deadline
            continue
        else:
            # Append to edf single-node schedule
            edf_single_node_schedule.append({
                "task_id": id,
                "node_id": 0,
                "end_time": end_time,
                "deadline": tasks_dict[id]['deadline'],
                "start_time": start_time,
            })
            start_time = end_time  # Update the next available start time for the node

    return {"schedule": edf_single_node_schedule, "name": "EDF Single Node", "missed_deadline" : missed_deadline}


def ll_multinode(application_data, platform_data):
    """
    Schedule jobs on a distributed system with multiple compute nodes using the Least Laxity (LL) strategy.
    This function schedules jobs based on their laxity, with the job having the least laxity being scheduled first.

    .. todo:: Implement Least Laxity (LL) algorithm to schedule jobs on multiple node in a distributed system.

    Args:
        application_data (dict): Job data including dependencies represented by messages between jobs.

    Returns:
        list of dict: Contains the scheduled job details, each entry detailing the node assigned, start and end times,
                      and the job's deadline.

    """
    tasks = application_data.get("tasks")
    messages = application_data.get("messages")
    nodes = platform_data.get("nodes")

    node_dict = {}
    ready_node = []
    # Initialize the node_dict and ready_node list
    for node in nodes:
        # Process only the compute nodes
        if (node['type'] == 'compute'):
            node_dict[node['id']] = {'waiting_time' : 0, 'tasks' : []}
            ready_node.append((0, node['id']))  # Append each node available time and node id

    Graph = nx.DiGraph()  # Create directed graph object

    # Create a dictionary holding main task parameters {id, deadline, worst_case_execution_time, laxity}
    tasks_dict = {}
    for task in tasks:
        laxity = task['deadline'] - task['wcet']
        tasks_dict[task['id']] = {'deadline' : task['deadline'], 'wcet' : task['wcet'], 'laxity': laxity}

    # Add tasks to the graph
    for task in tasks:
        Graph.add_node(task['id'])

    # Add edges to the graph according to the dependencies
    for message in messages:
        Graph.add_edge(message['sender'], message['receiver'])

    root_task = []  # To add the available root tasks laxity and id
    added_tasks = []  # To track tasks being processed

    ready_tasks = 0  # To track the tasks ready to be scheduled
    least_waiting_time = 0  # To track the least node waiting time to equalize the other unused nodes
    nearest_task_end_time = 0
    edges_release_time = []  # To track each predecessor end time to remove the edges from the graph
    ll_multi_node_schedule = []
    task_end_time = []  # To track each task end time to get removed from the graph

    while Graph.number_of_nodes() != 0:    
        # Get roots
        for task in Graph:
            # Check if task has no in edges and not added to the queue
            if (Graph.in_degree()[task] == 0 and not (added_tasks.count(task))):
                root_task.append((tasks_dict[task]['laxity'], task))
                added_tasks.append(task)
                ready_tasks += 1

        root_task.sort(reverse=True)  # Sort according to the earliest deadline in reverse order

        temp_queue = []  # Queue to update the ready_node queue the next iteration
        waiting_time_list = []  # List to track used nodes waiting time to be available
        for n in ready_node:
            ready_node_id = n[1]
            if ready_tasks:
                ready_tasks -= 1
                ll_task = root_task.pop()[1]  # Get the task with least laxity
                task_wcet = tasks_dict[ll_task]['wcet']
                node_dict[ready_node_id]['tasks'] += [ll_task]  # Append task to the available node's task list
                start_time = node_dict[ready_node_id]['waiting_time']  # Get the starting time from the available node dict
                node_dict[ready_node_id]['waiting_time'] += task_wcet  # Update the node waiting time
                temp_queue.append((node_dict[ready_node_id]['waiting_time'], ready_node_id))  # Update the temp_queue
                waiting_time_list.append(node_dict[ready_node_id]['waiting_time'])
                least_waiting_time = min(waiting_time_list)  # Get the minimum node waiting time

                outEdges = Graph.out_edges(ll_task)
                edges_release_time.append((ll_task, start_time + task_wcet, list(outEdges)))  # Append task id, end time, list of edges
                # Append each task end time without duplication
                if not task_end_time.count(start_time + task_wcet):
                    task_end_time.append(start_time + task_wcet)
                
                ll_multi_node_schedule.append({
                    "task_id": ll_task,
                    "node_id": ready_node_id,
                    "end_time": start_time + task_wcet,
                    "deadline": tasks_dict[ll_task]['deadline'],
                    "start_time": start_time,
                })
            else:  # Update the unused nodes with the least waiting time
                node_dict[ready_node_id]['waiting_time'] = max(least_waiting_time, node_dict[ready_node_id]['waiting_time'])
                temp_queue.append((node_dict[ready_node_id]['waiting_time'], ready_node_id))
        
        task_end_time.sort(reverse=True)  # Sort the task_end_time to get the nearest time at the end 
        if len(task_end_time):
            nearest_task_end_time = task_end_time.pop()
            # Check for each task release time
            eIdx = 0
            while eIdx < len(edges_release_time):
                # Check if the task has already been executed
                if (edges_release_time[eIdx][1] <= nearest_task_end_time):
                    Graph.remove_edges_from(edges_release_time[eIdx][2])  # Remove task out edges from the graph 
                    Graph.remove_node(edges_release_time[eIdx][0])  # Remove task from the graph
                    edges_release_time.pop(eIdx)
                else:
                    eIdx += 1  # Increment the loop iterator

        # Update all nodes waiting time with the nearest_task_end_time
        if (least_waiting_time < nearest_task_end_time):
            temp_queue = []
            for n in ready_node:
                node = n[1]
                node_dict[node]['waiting_time'] = nearest_task_end_time
                temp_queue.append((node_dict[node]['waiting_time'], node))

        # Update ready_node for next iteration
        ready_node = temp_queue  
        ready_node.sort()

    return {"schedule": ll_multi_node_schedule, "name": "LL Multi Node"}


def ldf_multinode(application_data, platform_data):
    """
    Schedule jobs on a distributed system with multiple compute nodes using the Latest Deadline First(LDF) strategy.
    This function schedules jobs based on their periods and deadlines, with the shortest period job being scheduled first.

    .. todo:: Implement Latest Deadline First(LDF) algorithm to schedule jobs on multiple nodes in a distributed system.

    Args:
        application_data (dict): Job data including dependencies represented by messages between jobs.
        platform_data (dict): Contains information about the platform, nodes and their types, the links between the nodes and the associated link delay.

    Returns:
        list of dict: Contains the scheduled job details, each entry detailing the node assigned, start and end times,
                      and the job's deadline.

    """
    tasks = application_data.get("tasks")
    messages = application_data.get("messages")
    nodes = platform_data.get("nodes")

    G1 = nx.DiGraph()
    G2 = nx.DiGraph()

    for task in tasks:
        G1.add_node(task["id"], wcet=task["wcet"], deadline=task["deadline"])
        G2.add_node(task["id"], wcet=task["wcet"], deadline=task["deadline"])

    for message in messages:
        G1.add_edge(message["sender"], message["receiver"])
        G2.add_edge(message["sender"], message["receiver"])

    sortedLDF = []
    ldf_multinode_schedule = []

    outDegree = [node for node in G1.nodes if G1.out_degree(node) == 0]

    while outDegree:
        outDegree.sort(key=lambda node: G1.nodes[node]["deadline"], reverse=True)

        task_id = outDegree.pop(0)
        task = G1.nodes[task_id]

        sortedLDF.insert(
            0,
            {
                "task_id": task_id,
                "node_id": 1,  
                "deadline": task["deadline"],
            },
        )

        for predecessor in list(G1.predecessors(task_id)):
            G1.remove_edge(predecessor, task_id)
            if G1.out_degree(predecessor) == 0:
                outDegree.append(predecessor)

    rootNodes = [node for node in G2.nodes if G2.in_degree(node) == 0]
    sortedLDF_index = {entry["task_id"]: index for index, entry in enumerate(sortedLDF)}
    rootNodes.sort(
        key=lambda node: sortedLDF_index[node],
    )
    start_time = 0
    all_nodes = [
        {
            "id": node["id"],
            "type": node["type"],
            "ending_time": 0,
            "task_id": 0,
        }
        for node in nodes
        if node["type"] == "compute"
    ]
    while nx.number_of_edges(G2) > 0 or rootNodes:
        # Sort root nodes by the order of sortedLDF
        rootNodes.sort(
            key=lambda node: sortedLDF_index[node],
        )
        available_nodes = [
            node for node in all_nodes if node["ending_time"] <= start_time
        ]

        while rootNodes and available_nodes:
            available_node = available_nodes.pop(0)
            task_id = rootNodes.pop(0)
            task = G2.nodes[task_id]

            # Calculate end time based on start time and task's WCET
            end_time = start_time + task["wcet"]
            for node in all_nodes:
                if node["id"] == available_node["id"]:
                    node["ending_time"] = end_time
                    node["task_id"] = task_id
                    break

            # Check if the task meets its deadline
            if end_time > task["deadline"]:
                break
            else:
                # Append to multi-node schedule
                ldf_multinode_schedule.append(
                    {
                        "task_id": task_id,
                        "node_id": available_node["id"],
                        "start_time": start_time,
                        "end_time": end_time,
                        "deadline": task["deadline"],
                    }
                )

        # Update task IDs for removal from graph2
        for node in all_nodes:
            if node["ending_time"] == start_time + 1:
                for successor in list(G2.successors(node["task_id"])):
                    G2.remove_edge(node["task_id"], successor)
                    if G2.in_degree(successor) == 0:
                        rootNodes.append(successor)

        # Remove processed tasks and their edges from graph2
        # Update starting time for the next iteration
        start_time += 1
        
    return {"schedule": ldf_multinode_schedule, "name": "LDF Multi Node"}


def edf_multinode(application_data, platform_data):
    """
    Schedule jobs on a distributed system with multiple compute nodes using the Earliest Deadline First (EDF) strategy.
    This function processes application data to schedule jobs based on the earliest
    deadlines.

    .. todo:: Implement Earliest Deadline First(EDF) algorithm to schedule jobs on multiple nodes in a distributed system.

    Args:
        application_data (dict): Job data including dependencies represented by messages between jobs.
        platform_data (dict): Contains information about the platform, nodes and their types, the links between the nodes and the associated link delay.

    Returns:
        list of dict: Contains the scheduled job details, each entry detailing the node assigned, start and end times,
                      and the job's deadline.

    """
    tasks = application_data.get("tasks")
    messages = application_data.get("messages")
    nodes = platform_data.get("nodes")

    node_dict = {}
    ready_node = []

    # Initialize the node_dict and ready_node list
    for node in nodes:
        # Process only the compute nodes
        if (node['type'] == 'compute'):
            node_dict[node['id']] = {'waiting_time' : 0, 'tasks' : []}
            ready_node.append((0, node['id']))  # Append each node available time and node id

    Graph = nx.DiGraph()  # Create directed graph object

    # Create a dictionary holding main task parameters {id, deadline, worst_case_execution_time}
    tasks_dict = {}
    for task in tasks:
        tasks_dict[task['id']] = {'deadline' : task['deadline'], 'wcet' : task['wcet']}

    # Add tasks to the graph
    for task in tasks:
        Graph.add_node(task['id'])

    # Add edges to the graph according to the dependencies
    for message in messages:
        Graph.add_edge(message['sender'], message['receiver'])

    root_task = []  # To add the available root tasks earliest deadline and id
    added_tasks = []  # To track tasks being processed

    ready_tasks = 0  # To track the tasks ready to be scheduled
    least_waiting_time = 0  # To track the least node waiting time to equalize the other unused nodes
    nearest_task_end_time = 0
    edges_release_time = []  # To track each predecessor end time to remove the edges from the graph
    edf_multi_node_schedule = []
    task_end_time = []  # To track each task end time to get removed from the graph
    while Graph.number_of_nodes() != 0:    
        # Get roots
        for n in Graph:
            if (Graph.in_degree()[n] == 0 and not (added_tasks.count(n))):
                root_task.append((tasks_dict[n]['deadline'], n))
                added_tasks.append(n)
                ready_tasks += 1

        root_task.sort(reverse=True)  # Sort according to the earliest deadline in reverse order

        temp_queue = []  # Queue to update the ready_node queue the next iteration
        waiting_time_list = []  # List to track used nodes waiting time to be available
        for n in ready_node:
            ready_node_id = n[1]
            if ready_tasks:
                ready_tasks -= 1
                earliest = root_task.pop()[1]  # Get the task with earliest deadline
                task_wcet = tasks_dict[earliest]['wcet']
                node_dict[ready_node_id]['tasks'] += [earliest]  # Append task to the available node's task list
                start_time = node_dict[ready_node_id]['waiting_time']  # Get the starting time from the available node dict
                node_dict[ready_node_id]['waiting_time'] += task_wcet  # Update the node waiting time
                temp_queue.append((node_dict[ready_node_id]['waiting_time'], ready_node_id))  # Update the temp_queue
                waiting_time_list.append(node_dict[ready_node_id]['waiting_time'])
                least_waiting_time = min(waiting_time_list)  # Get the minimum node waiting time

                outEdges = Graph.out_edges(earliest)
                edges_release_time.append((earliest, start_time + task_wcet, list(outEdges)))  # Append task id, end time, list of edges
                # Append each task end time without duplication
                if not task_end_time.count(start_time + task_wcet):
                    task_end_time.append(start_time + task_wcet)

                edf_multi_node_schedule.append({
                    "task_id": earliest,
                    "node_id": ready_node_id,
                    "end_time": start_time + task_wcet,
                    "deadline": tasks_dict[earliest]['deadline'],
                    "start_time": start_time,
                })
            else:  # Update the unused nodes with the least waiting time
                node_dict[ready_node_id]['waiting_time'] = max(least_waiting_time, node_dict[ready_node_id]['waiting_time'])
                temp_queue.append((node_dict[ready_node_id]['waiting_time'], ready_node_id))
        
        task_end_time.sort(reverse=True)  # Sort the task_end_time to get the nearest time at the end 
        if len(task_end_time):
            nearest_task_end_time = task_end_time.pop()
            # Check for each task release time
            eIdx = 0
            while eIdx < len(edges_release_time):
                # Check if the task has already been executed
                if (edges_release_time[eIdx][1] <= nearest_task_end_time):
                    Graph.remove_edges_from(edges_release_time[eIdx][2])  # Remove task out edges from the graph 
                    Graph.remove_node(edges_release_time[eIdx][0])  # Remove task from the graph
                    edges_release_time.pop(eIdx)
                else:
                    eIdx += 1  # Increment the loop iterator

        # Update all nodes waiting time with the nearest_task_end_time
        if (least_waiting_time < nearest_task_end_time):
            temp_queue = []
            for n in ready_node:
                node = n[1]
                node_dict[node]['waiting_time'] = nearest_task_end_time
                temp_queue.append((node_dict[node]['waiting_time'], node))

        # Update ready_node for next iteration
        ready_node = temp_queue  
        ready_node.sort()

    return {"schedule": edf_multi_node_schedule, "name": "EDF Multi Node"}


