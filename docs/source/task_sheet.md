
# Lab 2 - Scheduling algorithms for Real Time Systems

## Introduction

Scheduling algorithms are of fundamental importance to the functioning of real-time embedded systems, which are integral to numerous critical applications, including automotive safety systems and healthcare monitoring devices. These systems rely on precise timing and immediate response to external events to function correctly and efficiently. The role of scheduling algorithms is to manage the execution priorities of tasks in a way that optimizes performance and ensures timely task completion. Inadequate scheduling can result in missed deadlines, system overloads, and failures, which in safety-critical systems could have severe consequences. 

Therefore, the application of effective scheduling algorithms is crucial for enhancing system reliability and overall performance. By ensuring that system resources are allocated efficiently and tasks are completed within their time constraints, scheduling algorithms help maintain the integrity and reliability of embedded systems.

![Intro image](https://upload.wikimedia.org/wikipedia/commons/0/0c/Thread_pool.svg)

*Figure 1: A pool of tasks completed based on scheduling*

## Objective

The primary goal outlined in this task sheet is to design and implement various scheduling algorithms within the context of real-time systems. The visualization frontend will serve as an educational tool, providing visual insights into how different algorithms schedule tasks. You will be able to interact with the application to modify task parameters, compare the outcomes of different scheduling algorithms, and gain a deeper understanding of their advantages and limitations in real-world scenarios.

## Background 

The implementation of scheduling algorithms requires basic knowledge of concepts like graph theory, operations on graphs, the different models used, and so on. This section describes in detail the concepts necessary to implement the scheduling algorithms.

### Graph Theory

[Graph Theory](https://www.baeldung.com/cs/graph-theory-intro#8-the-weighted-graph) is a branch of mathematics and computer science that studies the properties of graphs. A graph consists of vertices (or nodes) and edges (or arcs) that connect pairs of vertices. In short, a graph is defined as $G = (V, E)$, where $V$ is a set of nodes (also called vertices) and $E$ is a set of edges (also called links).

![alt text](./images/graphs-set.webp)
#### Components of a Graph

**Vertices** are the fundamental units of graphs and can represent entities such as cities, people, or points in space. The **degree of a vertex** in an undirected graph is the number of edges connected to it. In directed graphs, this is split into the indegree (number of edges directed into the vertex) and outdegree (number of edges directed out of the vertex). Another important component of a graph is the **edge**. Edges connect pairs of vertices and can represent relationships like roads between cities, friendships between people, etc. There are different types of graphs, which we will look into in the next section.
A **subgraph** is formed from a subset of the vertices and edges of a graph. A **path** is a sequence of edges that connect a sequence of vertices. A path that starts and ends at the same vertex, with no repeated edges or vertices (except the starting/ending vertex), is called a **cycle**.


#### Types of Graphs

##### Directed and Undirected Graphs

Directed Graphs are those whose edges have a direction, indicating a one-way relationship. Directed graphs have the characteristic that they model real-world relationships well, for which we cannot freely interchange the subject and the object. Undirected graphs are those whose edges do not have a direction. The edges indicate a two-way relationship in that each edge can be traversed in both directions. As a general rule, if we are not sure whether a graph should be directed or undirected, then the graph is directed. Undirected Graphs are those whose edges do not have a direction. Undirected graphs allow their traversal between any two vertices connected by an edge. The same is not necessarily true for directed graphs.

Directed Acyclic Graph (DAG) is an incredibly valuable instrument for representing and managing the relationships and dependencies among various tasks that need to be scheduled. In such a graph, each node represents a task, and the directed edges between these nodes indicate the precedence relationships, meaning that one task must be completed before another can begin. 

In DAGs used for scheduling, each node represents an individual task or task. These tasks often have associated attributes like duration, resource requirements, or deadlines. The directed edges between the nodes represent the precedence constraints. An edge from node A to node B implies that task A must be completed before task B can start. This directionality is critical for ensuring tasks are performed in the correct order. The acyclic property of DAGs is crucial in scheduling because it guarantees that there are no circular dependencies, which would make scheduling impossible. In practical terms, this means there can be no situation where task A depends on task B while task B simultaneously depends on task A, either directly or indirectly.

![alt text](./images/image-2.png)

*Figure 2: A directed acyclic graph (DAG) shows precedences, showing that process 1 must complete before processes 2 and 3 can be started, etc.*

##### Weighted and Unweighted Graphs

Weighted Graphs are those whose edges have weights or costs associated with them, which can represent distance, cost, or any other metric that needs to be tracked. A typical weighted graph commonly used in machine learning is an artificial neural network. We can conceptualize neural networks as directed weighted graphs on which each vertex has an extra activation function assigned to it. Unweighted graphs are those whose edges do not have any weights associated with them.

##### Connected and Disconnected Graphs

We can also discriminate graphs on the basis of the characteristics of their paths. For example, we can discriminate according to whether there are paths that connect all pairs of vertices, or whether there are pairs of vertices that do not have any paths between them. We call a graph connected if there is at least one path between any two of its vertices. Similarly, we say that a graph is disconnected if there are at least two vertices separated from one another.


#### Graphs And NetworkX
Graph operations are fundamental to manipulating and analyzing data structured as graphs. In Python, you can manage these operations using libraries like NetworkX, which provides comprehensive support for creating and manipulating graphs. Here is a breakdown of common graph operations and the necessary APIs to perform them in Python using NetworkX. Below are some operations that are performed on graphs as part of the lab:
- Creating an empty graph
``` BASH
import networkx as nx

# Create a directed graph
D = nx.DiGraph()
```
- Insertion of Nodes/Edges in the graph: Insert a node or edge into the graph.
``` BASH
# Insert a single node
G.add_node(1)
# Insert an edge between node 1 and 2
G.add_edge(1, 2)
```
- Deletion of Nodes/Edges in the graph: Delete a node from the graph.
``` BASH
# Remove a node
G.remove_node(2)

# Remove an edge between 1 and 3
G.remove_edge(1, 3)
```
- Reading Graph Data: Read graph data from various formats such as JSON. For example, the  **application_data** is a python dictionary that already contains details of tasks and their interdependencies. You have to create the graph networkx from this application_data.
- Searching on Graphs: Search an entity in the graph.
``` BASH
# Use BFS to find all nodes reachable from node 1
reachable_from_1 = list(nx.bfs_edges(G, 1))
```
- Sorting Graph Data: Sorting is not a direct operation in graph theory as it is in array manipulation, but you might sort nodes or edges based on attributes like degree, weight, etc.
``` BASH
# Sort nodes by degree
sorted_nodes_by_degree = sorted(G.nodes(), key=lambda x: G.degree(x))
```
 

#### Graph Traversal Algorithms

Some of the commonly used graph traversal algorithms are [Breadth First Search (BFS)](https://www.geeksforgeeks.org/breadth-first-search-or-bfs-for-a-graph/) and Depth First Search (DFS) algorithms. BFS starts at a root node and explores all neighbors at the present depth before moving on to nodes at the next depth level. It uses a queue to keep track of the next vertex to start a search when a dead end is reached in any iteration. BFS finds the shortest path in an unweighted graph. In networking, BFS is used for broadcasting packets of information to all nodes of a network. [DFS](https://www.geeksforgeeks.org/depth-first-search-or-dfs-for-a-graph/) uses a stack to explore as far as possible along each branch before backtracking. It goes deep into a graph by exploring a node's children before visiting its siblings.

#### Traversing Shortest Path in a Graph

Algorithms like Dijkstra's Algorithm find the shortest path from a single source to all other vertices in a graph. Also, the Bellman-Ford Algorithm computes the shortest paths from a single source vertex to all of the other vertices in a graph with edge weights that may be negative. An algorithm that efficiently combines both Dijkstra and Bellman-Ford algorithms is called Johnson's Algorithm.

                        
#### Minimum Spanning Tree of a graph
[Minimum Spanning Tree (MST)](https://www.geeksforgeeks.org/prims-minimum-spanning-tree-mst-greedy-algo-5/) of a graph is a fundamental concept in graph theory used primarily in network design and optimization. The MST is a subset of the edges of a weighted, undirected graph that connects all the vertices together without any cycles and with the minimal possible total edge weight. This tree spans all the vertices in the graph, meaning that every vertex is included, making it a spanning tree. The minimum aspect refers to the sum of the weights of the edges in the tree being as small as possible. There are several algorithms available to find an MST in a graph, with Kruskal's Algorithm and Prim's Algorithm being the most prominent. Consider a graph where the vertices represent cities, and the edges represent possible roads between them, with weights indicating the cost to build. An MST would provide a way to connect all the cities while minimizing the total construction cost. This could be crucial for governmental budgeting and planning in infrastructure projects.

### Application and Platform Models

The application model defines how different tasks in a system interact and communicate with each other. It includes the properties of tasks and the messages that are exchanged between them. The tasks are represented by nodes, and in Figure 2, nodes 1, 2, 3, 4, 5, and 6 indicate tasks. The arrows indicate the direction of communication, where one task sends a message to another task. 

The platform model represents the hardware and communication infrastructure of an embedded system. It includes nodes (such as compute units, routers, sensors, and actuators) and links that connect these nodes.

In Figure 3, the arrows between the nodes represent the communication links, with the numbers at the links indicating the link delay. Here, we define that a communication link can only exist between a router and another node, which can either be a compute node, sensor, actuator, or another router. 

![alt text](./images/pl_model.PNG)

Figure 3: Platform model

In multi-node scheduling, the tasks can be distributed across different compute nodes. They can be scheduled on different compute nodes based on the availability and the communication delays between the nodes. However, the tasks must be scheduled in a way that all precedence constraints are maintained. The communication delays between nodes must also be considered while scheduling the tasks. For this, the delays can be added to the execution time of the tasks while calculating the completion time of the tasks.


## Tasks

- Implement the following [scheduling algorithms](scheduling_algorithms.md) as outlined in the [to-do list](todo.rst). These algorithms should take json inputs describing the platform model and application model. The input json conforms to the JSON schema defined in the [input json schema](README.md#api-input-schema-for-schedule-jobs). They should produce a schedule output that adheres to the JSON schema defined in the [output json schema](README.md#output-schema-for-schedule-jobs).
    - Latest Deadline First Single Node (LDF).
    - Earliest Deadline First Single Node (EDF).
    - Least Laxity Multi Node (LLF).
    - Latest Deadline First Multi Node.
    - Earliest Deadline First Multi Node.

- Test your implementation using the provided test cases. The test cases are designed to evaluate the correctness of your scheduling algorithms. Ensure that your implementation passes all the test cases. Run the tests using the following command from the root of project directory:
  ``` BASH
    pytest
  ```
  To automatically run the tests after each change in algorithm, you can run the following command from the project root directory:

  ``` BASH
    pip install pytest-watch
    ptw
  ```

- For this lab, you can ignore the link delay and bandwidth fields in the platform model. You can assume that the communication between the nodes is instantaneous. There is no bandwidth constraint, and all nodes are connected to a single router. However, as a bonus, you are encouraged to consider the link delays and communication paths as defined in the platform model.


## Documentation and Reporting

As you test and optimize, document your findings. Note down what worked, what did not,and how you adjusted your approach. This documentation will be invaluable for your final report and provides a clear record of your problem solving process. Testing and optimization are as much a part of the learning process as the initial development. They provide insight into the practical challenges and the importance of iterative design.

### Documenting Code

- **Code Comments**:Include comprehensive comments throughout your python code. Explain the purpose of functions, logic behind critical sections, and meanings of key variables. Use inline comments for complex lines of code to clarify their functionality.

- **Readable Structure**:Organize your code  logically. Group related functionalities into functions or classes and use clear, descriptive names for variables and functions. Ensure your script follows a consistent coding style for ease of reading and maintenance.

### Writing Lab Report
 
Use the cover sheet template/report template provided by in moodle.

- Only the title page should show the logos in the header
- Use proper paper geometry, e.g., margins: right, left, top 2.5 cm, bottom 2 cm.
- Font size 11-12, line spacing between 1.15 and 1.5.

#### Goal of the report:

- The report summarizes and presents your work on the exercises and answers all given questions from the task sheets
- Use proper line of arguments
- In a step by step procedure, the report introduces the task, raises the questions, shows the solutions (including necessary intermediate results) and makes conclusions
- Readers (e.g., other students, supervisors and the professor) who have not attended the course and do not know about any implementations must be able to understand the report.

#### In the report

- Use meaningful headings for different sections (headings should be numbered and larger than the text, not bold and not underlined)
- Use table of contents (and list of figures)
- Number the pages
- When answering the questions, number and repeat the questions before providing answers below each corresponding question.

##### When presenting a plot or a figure

- Carefully decide, what you intend to show/highlight/demonstrate with the figure
- Introduce the figure with direct referencing in the text (e.g., Figure 1 shows …)
- Center the figure horizontally
- Label the figure
- Label the axes (description and unit), appropriate font size
- Use a grid if necessary
- Use multiple colors or types for different lines within a plot
- Use a legend if necessary
- Describe plots and figures in detail, especially focus on the signal characteristics
- Provide explanations for your plot descriptions
- Make conclusions from the plot
- Figures and tables should be upright if possible, so that the reader does not need to rotate the page
- Use appropriate file formats for figures (e.g. latex: eps, pdf, svg; word: emf, wmf, svg, (large-scaled png))

##### When presenting a formula/equation

Use proper visualization, e.g., in MS Word use the equation editor (insert -> equation) or use the LaTeX equation option
- Center equations horizontally
- Number the equation
- Use only symbols in equations and no written words
- Explain all used symbols in the text after the equation. The symbols should have the same font and should not look different (compare: $\beta$ and β).

##### Hand in the report and all necessary working files

- Check for grammar and spelling mistakes before submission.
- The report should be a single document file covering the complete exercise.
- Provide commented programming code/modeling files along with your report (.py files).
- The programs must run without errors.
- Prepare zipped archive (.zip, .rar, .7z, etc.) containing all working files and the report
- Upload the zipped archive to the moodle platform.

## Deliverables

- algorithms.py file with scheduling algorithm implemented.
- Lab report in pdf format.

## Evaluation Criterion

- Well-structured and logically organized code.
- Comprehensive comments explaining the purpose and logic of the code.
- Quality of the report.

## References
- [Introduction to Graph Theory](https://www.baeldung.com/csgraph-theory-intro#8-the-weighted-graph)
- [Basic Components, Shortest path, Search algorithms, Minimum Spanning Tree](https://www.geeksforgeeks.org/transpose-graph/?ref=lbp)
- [Basics of Graph theory](https://medium.com/basecs/a-gentle-introduction-to-graph-theory-77969829ead8)
