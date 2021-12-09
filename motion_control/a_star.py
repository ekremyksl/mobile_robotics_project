from dijkstar import Graph, find_path
import math

def dijkstra(A,B):
    for i in range(0,len(A)):
        for j in range(0,len(A)):
            if A[i][j] !=0:
                A[i][j] = math.sqrt((B[j][0]-B[i][0])**2 + (B[i][1]-B[j][1])**2)



    #print('Matrix:',A)
    #print('length of A:', len(A))

    #print('Coodrinates:',B)
    #print('length of B:', len(B))

    def heuristic_func(u, v, edge, prev_edge):
        length_to_goal = math.sqrt((B[v][0]-B[-1][0])**2 + (B[v][1]-B[-1][1])**2)
        # print(u,v,length_to_goal)
        return length_to_goal

    graph = Graph()
    for i in range(0,len(A)):
        for j in range(0,len(A)):
            if A[i][j] != 0:
                graph.add_edge(i, j, A[i][j])
    #print('graph:',graph)

    # find_path(graph, 0, len(A)-1)
    nodes, edges, costs, total_cost = find_path(graph, 0, len(A)-1, heuristic_func=heuristic_func)
    #print('Nodes to for optimal path:',nodes)
    #print('Edges:', edges)
    #print('Costs:',costs)
    #print('Total cost:', total_cost)

    points = list()
    for point in nodes:
        points.append([B[point][0], B[point][1]])

    #print('Coordinates:',points)
    return points
