from math import log, sqrt
import numpy as np

def backtrack(pnode, cnode):
    for path_ucb in reversed(pnode.path_ucb):
        ind = path_ucb[0][3]
        path_ucb[0][0] += cnode.reward
        path_ucb[0][1] += 1
        path_ucb[0][2] -= 1
        path_ucb[ind+1][0] += cnode.reward
        path_ucb[ind+1][1] += 1
        path_ucb[ind+1][2] -= 1
    return pnode

def compare_ucb(pnode):
    for path_ucb in pnode.path_ucb:
        ucb = []
        for i in range(len(path_ucb)-1):
            ind = path_ucb[0][3]
            ucb.append((path_ucb[i+1][0]+0)/(path_ucb[i+1][1]+path_ucb[i+1][2]) +
                       1.0*sqrt(2*log(path_ucb[0][1]+path_ucb[0][2])/(path_ucb[i+1][1]+path_ucb[i+1][2])))
        new_ind = np.argmax(ucb)
        if ind != new_ind:
            back_flag = 1
            break
        else:
            back_flag = 0
    return back_flag

def update_selection_ucbtable(node, ind):
    table = []
    final_table = []
    node_info = store_info(node)
    node_info.append(ind)
    table.append(node_info)
    for i in range(len(node.childNodes)):
        child_info = store_info(node.childNodes[i])
        table.append(child_info)
    if node.state == ['&']:
        final_table.append(table)
    else:
        final_table.extend(node.path_ucb)
        final_table.append(table)
    return final_table

def store_info(node):
    table = [node.wins, node.visits, node.num_thread_visited]
    return table
