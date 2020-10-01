from math import log, sqrt
import numpy as np

def backtrack_tdsdfuct_all(info_table,reward):
    for path_ucb in reversed(info_table):
        ind = path_ucb[0][3]
        path_ucb[0][0] += reward
        path_ucb[0][1] += 1
        path_ucb[0][2] -= 1
        path_ucb[ind+1][0] += reward
        path_ucb[ind+1][1] += 1
        path_ucb[ind+1][2] -= 1
    return info_table

def backtrack_tdsdfuct_top2(info_table,reward):
    for path_ucb in reversed(info_table):
        ind = 0
        path_ucb[0][0] += reward
        path_ucb[0][1] += 1
        path_ucb[0][2] -= 1
        path_ucb[ind+1][0] += reward
        path_ucb[ind+1][1] += 1
        path_ucb[ind+1][2] -= 1
    return info_table

def backtrack_mpmcts_all(pnode, cnode):
    for path_ucb in reversed(pnode.path_ucb):
        ind = path_ucb[0][3]
        path_ucb[0][0] += cnode.reward
        path_ucb[0][1] += 1
        path_ucb[0][2] -= 1
        path_ucb[ind+1][0] += cnode.reward
        path_ucb[ind+1][1] += 1
        path_ucb[ind+1][2] -= 1
    return info_table

def backtrack_mpmcts_top2(pnode, cnode):
    for path_ucb in reversed(pnode.path_ucb):
        ind = 0
        path_ucb[0][0] += cnode.reward
        path_ucb[0][1] += 1
        path_ucb[0][2] -= 1
        path_ucb[ind+1][0] += cnode.reward
        path_ucb[ind+1][1] += 1
        path_ucb[ind+1][2] -= 1
    return info_table



def compare_ucb_tdsdfuct_all(info_table,pnode):
    #print ("check info_table:",info_table)
    for path_ucb in info_table:
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


def compare_ucb_tdsdfuct_top2(info_table,pnode):
    #print ("check info_table:",info_table)
    for path_ucb in info_table:
        ucb = []
        for i in range(len(path_ucb)-1):
            #ind = path_ucb[0][3]
            ind=0
            ucb.append((path_ucb[i+1][0]+0)/(path_ucb[i+1][1]+path_ucb[i+1][2]) +
                       1.0*sqrt(2*log(path_ucb[0][1]+path_ucb[0][2])/(path_ucb[i+1][1]+path_ucb[i+1][2])))
        new_ind = np.argmax(ucb)
        if new_ind!=ind:
            back_flag = 1
            break
        else:
            back_flag = 0
    return back_flag

def compare_ucb_mpmcts_top2(pnode):
    #print ("check info_table:",info_table)
    for path_ucb in pnode.path_ucb:
        ucb = []
        for i in range(len(path_ucb)-1):
            #ind = path_ucb[0][3]
            ind=0
            ucb.append((path_ucb[i+1][0]+0)/(path_ucb[i+1][1]+path_ucb[i+1][2]) +
                       1.0*sqrt(2*log(path_ucb[0][1]+path_ucb[0][2])/(path_ucb[i+1][1]+path_ucb[i+1][2])))
        new_ind = np.argmax(ucb)
        if new_ind!=ind:
            back_flag = 1
            break
        else:
            back_flag = 0
    return back_flag


def compare_ucb_mpmcts_all(pnode):
    #print ("check info_table:",info_table)
    for path_ucb in pnode.path_ucb:
        ucb = []
        for i in range(len(path_ucb)-1):
            ind = path_ucb[0][3]
            ucb.append((path_ucb[i+1][0]+0)/(path_ucb[i+1][1]+path_ucb[i+1][2]) +
                       1.0*sqrt(2*log(path_ucb[0][1]+path_ucb[0][2])/(path_ucb[i+1][1]+path_ucb[i+1][2])))
        new_ind = np.argmax(ucb)
        if new_ind!=ind:
            back_flag = 1
            break
        else:
            back_flag = 0
    return back_flag




def update_selection_ucbtable_tdsuct_all(node_table,node, ind):
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
        final_table.extend(node_table)
        final_table.append(table)
    return final_table

def update_selection_ucbtable_tdsdfuct_top2(node_table,node, ind):
    ind=0
    table=[]
    final_table = []
    node_info = store_info(node)
    node_info.append(ind)
    table.append(node_info)
    ucb=[]
    for i in range(len(node.childNodes)):
        ucb.append((node.childNodes[i].wins +
            node.childNodes[i].virtual_loss) /
                       (node.childNodes[i].visits +node.childNodes[i].num_thread_visited) +
                       1.0 *sqrt(2 *log(node.visits +node.num_thread_visited) /
                            (node.childNodes[i].visits +node.childNodes[i].num_thread_visited)))
    top2index=sorted(range(len(ucb)), key=lambda i: ucb[i])[-2:]
    #print ("top2best:",top2index)

    for i in range(len(top2index)):
        child_info = store_info(node.childNodes[i])
        table.append(child_info)
    if node.state == ['&']:
        final_table.append(table)
    else:
        final_table.extend(node_table)
        final_table.append(table)

    return final_table



def update_selection_ucbtable_mpmcts_top2(node, ind):
    ind=0
    table = []
    final_table = []
    node_info = store_info(node)
    node_info.append(ind)
    table.append(node_info)
    ucb=[]
    for i in range(len(node.childNodes)):
        ucb.append((node.childNodes[i].wins +
            node.childNodes[i].virtual_loss) /
                       (node.childNodes[i].visits +node.childNodes[i].num_thread_visited) +
                       1.0 *sqrt(2 *log(node.visits +node.num_thread_visited) /
                            (node.childNodes[i].visits +node.childNodes[i].num_thread_visited)))
    top2index=sorted(range(len(ucb)), key=lambda i: ucb[i])[-2:]
   # print ("top2best:",top2index)

    for i in range(len(top2index)):
        child_info = store_info(node.childNodes[i])
        table.append(child_info)
    if node.state == ['&']:
        final_table.append(table)
    else:
        final_table.extend(node.path_ucb)
        final_table.append(table)
    return final_table


def update_selection_ucbtable_mpmcts_all(node, ind):
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
