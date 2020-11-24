#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: assgin.py
# Author: Qian Ge <geqian1001@gmail.com>

import numpy as np

def assign_weight_count_all_0_case_1(cell, in_dim, out_dim):
    """ Parameters for counting all the '0' in the squence

    Input node only receives digit '0' and all the gates are
    always open.

    Args:
        in_dim (int): dimension of input
        out_dim (int): dimension of internal state and output
    """
    param_dict = {}
    param_dict['wgx'] = [[100.] if i == 0 else [0.] for i in range(10)]
    param_dict['wgh'] = np.zeros((out_dim, out_dim))
    param_dict['bg'] = np.zeros((1, out_dim))

    param_dict['wix'] = np.zeros((in_dim, out_dim))
    param_dict['wih'] = np.zeros((out_dim, out_dim))
    param_dict['bi'] =  100. * np.ones((1, out_dim))

    param_dict['wfx'] = np.zeros((in_dim, out_dim))
    param_dict['wfh'] = np.zeros((out_dim, out_dim))
    param_dict['bf'] = 100. * np.ones((1, out_dim))

    param_dict['wox'] = np.zeros((in_dim, out_dim))
    param_dict['woh'] = np.zeros((out_dim, out_dim))
    param_dict['bo'] = 100. * np.ones((1, out_dim))

    for key in param_dict:
        cell.set_config_by_name(key, param_dict[key])

def assign_weight_count_all_case_2(cell, in_dim, out_dim):
    """ Parameters for counting all the '0' in the squence
    
    Input node receives all the digits '0' but input gate only 
    opens for digit '0'. Other gates are always open.

    Args:
        in_dim (int): dimension of input
        out_dim (int): dimension of internal state and output
    """
    param_dict = {}
    param_dict['wgx'] = np.zeros((in_dim, out_dim))
    param_dict['wgh'] = np.zeros((out_dim, out_dim))
    param_dict['bg'] = 100. * np.ones((1, out_dim))

    param_dict['wix'] = [[100.] if i == 0 else [-100.] for i in range(10)]
    param_dict['wih'] = np.zeros((out_dim, out_dim))
    param_dict['bi'] =  np.zeros((1, out_dim))

    param_dict['wfx'] = np.zeros((in_dim, out_dim))
    param_dict['wfh'] = np.zeros((out_dim, out_dim))
    param_dict['bf'] = 100. * np.ones((1, out_dim))

    param_dict['wox'] = np.zeros((in_dim, out_dim))
    param_dict['woh'] = np.zeros((out_dim, out_dim))
    param_dict['bo'] = 100. * np.ones((1, out_dim))

    for key in param_dict:
        cell.set_config_by_name(key, param_dict[key])

#Task 2
def assign_weight_count_task_2(cell, in_dim, out_dim):
    """ Parameters for counting number of 0 after receiving the first 2 in the sequence;
    Internal state consists of two channels:
        1-st : number of '0' after receiving the first '2' (output)
        2-nd : number of '2'
        
    Input node's first channel is activated when '0' is received and its second channel is activated when '2' is received,
    while either channel is closed for all other situation.
        
    Input gate's first channel is open only when previous state's second channel is activated, i.e. model encountered '2' before.
    Its second channel is open when either current input consists of '2' or previous state's second channel is activated. 
    
    All channels of Output gate and Forget gate are always open.
    
    Args:
        in_dim (int): dimension of input
        out_dim (int): dimension of internal state and output
    """
    param_dict = {}
    param_dict['wgx'] = [[100., 0.] if i == 0 else [0., 100.] if i == 2 else [0., 0.] for i in range(10)]
    param_dict['wgh'] = np.zeros((out_dim, out_dim))
    param_dict['bg'] = np.zeros((1, out_dim))

    param_dict['wix'] = [[-100., 100.] if i == 2 else [-100., -100.] for i in range(10)]
    param_dict['wih'] = [[0., 200.], [200., 200.]]
    param_dict['bi'] = np.zeros((1, out_dim))

    #param_dict['wfx'] = [[100., -100.] if i == 2 else [100., 100.] for i in range(10)]
    param_dict['wfx'] = np.zeros((in_dim, out_dim))
    param_dict['wfh'] = np.zeros((out_dim, out_dim))
    param_dict['bf'] = 100. * np.ones((1, out_dim))
    #param_dict['bf'] = np.zeros((1, out_dim))

    param_dict['wox'] = np.zeros((in_dim, out_dim))
    param_dict['woh'] = np.zeros((out_dim, out_dim))
    param_dict['bo'] = 100. * np.ones((1, out_dim))

    for key in param_dict:
        cell.set_config_by_name(key, param_dict[key])

#Task 3
def assign_weight_count_task_3(cell, in_dim, out_dim):
    """ Parameters for counting number of 0 after receiving the first 2 in the sequence, but reset counter after receiving 3;
    Internal state consists of two channels:
        1-st : number of '0' after receiving the first '2' (output)
        2-nd : number of '2'
        
    Input node's first channel is activated when '0' is received and its second channel is activated when '2' is received,
    while either channel is deactivated for all other situation.
        
    Input gate's first channel is open only when previous state's second channel is activated, i.e. model encountered '2' before.
    It's second channel is open when either current input consists of '2' or previous state's second channel is activated. 
    
    Both channels of Forget gate are closed when there is '3' in current input.
    
    All channels of Output gate are always open.

    Args:
        in_dim (int): dimension of input
        out_dim (int): dimension of internal state and output
    """
    param_dict = {}
    param_dict['wgx'] = [[100., 0.] if i == 0 else [0., 100.] if i == 2 else [0., 0.] for i in range(10)]
    param_dict['wgh'] = np.zeros((out_dim, out_dim))
    param_dict['bg'] = np.zeros((1, out_dim))

    param_dict['wix'] = [[-100., 100.] if i == 2 else [-100., -100.] for i in range(10)]
    param_dict['wih'] = [[0., 200.], [200., 200.]]
    param_dict['bi'] = np.zeros((1, out_dim))

    param_dict['wfx'] = [[-100., -100.] if i == 3 else [100., 100.] for i in range(10)]
    #param_dict['wfx'] = np.zeros((in_dim, out_dim))
    param_dict['wfh'] = np.zeros((out_dim, out_dim))
    #param_dict['bf'] = 100. * np.ones((1, out_dim))
    param_dict['bf'] = np.zeros((1, out_dim))

    param_dict['wox'] = np.zeros((in_dim, out_dim))
    param_dict['woh'] = np.zeros((out_dim, out_dim))
    param_dict['bo'] = 100. * np.ones((1, out_dim))

    for key in param_dict:
        cell.set_config_by_name(key, param_dict[key])        
        