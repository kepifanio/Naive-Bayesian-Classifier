# Written by: Katherine Epifanio
# Date: 04-27-21

import numpy
import math
from tabulate import tabulate

data = []
pdf = []
bird_pdf = []
airpl_pdf = []
LOG_TRANSITION = math.log(0.9)
result_tbl = []
result_tbl.append(["#", "AIRPLANE PROBABILITY", "BIRD PROBABILITY", "CLASSIFICATION"])

# Parse up the data from "data.txt" to produce a 10-element list
# comprising 10 data arrays.
def parse_data():
    num_elems = len(data)
    new_list = []
    sub_list = []
    count = 0
    for i in range(num_elems):
        sub_list.append(data[i])
        count += 1
        if count == 300:
            new_list.append(sub_list)
            count = 0
            sub_list = []
    return new_list

# Grabs data points from text files and stores them.
def get_data(filename, list):
    file = open(filename, 'r')
    Lines = file.readlines()
    for line in Lines:
        curr_line = line.split(",")
        for data in curr_line:
            list.append(float(data))

# Parses bird and airplane data to separate lists.
def parse_pdf(bird, airplane):
    for i in range(400):
        bird.append(pdf[i])
        airplane.append(pdf[i + 400])

# Check for NaN value stored in array
def nan_check(velocity, type_arr):
    i = velocity * 2
    if math.isnan(i):
        return 0
    else:
        return type_arr[int(i)]

# Calculates product of all probability values
def mult_probs(data_arr, type):
    # Append all non-NaN probability logs to list
    logs = []
    arr_len = len(data_arr)
    count = 0
    for i in range(arr_len):
        prob = nan_check(data_arr[i], type)
        if prob != 0:
            curr_log = math.log(prob) + LOG_TRANSITION
            logs.append(curr_log)
    # Calculate log sum to return
    max_val = max(logs)
    log_len = len(logs)
    for i in range(log_len):
        new_val = math.exp(logs[i] - max_val)
        logs[i] = new_val
    logs_sum = math.log(sum(logs))
    return logs_sum + max_val

# Determine classification, factoring in standard deviation
def stdev(v, type):
    if v < 4:
        if type == "airplane":
            return 1
        else:
            return -1
    else:
        if type == "bird":
            return 1
        else:
            return -1

# Store our probability and classification results in table
def store_results(a_final, b_final, obsv_num):
    if a_final < b_final:
        type = "BIRD"
    else:
        type = "AIRPLANE"
    result_tbl.append([obsv_num, a_final, b_final, type])

# Print calculated probabilities and resulting classification
def print_results():
    print("\n\n")
    print("                   ------ FLYING-OBJECT RADAR ------")
    print("  * Classifying birds vs. airplanes via Naive Bayesian Classification * \n\n")
    print(tabulate(result_tbl, headers="firstrow", tablefmt="fancy_grid"))
    print("\n\n")

# Drives iteration through data and classification
def naive_bayesian():
    # Read in data and store in lists
    get_data("data.txt", data)
    get_data("pdf.txt", pdf)
    data_list = parse_data()
    parse_pdf(bird_pdf, airpl_pdf)
    list_len = len(data_list)
    obsv_num = 1
    for i in range(list_len):
        air_prob = mult_probs(data_list[i], airpl_pdf)
        bird_prob = mult_probs(data_list[i], bird_pdf)
        # Log-Sum-Exp
        denominator = air_prob + bird_prob
        a_final = air_prob - denominator
        b_final = bird_prob - denominator
        # Factor in standard deviation
        v = numpy.nanstd(data_list[i])
        a_stdev = stdev(v, "airplane")
        b_stdev = stdev(v, "bird")
        # Final probability values
        a_final = a_final + a_stdev
        b_final = b_final + b_stdev
        store_results(a_final, b_final, obsv_num)
        obsv_num += 1

    print_results()

naive_bayesian()
