import numpy as np
import math

def compute_syndrome(H, hard_bits):
    check_count = H.shape[0]
    syndrome = np.zeros(check_count, dtype=np.int8)
    for row in range(check_count):
        value = 0
        for col in range(H.shape[1]):
            if H[row, col] != 0 and hard_bits[col] != 0:
                value ^= 1
        syndrome[row] = value
    return syndrome

def decode_ldpc_sum_product(received_symbols, noise_variance, iteration_count, H, check_edge_start, edge_variable, variable_edges, variable_edge_start, llr_clip=20.0, message_damping=0.20):
    variable_count = len(received_symbols)
    edge_count = len(edge_variable)
    channel_llr = np.empty(variable_count, dtype=np.float64)
    for i in range(variable_count):
        value = (2.0 / noise_variance) * received_symbols[i]
        if value > llr_clip:
            value = llr_clip
        elif value < -llr_clip:
            value = -llr_clip
        channel_llr[i] = value
    variable_to_check = np.empty(edge_count, dtype=np.float64)
    check_to_variable = np.zeros(edge_count, dtype=np.float64)
    new_check_to_variable = np.empty(edge_count, dtype=np.float64)
    posterior = np.empty(variable_count, dtype=np.float64)
    posterior_history = np.empty((iteration_count, variable_count), dtype=np.float64)
    for edge in range(edge_count):
        variable_to_check[edge] = channel_llr[edge_variable[edge]]
    for iteration_index in range(iteration_count):
        for check_index in range(len(check_edge_start) - 1):
            start = check_edge_start[check_index]
            end = check_edge_start[check_index + 1]
            degree = end - start
            if degree == 1:
                new_check_to_variable[start] = 0.0
                continue
            tanh_product_total_sign = 1.0
            tanh_abs = np.empty(degree, dtype=np.float64)
            for local in range(degree):
                value = variable_to_check[start + local] * 0.5
                if value > 10.0:
                    value = 10.0
                elif value < -10.0:
                    value = -10.0
                t = math.tanh(value)
                if t < 0.0:
                    tanh_product_total_sign *= -1.0
                    tanh_abs[local] = -t
                else:
                    tanh_abs[local] = t
            for local in range(degree):
                product_abs = 1.0
                sign_value = tanh_product_total_sign
                value_here = variable_to_check[start + local] * 0.5
                if value_here > 10.0:
                    value_here = 10.0
                elif value_here < -10.0:
                    value_here = -10.0
                t_here = math.tanh(value_here)
                if t_here < 0.0:
                    sign_value *= -1.0
                for other in range(degree):
                    if other != local:
                        product_abs *= tanh_abs[other]
                product = sign_value * product_abs
                if product > 0.999999:
                    product = 0.999999
                elif product < -0.999999:
                    product = -0.999999
                updated = 2.0 * np.arctanh(product)
                previous = check_to_variable[start + local]
                damped = (1.0 - message_damping) * updated + message_damping * previous
                if damped > llr_clip:
                    damped = llr_clip
                elif damped < -llr_clip:
                    damped = -llr_clip
                new_check_to_variable[start + local] = damped
        for edge in range(edge_count):
            check_to_variable[edge] = new_check_to_variable[edge]
        for variable_index in range(variable_count):
            value = channel_llr[variable_index]
            start = variable_edge_start[variable_index]
            end = variable_edge_start[variable_index + 1]
            for local in range(start, end):
                edge = variable_edges[local]
                value += check_to_variable[edge]
            if value > llr_clip:
                value = llr_clip
            elif value < -llr_clip:
                value = -llr_clip
            posterior[variable_index] = value
            posterior_history[iteration_index, variable_index] = value
        hard = (posterior < 0.0).astype(np.int8)
        if np.all(compute_syndrome(H, hard) == 0):
            for fill_index in range(iteration_index + 1, iteration_count):
                posterior_history[fill_index, :] = posterior
            return posterior_history
        for variable_index in range(variable_count):
            start = variable_edge_start[variable_index]
            end = variable_edge_start[variable_index + 1]
            for local in range(start, end):
                edge = variable_edges[local]
                value = posterior[variable_index] - check_to_variable[edge]
                if value > llr_clip:
                    value = llr_clip
                elif value < -llr_clip:
                    value = -llr_clip
                variable_to_check[edge] = value
    return posterior_history
