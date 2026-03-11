import numpy as np
from configuration_parameters import *

# ============================================================
# TRELLIS TABLE GENERATION
# ============================================================

def build_recursive_systematic_convolutional_tables():
    """
    Builds the trellis tables for a Recursive Systematic Convolutional (RSC) encoder.
    Returns tables for next states, parity bits, BPSK signs, and trellis connectivity.
    """
    next_state_table = np.zeros((NUMBER_OF_STATES,2),dtype=int)
    parity_output_table = np.zeros((NUMBER_OF_STATES,2),dtype=int)
    input_symbol_sign_table = np.zeros((NUMBER_OF_STATES,2))
    parity_symbol_sign_table = np.zeros((NUMBER_OF_STATES,2))
    previous_states = [[] for _ in range(NUMBER_OF_STATES)]
    next_states = [[] for _ in range(NUMBER_OF_STATES)]

    for state in range(NUMBER_OF_STATES):
        register_bit0 = (state >> 1) & 1
        register_bit1 = state & 1
        for input_bit in (0,1):
            feedback_bit = input_bit ^ register_bit1
            parity_bit = feedback_bit ^ register_bit0 ^ register_bit1
            next_register0 = feedback_bit
            next_register1 = register_bit0
            next_state = (next_register0 << 1) | next_register1
            next_state_table[state,input_bit] = next_state
            parity_output_table[state,input_bit] = parity_bit
            input_symbol_sign_table[state,input_bit] = 1.0 if input_bit == 0 else -1.0
            parity_symbol_sign_table[state,input_bit] = 1.0 if parity_bit == 0 else -1.0
            previous_states[next_state].append((state,input_bit))
            next_states[state].append((next_state,input_bit))

    return next_state_table, parity_output_table, input_symbol_sign_table, parity_symbol_sign_table, previous_states, next_states

NEXT_STATE_TABLE, PARITY_TABLE, INPUT_SIGN_TABLE, PARITY_SIGN_TABLE, PREVIOUS_STATES, NEXT_STATES = build_recursive_systematic_convolutional_tables()

# ============================================================
# CONVOLUTIONAL ENCODER
# ============================================================

def convolutional_encode_generator_7_5(input_bits):
    """
    Encode a block of bits using a (7,5) convolutional code.
    """
    bits = np.concatenate([input_bits, np.zeros(CONVOLUTIONAL_MEMORY_ORDER,dtype=np.int8)])
    register0 = 0
    register1 = 0
    encoded_output = []

    for input_bit in bits:
        output_bit0 = input_bit ^ register0 ^ register1
        output_bit1 = input_bit ^ register1
        encoded_output.extend((output_bit0, output_bit1))
        register1 = register0
        register0 = int(input_bit)

    return np.array(encoded_output,dtype=np.int8)

# ============================================================
# TURBO ENCODER
# ============================================================

def turbo_encode_information_bits(info_bits, interleaver):
    """
    Encode information bits using a Turbo code (two RSC encoders with interleaver).
    Returns systematic and parity bits from both encoders.
    """
    info_bits = np.asarray(info_bits, dtype=np.int8)
    # First encoder
    sys_bits1 = info_bits.copy()
    par_bits1 = convolutional_encode_generator_7_5(info_bits)[1::2]  # take parity
    # Interleaved for second encoder
    interleaved_bits = info_bits[interleaver]
    sys_bits2 = interleaved_bits.copy()
    par_bits2 = convolutional_encode_generator_7_5(interleaved_bits)[1::2]
    return sys_bits1, par_bits1, sys_bits2, par_bits2
        register0 = int(input_bit)

    return np.array(encoded_output,dtype=np.int8)
