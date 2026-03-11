import numpy as np
from configuration_parameters import *

# ============================================================
# TRELLIS TABLE GENERATION
# ============================================================

def build_recursive_systematic_convolutional_tables():
    """
    Builds the trellis tables for a Recursive Systematic Convolutional (RSC) encoder.
    
    Returns:
        next_state_table: next state for each current state and input bit
        parity_output_table: parity output for each state and input
        input_symbol_sign_table: BPSK mapping for input bits
        parity_symbol_sign_table: BPSK mapping for parity bits
        previous_states: list of previous states leading to each state
        next_states: list of possible next states from each state
    """

    next_state_table = np.zeros((NUMBER_OF_STATES,2),dtype=int)
    parity_output_table = np.zeros((NUMBER_OF_STATES,2),dtype=int)

    input_symbol_sign_table = np.zeros((NUMBER_OF_STATES,2))
    parity_symbol_sign_table = np.zeros((NUMBER_OF_STATES,2))

    previous_states = [[] for _ in range(NUMBER_OF_STATES)]
    next_states = [[] for _ in range(NUMBER_OF_STATES)]

    # Loop through all states
    for state in range(NUMBER_OF_STATES):

        # Extract shift register bits from state
        register_bit0 = (state >> 1) & 1
        register_bit1 = state & 1

        for input_bit in (0,1):

            # Feedback for recursive systematic encoder
            feedback_bit = input_bit ^ register_bit1

            # Parity calculation (7,5 polynomial)
            parity_bit = feedback_bit ^ register_bit0 ^ register_bit1

            # Update shift registers for next state
            next_register0 = feedback_bit
            next_register1 = register_bit0

            # Combine bits to get new state
            next_state = (next_register0 << 1) | next_register1

            # Fill tables
            next_state_table[state,input_bit] = next_state
            parity_output_table[state,input_bit] = parity_bit

            # Map bits to BPSK (+1/-1)
            input_symbol_sign_table[state,input_bit] = 1.0 if input_bit == 0 else -1.0
            parity_symbol_sign_table[state,input_bit] = 1.0 if parity_bit == 0 else -1.0

            # Record trellis connectivity
            previous_states[next_state].append((state,input_bit))
            next_states[state].append((next_state,input_bit))

    return next_state_table, parity_output_table, input_symbol_sign_table, parity_symbol_sign_table, previous_states, next_states

# Build tables once globally
NEXT_STATE_TABLE, PARITY_TABLE, INPUT_SIGN_TABLE, PARITY_SIGN_TABLE, PREVIOUS_STATES, NEXT_STATES = build_recursive_systematic_convolutional_tables()


# ============================================================
# CONVOLUTIONAL ENCODER FUNCTION
# ============================================================

def convolutional_encode_generator_7_5(input_bits):
    """
    Encode a block of bits using a (7,5) convolutional code in octal.
    
    Args:
        input_bits: array of information bits (0 or 1)
    
    Returns:
        encoded_bits: output bits of the convolutional encoder
    """

    # Append memory bits (zeros) to flush the encoder
    bits = np.concatenate([input_bits, np.zeros(CONVOLUTIONAL_MEMORY_ORDER,dtype=np.int8)])

    register0 = 0
    register1 = 0

    encoded_output = []

    for input_bit in bits:

        # Calculate the two outputs of the convolutional encoder
        output_bit0 = input_bit ^ register0 ^ register1
        output_bit1 = input_bit ^ register1

        encoded_output.extend((output_bit0, output_bit1))

        # Update shift registers
        register1 = register0
        register0 = int(input_bit)

    return np.array(encoded_output,dtype=np.int8)
