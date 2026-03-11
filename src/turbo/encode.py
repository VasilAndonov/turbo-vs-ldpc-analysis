import numpy as np

# -----------------------------
# Encoder configuration
# -----------------------------
MEMORY_LENGTH = 2          # Memory length of the RSC encoder
TAIL_LENGTH = 2            # Number of tail bits to terminate encoder
NUMBER_OF_STATES = 4       # 2^memory_length

# -----------------------------
# Build RSC trellis tables
# -----------------------------
def build_rsc_trellis():
    """
    Builds the trellis tables for a 4-state RSC encoder with feedback polynomial 5 (octal) and feedforward 7 (octal).
    Returns:
        next_state_table : Next state for each current state and input bit
        parity_table     : Parity bit for each current state and input bit
        systematic_sign  : Sign mapping for systematic bits (for LLR calculations)
        parity_sign      : Sign mapping for parity bits (for LLR calculations)
        previous_states  : List of previous states and input bits leading to a given state
        next_states      : List of next states and input bits from a given state
    """
    next_state_table = np.zeros((NUMBER_OF_STATES, 2), dtype=int)
    parity_table = np.zeros((NUMBER_OF_STATES, 2), dtype=int)
    systematic_sign = np.zeros((NUMBER_OF_STATES, 2), dtype=float)
    parity_sign = np.zeros((NUMBER_OF_STATES, 2), dtype=float)
    previous_states = [[] for _ in range(NUMBER_OF_STATES)]
    next_states = [[] for _ in range(NUMBER_OF_STATES)]

    for state in range(NUMBER_OF_STATES):
        state_bit0 = (state >> 1) & 1
        state_bit1 = state & 1
        for input_bit in (0, 1):
            # Recursive bit: feedback polynomial 101 (octal)
            recursive_bit = input_bit ^ state_bit1
            # Parity bit: feedforward polynomial 111 (octal)
            parity_bit = recursive_bit ^ state_bit0 ^ state_bit1
            next_state_bit0 = recursive_bit
            next_state_bit1 = state_bit0
            next_state = (next_state_bit0 << 1) | next_state_bit1

            next_state_table[state, input_bit] = next_state
            parity_table[state, input_bit] = parity_bit
            systematic_sign[state, input_bit] = 1.0 if input_bit == 0 else -1.0
            parity_sign[state, input_bit] = 1.0 if parity_bit == 0 else -1.0
            previous_states[next_state].append((state, input_bit))
            next_states[state].append((next_state, input_bit))

    return next_state_table, parity_table, systematic_sign, parity_sign, previous_states, next_states


# Build global trellis tables
NEXT_STATE_TABLE, PARITY_TABLE, SYSTEMATIC_SIGN_TABLE, PARITY_SIGN_TABLE, PREVIOUS_STATES, NEXT_STATES = build_rsc_trellis()

# -----------------------------
# Encoder functions
# -----------------------------
def rsc_tail_bits(current_state):
    """
    Generate tail bits to force the encoder back to the zero state.
    """
    tail_bits = []
    state = current_state
    for _ in range(TAIL_LENGTH):
        input_bit = state & 1
        tail_bits.append(input_bit)
        state = NEXT_STATE_TABLE[state, input_bit]
    return tail_bits


def rsc_encode_terminated(information_bits):
    """
    Encode information bits using a terminated RSC encoder.
    """
    information_bits = np.asarray(information_bits, dtype=np.int8)
    systematic_bits = []
    parity_bits = []
    current_state = 0

    # Encode information bits
    for bit in information_bits:
        systematic_bits.append(int(bit))
        parity_bits.append(PARITY_TABLE[current_state, bit])
        current_state = NEXT_STATE_TABLE[current_state, bit]

    # Add tail bits
    tail_bits = rsc_tail_bits(current_state)
    for bit in tail_bits:
        systematic_bits.append(int(bit))
        parity_bits.append(PARITY_TABLE[current_state, bit])
        current_state = NEXT_STATE_TABLE[current_state, bit]

    return np.array(systematic_bits, dtype=np.int8), np.array(parity_bits, dtype=np.int8)


def turbo_encode(information_bits, interleaver):
    """
    Turbo encode using two parallel RSC encoders and an interleaver.
    Returns:
        systematic_1, parity_1, systematic_2, parity_2
    """
    information_bits = np.asarray(information_bits, dtype=np.int8)
    systematic_bits_1, parity_bits_1 = rsc_encode_terminated(information_bits)
    interleaved_bits = information_bits[interleaver]
    systematic_bits_2, parity_bits_2 = rsc_encode_terminated(interleaved_bits)
    return systematic_bits_1, parity_bits_1, systematic_bits_2, parity_bits_2
