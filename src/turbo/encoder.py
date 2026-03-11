import numpy as np

MEMORY = 2
TAIL = 2
NSTATES = 4

def build_rsc_tables():
    next_state = np.zeros((NSTATES, 2), dtype=int)
    parity = np.zeros((NSTATES, 2), dtype=int)
    input_sign = np.zeros((NSTATES, 2), dtype=float)
    parity_sign = np.zeros((NSTATES, 2), dtype=float)
    prev_states = [[] for _ in range(NSTATES)]
    next_states = [[] for _ in range(NSTATES)]

    for state in range(NSTATES):
        s0 = (state >> 1) & 1
        s1 = state & 1
        for u in (0, 1):
            r = u ^ s1  # feedback 101
            p = r ^ s0 ^ s1  # feedforward 111
            ns0 = r
            ns1 = s0
            ns = (ns0 << 1) | ns1
            next_state[state, u] = ns
            parity[state, u] = p
            input_sign[state, u] = 1.0 if u == 0 else -1.0
            parity_sign[state, u] = 1.0 if p == 0 else -1.0
            prev_states[ns].append((state, u))
            next_states[state].append((ns, u))
    return next_state, parity, input_sign, parity_sign, prev_states, next_states

NEXT_STATE, PARITY, INPUT_SIGN, PARITY_SIGN, PREV_STATES, NEXT_STATES = build_rsc_tables()


def rsc_tail_bits(state):
    tail = []
    st = state
    for _ in range(TAIL):
        s1 = st & 1
        u = s1
        tail.append(u)
        st = NEXT_STATE[st, u]
    return tail


def rsc_encode_terminated(info_bits):
    info_bits = np.asarray(info_bits, dtype=np.int8)
    sys = []
    par = []
    state = 0
    for b in info_bits:
        sys.append(int(b))
        par.append(PARITY[state, b])
        state = NEXT_STATE[state, b]

    tails = rsc_tail_bits(state)
    for b in tails:
        sys.append(int(b))
        par.append(PARITY[state, b])
        state = NEXT_STATE[state, b]

    return np.array(sys, dtype=np.int8), np.array(par, dtype=np.int8)


def turbo_encode(info_bits, interleaver):
    info_bits = np.asarray(info_bits, dtype=np.int8)
    sys1, par1 = rsc_encode_terminated(info_bits)
    interleaved = info_bits[interleaver]
    sys2, par2 = rsc_encode_terminated(interleaved)
    return sys1, par1, sys2, par2


def conv_encode_75(bits):
    bits = np.asarray(bits, dtype=np.int8)
    bits = np.concatenate([bits, np.zeros(MEMORY, dtype=np.int8)])
    s0 = 0
    s1 = 0
    out = []
    for u in bits:
        c0 = u ^ s0 ^ s1  # 111
        c1 = u ^ s1       # 101
        out.extend((c0, c1))
        s1 = s0
        s0 = int(u)
    return np.array(out, dtype=np.int8)
