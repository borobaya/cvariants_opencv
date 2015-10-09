IMAGE_SIZE = 500

BRISK_THRESHOLD = 30
BRISK_OCTAVES = 3
BRISK_SCALE = 1.0

N_CLUSTERS = 128

# 8-bit LSH components
RP_N_COMPONENTS = 2
# 4-bit LSH components (what is used in ES)
RP_N_NIBBLES = RP_N_COMPONENTS * 2
# number LSH component bits
RP_N_BITS = RP_N_COMPONENTS * 8

# RP has no model state beside the seed for the RNG. NO NOT CHANGE.
RP_RAND_STATE = 10
