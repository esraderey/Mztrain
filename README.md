# Mztrain
Train large models in 6-10 GB instead of 32 GB. SVD factors (U, S, V) are trained as first-class parameters, not compressed after the fact. Includes progressive rank growth, ElasticRank (bidirectional rank: grow/sleep/revive/prune), Loss-Guard rollback, INT8 optimizer states, gradient compression, and a GPU-validated VRAM governor.
