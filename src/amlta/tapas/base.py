from amlta.config import config

tapas_ft_dir = config.data_dir / "tapas-ft"
tapas_ft_checkpoints_dir = tapas_ft_dir / "checkpoints"

id2aggregation = {0: "NONE", 1: "SUM", 2: "AVERAGE", 3: "COUNT"}
