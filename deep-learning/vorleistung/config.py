#### Mandatory configuration ####

PROJECT_ROOT = "/Users/luca/Projects/ms-data-science/deep-learning/vorleistung"
DATASET_PATH = "/Users/luca/Projects/ms-data-science/deep-learning/vorleistung/data" # this path should contain the EuroSAT_MS directory


#### Optional configuration ####

# Model used by reproduction script
# Final models can be downloaded at: https://download-directory.github.io/?url=https://github.com/luca-heitmann/data-science-courses/tree/main/deep-learning/vorleistung/final_results
REPRODUCTION_MODEL_PATH = PROJECT_ROOT + "/final_results/2025-12-20_20-26-13-task2-results/model.pkl"

# Parameters for training
SEED = 3778660
TRAIN_SIZE = 2500
TEST_SIZE = 2000
VAL_SIZE = 1000
