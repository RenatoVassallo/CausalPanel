import datetime
today = datetime.datetime.now().strftime("%Y%m%d")

# Parameters
# ================================================================================
frame_size   = [18, 18]  # how large of a window 'frame' to sample
buffer_size  = [18, 18]  # how many months to ensure that the control group doesn't contain treatment.
shift_amount = buffer_size[0] + 1  # number of months to shift the covariates.
extension    = 0  # how many months to add to the end of each conflict.
use_recall_tuned_params = True  # whether or not to use the recall-tuned hyperparameters
matching_params = {
                    "k": 5,
                    "d": 0.1,
                    "t": 10,
                }  # matching parameters for the nearest neighbors matching algorithm.
ends = [3, 6, 12, 24, 60]  # different columns to create (for example, the average of fatalities in the past 3 months). Each integer in the list is a different rolling mean length.

split_size = 37

# Directories
# ================================================================================
id_cols = ["isocode", "period"]
cleaned_input_path = "input/cleaned/"
raw_input_path = "input/raw/"
output_path = "output/"