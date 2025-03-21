import os

data_directory         = "./train/"
derived_data_directory = "./data/"
derived_test_data_directory = "./test_data/"

try:
    user = os.environ['USER']
except:
    user = "noUserFound"

tmp_mem_directory      = "/dev/shm/%s/"%(user)


plot_directory  = "./plots/"
model_directory = "./models/"
data_directory  = "./data/"
