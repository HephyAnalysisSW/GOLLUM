import os

try:
    user = os.environ['USER']
except:
    user = "noUserFound"

tmp_mem_directory      = "/dev/shm/%s/"%(user)

plot_directory  = "./plots/"
model_directory = "./models/"
data_directory  = "./data/"
