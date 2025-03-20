import os

def list_files_in_directory(directory_path):
    toyList = []
    for file in os.listdir(directory_path):
        if os.path.isfile(os.path.join(directory_path, file)):
            passedAll = True
            for veto in ["ttbar", "htautau", "diboson", "ztautau"]:
                if veto in file:
                    passedAll = False
            if passedAll:
                toyList.append(file)
    return toyList


toyPath = "/scratch-cbe/users/robert.schoefbeck/Higgs_uncertainty/data/lowMT_VBFJet/"
toys = list_files_in_directory(toyPath)

cmd_save =         'python runInference.py --config config_reference.yaml --save --logLevel DEBUG --modify CSI.save=False Toy_name="<TOYNAME>" Save.Toy.filename="<TOYNAME>.h5"'
cmd_save_nominal = 'python runInference.py --config config_reference.yaml --save --logLevel DEBUG --modify CSI.save=False'
saveSubmission = "toySubmission_save.sh"
with open(saveSubmission, "w") as file:
    for t in toys:
        toyname = t.replace(".h5", "")
        if "nominal" in toyname:
            file.write(cmd_save_nominal+"\n")
        else:
            file.write(cmd_save.replace("<TOYNAME>", toyname)+"\n")


muVals = [
    0.1,
    0.3,
    0.5,
    0.75,
    1.0,
    1.25,
    1.5,
    1.75,
    2.0,
    2.25,
    2.5,
    3.0,
]
cmd_fit =         'python runInference.py --config config_reference.yaml --predict --asimov_mu <MU> --postfix <TOYNAME> --modify Toy_name="<TOYNAME>" Save.Toy.filename="<TOYNAME>.h5"'
cmd_fit_nominal = 'python runInference.py --config config_reference.yaml --predict --asimov_mu <MU>'
fitSubmission = "toySubmission_fit.sh"
with open(fitSubmission, "w") as file:
    for t in toys:
        toyname = t.replace(".h5", "")
        for mu in muVals:
            if "nominal" in toyname:
                cmd = cmd_fit_nominal.replace("<MU>", "%.2f"%mu)
            else:
                cmd = cmd_fit.replace("<TOYNAME>", toyname).replace("<MU>", "%.2f"%mu)
            file.write(cmd+"\n")
