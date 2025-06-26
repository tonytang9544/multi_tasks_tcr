import platform

if platform.system() == "Linux":
    dataset_root = ""
elif platform.system() == "Darwin":
    dataset_root = "/Users/tangm/The Francis Crick Dropbox/Minzhe Tang/Tony/After-PhD/Machine_learning_MSc/UCL-AI_for_biomed/Course_material/thesis_project/TCR_project/dataset"
