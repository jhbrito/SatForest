from data import prepare_dataset_V1

dataset_path = "C:/Tesselo/data/tesselo-training-tiles"
clean_paths_file = "./clean_paths_V1.txt"
data_stats_file = "./data_stats_V1.txt"

trainSet, testSet, dataStats = prepare_dataset_V1(datasetPath=dataset_path,
                                                  cleanTilesFile=clean_paths_file,
                                                  dataStatsFile=data_stats_file)
