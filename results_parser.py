import includes.dataset_helper as dsf
import json
import numpy
import pprint
import os

def merge_arrays(array, reduce_mean=True):
    output = {}
    for item in array:
        for param in item:
            if param not in output:
                output[param] = []
            output[param].append(item[param])
    if reduce_mean:
        for param in output:
            output[param] = numpy.mean(output[param])
    return output

tag={'ae':'bardotAECluster','cnn':'bardotCnnCluster', 'kmeans':'kmeansCluster'}


inputRuns = {
        'ae':
        {
            'maxpool': ['1523627556','1523589861','1523614986','1523564565','1523602422'],#['1522935015','1523163701','1523213649','1522956840','1523189577'],
            'maxpool_argmax': ['1522907085','1522990806','1522969504','1523146707','1523059144'],
            'stride': ['1522920310','1523207005','1523034329','1523085735','1523004090'],
            'maxpool_argmax_rotation': ['1522885275','1523010774','1523072387','1523040999','1523176257']
        },
        'cnn':
            {
            'maxpool': ['1522926995','1523100868','1522982781','1523117321','1523133853'],
            'stride': ['1522902318','1523112590','1523028776','1523096133','1523024006'],
            'stride_rotation': ['1522880447','1523202167','1523141886','1523054274','1523129032'],
            'stride_600_200': ['1522898538','1523092362','1523159958','1523125319','1523108876']
        },
        'kmeans':
            {
            'main': ['1523660289','1523666332','1523663307','1523657280','1523654278']#['1523411363','1523408328','1523405297','1523402270','1523399267']
        }
}
datasetNames = ['bardot_5', 'bardot_5_aug_5']

def obtainClustQuality(method,rangeClust=[3,8]):
    results = {}
    for typeName in inputRuns[method]:
        #typeName = 'stride_600_200'
        results[typeName] = {}
        for datasetName in datasetNames:
            if datasetName not in results[typeName]:
                results[typeName][datasetName] = {}
            for clusterNum in range(rangeClust[0],rangeClust[1]):
                if clusterNum not in results[typeName][datasetName]:
                    results[typeName][datasetName][clusterNum] = []
                for runNum in inputRuns[method][typeName]:
                    result = json.load(open("result/{}/{}/benchmark-kmeans-5-{}_{}.json".format(tag[method], runNum, clusterNum, datasetName)))
                    results[typeName][datasetName][clusterNum].append(result)
                results[typeName][datasetName][clusterNum] = merge_arrays(results[typeName][datasetName][clusterNum], reduce_mean=True)
        #break

    pprint.pprint(results)

def obtainClustNumAssess(method):
    results = {}
    clustTypes = ['dbscan', 'hdbscan']

    for typeName in inputRuns[method]:
        #typeName='stride_600_200'
        results[typeName] = {}
        for runNum in inputRuns[method][typeName]:
            for datasetName in datasetNames:
                if datasetName not in results[typeName]:
                    results[typeName][datasetName] = {}
                for clustType in clustTypes:
                    if clustType not in results[typeName][datasetName]:
                        results[typeName][datasetName][clustType] = []
                    #print("benchmark-{}-5-{}_bardot_5_aug_5.json".format(clustType, 1111, datasetName))
                    found = False
                    for i in range(1,15):
                        if os.path.isfile("result/{}/{}/benchmark-{}-5-{}_{}.json".format(tag[method], runNum, clustType, i, datasetName)):
                            print("result/{}/{}/benchmark-{}-5-{}_{}.json".format(tag[method], runNum, clustType, i, datasetName))
                            results[typeName][datasetName][clustType].append(i)
                            found=True
                            break
                    if not found:
                        raise Exception("result/{}/{}/benchmark-{}-5-X_bardot_5_aug_5.json".format(tag[method], runNum, clustType, datasetName))
        for dataset in results[typeName]:
            for clustAlg in results[typeName][dataset]:
                results[typeName][dataset][clustAlg] = numpy.mean(results[typeName][dataset][clustAlg])
        #break

    pprint.pprint(results)

#obtainClustNumAssess('kmeans')
obtainClustQuality('ae',rangeClust=[5,6])
#autoencoder maxpool silhouette:
#a
#0.3268731415271759 0.3352903246879578 0.3133940637111664 0.30055888891220095 0.2891226291656494
#b
#0.35952613353729246 0.39282880425453187 0.4373749911785126 0.4374109089374542 0.41568037271499636

#cnn stride 600-200 silhouette
#0.5265955805778504 0.6854930520057678 0.8477516531944275 0.7870886445045471 0.7183929920196533
#0.5501237034797668 0.5470641851425171 0.5662600040435791 0.5636135816574097 0.5705798506736756

#kmeans
#0.5528432726860046 0.5835192441940308 0.5791624546051025 0.5621180653572082 0.5587791800498962
#0.6686994433403015 0.6359883546829224 0.572857940196991 0.5386942505836487 0.5340792298316955