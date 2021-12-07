import logging
import os

import time
from dataclasses import dataclass
from typing import Optional, List, Dict
from multiprocessing import Pool, Manager
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from tqdm import tqdm
from copy import deepcopy

from loggibud.v1.types import (
    CVRPInstance,
    CVRPSolution,
    CVRPSolutionVehicle,
    Delivery,
    JSONDataclassMixin,
)

from loggibud.v1.baselines.shared.ortools import (
    solve as ortools_solve,
    ORToolsParams,
)

import matplotlib.pyplot as plt
from project.utils import create_instanceCVRP, createUC, dictOffilinePA0, dictOffilineDF0, dictOffilineRJ0
from math import sqrt
from loggibud.v1.eval.task1 import evaluate_solution

logger = logging.getLogger(__name__)

@dataclass
class CHParams(JSONDataclassMixin):
    num_clusters: Optional[int] = None
    ortools_tsp_params: Optional[ORToolsParams] = ORToolsParams(
                max_vehicles=1,
                time_limit_ms=1_000,
            )
    num_ucs: Optional[int] = 28
    seed: int = 0
    @classmethod
    def get_baseline(cls):
        return cls(
            seed = 0,
            ortools_tsp_params=ORToolsParams(
                max_vehicles=1,
                time_limit_ms=1_000,
            ),
        )

@dataclass
class CHModel:
    params: CHParams
    clustering: KMeans


def pretrain(
    instances: List[CVRPInstance], params: Optional[CHParams] = None
) -> CHModel:
    params = params or CHParams.get_baseline()

    points = np.array(
        [
            [d.point.lng, d.point.lat]
            for instance in instances
            for d in instance.deliveries
        ]
    )

    num_clusters = params.num_clusters if params.num_clusters else metodoCotovelo(points)

    logger.info(f"Clustering instance into {num_clusters} subinstances")
    clustering = KMeans(num_clusters, random_state=params.seed)
    clustering.fit(points)

    return CHModel(
        params=params,
        clustering=clustering,
    )


def numero_cluster(error_rate):
    x1, y1 = 1, error_rate[0]
    x2, y2 = 28, error_rate[len(error_rate)-1]
    logger.info("Calculando as distancias referente ao error_rate")
    distances = []
    for i in range(len(error_rate)):
        x0 = i+2
        y0 = error_rate[i]
        numerator = abs((y2 - y1)*x0 - (x2 - x1)*y0 + x2*y1 - y2*x1)
        denominator = sqrt((y2 - y1)**2 + (x2 - x1)**2)
        distances.append(numerator/denominator)
    logger.info("Retornando o número de cluster.")
    return distances.index(max(distances)) + 2


def metodoCotovelo(points):
    logger.info("Calculando o error_rate")
    error_rate = []
    for i in range(2,29):
        kmeans = KMeans(n_clusters=i)
        kmeans.fit(points)
        error_rate.append(kmeans.inertia_)
    #name = "error_rate_df-0"
    #logger.info("Plotando o error_rate")

    #plt.plot(range(2, 29), error_rate, color="blue")
    #plt.savefig (name)
    return numero_cluster(error_rate)



def distributing(m: int, tamClusters: List[int], sum_p: int, ordenado: List[int]):
    arredondamento = {i: int(np.ceil(m * tamClusters[i]/sum_p)) for i in ordenado}
    soma = sum(filter(lambda elem:elem,(map(lambda dic:int(dic),arredondamento.values()))))
    xy = sorted(arredondamento, key=arredondamento.get)

    while soma != m:
        for i in xy:
            if arredondamento[i] > 1:
                arredondamento[i] = arredondamento[i] - 1
                break
        xx = sorted(arredondamento, key=arredondamento.get, reverse=True)
        arredondamento = {i: arredondamento[i] for i in xx}
        xy = sorted(arredondamento, key=arredondamento.get)
        soma = sum(filter(lambda elem:elem,(map(lambda dic:int(dic),arredondamento.values()))))
        print(arredondamento)

    print("distribuicao final",arredondamento)
    
    # transformando em list
    distribute = []
    for i in arredondamento:
        for j in range(arredondamento[i]):
            distribute.append(i)
    return distribute, arredondamento



def uc_distribute(m: int, tamClusters: List[int]):
    tam_pools = {i: tamClusters[i] for i in range(len(tamClusters))}
    ordenado = sorted(tam_pools, key = tam_pools.get, reverse=True)
    sum_clusters = sum([tamClusters[i] for i in range(len(tamClusters))])
    distribuicao, dictDistribuicao = distributing(m, tamClusters, sum_clusters, ordenado) # [1, 0, 2, 4]
    return distribuicao, dictDistribuicao


def modelsUC(instances: List[CVRPInstance], dictDistribuicao: Dict) -> List[KMeans]:
    models = []
    for cluster in range(len(instances)):
        points = []
        print(cluster)
        for delivery in instances[cluster].deliveries:
            points.append([delivery.point.lng, delivery.point.lat])
        modelUC = KMeans(n_clusters=dictDistribuicao[cluster], random_state=0).fit(points)
        models.append(modelUC)

    return models



def modelsUC2(deliveries: List[Delivery], dictDistribuicao: Dict, num_clusters: int, pointsClusters) -> List[KMeans]:
    models = [KMeans(n_clusters=dictDistribuicao[cluster], random_state=0) for cluster in range(num_clusters) ]
    for cluster in range(num_clusters):
        models[cluster].fit(pointsClusters[cluster])
    return models



def qtdClusters(deliveries, model):
    soma = [0 for i in range(model.clustering.n_clusters)]
    pointsClusters = [[] for cluster in range(model.clustering.n_clusters) ]

    for delivery in deliveries:
        point = [delivery.point.lng, delivery.point.lat]
        cluster = model.clustering.predict([point])[0]
        soma[cluster] += 1
        pointsClusters[cluster].append(point)

    return soma, pointsClusters



def aloc(instance: CVRPInstance, model: CHModel, 
params: CHParams, distribuicao: List[int], models: List[KMeans]) -> CVRPSolution:
    UCS = [createUC() for i in range(params.num_ucs)]
    R = [] # conjunto de entregas   
    Q = instance.vehicle_capacity # capacidade das ucs
    vehicles = []

    for delivery in instance.deliveries:
        #predict
        point = [delivery.point.lng, delivery.point.lat]
        #print(model)
        cluster = model.clustering.predict([point])[0]
        clusterUC = models[cluster].predict([point])[0]
       # print("d3")
        sub = [i for i in range(len(distribuicao)) if distribuicao[i] == cluster]
       # print("d4")
        for i in range(len(distribuicao)):
            if distribuicao[i] == cluster:
                clusterUC -= 1
                flag = i
            #    flag = distribuicao[i]
            # [0,0,0,0,0,0,1,1,1,1,2,2,2,3,3,3,3,4,4,4]

            if clusterUC < 0:
                j_min = flag
                break
        
      #  print(f"cluster: {cluster}", end= " ")
      #  for j in sub:
      #      print(f"uc {j}: {len(UCS[j].deliveries)}  || ", end= " ")
      #  print("\n")

        if UCS[j_min].C + delivery.size > Q:
            logger.info(f"Despachando Unidade de Carregamento {j_min}.")
            inst = create_instanceCVRP(instance, UCS[j_min].deliveries, instance.name, 3)
            R.append(inst)
            UCS[j_min].C = 0
            UCS[j_min].deliveries = []


        UCS[j_min].C = UCS[j_min].C + delivery.size
        UCS[j_min].deliveries.append(delivery)

    logger.info("Despachando Unidades de Carregamento que não chegaram ao limite.")
    for i in range(model.clustering.n_clusters): # ucs restantes que nao atingiram a capacidade Q
        sub = [j for j in range(len(distribuicao)) if distribuicao[j]==i]
        for j in sub:
            if len(UCS[j].deliveries) > 0:
                inst = create_instanceCVRP(instance, UCS[j].deliveries, instance.name, 3)
                R.append(inst)
                
    for inst in R:
        sol = ortools_solve(inst, params.ortools_tsp_params)# TSP
        while not isinstance(sol, CVRPSolution):
            logger.info(f"SOLUÇÃO NONETYPE. Buscando novamente. {instance.name}")
            sol = ortools_solve(inst, params.ortools_tsp_params)# TSP
        vehicles.append(CVRPSolutionVehicle(instance.origin, sol.deliveries))

    return CVRPSolution(
        name=instance.name,
        vehicles= vehicles,
    )



if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    parser = ArgumentParser()

    parser.add_argument("--train_instances", type=str, required=True)    
    parser.add_argument("--eval_instances", type=str, required=True)
    parser.add_argument("--output", type=str)
    parser.add_argument("--params", type=str)

    args = parser.parse_args()

    # Load instance and heuristic params.
    eval_path = Path(args.eval_instances)
    eval_path_dir = eval_path if eval_path.is_dir() else eval_path.parent
    eval_files = (
        [eval_path] if eval_path.is_file() else list(eval_path.iterdir())
    )

    train_path = Path(args.train_instances)
    train_path_dir = train_path if train_path.is_dir() else train_path.parent
    train_files = (
        [train_path] if train_path.is_file() else list(train_path.iterdir())
    )


    params = CHParams.from_file(args.params) if args.params else CHParams.get_baseline()
    print(params)
    train_instances = [CVRPInstance.from_file(f) for f in train_files[:240]]

    logger.info("Pretraining on training instances.")

    # primeiro nivel
    model = pretrain(train_instances, params)


    cidade = str(eval_path).split("/")
    cidade = cidade[len(cidade)-1]
    out = f"{args.output}/{cidade}/clusters_{model.clustering.n_clusters}" if args.output else None
   # outInstances = f"{args.output}/instances" if args.output else None
    
    output_dir = Path(out or ".")
    output_dir.mkdir(parents=True, exist_ok=True)

    # juntando todas as entregas de todas as instancias
    deliveries = np.array(
        [
            d
            for instance in train_instances
            for d in instance.deliveries
        ]
    )

    logger.info(f"Total de entregas: {len(deliveries)}")

    

    #instanciaGeral =  create_instanceCVRP(CVRPInstance.from_file(eval_files[0]), deliveries, "instanceGeral", 1)
    #instancesTest = [instanciaGeral]
    #logger.info("Separando as entregas em seus respectivos clusters")
    #points_clusters = [[] for i in range(model.clustering.n_clusters)]
    #for delivery in deliveries:
    #    cluster = model.clustering.predict([[delivery.point.lng, delivery.point.lat]])[0]
    #    points_clusters[cluster].append(delivery)

    #logger.info("Criando as instances_clusters")
    #instances = []
    #for cluster in range(len(points_clusters)):
     #   name = f"cluster_{cluster}"
    #    instanceCluster = create_instanceCVRP(train_instances[0], points_clusters[cluster], name, 1)
    #    instances.append(instanceCluster)
    #    print(len(instanceCluster.deliveries))
    #print(len(instances))

    somaClusters, pointsClusters = qtdClusters(deliveries, model)
    print(somaClusters)

    logger.info("Fazendo a distribuição das Unidades de Carregamento")
    #distribuicao, dictDistribuicao = uc_distribute(params.num_ucs, instances)
    #print()

    # distribuicao 
    distribuicao, dictDistribuicao = uc_distribute(params.num_ucs, somaClusters)
    # models = modelsUC(instances, dictDistribuicao)

    # segundo nivel
    models = modelsUC2(deliveries, dictDistribuicao, model.clustering.n_clusters, pointsClusters)
    
    #plotModels(models, deliveries, model)
    manager = Manager()
    results = manager.list()

    def solve(file):
        instance = CVRPInstance.from_file(file)
        logger.info(f"Alocando entregas: {instance.name}")
        solution = aloc(instance, model, params, distribuicao, models)
        distance = evaluate_solution(instance, solution)
        #solution.to_file((output_dir / f"{instance.name}.json"))
        print(distance)
        res = (instance.name, distance)
        results.append(res)


    inicio = time.time()

    # caso haja problema no tqdm
    for eval in eval_files:
        solve(eval)

    # Run solver on multiprocessing pool.
    #with Pool(os.cpu_count()) as pool:
    #    list(tqdm(pool.imap(solve, eval_files), total=len(eval_files)))

    final = time.time()



    # AVALIANDO 

    print(f"{eval_path}_{model.clustering.n_clusters}", "Clusterizacao Hierarquica")

    porcs = []
    for instance, distance in results:
        porc = (distance/dictOffilinePA0[instance])*100 - 100
        porcs.append(porc)
        print(f"{instance} ({distance} km)")
    soma = 0
    for p in porcs:
        soma += p
    
    print("media:",soma/len(porcs))

    print("tempo: ", final - inicio)
