import logging
import os
from multiprocessing import Pool, Manager
from pickle import LIST
from tqdm import tqdm

from argparse import ArgumentParser
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple

import numpy as np
from sklearn.cluster import KMeans

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

from loggibud.v1.eval.task1 import evaluate_solution

from project.utils import (
    dictOffilineDF0, 
    dictOffilinePA0,
)

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

@dataclass
class UCModel:
    C: int
    phi: List[Delivery]
    deliveries: List[Delivery]

    @classmethod
    def get_baseline(cls):
        return cls(
            C= 0,
            phi= [],
            deliveries= []
        )

@dataclass
class Params(JSONDataclassMixin):
    num_clusters: Optional[int] = None
    ortools_tsp_params: Optional[ORToolsParams] = ORToolsParams(
                max_vehicles=1,
                time_limit_ms=1_000,
            )
    NUM_UCS: Optional[int] = 28
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
class ParamsModel:
    params: Params
    clustering: KMeans
    subclustering: Optional[List[KMeans]] = None
    subinstance: Optional[CVRPInstance] = None
    list_distribute: Optional[List[int]] = None
    dict_distribute: Optional[Dict[int, int]] = None


# o balanceamento busca remover UC de um cluster que está mais longe do numero anterior.
# exemplo: C1: 8.07 UCS (ceil -> 9 UCS)  | C2: 10.96 (ceil -> 11 UCS)
#          C1: 9 - 8.07 = 0.93   |  C2: 11 - 10.96 = 0.04
# com isso, uma UC será removida do cluster C1 (irá para 8 UCS).
def get_distributing(
    NUM_UCS: int, y_pred: List[int], num_clusters: int 
) -> Tuple[list, Dict[int, int]]:

    total_amount_deliveries = len(y_pred)
    unique, counts = np.unique(y_pred, return_counts=True)
    no_rounding = {i: NUM_UCS * counts[i]/total_amount_deliveries for i in range(num_clusters)}
    rounding = {i: int(np.ceil(NUM_UCS * counts[i]/total_amount_deliveries)) 
                for i in range(num_clusters)}
    sum_distribute = sum(filter(lambda elem:elem,(map(lambda dic:int(dic),rounding.values()))))
    max = -1 ; removes = []

    while sum_distribute > NUM_UCS:
        for i in range(num_clusters):
            aux = rounding[i] - no_rounding[i]
            if aux > max and rounding[i] > 1 and i not in removes:
                max = aux
                remove_cluster = i
        removes.append(remove_cluster)
        if max == -1:
            removes = []
        else:
            rounding[remove_cluster] = rounding[remove_cluster] - 1
            no_rounding[remove_cluster] = no_rounding[remove_cluster] - 1
            sum_distribute = sum_distribute - 1 ; max = -1
    
    rounding = {i: rounding[i] for i in sorted(rounding, key=rounding.get, reverse=True)}
    distribute = [i for i in rounding for j in range(rounding[i])]

    return distribute, rounding


# o balanceamento busca remover UC do primeiro cluster que tem mais que uma UC.
# exemplo: C1: 10   |   C2: 3    |   C3:   1    |    C4: 2
# sera removido uma UC do cluster C4
def get_distributing2(
    NUM_UCS: int, y_pred: List[int]
):
    unique, counts = np.unique(y_pred, return_counts=True)
    tam_pools = {i: counts[i] for i in range(len(counts))}
    ordenado = sorted(tam_pools, key = tam_pools.get, reverse=True)
    sum_clusters = sum([counts[i] for i in range(len(counts))])
    
    arredondamento = {i: int(np.ceil(NUM_UCS * counts[i]/sum_clusters)) for i in ordenado}
    soma = sum(filter(lambda elem:elem,(map(lambda dic:int(dic),arredondamento.values()))))
    xy = sorted(arredondamento, key=arredondamento.get)

    while soma != NUM_UCS:
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




def pretrain(
    instances: List[CVRPInstance], params: Optional[Params] = None
) -> ParamsModel:
    params = params or Params.get_baseline()

    points = np.array(
        [
            [d.point.lng, d.point.lat]
            for instance in instances
            for d in instance.deliveries
        ]
    )

    num_clusters = params.num_clusters

    # criando modelo do primeiro nivel da clusterização
    logger.info(f"Clustering instance into {num_clusters} subinstances")
    clustering = KMeans(num_clusters, init='k-means++', random_state=params.seed)

    y_pred = clustering.fit_predict(points)

    list_distribute, dict_distribute = get_distributing(params.NUM_UCS, y_pred, num_clusters)
    # list_distribute, dict_distribute = get_distributing2(params.NUM_UCS, y_pred)

    # criando modelos do segundo nivel da clusterização
    subclusterings = [KMeans(dict_distribute[i], init='k-means++', random_state=params.seed)
                        .fit(points[np.in1d(y_pred, [0])]) for i in range(num_clusters) ]

    print(dict_distribute)
    print(subclusterings)
    return ParamsModel(
        params=params,
        list_distribute=list_distribute,
        dict_distribute=dict_distribute,
        clustering=clustering,
        subclustering=subclusterings,
    )

def instances_icvrp(
    instance: CVRPInstance, model: ParamsModel, UCS: List[UCModel], instances_cvrp: List[CVRPInstance]
) -> List[CVRPInstance]:
    # ucs restantes que nao atingiram a capacidade Q
    for i in range(model.clustering.n_clusters): 
        sub = [j for j in range(len(model.list_distribute)) if model.list_distribute[j]==i]
        for j in sub:
            if len(UCS[j].deliveries) > 0:
                inst = CVRPInstance(name = instance.name,
                                    region= "",
                                    origin= instance.origin,
                                    vehicle_capacity = 3 * instance.vehicle_capacity,
                                    deliveries= UCS[j].deliveries
                                )
                instances_cvrp.append(inst)
    return instances_cvrp


def solutions_icvrp(
    model: ParamsModel, instances_cvrp: List[CVRPInstance], solutions_cvrp: List[CVRPSolution]
) -> List[CVRPSolution] :
    for inst in instances_cvrp:
        sol = ortools_solve(inst, model.params.ortools_tsp_params)# TSP
        while not isinstance(sol, CVRPSolution):
            print(f"SOLUÇÃO NONETYPE. Buscando novamente. {inst.name}")
            sol = ortools_solve(inst, model.params.ortools_tsp_params)# TSP
        solutions_cvrp.append(CVRPSolutionVehicle(inst.origin, sol.deliveries))
    return solutions_cvrp


def alocation(instance: CVRPInstance, model: ParamsModel) -> CVRPSolution:
    UCS = [UCModel.get_baseline() for i in range(model.params.NUM_UCS)]
    instances_cvrp = [] ; solutions_cvrp = [] #  # conjunto de entregas e soluções

    for delivery in instance.deliveries:
        w = delivery.size
        cluster = model.clustering.predict([[delivery.point.lng, delivery.point.lat]])[0]
        subcluster = model.subclustering[cluster].predict([[delivery.point.lng, delivery.point.lat]])[0]

        j_min = model.list_distribute.index(cluster) + subcluster

        if UCS[j_min].C + w > instance.vehicle_capacity:
            logger.info(f"Despachando Unidades de Carregamento {j_min}.")
            instances_cvrp.append(CVRPInstance(name = instance.name,
                                    region= "",
                                    origin= instance.origin,
                                    vehicle_capacity = 3 * instance.vehicle_capacity,
                                    deliveries= UCS[j_min].deliveries))
            UCS[j_min].C = 0
            UCS[j_min].deliveries = []

        UCS[j_min].C = UCS[j_min].C + w
        UCS[j_min].deliveries.append(delivery)

    instances_cvrp = instances_icvrp(instance, model, UCS, instances_cvrp)
    solutions_cvrp = solutions_icvrp(model, instances_cvrp, solutions_cvrp )

    return CVRPSolution(
        name=instance.name,
        vehicles= solutions_cvrp,
    )


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    parser = ArgumentParser()

    parser.add_argument("--train_instances", type=str, required=True)    
    parser.add_argument("--eval_instances", type=str, required=True)
    # parser.add_argument("--output", type=str)
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

    params = Params.from_file(args.params) if args.params else Params.get_baseline()
    print(params)

    train_instances = [CVRPInstance.from_file(f) for f in train_files[:240]]

    logger.info("Pretraining on training instances.")
    model = pretrain(train_instances, params)

    manager = Manager()
    results = manager.list()

    def solve(file):
        instance = CVRPInstance.from_file(file)
        logger.info(f"Alocando entregas: {instance.name}")
        solution = alocation(instance, model)
        distance = evaluate_solution(instance, solution)
        #solution.to_file((output_dir / f"{instance.name}.json"))
        print(distance)
        res = (instance.name, distance)
        results.append(res)


    # Run solver on multiprocessing pool.
    with Pool(os.cpu_count()) as pool:
        list(tqdm(pool.imap(solve, eval_files), total=len(eval_files)))

    print(f"{eval_path}_{model.clustering.n_clusters}", "AS")

    porcs = []
    for instance, distance in results:
        porc = (distance/dictOffilinePA0[instance])*100 - 100
        porcs.append(porc)
        print(f"{instance} ({distance} km)")
    sum_distribute = 0
    for p in porcs:
        sum_distribute += p
    
    print("media:",sum_distribute/len(porcs))
    
