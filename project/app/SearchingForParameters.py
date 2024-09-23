import os
from tqdm import tqdm

from pathlib import Path
from multiprocessing import Pool, Manager
from sklearn.model_selection import train_test_split
import numpy as np

from datetime import datetime
import json


from loggibud.v1.eval.task1 import evaluate_solution

from app.types import CVRPInstance

from typing import Optional

from app.ClusteringWithLevels import (
    ClusteringWithLevels,
    Params,
)

from app.DeliveryAllocation import DeliveryAllocation

from project.utils import create_instanceCVRP

class SearchingForParameters:
    def __init__(
            self, 
            directory: str, 
            size_test: Optional[float] = 0.1, 
            count_results:Optional[int] = 10
        ) -> None:
        self.directory=directory

        self.path_train = 'data/cvrp-instances-1.0/train/' + self.directory
        self.path_dev = 'data/cvrp-instances-1.0/dev/' + self.directory
        self.path_informacoes = f"./app/parameters2/{self.directory}_params.json"

        self.clusters_numbers= range(4,15)
        self.lus_numbers=range(10,29)
        self.size_test=size_test
        self.count_results=count_results


    def save_params(self,  count_batchs:int):
        informacoes = {
            "directory": self.directory,
            "path": self.path_train,
            "best_result": self.results_parameters[0],
            f"results_top_{self.count_results}": self.results_parameters[:self.count_results],
            "path_info": self.path_informacoes,
            "timestamp": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
            "mean_deliveries_instance_test": self.mean_deliveries,
            "total_deliveries": len(self.deliveries),
            "sum_deliveries_test": self.total_deliveries_dev,
            "count_batchs": count_batchs
        }

        # Salve o dicionário em um arquivo JSON
        with open(self.path_informacoes, "w") as arquivo:
            json.dump(informacoes, arquivo, indent=4)

    def loading_instances(self, path: str) -> list:
        path = Path(path)
        files = (
            [path] if path.is_file() else list(path.iterdir())
        )
        return [CVRPInstance.from_file(f) for f in files[:240]]

    def delivery_mean_per_instance(self, instances: list) -> float:
        self.total_deliveries_dev = sum(len(instance.deliveries) for instance in instances)
        return self.total_deliveries_dev/len(instances) 

    def defining_batches(self, X_test: list):
        path_dev = 'data/cvrp-instances-1.0/dev/' + self.directory
        instances_dev = self.loading_instances(path_dev)
        self.mean_deliveries=self.delivery_mean_per_instance(instances_dev)
        batches = [create_instanceCVRP(instances_dev[0], X_test[i:i + int(self.mean_deliveries)], f"batch_{i}", 1) 
                  for i in range(0, len(X_test), int(self.mean_deliveries))]
        return batches

    def organizing_results(self):
        distances = {(c, LU): 0 for c in self.clusters_numbers for LU in self.lus_numbers}
        for c, LU, distance in self.distances_list:
            distances[c, LU] += distance
        ordering = sorted(distances.items(), key=lambda item: item[1])
        for value in ordering:
            if value[1]>0:
                self.results_parameters.append({"num_clusters": value[0][0], "num_loadings_units": value[0][1], "distance": value[1]})
        
    def solve(self, instance):
        for c in self.clusters_numbers:
            for LU in self.lus_numbers:
                if LU >= c: 
                    params={"num_clusters": c, "num_loadings_units": LU}
                    params = Params.create_dict(params) if params else Params.get_baseline()
                    print(params)
                    model=ClusteringWithLevels(self.points, params)
                    solution=DeliveryAllocation(model, instance).application()
                    distance = evaluate_solution(instance, solution)
                    self.distances_list.append([c,LU, distance])

    def application(self):

        instances_train = self.loading_instances(self.path_train)
        self.deliveries= [d for instance in instances_train for d in instance.deliveries]
        X_train, X_test = train_test_split(self.deliveries, test_size=self.size_test, random_state=1609)
        self.points = np.array(
                [
                    [d.point.lng, d.point.lat]
                    for d in X_train
                ]
            )
        batches=self.defining_batches(X_test)

        manager = Manager()
        self.distances_list=manager.list()
        self.results_parameters=[]

        # Run solver on multiprocessing pool.
        with Pool(os.cpu_count()) as pool:
            list(tqdm(pool.imap(self.solve, batches), total=len(batches)))


        self.organizing_results()
        self.save_params(len(batches))
        return self.results_parameters[0]
        


if __name__ == "__main__":
    directories = ['pa-0']
    for dir in directories:
        params = SearchingForParameters(dir).application()
