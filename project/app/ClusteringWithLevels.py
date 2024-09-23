from typing import Optional, List, Dict
from dataclasses import dataclass

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np

from app.types import JSONDataclassMixin
from app.shared.ortools import ORToolsParams


@dataclass
class Params(JSONDataclassMixin):
    num_clusters: Optional[int] = None
    ortools_tsp_params: Optional[ORToolsParams] = ORToolsParams(
                max_vehicles=1,
                time_limit_ms=1_000,
            )
    num_loadings_units: Optional[int] = 28
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
    
    @classmethod
    def create_dict(cls, p: Dict):
        return cls(
            num_clusters=p['num_clusters'],
            num_loadings_units=p['num_loadings_units'],
            seed = 0,
            ortools_tsp_params=ORToolsParams(
                max_vehicles=1,
                time_limit_ms=1_000,
            ),
        )

    

class ClusteringWithLevels:
    params: Params
    clustering: KMeans
    subclustering: List[KMeans]

    def __init__(self, points: np.array, params: Params) -> None:
        self.points=points
        self.params=params

        self.clustering=self.__create_cluster(self.params.num_clusters, self.params.seed)
        self.initial_predictions=self.clustering.fit_predict(self.points)

        self.distribution_loadings_units_dict=self.__distribution_loadings_units()
        self.distribution_loadings_units_list=[k for k, v in self.distribution_loadings_units_dict.items() for _ in range(v)]
        self.subclustering=self.__create_subclusterings()

        self.dict_centroides= self.__get__dict_centroides()
        """
        self.scaler=StandardScaler()
        self.dict_centroides= self.scaler.fit_transform(list(self.__get__dict_centroides().values()))
        self.dict_centroides= {index: value for index, value in enumerate(self.dict_centroides)}
        """


    
    def __create_cluster(self, num_clusters, seed) -> KMeans:
        """ Creating a KMeans clustering model. """

        return KMeans(num_clusters, init='k-means++', random_state=seed, n_init=10)
    
    def __create_subclusterings(self) -> List[KMeans]:
        """ Creating a list of clustering models. """

        return [ self.evaluate_cluster(cluster) for cluster in range(self.params.num_clusters) ]

    def __distribution_loadings_units(self) -> dict:
        """ 
        Distribution of loading units 
        based on the number of points in each cluster of the level one model.
        """

        unique, clusters_count = np.unique(self.initial_predictions, return_counts=True)
        deliveries_total= sum(clusters_count)
        clusters_count = {i: clusters_count[i] for i in range(len(clusters_count))}
        distribution = {i: 0 for i in range(len(clusters_count))}

        ordened_values = {i: clusters_count[i] for i in sorted(clusters_count, key = clusters_count.get, reverse=True)}
        total_distributed_LUs = 0
        for cluster, count in ordened_values.items():
            deliveries_percentage  = count / deliveries_total
            loadings_units_count = int(np.ceil(self.params.num_loadings_units * deliveries_percentage))
            distribution[cluster] = loadings_units_count
            total_distributed_LUs = total_distributed_LUs + loadings_units_count

        clusters_bigger_one = sorted(distribution, key = distribution.get, reverse=False)
        while total_distributed_LUs > self.params.num_loadings_units:
            for cluster in clusters_bigger_one:
                if distribution[cluster] > 1:
                    distribution[cluster] = distribution[cluster] - 1
                    total_distributed_LUs = total_distributed_LUs - 1
                    break
        return distribution
    
    def __get__dict_centroides(self) -> dict:
        centroides = {}
        key=0
        for cluster in self.subclustering:
            for subcluster in cluster.cluster_centers_:
                centroides[key] = subcluster
                key += 1   
        return centroides

    
    def evaluate_cluster(self, cluster) -> KMeans:
        """ Criação e treinamento de um modelo clusterizado do segundo nível. """
        
        cluster_points = [self.points[i] for i in range(len(self.initial_predictions)) if self.initial_predictions[i] == cluster]

        # Cria um modelo KMeans para o subagrupamento
        subclustering = self.__create_cluster(self.distribution_loadings_units_dict[cluster], self.params.seed)

        # Treina o modelo com os pontos do cluster
        subclustering.fit(cluster_points)

        return subclustering