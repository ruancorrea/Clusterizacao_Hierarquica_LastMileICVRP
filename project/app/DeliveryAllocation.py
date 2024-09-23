from typing import List
import random 
import math
import numpy as np

from app.types import (
    LoadingUnitModel,
    CVRPInstance,
    CVRPSolution,
    Delivery,
    CVRPSolutionVehicle,
    Point
)

from app.ClusteringWithLevels import ClusteringWithLevels
from app.shared.ortools import (
    solve as ortools_solve,
)

class DeliveryAllocation:
    def __init__(self, model: ClusteringWithLevels, instance: CVRPInstance) -> None:
        self.model=model
        self.instance=instance
        self.dispatches=[]
        self.loadings_units = self.__create_loadings_units()

    def __create_loadings_units(self) -> List[LoadingUnitModel]:
        """ Starting loading units emptied. """

        return [LoadingUnitModel.get_baseline() for i in range(self.model.params.num_loadings_units)]
    
    def __create_instanceCVRP(self, deliveries: List[Delivery]) -> CVRPInstance:
        """ Creating an instanceCVRP with deliveries. """

        return CVRPInstance(
                    name = self.instance.name,
                    region= "",
                    origin= self.instance.origin,
                    vehicle_capacity = self.instance.vehicle_capacity,
                    deliveries= deliveries
                )
    
    def solve(self, dispatch: CVRPInstance) -> CVRPSolutionVehicle:
        """ Generate route from a tsp solution. """

        solution=None
        while True:
            solution=ortools_solve(dispatch, self.model.params.ortools_tsp_params)# TSP
            if isinstance(solution, CVRPSolution):
                break
            else:
                random.shuffle(dispatch.deliveries)
        return CVRPSolutionVehicle(self.instance.origin, solution.deliveries)
    
    def euclidean_distance(self, point1, point2):
        #return round(math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2), 8)
        return np.sqrt(np.sum((point1 - point2) ** 2))
    
    def manhattan_distance(self, point1, point2):
        return sum(abs(a - b) for a, b in zip(point1, point2))
    
    def choice_of_loading_unit_v2(self, point: Point) -> int:
        """ 
        Application of clustered models to find the delivery loading unit. 
        1. Prediction in the level one model to choose the level two model
        2. Prediction in the chosen model from level two to choose the loading unit
        """

        #p = self.model.scaler.transform([[point.lng, point.lat]])[0]
        p = [point.lng, point.lat]

        loading_unit = min(
            self.model.dict_centroides, 
            key=lambda k: self.euclidean_distance(self.model.dict_centroides[k], p))

        return loading_unit
        
    def choice_of_loading_unit(self, point: Point) -> int:
        """ 
        Application of clustered models to find the delivery loading unit. 
        1. Prediction in the level one model to choose the level two model
        2. Prediction in the chosen model from level two to choose the loading unit
        """

        cluster = self.model.clustering.predict([[point.lng, point.lat]])[0]
        subcluster = self.model.subclustering[cluster].predict([[point.lng, point.lat]])[0]
        loading_unit = self.model.distribution_loadings_units_list.index(cluster) + subcluster
        return loading_unit

    def delivery_evaluate(self, delivery: Delivery):
        """ 
        Analyze the weight of the delivery:
        If adding the delivery exceeds the capacity of the loading unit
        then the previous LU deliveries will be dispatched and it will be emptied. 
        In the end, delivery will be added to LU.
        """

        index = self.choice_of_loading_unit(delivery.point)
        if self.loadings_units[index].capacity + delivery.size > self.instance.vehicle_capacity:
            self.dispatches.append(self.loadings_units[index].deliveries)
            self.loadings_units[index]=LoadingUnitModel.get_baseline()
        self.loadings_units[index].capacity += delivery.size
        self.loadings_units[index].deliveries.append(delivery)

    def application(self):
        """ 
            1. Definition of the loading unit for each delivery
            1.1. Dispatch of loading units with maximum capacity reached
            2. Dispatch of loading units that have not reached maximum capacity
            3. Organization of routes for each dispatch
        """

        for delivery in self.instance.deliveries:
            self.delivery_evaluate(delivery) 


        for loading_unit in self.loadings_units:
            if len(loading_unit.deliveries)>0:
                self.dispatches.append(loading_unit.deliveries)

        return CVRPSolution(
            name=self.instance.name,
            vehicles=[ self.solve(self.__create_instanceCVRP(dispatch)) for dispatch in self.dispatches ],
        )
