import os
from src.logger import logging
from src.entity import DataTransformationConfig
from src.config.configuration import ConfigurationManager
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from src.exception import CustomException
from sklearn.preprocessing import OrdinalEncoder,StandardScaler
import os
from src.utils import save_object
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

class DataTransformation:
    def __init__(self,config: DataTransformationConfig):
        self.config = config

    def get_data_transformation_obj(self):
        date="Date"
        categorical_col=["Seasons","Holiday","Functioning Day"]
        numerical_col=["Rented Bike Count",]


    





