import streamlit as st
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os 
import plotly.express as px
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.linear_model import LinearRegression
import plotly as plt
import tensorflow as tf

from tensorflow import keras
import keras.layers
import keras.models
import statsmodels.api as sm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from  statsmodels.tsa.stattools import adfuller
from pmdarima.arima import auto_arima
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima_model import ARIMA


warnings.filterwarnings('ignore')


st.title('Fuel Price and Growth Analysis')
st.image("fossil.jpg")


    












