{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## **Problem Statement**\n",
    "# Predicting Turbine Energy Yield (TEY) using ambient variables as features."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 103,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Importing Libraries\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import plotly.graph_objects as go\n",
    "from plotly.subplots import make_subplots\n",
    "import seaborn as sns\n",
    "from collections import Counter\n",
    "sns.set_style('darkgrid')\n",
    "from imblearn.pipeline import Pipeline\n",
    "from dataprep.eda import plot, plot_correlation, plot_missing, create_report\n",
    "import plotly.express as px\n",
    "from plotly.offline import plot as off\n",
    "import plotly.figure_factory as ff\n",
    "import plotly.io as pio\n",
    "from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler, scale\n",
    "from sklearn.ensemble import RandomForestRegressor\n",
    "from sklearn.metrics import mean_squared_error\n",
    "from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, cross_val_score, \\\n",
    "                                    train_test_split, RandomizedSearchCV\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')\n",
    "from keras.optimizers import Adam\n",
    "from kerastuner.tuners import RandomSearch\n",
    "from tensorflow.keras import layers\n",
    "from tensorflow import keras\n",
    "from keras.models import Sequential\n",
    "from keras.layers import Dense\n",
    "from keras.layers import Dropout\n",
    "from keras.utils import np_utils\n",
    "from keras.constraints import maxnorm\n",
    "from keras.wrappers.scikit_learn import KerasRegressor, KerasClassifier\n",
    "from sklearn.feature_selection import SelectKBest\n",
    "from sklearn.feature_selection import mutual_info_regression"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>AT</th>\n",
       "      <th>AP</th>\n",
       "      <th>AH</th>\n",
       "      <th>AFDP</th>\n",
       "      <th>GTEP</th>\n",
       "      <th>TIT</th>\n",
       "      <th>TAT</th>\n",
       "      <th>TEY</th>\n",
       "      <th>CDP</th>\n",
       "      <th>CO</th>\n",
       "      <th>NOX</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>6.8594</td>\n",
       "      <td>1007.9</td>\n",
       "      <td>96.799</td>\n",
       "      <td>3.5000</td>\n",
       "      <td>19.663</td>\n",
       "      <td>1059.2</td>\n",
       "      <td>550.00</td>\n",
       "      <td>114.70</td>\n",
       "      <td>10.605</td>\n",
       "      <td>3.1547</td>\n",
       "      <td>82.722</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>6.7850</td>\n",
       "      <td>1008.4</td>\n",
       "      <td>97.118</td>\n",
       "      <td>3.4998</td>\n",
       "      <td>19.728</td>\n",
       "      <td>1059.3</td>\n",
       "      <td>550.00</td>\n",
       "      <td>114.72</td>\n",
       "      <td>10.598</td>\n",
       "      <td>3.2363</td>\n",
       "      <td>82.776</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>6.8977</td>\n",
       "      <td>1008.8</td>\n",
       "      <td>95.939</td>\n",
       "      <td>3.4824</td>\n",
       "      <td>19.779</td>\n",
       "      <td>1059.4</td>\n",
       "      <td>549.87</td>\n",
       "      <td>114.71</td>\n",
       "      <td>10.601</td>\n",
       "      <td>3.2012</td>\n",
       "      <td>82.468</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>7.0569</td>\n",
       "      <td>1009.2</td>\n",
       "      <td>95.249</td>\n",
       "      <td>3.4805</td>\n",
       "      <td>19.792</td>\n",
       "      <td>1059.6</td>\n",
       "      <td>549.99</td>\n",
       "      <td>114.72</td>\n",
       "      <td>10.606</td>\n",
       "      <td>3.1923</td>\n",
       "      <td>82.670</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>7.3978</td>\n",
       "      <td>1009.7</td>\n",
       "      <td>95.150</td>\n",
       "      <td>3.4976</td>\n",
       "      <td>19.765</td>\n",
       "      <td>1059.7</td>\n",
       "      <td>549.98</td>\n",
       "      <td>114.72</td>\n",
       "      <td>10.612</td>\n",
       "      <td>3.2484</td>\n",
       "      <td>82.311</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>15034</th>\n",
       "      <td>9.0301</td>\n",
       "      <td>1005.6</td>\n",
       "      <td>98.460</td>\n",
       "      <td>3.5421</td>\n",
       "      <td>19.164</td>\n",
       "      <td>1049.7</td>\n",
       "      <td>546.21</td>\n",
       "      <td>111.61</td>\n",
       "      <td>10.400</td>\n",
       "      <td>4.5186</td>\n",
       "      <td>79.559</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>15035</th>\n",
       "      <td>7.8879</td>\n",
       "      <td>1005.9</td>\n",
       "      <td>99.093</td>\n",
       "      <td>3.5059</td>\n",
       "      <td>19.414</td>\n",
       "      <td>1046.3</td>\n",
       "      <td>543.22</td>\n",
       "      <td>111.78</td>\n",
       "      <td>10.433</td>\n",
       "      <td>4.8470</td>\n",
       "      <td>79.917</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>15036</th>\n",
       "      <td>7.2647</td>\n",
       "      <td>1006.3</td>\n",
       "      <td>99.496</td>\n",
       "      <td>3.4770</td>\n",
       "      <td>19.530</td>\n",
       "      <td>1037.7</td>\n",
       "      <td>537.32</td>\n",
       "      <td>110.19</td>\n",
       "      <td>10.483</td>\n",
       "      <td>7.9632</td>\n",
       "      <td>90.912</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>15037</th>\n",
       "      <td>7.0060</td>\n",
       "      <td>1006.8</td>\n",
       "      <td>99.008</td>\n",
       "      <td>3.4486</td>\n",
       "      <td>19.377</td>\n",
       "      <td>1043.2</td>\n",
       "      <td>541.24</td>\n",
       "      <td>110.74</td>\n",
       "      <td>10.533</td>\n",
       "      <td>6.2494</td>\n",
       "      <td>93.227</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>15038</th>\n",
       "      <td>6.9279</td>\n",
       "      <td>1007.2</td>\n",
       "      <td>97.533</td>\n",
       "      <td>3.4275</td>\n",
       "      <td>19.306</td>\n",
       "      <td>1049.9</td>\n",
       "      <td>545.85</td>\n",
       "      <td>111.58</td>\n",
       "      <td>10.583</td>\n",
       "      <td>4.9816</td>\n",
       "      <td>92.498</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>15039 rows × 11 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "           AT      AP      AH    AFDP    GTEP     TIT     TAT     TEY     CDP  \\\n",
       "0      6.8594  1007.9  96.799  3.5000  19.663  1059.2  550.00  114.70  10.605   \n",
       "1      6.7850  1008.4  97.118  3.4998  19.728  1059.3  550.00  114.72  10.598   \n",
       "2      6.8977  1008.8  95.939  3.4824  19.779  1059.4  549.87  114.71  10.601   \n",
       "3      7.0569  1009.2  95.249  3.4805  19.792  1059.6  549.99  114.72  10.606   \n",
       "4      7.3978  1009.7  95.150  3.4976  19.765  1059.7  549.98  114.72  10.612   \n",
       "...       ...     ...     ...     ...     ...     ...     ...     ...     ...   \n",
       "15034  9.0301  1005.6  98.460  3.5421  19.164  1049.7  546.21  111.61  10.400   \n",
       "15035  7.8879  1005.9  99.093  3.5059  19.414  1046.3  543.22  111.78  10.433   \n",
       "15036  7.2647  1006.3  99.496  3.4770  19.530  1037.7  537.32  110.19  10.483   \n",
       "15037  7.0060  1006.8  99.008  3.4486  19.377  1043.2  541.24  110.74  10.533   \n",
       "15038  6.9279  1007.2  97.533  3.4275  19.306  1049.9  545.85  111.58  10.583   \n",
       "\n",
       "           CO     NOX  \n",
       "0      3.1547  82.722  \n",
       "1      3.2363  82.776  \n",
       "2      3.2012  82.468  \n",
       "3      3.1923  82.670  \n",
       "4      3.2484  82.311  \n",
       "...       ...     ...  \n",
       "15034  4.5186  79.559  \n",
       "15035  4.8470  79.917  \n",
       "15036  7.9632  90.912  \n",
       "15037  6.2494  93.227  \n",
       "15038  4.9816  92.498  \n",
       "\n",
       "[15039 rows x 11 columns]"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Import Dataset\n",
    "df = pd.read_csv('gas_turbines.csv')\n",
    "df"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. About Dataset<a class=\"anchor\" id=\"2\"></a>\n",
    "\n",
    "The dataset contains 36733 instances of 11 sensor measures aggregated over one hour (by means of average or sum) from a gas turbine. \n",
    "The Dataset includes gas turbine parameters (such as Turbine Inlet Temperature and Compressor Discharge pressure) in addition to the ambient variables.\n",
    "\n",
    "Attribute Information:\n",
    "\n",
    "The explanations of sensor measurements and their brief statistics are given below."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "|Variable|(Abbrivation)|Unit|Min|Max|Mean|\n",
    "|:------|:------:|:------|:------|:------|:------|\n",
    "|Ambient temperature |(AT)| C| 6.23| 37.10| 17.71|\n",
    "|Ambient pressure |(AP)| mbar |985.85 |1036.56 |1013.07|\n",
    "|Ambient humidity |(AH)| (%) |24.08 |100.20 |77.87|\n",
    "|Air filter difference pressure |(AFDP)| mbar |2.09 |7.61 |3.93|\n",
    "|Gas turbine exhaust pressure |(GTEP)| mbar |17.70 |40.72 |25.56|\n",
    "|Turbine inlet temperature |(TIT)| C |1000.85 |1100.89 |1081.43|\n",
    "|Turbine after temperature |(TAT)| C |511.04 |550.61 |546.16|\n",
    "|Compressor discharge pressure |(CDP)| mbar |9.85 |15.16 |12.06|\n",
    "|Turbine energy yield |(TEY)| MWH |100.02 |179.50 |133.51|\n",
    "|Carbon monoxide |(CO)| mg/m3 |0.00 |44.10 |2.37|\n",
    "|Nitrogen oxides |(NOx)| mg/m3 |25.90 |119.91 |65.29|"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": []
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Data Exploration <a class=\"anchor\" id=\"3\"></a>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Before doing anything else with the data let's see if there are any null values (missing data) in any of the columns."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "AT      False\n",
       "AP      False\n",
       "AH      False\n",
       "AFDP    False\n",
       "GTEP    False\n",
       "TIT     False\n",
       "TAT     False\n",
       "TEY     False\n",
       "CDP     False\n",
       "CO      False\n",
       "NOX     False\n",
       "dtype: bool"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.isnull().any()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "AT      0\n",
       "AP      0\n",
       "AH      0\n",
       "AFDP    0\n",
       "GTEP    0\n",
       "TIT     0\n",
       "TAT     0\n",
       "TEY     0\n",
       "CDP     0\n",
       "CO      0\n",
       "NOX     0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.isnull().sum()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- We have no missing data so all the entries are valid for use.\n",
    "\n",
    "- Now we can check the column names to get a better understanding of what features we will be basing our regression on."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 3.1 Descriptive Analysis<a class=\"anchor\" id=\"3.1\"></a>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(15039, 11)"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "AT      float64\n",
       "AP      float64\n",
       "AH      float64\n",
       "AFDP    float64\n",
       "GTEP    float64\n",
       "TIT     float64\n",
       "TAT     float64\n",
       "TEY     float64\n",
       "CDP     float64\n",
       "CO      float64\n",
       "NOX     float64\n",
       "dtype: object"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Checking the data types\n",
    "df.dtypes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "AT      12086\n",
       "AP        540\n",
       "AH      12637\n",
       "AFDP    11314\n",
       "GTEP     8234\n",
       "TIT       706\n",
       "TAT      2340\n",
       "TEY      4207\n",
       "CDP      3611\n",
       "CO      13096\n",
       "NOX     11996\n",
       "dtype: int64"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Unique values for every feature\n",
    "df.nunique()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 15039 entries, 0 to 15038\n",
      "Data columns (total 11 columns):\n",
      " #   Column  Non-Null Count  Dtype  \n",
      "---  ------  --------------  -----  \n",
      " 0   AT      15039 non-null  float64\n",
      " 1   AP      15039 non-null  float64\n",
      " 2   AH      15039 non-null  float64\n",
      " 3   AFDP    15039 non-null  float64\n",
      " 4   GTEP    15039 non-null  float64\n",
      " 5   TIT     15039 non-null  float64\n",
      " 6   TAT     15039 non-null  float64\n",
      " 7   TEY     15039 non-null  float64\n",
      " 8   CDP     15039 non-null  float64\n",
      " 9   CO      15039 non-null  float64\n",
      " 10  NOX     15039 non-null  float64\n",
      "dtypes: float64(11)\n",
      "memory usage: 1.3 MB\n"
     ]
    }
   ],
   "source": [
    "df.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(0, 11)"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df[df.duplicated()].shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>AT</th>\n",
       "      <th>AP</th>\n",
       "      <th>AH</th>\n",
       "      <th>AFDP</th>\n",
       "      <th>GTEP</th>\n",
       "      <th>TIT</th>\n",
       "      <th>TAT</th>\n",
       "      <th>TEY</th>\n",
       "      <th>CDP</th>\n",
       "      <th>CO</th>\n",
       "      <th>NOX</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "Empty DataFrame\n",
       "Columns: [AT, AP, AH, AFDP, GTEP, TIT, TAT, TEY, CDP, CO, NOX]\n",
       "Index: []"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df[df.duplicated()]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>count</th>\n",
       "      <th>mean</th>\n",
       "      <th>std</th>\n",
       "      <th>min</th>\n",
       "      <th>25%</th>\n",
       "      <th>50%</th>\n",
       "      <th>75%</th>\n",
       "      <th>max</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>AT</th>\n",
       "      <td>15039.0</td>\n",
       "      <td>17.764381</td>\n",
       "      <td>7.574323</td>\n",
       "      <td>0.522300</td>\n",
       "      <td>11.408000</td>\n",
       "      <td>18.1860</td>\n",
       "      <td>23.8625</td>\n",
       "      <td>34.9290</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>AP</th>\n",
       "      <td>15039.0</td>\n",
       "      <td>1013.199240</td>\n",
       "      <td>6.410760</td>\n",
       "      <td>985.850000</td>\n",
       "      <td>1008.900000</td>\n",
       "      <td>1012.8000</td>\n",
       "      <td>1016.9000</td>\n",
       "      <td>1034.2000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>AH</th>\n",
       "      <td>15039.0</td>\n",
       "      <td>79.124174</td>\n",
       "      <td>13.793439</td>\n",
       "      <td>30.344000</td>\n",
       "      <td>69.750000</td>\n",
       "      <td>82.2660</td>\n",
       "      <td>90.0435</td>\n",
       "      <td>100.2000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>AFDP</th>\n",
       "      <td>15039.0</td>\n",
       "      <td>4.200294</td>\n",
       "      <td>0.760197</td>\n",
       "      <td>2.087400</td>\n",
       "      <td>3.723900</td>\n",
       "      <td>4.1862</td>\n",
       "      <td>4.5509</td>\n",
       "      <td>7.6106</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>GTEP</th>\n",
       "      <td>15039.0</td>\n",
       "      <td>25.419061</td>\n",
       "      <td>4.173916</td>\n",
       "      <td>17.878000</td>\n",
       "      <td>23.294000</td>\n",
       "      <td>25.0820</td>\n",
       "      <td>27.1840</td>\n",
       "      <td>37.4020</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>TIT</th>\n",
       "      <td>15039.0</td>\n",
       "      <td>1083.798770</td>\n",
       "      <td>16.527806</td>\n",
       "      <td>1000.800000</td>\n",
       "      <td>1079.600000</td>\n",
       "      <td>1088.7000</td>\n",
       "      <td>1096.0000</td>\n",
       "      <td>1100.8000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>TAT</th>\n",
       "      <td>15039.0</td>\n",
       "      <td>545.396183</td>\n",
       "      <td>7.866803</td>\n",
       "      <td>512.450000</td>\n",
       "      <td>542.170000</td>\n",
       "      <td>549.8900</td>\n",
       "      <td>550.0600</td>\n",
       "      <td>550.6100</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>TEY</th>\n",
       "      <td>15039.0</td>\n",
       "      <td>134.188464</td>\n",
       "      <td>15.829717</td>\n",
       "      <td>100.170000</td>\n",
       "      <td>127.985000</td>\n",
       "      <td>133.7800</td>\n",
       "      <td>140.8950</td>\n",
       "      <td>174.6100</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>CDP</th>\n",
       "      <td>15039.0</td>\n",
       "      <td>12.102353</td>\n",
       "      <td>1.103196</td>\n",
       "      <td>9.904400</td>\n",
       "      <td>11.622000</td>\n",
       "      <td>12.0250</td>\n",
       "      <td>12.5780</td>\n",
       "      <td>15.0810</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>CO</th>\n",
       "      <td>15039.0</td>\n",
       "      <td>1.972499</td>\n",
       "      <td>2.222206</td>\n",
       "      <td>0.000388</td>\n",
       "      <td>0.858055</td>\n",
       "      <td>1.3902</td>\n",
       "      <td>2.1604</td>\n",
       "      <td>44.1030</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>NOX</th>\n",
       "      <td>15039.0</td>\n",
       "      <td>68.190934</td>\n",
       "      <td>10.470586</td>\n",
       "      <td>27.765000</td>\n",
       "      <td>61.303500</td>\n",
       "      <td>66.6010</td>\n",
       "      <td>73.9355</td>\n",
       "      <td>119.8900</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "        count         mean        std          min          25%        50%  \\\n",
       "AT    15039.0    17.764381   7.574323     0.522300    11.408000    18.1860   \n",
       "AP    15039.0  1013.199240   6.410760   985.850000  1008.900000  1012.8000   \n",
       "AH    15039.0    79.124174  13.793439    30.344000    69.750000    82.2660   \n",
       "AFDP  15039.0     4.200294   0.760197     2.087400     3.723900     4.1862   \n",
       "GTEP  15039.0    25.419061   4.173916    17.878000    23.294000    25.0820   \n",
       "TIT   15039.0  1083.798770  16.527806  1000.800000  1079.600000  1088.7000   \n",
       "TAT   15039.0   545.396183   7.866803   512.450000   542.170000   549.8900   \n",
       "TEY   15039.0   134.188464  15.829717   100.170000   127.985000   133.7800   \n",
       "CDP   15039.0    12.102353   1.103196     9.904400    11.622000    12.0250   \n",
       "CO    15039.0     1.972499   2.222206     0.000388     0.858055     1.3902   \n",
       "NOX   15039.0    68.190934  10.470586    27.765000    61.303500    66.6010   \n",
       "\n",
       "            75%        max  \n",
       "AT      23.8625    34.9290  \n",
       "AP    1016.9000  1034.2000  \n",
       "AH      90.0435   100.2000  \n",
       "AFDP     4.5509     7.6106  \n",
       "GTEP    27.1840    37.4020  \n",
       "TIT   1096.0000  1100.8000  \n",
       "TAT    550.0600   550.6100  \n",
       "TEY    140.8950   174.6100  \n",
       "CDP     12.5780    15.0810  \n",
       "CO       2.1604    44.1030  \n",
       "NOX     73.9355   119.8900  "
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.describe().T"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The features are not on the same scale. For example `AP` has a mean of `1013.19` and `CO` has a mean value of `1.97`. Features should be on the same scale for algorithms such as (gradient descent) to converge smoothly. Let's go ahead and check further."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "TEY     1.000000\n",
       "CDP     0.988473\n",
       "GTEP    0.977042\n",
       "TIT     0.891587\n",
       "AFDP    0.717995\n",
       "AP      0.146939\n",
       "NOX    -0.102631\n",
       "AH     -0.110272\n",
       "AT     -0.207495\n",
       "CO     -0.541751\n",
       "TAT    -0.720356\n",
       "Name: TEY, dtype: float64"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.corr()[\"TEY\"].sort_values(ascending=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Index(['AT', 'AP', 'AH', 'AFDP', 'GTEP', 'TIT', 'TAT', 'TEY', 'CDP', 'CO',\n",
       "       'NOX'],\n",
       "      dtype='object')"
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "numerical_features = df.describe(include=[\"int64\",\"float64\"]).columns\n",
    "numerical_features"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Data Visualization<a class=\"anchor\" id=\"4\"></a>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": []
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "+ ###  Univariate plots<a class=\"anchor\" id=\"4.1\"></a>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Next let's get the distribution of animal data across the types of classes."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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CkBERKJLXwYnImIoFYCIiKFUACIihtL7Lk9j+vTpeL1eAM477zzWrFkDwOrVqxk9ejR5eXkArFq1it27d+PxeAAoKyvrvN5g5ps5cyYrV67E5XLhdrspKSnh7LPPprq6mqqqKmJiYigoKCAnJ+eMZ+tLPrus3+zZs1m+fDmWZTF27FiWL1+Oy+WKyvr1Nptd1u7E78bzzz/P5s2b2bJlC4Bt7nvd5bPL+t18883ccccdXHDBBQDk5eXxve9978yvnyVdamtrs6699tqwsaamJmv27NnWt7/9bauioqJz/MYbb7Sampqinu+mm26y3n77bcuyLKuystJavXq19emnn1rf//73rfb2duuLL77o/Ltd8lmWfdavoKDAev311y3LsqzCwkLrhRdeiMr69TabZdln7SzLst5++21r5syZ1vXXX29ZlmWr+15X+SzLPutXXV1tbdq0KWxsMNZPp4C6UV9fT2trK7fccgszZ86krq4Ov9/P/PnzufbaazvndXR0cODAAVasWMGNN97Ik08+GbV8paWlXHzxxQCEQiHi4uLYu3cv48ePx+124/V6SUtLo76+3jb57LR+69ev5/LLLycQCNDY2MhZZ50VlfXrbTY7rd2RI0d48MEHWbp0aec8O933uspnp/V76623+POf/8xNN93E0qVL8fl8g7J+OgXUjfj4eGbPns3111/P/v37mTNnDlu3bmXUqFGdH2ADaGlp4eabb2bWrFmEQiFmzpzJpZdeytixY6OSD2D37t1s3ryZxx9/nJdffjnskNbj8eDz+c5otr7ks9v6HTp0iFmzZpGUlMTo0aP58MMPB339epvNLms3e/ZsxowZw9KlS4mLi+uc5/P5bHHf6y6fXdZvzpw53HbbbVx//fVceumlbNiwgV/+8peMHTv2jK+fCqAbo0eP5vzzz8fhcDB69GhSUlJobGzk3HPPDZuXkJDAzJkzSUhIAOCKK66gvr7+jN+Jusv35ptvsmHDBh555BFSU1NP+coNv98/KOc4e5vvxC+eXdbv61//Oi+88AJPPPEEa9euZdq0aYO+fr3Ntnr1alus3UcffYTT6eTee++lvb2d9957j/vuu48rrrjCFve97vIVFRXZYv1SUlKYPHly52PLd77zHVauXMnEiRPP+PrpFFA3nnzySdauXQvA4cOH8fl8jBhx6tfG7t+/n/z8fEKhEMFgkN27d3PJJZdEJd/rr7/O5s2bKS8vZ9SoUQBkZmZSW1tLe3s7zc3NNDQ0DMrXbvQ2n53Wb8WKFezfvx/48tmW0+mMyvr1Nptd1u6CCy5g69atlJeXU1paykUXXcSyZctsc9/rLp9d1s/n8zF37lz27t0LwM6dO7nkkksGZf30SeBuBAIBlixZwkcffYTD4WDRokVkZWUBsH79es4+++zOdwFt3LiRrVu3Ehsby7XXXts5Ppj57r77bgoKCjj33HNJTk4G4PLLL+fOO++kurqaLVu2YFkWt99+O7m5ubbKZ4f1W7RoEQD3338/sbGxJCQksGrVKkaOHDno69eXbHZZuxO/GwcPHuSuu+6iuroawBb3vdPls8v6xcXFsXLlSmJjYzn77LNZuXIlSUlJZ3z9VAAiIobSKSAREUOpAEREDKUCEBExlApARMRQKgAREUPpg2AiPVi7di379u2jsbGRtrY2Ro0axfDhw1m2bBnTpk1j7dq1XHPNNfzud7/jqaee6vyw0Yn3lD/44IN87Wtfi/JeiJxKbwMV6aWnn36a999/v/N9+Rs2bKClpYW6ujrKy8s75538XnMRu9IpIJEIWJbFs88+y6xZswgGg7zzzjvRjiTSZyoAkQjs3LmTjIwMUlNT+dGPfsTjjz8e7UgifaYCEIlAdXU1Bw8eZPbs2Tz//PP88Y9/pLm5OdqxRPpELwKL9NHnn3/Onj17ePHFF3G5XADcc889PPPMM8ycOTPK6UR6T0cAIn307LPPMm3atM4Hf4AbbriBiooK9J4KGUr0LiAREUPpCEBExFAqABERQ6kAREQMpQIQETGUCkBExFAqABERQ6kAREQM9f8AsmoufIBZS4MAAAAASUVORK5CYII=\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "numerical_features=[feature for feature in df.columns if df[feature].dtypes != 'O']\n",
    "for feat in numerical_features:\n",
    "    skew = df[feat].skew()\n",
    "    sns.distplot(df[feat], kde= False, label='Skew = %.3f' %(skew), bins=30)\n",
    "    plt.legend(loc='best')\n",
    "    plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Some of the features are normally distributed. The features AH, CO, TITy and TATa exhibit the highest skew coefficients. Moreover, the distribution of Carbon Mono oxide (CO) and Turbine inlet temperature\t(TIT) and Turbine after temperature\t(TAT) seem to contain many outliers. Let's identify the indices of the observations containing outliers using Turkey's method."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "The dataset contains 1543 observations with more than 2 outliers\n"
     ]
    }
   ],
   "source": [
    "# Detect observations with more than one outlier\n",
    "\n",
    "def outlier_hunt(df):\n",
    "    \"\"\"\n",
    "    Takes a dataframe df of features and returns a list of the indices\n",
    "    corresponding to the observations containing more than 2 outliers. \n",
    "    \"\"\"\n",
    "    outlier_indices = []\n",
    "    \n",
    "    # iterate over features(columns)\n",
    "    for col in df.columns.tolist():\n",
    "        # 1st quartile (25%)\n",
    "        Q1 = np.percentile(df[col], 25)\n",
    "        \n",
    "        # 3rd quartile (75%)\n",
    "        Q3 = np.percentile(df[col],75)\n",
    "        \n",
    "        # Interquartile rrange (IQR)\n",
    "        IQR = Q3 - Q1\n",
    "        \n",
    "        # outlier step\n",
    "        outlier_step = 1.5 * IQR\n",
    "        \n",
    "        # Determine a list of indices of outliers for feature col\n",
    "        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index\n",
    "        \n",
    "        # append the found outlier indices for col to the list of outlier indices \n",
    "        outlier_indices.extend(outlier_list_col)\n",
    "        \n",
    "    # select observations containing more than 2 outliers\n",
    "    outlier_indices = Counter(outlier_indices)        \n",
    "    multiple_outliers = list( k for k, v in outlier_indices.items() if v > 2 )\n",
    "    \n",
    "    return multiple_outliers   \n",
    "\n",
    "print('The dataset contains %d observations with more than 2 outliers' %(len(outlier_hunt(df[numerical_features])))) "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "##### Mmm hmm! there exists observations with multiple outliers. \n",
    "\n",
    "Let's examine the boxplots for the several distributions"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Index(['AT', 'AP', 'AH', 'AFDP', 'GTEP', 'TIT', 'TAT', 'TEY', 'CDP', 'CO',\n",
       "       'NOX'],\n",
       "      dtype='object')"
      ]
     },
     "execution_count": 27,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "numerical_features"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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tyg5OVmZmZkqKyvj3AxxLe1Lzsvw4N+ft956q8466yzNmTNH8fHxnuWqq6s5N0NcS/uSczP0BXqevf322zV69GgNGDBATzzxhH7zm9+oX79+UXteEqxDVGZmpnr16iXDMJSZmamUlBQdPnxY3bp181kuMTFREydOVGJioiTpBz/4gXbv3s0TUQgJ1pf/+te/9MQTT+ipp55SWlqakpOTVVNT41mvpqaG7xaFoJb2p/vCgHMzdAXryx49eujNN9/Uiy++qCVLlmjYsGGcmyGupX25aNEizssw4N+fBw8eVExMjB588EHV1tZqz549evjhh/WDH/yAczPEtbQv77vvPs7NEBfoefayyy7zZJOhQ4dqwYIFuvDCC6P2vOSj4CHqpZde0pIlSyRJhw4dUnV1tbp06dJkuU8//VQTJkxQQ0OD6uvrtX37dvXv37+9y0UzAvXlP/7xDz333HMqKChQenq6JOm8887T+++/r9raWlVVVWnv3r3Kzs62s3QE0NL+5NwMfYH68v7779enn34q6fi77DExMZybYaClfcl5GR78+7N3797asGGDCgoKtHLlSvXt21dz587l3AwDLe1Lzs3QF+h59s4779QHH3wgSXrvvffUv3//qD4vDdM0TbuLQFN1dXWaPXu2Dh48KMMwNHPmTA0cOFCStGrVKnXu3NnzV8HXrFmjDRs2KDY2Vtddd51nOkKDf1/OmDFDU6dOVbdu3dSpUydJ0n/913/pnnvuUWFhodavXy/TNHXHHXdo+PDhNlcPf63pT87N0BboeVaSli1bptjYWCUmJmrhwoXq2rUr52aIa01fcl6Gvuaugfbv36/c3FwVFhZKEudmiGtNX3JuhrZAfRkfH68FCxYoNjZWnTt31oIFC5ScnBy15yXBGgAAAAAAC/goOAAAAAAAFhCsAQAAAACwgGANAAAAAIAFBGsAAAAAACwgWAMAAAAAYIHT7gIAAIh2S5Ys0a5du3T48GF9++23Sk9PV2pqqubOnathw4ZpyZIluvLKK/XKK6/oD3/4g2pra7Vnzx7P77yuWLFCZ5xxhs17AQBA9OLntgAACBEvv/yySkpKPL/D/MQTT+jo0aPasWOHCgoKPMv5//4rAACwFx8FBwAgBJmmqT/96U+65ZZbVF9fr08++cTukgAAQBAEawAAQtB7772n7OxspaWl6YYbbtDatWvt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FtsVAIBz4zvWAAAAAADYQLAGAAAAAMAGgjUAAAAAADYYpmmaThcBAAAAAECk4ow1AAAAAAA2EKwBAAAAALCBYA0AAAAAgA0EawAAAAAAbCBYAwAAAABgA8EaAAAAAAAbIipYv/322yooKJAkVVZWavr06crPz9eiRYvU3NwsSdqwYYMmT56sKVOmaPv27U6Wi3by366S9Oqrr2rOnDnW43/+85+6+eabNW3aND311FNOlIgL4L9dd+/erfz8fBUUFOjOO+/UsWPHJLG/Rir/bfvBBx9o+vTpmjZtmhYvXqympiZJbNtIFHwslqSXXnpJU6dOtR6zXSOP/3atqKjQddddp4KCAhUUFGjjxo2S2K6RyH+7Hj9+XPfcc49uueUWTZs2Tfv27ZPEdo1E/tt19uzZ1r46ZswYzZ49WxLbNayZEeLZZ581J02aZN58882maZrm97//ffN//ud/TNM0zYULF5qbN282jxw5Yk6aNMk8c+aM+cknn1h/I3wFb9clS5aYeXl55qxZs6w2N954o1lZWWk2Nzebd911l7lr1y6nykU7BW/XW265xXz33XdN0zTNX//61+ajjz7K/hqhgrftPffcY/7tb38zTdM0586dy7E4QgVvV9M0zXfffde89dZbrefYrpEneLtu2LDB/MUvfhHQhu0aeYK369y5c82XX37ZNE3TfPPNN83t27ezXSNQqOOwaZrmyZMnzRtvvNE8fPgw2zXMRcwZ68zMTK1atcp6XFFRoauvvlqSNHLkSL3xxht655139KUvfUlxcXHq0aOHMjMz9a9//cupktEOwdv18ssv1+LFi63HdXV1amhoUGZmpgzD0LXXXqs333zTgUpxPoK3a3l5uYYNGyZJampqUnx8PPtrhAretqtWrdJVV12lhoYGHT16VH369GHbRqDg7VpTU6OVK1eqpKTEeo7tGnmCt+uuXbv05z//WbfccotKSkpUV1fHdo1Awdv1rbfe0uHDh3X77bfrpZde0tVXX812jUDB29Vn1apV+s53vqP+/fuzXcNcxATrvLw8ud1u67FpmjIMQ5KUnJys2tpa1dXVqUePHlab5ORk1dXVdXmtaL/g7XrDDTdY21VqCdYpKSnWY9+2RngL3q79+/eX1PLi/9xzz+n2229nf41QwdvW5XKpurpakyZNUk1NjbKzs9m2Ech/uzY1NWnBggUq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\n",
      "text/plain": [
       "<Figure size 1008x1152 with 11 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "ot=df.copy() \n",
    "fig, axes=plt.subplots(11,1,figsize=(14,16),sharex=False,sharey=False)\n",
    "sns.boxplot(x='AT',data=ot,palette='crest',ax=axes[0])\n",
    "sns.boxplot(x='AP',data=ot,palette='crest',ax=axes[1])\n",
    "sns.boxplot(x='AH',data=ot,palette='crest',ax=axes[2])\n",
    "sns.boxplot(x='AFDP',data=ot,palette='crest',ax=axes[3])\n",
    "sns.boxplot(x='GTEP',data=ot,palette='crest',ax=axes[4])\n",
    "sns.boxplot(x='TIT',data=ot,palette='crest',ax=axes[5])\n",
    "sns.boxplot(x='TAT',data=ot,palette='crest',ax=axes[6])\n",
    "sns.boxplot(x='TEY',data=ot,palette='crest',ax=axes[7])\n",
    "sns.boxplot(x='CDP',data=ot,palette='crest',ax=axes[8])\n",
    "sns.boxplot(x='CO',data=ot,palette='crest',ax=axes[9])\n",
    "sns.boxplot(x='NOX',data=ot,palette='crest',ax=axes[10])\n",
    "plt.tight_layout(pad=2.0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:>"
      ]
     },
     "execution_count": 34,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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wQgAAACAgmAEAAAAQEEwAgAAAKBQX+sB2H0LFvw89977hyRJQ0NjmprGZPLkM2s7FAAAANBnCUb9QGvryjy16slkj7o8tfqpWo8DAAAA9HGCUX+xR10yckitpwAAAAD6Ae8wAgAAAKAgGAEAAABQEIwAAAAAKAhGAAAAABQEIwAAAAAKghEAAAAABcEIAAAAgIJgBAAAAEBBMAIAAACgIBgBAAAAUBCMAAAAACgIRgAAAAAUBCMAAAAACoJRH7NsWUuWLWup+mcAAACAgau+1gOwc1paliRJxo9vrupnAAAAgIHLCiMAAAAACoIRAAAAAAXBCAAAAICCYAQAAABAQTACAAAAoCAYAQAAAFAQjAAAAAAoCEYAAAAAFAQjAAAAAAqCEQAAAAAFwQgAAACAQn2tB6D6/vKXR/Lyyy9lypTTiv1DhgzJVVd9L42NI2o0GQAAANAbdSkYPfbYY7niiiuyatWq7LXXXtlrr70yderUXH755dmyZUseeeSR7LfffmlsbMyxxx6bgw46KN/97nczatSozu8YN25cZs6cmSlTpmTDhg0ZOnRokmTw4MGZM2dODjrooOr8hOTll1/a5v6NGzfmppt+kzPO+GwPTwQAAAD0ZjsMRhs2bMhZZ52Vyy67LO985zuTJPfee2++9rWvZf78+UmS6dOn56STTsqECROSJDfeeGM+9KEP5YILLtjmd86ZMyeHHXZYkuT666/Pz372s3zlK1/plh+I0uWXf/UNjy9e/L9z8smTrDICAAAAOu0wGP3+97/P+973vs5YlCRHHXVU5s2b1y0DtLe3Z++99+6W7xoI2tvb0tbWllmzLu7c19q6MtlceWVj/ea0rlnZefxPf/rjDr/TKiMAAADgtXYYjB5//PGMHj26c/uss87K2rVr88wzz+QXv/hFDj744G1+7tZbb80999zTuT1p0qScfPLJSZJp06Zl6NChqaury1ve8pZMnTp1N38Mdseddy4VjAAAAIBOOwxGBx98cO67777O7R/84AdJko9//OPp6OjY7ue6+kgaO6ehoTENDY2ZMePSzn2zZl2cFY/850qivQen6eAxncdf/6LrbTn22AlVmRUAAADomwbt6ISJEyfmrrvuyh/+8IfOfa2trVm1alXq6uqqORvd4K//+m92eM7JJ0/qgUkAAACAvmKHK4yGDRuWH/zgB7nqqqty5ZVXpqOjI/X19bnsssvy5je/ebufe/0jacOHD+9cnUTPueiir73hKqOJEz/ghdcAAABAYYfBKEkOPfTQfOtb39ru8dmzZxfbp556ak499dRtnvvqb1aj5+y55155+eWXtto/ZMgQq4sAAACArXQpGNG3veUtY5OkeO8RAAAAwPbs8B1GAAAAAAwsghEAAAAABcEIAAAAgIJgBAAAAEBBMAIAAACgIBgBAAAAUBCMAAAAACgIRgAAAAAUBCMAAAAACoIRAAAAAIX6Wg/AzmluPqFHPgMAAAAMXIJRHzN+fHOPfAYAAAAYuDySBgAAAEBBMAIAAACgIBgBAAAAUBCMAAAAACgIRgAAAAAUBCMAAAAACoIRAAAAAAXBCAAAAICCYAQAAABAQTACAAAAoCAYAQAAAFAQjAAAAAAo1Nd6ALrJpkry3MZX/rxvbUcBAAAA+jbBqB9oahqT9va2JElDQ2OamsbUdB4AAACgbxOM+oHJk8+s9QgAAABAP+IdRgAAAAAUBCMAAAAACoIRAAAAAAXBCAAAAICCYAQAAABAQTACAAAAoCAYAQAAAFAQjAAAAAAoCEYAAAAAFAQjAAAAAAqCEQAAAAAFwQgAAACAgmAEAAAAQKG+1gMAAAAAVMuCBT9Pa+vK7R5vb29LkjQ0NG7zeFPTmEyefGb3D9bLCUYAAABAv9XaujIrHnko2b9h2yesbkuSPJWN2zjWXr3BejnBCAAAAOjf9m/I4A+9f5uHNt96e5Js8/irxwYi7zACAAAAoCAYAQAAAFAQjAAAAAAoCEYAAAAAFAQjAAAAAAqCEQAAAAAFwQgAAACAgmAEAAAAQEEwAgAAAKAgGAEAAABQEIwAAAAAKAhGAAAAABQEIwAAAAAKghEAAABQU8uWtWTZspZaj9ElfWnW3VFf6wEAAACAga2lZUmSZPz45hpPsmN9adbdYYURAAAAAAXBCAAAAICCYAQAAABAQTACAAAAoCAYAQAAAFAQjAAAAAAoCEYAAAAAFAQjAAAAAAqCEQAAAAAFwQgAAACAgmAEAAAAQKG+1gMAAAAAu6atbU1mzLggjz32aKZNuyhHHHHUVse/971v5YtfPC+NjSO2+x2vP2fx4v+da6/9SdXnf70pU07L/PkLe/y6bK3bg9Hs2bNz//3359lnn81LL72UUaNGZcSIEbn77rtzxx135NOf/nS2bNmSRx55JPvtt18aGxtz7LHH5qyzzuruUQAAAKBfW7To13n00dYkydy538wPf3jtVscffHBFbrrpNznjjM9u9ztef8611/60qnPT+3V7MJo+fXqS5MYbb8wjjzySCy64IEly3HHHJUl+8YtfdJ530kknZcKECd09AgAAAPR7bW1r8u///m+d2+vWrcv999/bucqorW1Nbr/996lUKlm69Pc5+eRJW60y2tY5d9/9/yWp9OSPUrDKqHfwSBoAAAD0QYsW/Tpbtmwp9r12ldGiRb9OpfJK+KlUtmxzldG2zlm8+HfVH34HZs26uNu+q7V1ZTJkF1/hvP6ltLatLOZpbV2ZxsbGbpmtN/PSawAAAOiD7rrr9q32rVu3rjje0dGRJOno6Middy7d5ndsfU7tVhfRe1hhBAAAAH3QMce8P0uWlKuBhg0bVhxfunRJOjo6Ul9fn2OP3fqVMNs655UVRrWNRjNmXNpt3zVr1sVZ0f7Mrn14773S9KYDi3m6c/VTb2aFEQAAAPRBp5zysQwaVP7X+nPOOb84XldXlySpqxuUk0+etM3veP05Z5zxj1Wcmr5CMAIAAIA+qLFxRP72b/+uc3vYsGGdL7x+9fj733986urqMmHC8Vu98Hp750yc+IEkdT3xI2yTF173DlV7JO3UU08ttu+4445ie/bs2dW6NAAAAAwIp5zysTz88J/y2GOPFquLXnv8iSce2+bqojc654wz/jHXXvuTqsxM3+AdRgAAANBHNTaOyPe/f3Xa2tZv9/hFF31th9/x+nMmTvzAf6406hmvvheoO99dxO7xSBoAAAAABcEIAAAAgIJgBAAAAEBBMAIAAACgIBgBAAAAUBCMAAAAACgIRgAAAAAUBCMAAAAACoIRAAAAAAXBCAAAAIBCfa0HAAAAAAa25uYTaj1Cl/WlWXeHYAQAAADU1PjxzbUeocv60qy7wyNpAAAAABQEIwAAAAAKghEAAAAABcEIAAAAgIJgBAAAAEBBMAIAAACgIBgBAAAAUBCMAAAAACgIRgAAAAAUBCMAAAAACoIRAAAAAAXBCAAAAICCYAQAAABAob7WAwAAAABU1er2bL719u0ca0uSbR9f3Z40HFi9uXoxwQgAAADot5qaxrzh8fYMSZI0NDRufbDhwB1+vr8SjAAAAIB+a/LkM2s9Qp/kHUYAAAAAFAQjAAAAAAqCEQAAAAAFwQgAAACAgmAEAAAAQEEwAgAAAKAgGAEAAABQEIwAAAAAKAhGAAAAABQEIwAAAAAKghEAAAAABcEIAAAAgEJ9rQcAAOhLFiz4eVpbV27zWHt7W/bff2SmTZvZs0MBAHQzwQgAYCe0tq7MikceTt1+I7c6Vnn6qbS1tfX8UAAA3UwwAgDYSXX7jcweH/roVvs3/uKnNZgGAKD7eYcRAAAAAAXBCAAAAICCYAQAAABAQTACAAAAoCAYAQAAAFAQjAAAAAAoCEYAAAAAFAQjAAAAAAqCEQAAAAAFwQgAAACAgmAEAAAAQEEwAgAAAKAgGAEAAABQEIwAgAFv2bKWLFvW0i3ftXlzR7d9FwBArQhGAMCA19KyJC0tS7rluzo6OrrtuwAAakUwAgAAAKAgGAEAAABQEIwAAAAAKAhGAAAAABQEIwAAAAAKghEAAAAABcEIAAAAgIJgBAAAAEBBMAIAAACgIBgBAAAAUBCMAAAAACjU13oAAIDdcd9992TOnMs7t4cNG55169bu0ndNmXJa5s9f2F2jAQD0Wd0SjGbPnp37778/zz77bF566aWMGjUqI0aMyIwZM3LiiSdm9uzZ+eAHP5ibbropv/nNb/Lyyy/n4YcfzhFHHJEkufLKK3PQQQd1xygAwAAzd+63iu1djUUAAPyXbglG06dPT5LceOONeeSRR3LBBRckSX7wgx/kU5/6VK6//vp88IMfzMknn5yTTz45jz/+eM4///zMnz+/Oy4PAAxQ9913T9avX9et32mVEQBAFR9Jq1Qq+e1vf5vrr78+X/jCF/Lggw9m3Lhx1bocADAAvX51UXeZNevi7R5rbV2ZypAh2z64ZXO2bNmS9va2qswFANBTqvbS67vuuivjxo3Lfvvtl0mTJuW6666r1qUAgAGqu1cXAQDwiqqtMLrhhhvy+OOP5x//8R+zadOmrFixIhdccEH22Wefal0SABhg9t57WFWi0YwZl2732KxZF+dPbW3bPjhocAZVKmloaOz2mQAAelJVVhg9//zzueeee7Jw4cL89Kc/zbx583LiiSdm0aJF1bgcADBAnXPOebUeAQCgX6pKMPrtb3+bE088MYMHD+7c9/GPfzzXX399KpVKNS4JAAxARx759uy997Bu/U4vvAYA6OZH0k499dTtHjvqqKPyv/7X/0qSHHroobnhhhu689IAwAB1zjnnZc6cyzu3hw0bnnXr1tZwIgCAvq9q7zACAOgJRx759t1eFfTqb0V7o3cXAQAMJFX7LWkAAAAA9E2CEQAAAAAFwQgAAACAgmAEAAAAQEEwAgAAAKAgGAEAAABQEIwAAAAAKAhGAAAAABQEIwAAAAAKghEAAAAAhfpaDwAAUGvNzSd023fV19d36/cBANSCYAQADHjjxzd323cNHlzfrd8HAFALHkkDAAAAoCAYAQAAAFAQjAAAAAAoCEYAAAAAFAQjAAAAAAqCEQAAAAAFwQgAAACAgmAEAAAAQEEwAgAAAKAgGAEAAABQEIwAAAAAKAhGAAAAABQEIwAAAAAK9bUeAACgr6k8/1w23frbrQ90bErq/esVAND3+TcaAICd0NQ0ZrvH2uuS/fcf2XPDAABUiWAEALATJk8+s9YjAABUnXcYAQAAAFAQjAAAAAAoCEYAAAAAFAQjAAAAAAqCEQAAAAAFwQgAAACAgmAEAAAAQEEwAgAAAKAgGAEAAABQEIwAAAAAKAhGAAAAABQEIwAAAAAK9bUeAPq6BQt+ntbWlcW+9va2JElDQ+M2P9PUNCaTJ59Z3cEAAABgFwlGsJtaW1dmxSOPZPD+h3Tu27z6+STJM9lrq/M3r36yx2YDAACAXSEYQTcYvP8h2ftDX+jcXn/r1UlS7Hv9MQAAAOitvMMIAAAAgIJgBAAAAEBBMAIAAACgIBgBAAAAUBCMAAAAACgIRgAAAAAUBCMAAAAACoIRAAAAAAXBCAAAAICCYAQAAABAQTACAAAAoCAYAQAAAFAQjAAAAAAoCEYMWMuWtWTZspZaj7HL+vr8AAAA9F71tR4AaqWlZUmSZPz45hpPsmv6+vwAAAD0XlYYAQAAAFAQjAAAAAAoCEYAAAAAFAQjAAAAAAqCEQAAAAAFwQgAAACAgmAEAAAAQEEwAgAAAKAgGAEAAABQEIwAAAAAKAhGAAAAABTqaz0AVEtb25pcfvlX8/TTq97wvClTTsv8+Qt7aKrus2LFA0lemX9nTZ8+M0cccVR3jwQAAEA/sVvBaPbs2bn//vvz7LPP5qWXXsqoUaMyYsSILFu2LEcccURx7tVXX51TTz01s2fPzrvf/e4kyQMPPJAvf/nL+fWvf51hw4btziiwlUWLfr3DWDRQzZ37zfzwh9fWegwAAAB6qd0KRtOnT0+S3HjjjXnkkUdywQUX5PHHH8+qVasyf/78rc6fNWtWLrrooixatCiDBg3KRRddlNmzZ4tFdLu2tjVpaVnc5fP72iqjXVlV9Frr1q3L/fffa5URAAAA29Sjj6S9973vTXNzc77//e9nr732ysSJE/P2t7+9J0dggFi06NfZvHnzTn1m1qyLd+lara0rs2VI16PnlvUvprVt1S5fr7tYZQQAAMD2VCUYPfzww5kyZUrn9hFHHNG5Gum8887Lf//v/z2NjY356U9/Wo3LQ+666/Zaj9DrrVu3rtYjAAAA0EtVJRi99a1v3eYjaUmy5557ZuLEiRk5cmQGDx5cjctDjjnm/Vmy5Hc79ZkZMy7dpWvNmnVxHmp/qcvnD9p7nzS96YBdvl6y+4+kJfEoKAAAANs1qNYDQDWccsrHBMkdOOec82s9AgAAAL1UVYLRq4+kvfZ/HnvssWpcCrapsXFEmpsndvn8vvTC62T35x02bJgXXgMAALBd3fJI2qmnntr550MPPTT/8R//8Ybnn3POOd1xWXhDp5zysdx//715+ulVtR6l17G6CAAAgDfSo78lDXpSY+OIXHnl3O0ef/W3lO3Ou4Rq6fDD35ak784PAABA7+UdRgAAAAAUBCMAAAAACoIRAAAAAAXBCAAAAICCYAQAAABAQTACAAAAoCAYAQAAAFAQjAAAAAAoCEYAAAAAFAQjAAAAAAr1tR4AaqW5+YRaj7Bb+vr8AAAA9F6CEQPW+PHNtR5ht/T1+QEAAOi9PJIGAAAAQEEwAgAAAKAgGAEAAABQEIwAAAAAKAhGAAAAABQEIwAAAAAKghEAAAAABcEIAAAAgIJgBAAAAEBBMAIAAACgIBgBAAAAUBCMAAAAACgIRgAAAAAU6ms9APQHm1c/mfW3Xl1sJyn2FccaxvbYbAAAALCzBCPYTU1NY7ba1579kiQNDXtt/YGGsdv8DAAAAPQWghHspsmTz6z1CAAAANCtvMMIAAAAgIJgBAAAAEBBMAIAAACgIBgBAAAAUBCMAAAAACgIRgAAAAAUBCMAAAAACoIRAAAAAAXBCAAAAICCYAQAAABAQTACAAAAoCAYAQAAAFCor/UAbNucOZdl9ernctRR78jkyWfWehwAAABgABGMeqk///nhbNiwPg0NjbUeBQAAABhgPJIGAAAAQEEwAgAAAKAgGAEAAABQEIwAAAAAKAhGAAAAABQEIwAAAAAKghEAAAAABcEIAAAAgIJgBAAAAEBBMAIAAACgIBgBAAAAUBCMAAAAACgIRgAAAAAUBCMAAAAACoJRL7NsWUuWLWvp3G5vbyu2AQAAAKqtvtYDUGppWVJst7W1paVlScaPb67RRAAAAMBAY4URAAAAAAXBCAAAAICCYAQAAABAQTACAAAAoCAYAQAAAFAQjAAAAAAoCEYAAAAAFAQjAAAAAAqCEQAAAAAFwQgAAACAgmAEAAAAQKG+1gNQWrHigWJ7w4b1WbHigUyZclqSZP78hbUYCwAAABhAdisYPfTQQ7niiiuyYcOGrF+/Ps3NzTnllFPy0Y9+NEcccUQqlUo2btyYj3zkI5k8eXKS5Mgjj8w73/nOJElHR0cOO+ywXHLJJamv164AAAAAeoNdrjQvvPBCzj///MydOzdjxozJ5s2b86UvfSnLli3LW9/61syfPz9JsmnTppx99tk55JBDcsIJJ6ShoaHzWJKce+65aWlpycSJE3f/p+njXl1FtKNzrDICAAAAqmmX32G0ePHiHH300RkzZkySZPDgwZkzZ07e9773Feftscce+dSnPpV//dd/3eo7Nm3alPXr12fvvffe1TEAAAAA6Ga7vMLomWeeyahRo4p9w4YNyx577LHVuSNHjsyaNWuSJO3t7ZkyZUqSpK6uLhMmTMgxxxyzq2MAAAAA0M12ORgdcsgheeCB8gXNjz32WFatWrXVuU888UQOPvjgJNnqkTQAAAAAepddfiTt+OOPz+23355HH300ySuPl82ePTsPPvhgcd7GjRszb968/MM//MPuTQoAAABAj9jlFUbDhw/P7Nmzc9FFF6VSqWTdunU5/vjjM2HChFx11VWZMmVK6urq0tHRkQ9/+MM59thju3Pufmn+/IU7fPG1F14DAAAA1bZbv8v+yCOPzLx587ba/x//8R/b/cwdd9yxO5cEAAAAoMp2KxjR/Q4//G1JktbWldmwYX2GDt07TU1jMmPGpTWeDAAAABgodvkdRgAAAAD0T4IRAAAAAAXBCAAAAICCYAQAAABAQTACAAAAoCAYAQAAAFAQjAAAAAAoCEYAAAAAFAQjAAAAAAqCEQAAAAAFwaiXaW4+Ic3NJ3RuNzY2FtsAAAAA1SYY9TLjxzdn/Pjmzu2GhsZiGwAAAKDaBCMAAAAACoIRAAAAAAXBCAAAAICCYAQAAABAQTACAAAAoCAYAQAAAFAQjAAAAAAoCEYAAAAAFAQjAAAAAAqCEQAAAAAFwQgAAACAgmAEAAAAQEEwAgAAAKAgGAEAAABQqK/1AGzbYYe9NatXP5empjG1HgUAAAAYYASjXmratJm1HgEAAAAYoDySBgAAAEBBMAIAAACgIBgBAAAAUBCMAAAAACgIRgAAAAAUBCMAAAAACoIRAAAAAAXBCAAAAICCYAQAAABAQTACAAAAoCAYAQAAAFAQjAAAAAAoCEYAAAAAFOprPcBA89WvXpRVq57OUUe9I5Mnn1nrcQAAAAC2Ihj1sD/96cGsX78uDQ2NtR4FAAAAYJs8kgYAAABAQTACAAAAoCAYAQAAAFAQjAAAAAAoCEYAAAAAFAQjAAAAAAqCEQAAAAAFwQgAAACAgmAEAAAAQEEwAgAAAKAgGAEAAABQEIwAAAAAKAhGAAAAABQEox60bFlLOjo2FdvLlrXUcCIAAACArdXXeoCBpKVlSTo6OortJBk/vrlWIwEAAABsxQojAAAAAAqCEQAAAAAFwQgAAACAgmAEAAAAQEEwAgAAAKAgGAEAAABQEIwAAAAAKAhGAAAAABQEIwAAAAAKghEAAAAABcEIAAAAgEJ9rQcYyFaseCBJMmXKaTs8t6GhIfvu25Cnn16VpC4HHHBg9tijPpXKK8fr6+tz7rlT097elssum5lKpZJKpZJBgwZlxIj9s3r1c+no2JRKpZLp02fmzW8elW9/+xtJ6nLuuVPT2Dgi9913T77xjVl585sPzbRpM9PYOKJ6PzwAAADQa1UlGD300EO54oorsmHDhqxfvz7Nzc0555xzsmbNmsyZMydPPvlkNm/enDe96U2ZPn16DjjggGqM0a+0t7envb29c/uJJx7b6pybbvpN/vjH+/Pyyy8X+1eterLYnjv3mzn66OPy5z8/3Pm5M874bObO/VYqlUoef/yxzn0AAADAwNPtweiFF17I+eefn7lz52bMmDHZvHlzvvSlL+WXv/xlbr311nzmM5/J3/3d3yVJ7rzzznzuc5/LwoULM3jw4O4epVd7dXVRd2ppWZyOjo4dnrdu3br8+7//W+f20qVLcvjhf5P169d17vv3f1+ck0+eZJURAAAADEDd/g6jxYsX5+ijj86YMWOSJIMHD86cOXNy5JFHZp999umMRUly7LHHZvTo0fm///f/dvcYvVJ7e1u2bNlSte/vSix61Wvn6OjoyA9/+L3i+ObNHbnppt9022wAAABA39HtweiZZ57JqFGjin3Dhg3L448/vtX+JBk1alSefPLJrfbTcyqVSjZv3jo23Xnn0hpMAwAAANRatwejQw45JKtWrSr2PfbYYxk5cmSeeOKJrc5vbW3Nm970pu4eo1dqaGjMoEG97xfT1dXVZfDgrZ9OPPbYCTWYBgAAAKi1bq8Xxx9/fG6//fY8+uijSZJNmzZl9uzZeeihh/Lcc89lyZIlnecuXbo0ra2tee9739vdYwxI9fVdfyXVa8NVfX19Pv/5LxbHBw+uz8knT+q22QAAAIC+o9tfej18+PDMnj07F110USqVStatW5fjjz8+n/zkJ/P3f//3+ed//uf86Ec/SpIcfPDBueaaawbcC6+T5PDD39btL75ubp6YP/7x/jz55ONveN6wYcNy9NHHZcmS3yVJJkw4Ie9733H5+c9/3Pni67/924leeA0AAAADVLcHoyQ58sgjM2/evK3277///rnqqquqccl+r6GhIfvu25Cnn16VpC4HHHBg9tijPpXKK8fr619ZEXT88RNz2WUzU6lUUqlUMmjQoIwYsX9Wr34uHR2bUqlUcs455+fNbx6V1tZHktR1riQ655zz8o1vzMqb33yo1UUAAAAwgNVVKq8mh95r06bNaWtbX+sxdtusWRfnwQdXZMuWLTn88Ld17p8x49IaTgXdp7Fx737xzypsj3uc/sz9TX/m/qa/c4+zqw44YJ/tHut9b2AGAAAAoKYEIwAAAAAKghEAAAAABcEIAAAAgIJgBAAAAEBBMAIAAACgIBgBAAAAUBCMAAAAACgIRgAAAAAUBCMAAAAACvW1HmAgaW4+IY888nA2btzYuQ0AAADQ21hh1IPGj29Off0exfb48c01nAgAAABga4IRAAAAAAXBCAAAAICCYAQAAABAQTACAAAAoCAYAQAAAFAQjAAAAAAoCEYAAAAAFAQjAAAAAAqCEQAAAAAFwQgAAACAgmAEAAAAQEEwAgAAAKAgGAEAAABQEIwAAAAAKNTXeoCB5q//elxWrXo6TU1jaj0KAAAAwDYJRj3sa1+7PG1t62s9BgAAAMB2eSQNAAAAgIJgBAAAAEBBMAIAAACgIBgBAAAAUBCMAAAAACgIRgAAAAAUBCMAAAAACoIRAAAAAAXBCAAAAICCYAQAAABAQTACAAAAoCAYAQAAAFAQjAAAAAAo1Nd6APqXBQt+ntbWlZ3b7e1tSZKGhsZtnt/UNCaTJ59Z/cEAAACALhOM6FatrSvz6MOPZPQ+BydJ2l5cnSRpeGnIVuc++uKqHp0NAAAA6BrBiG43ep+D85WjP5Mk+frynyVJ5/ZrvXoMAAAA6F28wwgAAACAgmAEAAAAQEEwAgAAAKAgGAEAAABQEIwAAAAAKAhGAAAAABQEIwAAAAAKghEAAAAABcEIAAAAgIJgBAAAAEBBMAIAAACgIBgBAAAAUBCM6LRsWUuWLWup9Rg7ra/ODQAAAL1Vfa0HoPdoaVmSJBk/vrnGk+ycvjo3AAAA9FZWGAEAAABQEIwAAAAAKAhGAAAAABQEIwAAAAAKghEAAAAABcEIAAAAgIJgBAAAAEBBMAIAAACgIBgBAAAAUBCMAAAAACgIRgAAAAAU6ms9AN2vrW1Nvve9b+WLXzwvjY0juvSZ1ta/5E9/WpGmpjHVHa4KVqx4IEkyZcppVb3O0KFD841vfKfLf6cAAADQV+3SCqPly5fnPe95T5566qnOfVdeeWVuvPHGrFu3Lpdffnn+x//4H5kyZUo+//nP5y9/+UuS5I477shHPvKRvPTSS0mSp59+Oh/+8Ifz9NNPd8OPwqsWLfp1HnxwRW666Tdd/szVV383lcqWPPnk41WcrG/bsGHDTv2dAgAAQF+1y4+k7bHHHvnKV76SSqVS7J85c2aamppy3XXXZf78+Tn33HNz9tln58UXX8xxxx2X8ePHZ/bs2dm0aVPOO++8TJ8+PQcddNBu/yC8oq1tTW6//fepVCpZuvT3aWtbs8PPtLb+pTMUbdy4MY8+urLKU3afs8/+bI9eb/Hi/92lv1MAAADoy3b5kbT3ve992bJlS6677rpMnjw5SbJmzZo8+OCD+eY3v9l53uGHH57jjz8+v/vd7zJp0qScd955+eQnP5kvfOELOfbYY3Pcccft/k9Bp0WLft0Z8SqVLbnppt/kjDPeOKpcffV3i+1LL52RsWPfukvXb21dmca6vbt0bvvLa9PW+kxmzbp4l66VJC+80L7Ln91VXfk7BQAAgL5st156fckll+Taa6/NypUrkyRbtmzJqFGjtjpv1KhRefLJJ5O8sjLp4x//eO68886ceuqpu3N5tuGuu25PR0dHkqSjoyN33rl0h595/WNoGzdurMps/UVX/k4BAACgL9utl16PGDEiF154YaZPn553vetd2bRpU2cYeq3W1tYcdthhSZInnngiP/nJTzJ16tRMnTo18+bNy+DBg3dnDF7jmGPen6VLl6SjoyP19fU59tgJO/zMIYccWkSjN795VGbMuHSXrj9r1sXJ0+u7dG7DnsPTcNCBu3ytpPovut6WrvydAgAAQF+2WyuMkuSEE07IW97ylixatCgHH3xwRo8eneuuu67z+P33358lS5bkxBNPzMaNG3PuuefmwgsvzBlnnJE3velN+d73vre7I/Aap5zysdTV1SVJ6uoG5eSTJ+3wM1/4wj+94XZvtu++DT1+za78nQIAAEBfttvBKElmzJiRvfbaK0kyZ86cPPTQQznttNNy+umn5zvf+U6uvvrq7LvvvpkzZ07e/e53p7m5Ockrj7TddtttWb58eXeMQZLGxhF5//uPT11dXSZMOL5LvwK+qektOeSQQ5MkQ4YMyejRY6o8Zff5/vd/0qPXmzjxA136OwUAAIC+bJceSTv66KNz9NFHd24PHz48v//97zu3L7nkkm1+bubMmcX28OHD87vf/W5XRuANnHLKx/LEE4/t1EqYL3zhnzJz5vTOcMTWhg4danURAAAAA8JuvcOI3qmxcUQuuuhrO/WZpqa35K//+vAqTVRdhx/+tiTZrXchAQAAAP+lWx5JAwAAAKD/EIwAAAAAKAhGAAAAABQEIwAAAAAKghEAAAAABcEIAAAAgIJgBAAAAEBBMAIAAACgIBgBAAAAUBCMAAAAACjU13oAeo/m5hNqPcIu6atzAwAAQG8lGNFp/PjmWo+wS/rq3AAAANBbeSQNAAAAgIJgBAAAAEBBMAIAAACgIBgBAAAAUBCMAAAAACgIRgAAAAAUBCMAAAAACoIRAAAAAAXBCAAAAICCYAQAAABAQTACAAAAoCAYAQAAAFAQjAAAAAAo1Nd6APqfR19cla8v/1mSpPXFp5Kkc/v1540+aGyPzgYAAADsmGBEt2pqGlNsN+61/yt/aNh7q3NHHzR2q/MBAACA2hOM6FaTJ59Z6xEAAACA3eQdRgAAAAAUBCMAAAAACoIRAAAAAAXBCAAAAICCYAQAAABAQTACAAAAoFBXqVQqtR4CAAAAgN7DCiMAAAAACoIRAAAAAAXBCAAAAICCYAQAAABAQTACAAAAoCAYAQAAAFCor/UAA8WWLVtyySWX5E9/+lOGDBmSyy+/PE1NTbUeC3bKpk2bcuGFF+aJJ57Ixo0bc9ZZZ+Wtb31rpk+fnrq6uvzVX/1VLr744gwaNCg33HBDfvWrX6W+vj5nnXVWjj/++FqPD12yevXqnHrqqfnZz36W+vp69zf9yo9+9KMsWbIkmzZtyic+8Ym8973vdY/TL2zatCnTp0/PE088kUGDBuWyyy7zn+H0G/fcc0+uvPLKzJ8/P62trV2+r1966aVMnTo1q1evzrBhwzJnzpzst99+tf5x6EOsMOoh//Zv/5aNGzfmX/7lX/LlL385s2fPrvVIsNNuvvnmNDY25vrrr8+Pf/zjXHbZZfn617+ec889N9dff30qlUoWL16cZ599NvPnz8+vfvWr/PSnP803v/nNbNy4sdbjww5t2rQpX/3qV7PXXnslifubfmX58uX5f//v/+WXv/xl5s+fn1WrVrnH6TdaWlrS0dGRX/3qVzn77LPz7W9/2/1Nv/DjH/84F110UV5++eUkO/fvJr/85S8zbty4XH/99Tn55JNz9dVX1/inoa8RjHrI3Xffnfe///1Jkne84x257777ajwR7Ly///u/z5e+9KXO7cGDB+f+++/Pe9/73iTJhAkTcuedd+bee+/NO9/5zgwZMiT77LNPRo8enRUrVtRqbOiyOXPm5PTTT8+BBx6YJO5v+pVly5Zl3LhxOfvss/P5z38+f/u3f+sep994y1veks2bN2fLli1Zu3Zt6uvr3d/0C6NHj87cuXM7t3fmvn7tfwedMGFC7rrrrpr8DPRdglEPWbt2bYYPH965PXjw4HR0dNRwIth5w4YNy/Dhw7N27dr80z/9U84999xUKpXU1dV1Hn/xxRezdu3a7LPPPsXn1q5dW6uxoUtuvPHG7Lfffp3/YpXE/U2/smbNmtx33335zne+k0svvTQXXHCBe5x+Y++9984TTzyRD37wg5k5c2amTJni/qZf+MAHPpD6+v96k8zO3Nev3f/qubAzvMOohwwfPjzr1q3r3N6yZUvxDz70FU899VTOPvvsfPKTn8yHP/zhXHHFFZ3H1q1bl3333Xer+33dunXF/xOD3ug3v/lN6urqctddd+WPf/xjpk2blueff77zuPubvq6xsTFjx47NkCFDMnbs2Oy5555ZtWpV53H3OH3Ztddem/Hjx+fLX/5ynnrqqXz605/Opk2bOo+7v+kvBg36rzUfO7qvX7v/1XNhZ1hh1EPe9a53ZenSpUmSP/zhDxk3blyNJ4Kd99xzz+Uzn/lMpk6dmo997GNJkre97W1Zvnx5kmTp0qV5z3vek6OOOip33313Xn755bz44ov585//7J6n17vuuuuyYMGCzJ8/P3/zN3+TOXPmZMKECe5v+o13v/vduf3221OpVPL0009nw4YNOeaYY9zj9Av77rtvZ/hpaGhIR0eHf0ehX9qZ+/pd73pXWlpaOs9997vfXcvR6YPqKpVKpdZDDASv/pa0Bx98MJVKJf/8z/+cww47rNZjwU65/PLL8z//5//M2LFjO/fNmDEjl19+eTZt2pSxY8fm8ssvz+DBg3PDDTfkX/7lX1KpVPK5z30uH/jAB2o4OeycKVOm5JJLLsmgQYMyc+ZM9zf9xje+8Y0sX748lUol5513Xg499FD3OP3CunXrcuGFF+bZZ5/Npk2b8qlPfSpHHnmk+5t+4fHHH8/555+fG264IX/5y1+6fF9v2LAh06ZNy7PPPps99tgjV111VQ444IBa/zj0IYIRAAAAAAWPpAEAAABQEIwAAAAAKAhGAAAAABQEIwAAAAAKghEAAAAABcEIAAAAgIJgBAAAAEBBMAIAAACg8P8DAhbtOXqQr8kAAAAASUVORK5CYII=\n",
      "text/plain": [
       "<Figure size 1440x1152 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "#outlier\n",
    "plt.figure(figsize=(20,16))\n",
    "sns.boxplot(data=df[numerical_features], orient=\"h\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Text(0.5, 1.0, 'Histogram of the Target Column')"
      ]
     },
     "execution_count": 37,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAtYAAAJZCAYAAACN9nLLAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjUuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/YYfK9AAAACXBIWXMAAAsTAAALEwEAmpwYAABt0UlEQVR4nO3deZyNdf/H8fd1tjmzGpM1W9aIJrskikKblCJGc1da/Up32gihJKm4K3e6W++7WyE3uqubSim0SNIipEWWouzDrGe9fn+cjAbDmLnOuWbG6/l4yJlzrnNdn/PpOPOe73yv72WYpmkKAAAAQJk47C4AAAAAqAwI1gAAAIAFCNYAAACABQjWAAAAgAUI1gAAAIAFCNYAAACABQjWACqUU089VXv27Cly3/z583XzzTdLkp588kn997//Peo+/v73v+v999+PVolR9d133+n8889Xv3799OuvvxZ57M+va+TIkXrxxRdLfZznnntOffv2Vd++fdWmTRv16NGj8OstW7aU6TUcy5AhQw77f3zA/v379dBDD6lPnz7q27evLrvsMv3nP/855j5XrFihSy65xOpSAaAIl90FAICV/vrXvx5zmxUrVqhJkyYxqMZ6ixcvVqdOnTRx4sTDHrPydd1000266aabJEmZmZkaPHiwLrjgAkv2fSyffPLJEe/3+Xy6+uqr1adPH73++utyuVzaunWrrr32WklS//79Y1IfABSHYA2gUhk5cqSaNm2q66+/Xk899ZTee+89ud1uVa1aVZMmTdJ7772nNWvW6NFHH5XT6dSZZ56pBx54QOvXr5dhGOratavuvPNOuVwuLV26VI8//rgcDodatGihTz/9VDNnztTnn3+uuXPnKj8/X0lJSXr22Wc1fvx4bd68WVlZWUpMTNTjjz+uRo0aKTMzUy1bttTXX3+tPXv2aMCAAdq1a5c+//xz5efn64knntCpp5562Ot4+umntWDBAjmdTjVs2FD333+/li9frlmzZikUCqmgoEBTpkwp3P7VV18t8rok6auvvtLAgQO1a9cuNW3aVFOmTFFCQoI2bNigiRMnKisrS6FQSJmZmbryyitL1N9wOKyHH35Y33zzjXJzc2Waph566CG1a9dOI0eOVFZWln755Rede+65uv7663Xfffdpy5YtSk1NVfXq1dW0aVMNGzas2Bruu+8+SdI111yj5557TrVr1y489sKFC5WQkKAbb7yx8L46deroiSeeUCAQkCT9+OOPevDBB5WVlSXDMDRkyBBddtllxb5HDv26R48euuSSS/TZZ59p3759uuGGG/Tll19q7dq1crlceuaZZ1SzZk316NFDl19+uZYvX67ffvtNffv21R133FGiHgKovAjWACqca665Rg7HwZls+/btOyyc/vbbb3r55Ze1fPlyeTwevfTSS1q9erUGDx6sd955R4MHD1bPnj01YsQIpaam6q233lIgENDQoUP10ksvqX///rr33nv18ssvq3nz5nr99df1+uuvF+7/p59+0gcffKCkpCS98847SklJ0WuvvSZJGjt2rF599VXdf//9kqStW7dq9uzZ+uabbzRgwAA988wzGjlypB5++GG98sormjBhQpHa582bp48++khz585VQkKCpk2bVji1Y/Pmzdq7d6/Gjh1b5DmHvq7Fixdr+/bt+ve//y2Px6P+/ftr0aJFuuSSS3T77bfr0UcfVcuWLZWdna2rrrpKTZo0UevWrY/Z+2+++UY7duzQa6+9JofDoeeee07PP/+82rVrJ0kqKCjQggULJEl33nmnmjRpomeffVY7duxQv3791LRpUwWDwWJrmDRpkubPn6+XX35ZaWlpRY69Zs0atW3b9rCaWrZsKUkKBoMaOnSo7r33XvXq1Uvbt29X//791aBBg2O+rj/z+XyaM2eOFi5cqLvuukuvv/66mjdvrltvvVWvv/66brnlFklSXl6eZs6cqe3bt6tnz5664oorVK9eveM6FoDKhWANoMI5NHTNnz9f7777bpFtatasqebNm+vyyy9Xt27d1K1bN3Xu3PmwfS1btkyzZs2SYRjyeDwaOHCgXn75ZTVs2FCNGzdW8+bNJUmXX365HnroocLnnXrqqUpKSpIkXXDBBapXr55mzJihzZs36/PPP1ebNm0Kt+3Zs6ckFYaurl27SpLq16+vzz///Ig19evXTwkJCZKkv/zlL/rHP/4hv99/XH06//zzFR8fL0lq2rSp9uzZo02bNmnLli0aNWpU4XYFBQVat25diYJ1mzZtVKVKFc2ePVu//PKLVqxYocTExMLHDwRsSVq6dGnhDyM1atQonEpS2hoMw5BpmsU+vmnTJvl8PvXq1UtS5D3Qq1cvffTRR+rUqdMxX9sBB55fr149VatWrfA9UL9+fe3bt69wu/POO6/wOCeddJL27dtHsAZOcARrAJWSw+HQK6+8om+//VbLly/Xww8/rK5du+ree+8tsl04HJZhGEW+DgaDcjqdh4W4P4+SHwi9kjRz5kzNmTNHgwcPVp8+fZSamlrkxEKPx1NkP263+6i1F1fT8XK5Dn7EHwiloVBIycnJeuONNwof27Vrl5KTk0u0zyVLlmjixIm67rrrdN5556lRo0Z68803Cx//c19cLleRHh7oX2lraN26tV599dXD7l+8eLG++OILXXbZZUX6JkmmaR7Wu0MD+oFpJAf8+f/X0f5fxcXFFbtPACcmVgUBUCmtX79el1xyiRo3bqybb75Z1157rb799ltJktPpLAxbZ599tl555RWZpim/3685c+borLPOUtu2bbVp0yatX79ekvTuu+9q//79hwU3Sfr44491+eWXq3///mrYsKE++OADhUKhUtfetWtXzZs3T3l5eZKkGTNmqEOHDocF9EP9+XUVp2HDhvJ6vYWh9rffftMll1yiNWvWlKi2Tz75RN27d1dGRoZatWql999/v9jXes4552ju3LmSpL179+r999+XYRjHrKG419GrVy/l5OTo+eefLzzmL7/8okceeUSNGzdWo0aN5HK5tGjRIknS9u3b9e677+qss84qsp+qVasWHmv79u1H/K0BAJQGI9YAKqXmzZvrwgsv1BVXXKGEhAR5vV6NGTNGktSjRw9NnTpVgUBAY8aMKVy+LRAIqGvXrrrlllvk8Xg0depUjRgxQg6HQ61atZLL5SqcWvFnQ4YM0dixYwtDZOvWrfXDDz+UuvYrr7xSv/32m/r3769wOKwGDRro8ccfP+bz/vy6iuPxeDR9+nRNnDhRL7zwgoLBoP76178WmcJxNAMHDtRdd92lPn36KBgMqkuXLlq0aJHC4fBh2953330aM2ZM4Sj+ySefLK/Xe8waLrjgAmVmZmratGlq1qxZkdr/+c9/6rHHHlOfPn3kdDrldDo1dOhQ9evXT5I0ffp0PfTQQ5o2bZpCoZBuvfVWnXnmmVqxYkXhfjIzM3X33Xerd+/eqlu3rs4888wSvXYAOBbD5HdXAHCYnJwcTZ8+XcOGDVN8fLzWrl2rm2++WR999NERR61xuFdffVWnnXaa2rRpI7/fr4yMDA0bNkznnHOO3aUBQFQwYg0AR5CUlCS3260rr7xS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NARrAAAAwAIEawAAAMACBGsAAADAAgRrAAAAwAIEawAAAMACBGsAAADAAgRrAAAAwAIEawAAAMACBGsAAADAAgRrAAAAwAIEawAAAMACBGsAAADAAgRrAAAAwAIEawAAAMACBGsAAADAAgRrAAAAwAIEawAAAMACBGsAAADAAgRrAAAAwAIEawAAAMACBGsAAADAAgRrAAAAwAIEawAAAMACBGsAAADAAgRrAAAAwAIEawAAAMACBGsAAADAAgRrAAAAwAIEawAAAMACBGsAAADAAgRrAAAAwAIEawAAAMACBGsAAADAAgRrAAAAwAIEawAAAMACBGsAAADAAgRrAAAAwAIEawAAAMACBGsAAADAAgRrAAAAwAIEawAAAMACBGsAAADAAgRrAAAAwAIEawAAAMACBGsAAADAAgRrAAAAwAIEawAAAMACBGsAAADAAgRrAAAAwAIEawAAAMACBGsAAADAAgRrAAAAwAIEawAAAMACBGsAAADAAgRrAAAAwAIEawAAAMACBGsAAADAAgRrAAAAwAIuq3cYCAQ0atQobd26VX6/X0OHDlWtWrV0yy236JRTTpEkDRo0SBdddJHmzJmj2bNny+VyaejQoerevbsKCgp0zz33aPfu3UpMTNTkyZOVlpZmdZkAAACApSwP1m+++aZSU1P12GOPae/evbr88st166236rrrrtOQIUMKt9u5c6dmzJihefPmyefzKSMjQ126dNGsWbPUrFkzDRs2TAsWLND06dM1ZswYq8sEAAAALGX5VJALLrhAf/3rXwu/djqdWrNmjZYsWaLBgwdr1KhRysnJ0erVq9WmTRt5PB4lJyerfv36Wr9+vVatWqWuXbtKkrp166bly5dbXSIAAABgOctHrBMTEyVJOTk5uv3223XHHXfI7/erf//+atWqlZ555hk9/fTTat68uZKTk4s8LycnRzk5OYX3JyYmKjs7u0THdToNpaYmWP1yjnFMR8yPWRnRx7Kjh9aIZR8d/gJ5ve7iH3cYSk2Nj0ktVuP9aA36WHb00Br0seQsD9aS9Ntvv+nWW29VRkaG+vTpo/379yslJUWS1LNnT02YMEHt27dXbm5u4XNyc3OVnJyspKSkwvtzc3MLn3csoZCprKw861/MUaSmJsT8mJURfSw7emiNWPYxzWuooCBQ7OPecOw/06zC+9Ea9LHs6KE16GNR1asnF/uY5VNBdu3apSFDhuiee+7RlVdeKUm6/vrrtXr1aknS8uXL1bJlS6Wnp2vVqlXy+XzKzs7Whg0b1KxZM7Vt21ZLly6VJC1btkzt2rWzukQAAADAcpaPWP/jH//Q/v37NX36dE2fPl2SNHLkSD388MNyu92qVq2aJkyYoKSkJGVmZiojI0OmaWr48OGKi4vToEGDNGLECA0aNEhut1tTpkyxukQAAADAcoZpmqbdRVghEAgxFaSCoo9lRw+tEfOpIA9OLPZx79jR2lNQMT+eeT9agz6WHT20Bn0sKqZTQQAAAIATEcEaAAAAsADBGgAAALAAwRoAAACwAMEaAAAAsADBGgAAALAAwRoAAACwAMEaAAAAsADBGgAAALAAwRoAAACwAMEaAAAAsADBGgAAALAAwRoAAACwAMEaAAAAsADBGgAAALAAwRoAAACwAMEaAAAAsADBGgAAALAAwRoAAACwAMEaAAAAsADBGgAAALAAwRoAAACwAMEaAAAAsADBGgAAALAAwRoAbDZ7tkezZ3vsLgMAUEYEawAAAMACLrsLAIATXXy8aXcJAAALEKwBwGZ9+wbsLgEAYAGmggAAAAAWIFgDgM2WLXNp2TJ+gQgAFR2f5ABgs23bDo5xXBD/qiRpiUbbVQ4AoJQYsQYAAAAsQLAGAAAALECwBgAAACzAHGsAsFly8sF1rOuEU2ysBABQFgRrALDZxRcfXMf6RV8fGysBAJQFU0EAAAAACxCsAcBmH3zg0gcfRH6BeK/nfd3red/migAApcFUEACw2Y4dB8c4Vju321gJAKAsGLEGAAAALECwBgAAACxAsAYAAAAswBxrALBZWlq48HaTcJqNlQAAyoJgDQA269UrWHj7774LbawEAFAWTAUBAAAALFCiYD19+vQiX0+ZMiUqxQDAiWjRIpcWLYr8AvG2uLd1W9zbNlcEACiNo04F+c9//qO5c+dqw4YNWrZsmSQpFAopGAzqrrvuikmBAFDZ7dlzcIzjJ8ceGysBAJTFUYN137591blzZz377LO65ZZbJEkOh0MnnXRSTIoDAAAAKoqjTgXxeDyqW7euHnjgAe3evVvbtm3Tr7/+qm+++SZW9QEAAAAVQolWBbn99tu1e/du1a5dW5JkGIY6dOgQ1cIAAACAiqREwXrXrl2aPXt2tGsBgBNSjRoH17FOD9W0sRIAQFmUKFg3bNhQ27dvV82afOADgNV69Di4jvWj/vNtrAQAUBYlCtarVq1S9+7dlZZ28IpgH3/8cdSKAgAAACqaEgXrRYsWRbsOADhhLVjgliRdfHFA18e9JUl6VaPtLAkAUAolCtb33XffYfdNmjTJ8mIA4ESUnW0U3t7q2G9jJQCAsihRsL7oooskSaZpat26ddqxY0dUiwIAAAAqmhIF665duxbe7tatm4YMGRK1ggAAAICKqETB+s8nKu7cuVO7du2KWkEAAABARVSiYL1gwYLC2x6PRw8//HCx2wYCAY0aNUpbt26V3+/X0KFD1aRJE40cOVKGYahp06YaN26cHA6H5syZo9mzZ8vlcmno0KHq3r27CgoKdM8992j37t1KTEzU5MmTi6xGAgCVzcknH1zHulOojo2VAADKokTBetKkSfrhhx/0008/qWHDhmrRokWx27755ptKTU3VY489pr179+ryyy9X8+bNdccdd6hTp04aO3asFi9erNatW2vGjBmaN2+efD6fMjIy1KVLF82aNUvNmjXTsGHDtGDBAk2fPl1jxoyx7AUDQHnTrdvBdawf8J9rXyEAgDIpUbCeMWOG/ve//yk9PV0vvfSSLrzwQl1//fVH3PaCCy5Q7969C792Op1au3atOnbsKCkyR/uTTz6Rw+FQmzZt5PF45PF4VL9+fa1fv16rVq3SDTfcULjt9OnTy/oaAQAAgKhzlGSj//3vf3r11Vc1evRozZo1SwsXLix228TERCUlJSknJ0e333677rjjDpmmKcMwCh/Pzs5WTk6OkpOTizwvJyenyP0HtgWAyuyNN9x6443IWtYZ3vnK8M63uSIAQGmUaMTaNE25XJFN3W633G73Ubf/7bffdOuttyojI0N9+vTRY489VvhYbm6uUlJSlJSUpNzc3CL3JycnF7n/wLYl4XQaSk1NKNG2VnE6HTE/ZmVEH8uOHlojln10+Avk9UY+S32+yBiH1+tWltMXedxhKDU1Pia1WI33ozXoY9nRQ2vQx5IrUbBu166dbr/9drVr106rVq1SmzZtit12165dGjJkiMaOHavOnTtLkk477TStWLFCnTp10rJly3TmmWcqPT1dTzzxhHw+n/x+vzZs2KBmzZqpbdu2Wrp0qdLT07Vs2TK1a9euRC8kFDKVlZVXom2tkpqaEPNjVkb0sezooTVi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\n",
      "text/plain": [
       "<Figure size 864x720 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.set_style(\"darkgrid\")\n",
    "plt.figure(figsize=(12,10))\n",
    "sns.histplot(data=df,x=\"TEY\",color=\"red\",kde=True)\n",
    "plt.axvline(x=df[\"TEY\"].mean(),ymax=0.55,color=\"green\",linestyle='--',label=\"Mean\")\n",
    "plt.axvline(x=df[\"TEY\"].median(),ymax=0.56,color=\"purple\",linestyle='--',label=\"Median\")\n",
    "plt.legend()\n",
    "plt.title(\"Histogram of the Target Column\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Text(0.5, 1.0, 'Cumulative of the Target Column')"
      ]
     },
     "execution_count": 38,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAtwAAAJZCAYAAACa1OICAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjUuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/YYfK9AAAACXBIWXMAAAsTAAALEwEAmpwYAABgVElEQVR4nO3deXwU9cHH8e/s7JFkkxACkTscSpTDcIMo4FXF+0BFEkvrU62ttVipWrzReqCPShUsVq22fUBA6lEPvBFBDpFDQALIfR9yk3PPef6IpoQzgd2d3c3n/Xr5kkxm9vfdnxvy7fQ3M4ZlWZYAAAAARIXD7gAAAABAMqNwAwAAAFFE4QYAAACiiMINAAAARBGFGwAAAIgiCjcAAAAQRRRuAAkrFArpH//4hwYMGKArr7xSl1xyiZ5++mn5/f6oj71p0yZ16dLlmPv9+9//1uuvvy5JmjBhgl5++eVoR9OLL76oc845R/fee2+17cXFxfrFL35R9fWpp56q3bt3H/c4gwYNqpr3du3a6corr9SVV16pO++887hfsyYWL16shx566IjfX7BggW666SZdeeWVuvzyy3XLLbdoxYoVx3zde+65R6+++mokowKAJMlpdwAAOF4PP/yw9u3bp3/961/KyMhQWVmZ7rrrLt1///16+umn7Y4nSZo/f77atm0rSSooKIjJmG+++aaeeeYZde/evdr2ffv26bvvvovYOBMnTpRU+T8+Lr/8cr377rsRe+2jWbVqlbZv337Y782dO1d33323XnjhBXXs2FGS9N5772nw4MH66KOPlJ2dHZOMAHAgCjeAhLRp0ya9//77mjFjhtLT0yVJaWlpeuSRR7RgwQJJlWcs27Ztq5tuuumQr8877zxddtll+vrrr7Vv3z7dfPPNWrBggYqKiuR0OvXiiy+qUaNGOu+88/T888/r9NNPl6Sqr+vXr1+VZefOnXrooYe0a9cu7dixQ82aNdNzzz2nBQsW6IsvvtDMmTOVkpKi3bt3a8+ePTrvvPP01FNP6f3335ck7d+/X+eff74+//xzVVRU6M9//rO2bt2qQCCgSy+9VL/97W8Pef/btm3Tww8/rM2bN8uyLF111VW6+eabdccdd2j79u26//779Yc//EGXXHJJ1TH33nuvKioqdOWVV+rtt9+WJI0ePVqLFi3S3r17ddNNN+mGG26QVHlmfsKECQqHw8rKytKDDz6ok08+uUb/bdauXas///nPKi0t1Y4dO3Taaafpueeek8fjUceOHXX++edr+fLleuaZZ7R7924988wzcjgcateunWbNmqXx48erefPmh82QlpamUaNGqbi4WPfee69GjBhRbexRo0bpd7/7XVXZlqQrrrhCHo9HoVBIkvTGG29o7NixcjgcatiwoR588EG1bt262uuceuqpmj17dlVB/+nrlStXauTIkWrSpInWrl2r1NRU3XLLLRo7dqzWrl2rCy+8UPfdd5/mzJmjv/zlL2rRooVWrlypYDCoRx55RN26davRHAJIMhYAJKCPP/7Yuuaaa466z7Bhw6y///3vh/363HPPtZ544gnLsixr8uTJ1mmnnWYtW7bMsizL+t3vfme9+OKLVfstXry46jV++nrjxo1W586dLcuyrH/+85/WSy+9ZFmWZYXDYevmm2+2Xn311UPGHDVqlPXII49Y4XC42uu+/vrr1p133mlZlmUNHjzYmjJlimVZllVRUWENHjzYmjx58iHv7YYbbrBee+01y7Isa//+/dbll19uffDBB4fN/JMDM1uWZeXl5VXlLCoqsjp27Gj5/X5rzpw5VmFhoVVWVmZZlmV99dVX1kUXXXTEeT74dZ988knrP//5j2VZluX3+63LLrvM+vjjj6vGfOeddyzLsqzdu3dbPXv2rJr3t99+28rLy7M2btx41AxvvfWWdcsttxw2S+fOna2VK1ceMeusWbOsn/3sZ9auXbuqXuviiy+2wuFwtf9WeXl5Vfsc+PXXX39ttWvXzioqKrIsy7Juuukm6/rrr7d8Pp+1a9cuq0OHDta2bduq9lu6dKllWZb16quvWjfccMMRcwFIbpzhBpCQHA6HwuHwCb3GhRdeKElq0aKFGjZsqNNOO02SlJubq3379tX4dX75y19q3rx5+sc//qF169Zp5cqV6tSp0xH3NwxD11xzjd555x2dfvrpevvtt/WnP/1JZWVlmjt3rvbt26fnn39eklRWVqbly5dXO1NdVlamBQsW6LXXXpMkZWRkaMCAAZo+fbouvfTSWs3BZZddJklq166d/H6/SkpK9OWXX2r9+vUaNGhQ1X779+/X3r17lZWVdczXvPvuuzVz5ky98sorWrdunX744QeVlZVVff+npS7z5s3TySefXDXvV199tR577DFJOmqG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5bZSqFUrWekZNgcGAACoOxKqcIfDYT388MP6/vvv5Xa79dhjj6lly5Z2x4q4I67DLi+XVq+WVqyQli2TFi+WFi2SVq6UwuHKfdLTpewGCvQ8Q+GmzRRq0lRK45JHAAAAuyRU4f7888/l9/v1xhtvaOHChXryySf14osv2h0rckIhObZvk7lqjyqeeEpG8X459u2TY89uGXt2y9i/X8YBu4frZSmck6Nwr94K55ykcM5JSnt5jCqG3W/bWwAAAEB1CVW458+fr759+0qSOnfurCVLltic6CjCYRmlJZUlef9+GcXFchTvk7F7txw7d8qxa6eMnTvk2LVTjp075Ni2TY5tW2WEQpKklB9fxvJ4FK6frXCzFgp3rK9w/WxZ9bMVzq4vuT2HjutwxO49AgAA4JgSqnCXlJQoPT296mvTNBUMBuV0xtfbqHftlXJ99WXVGurDcjqlnBzppJMq/93utMp7X7doIeXmqmzSW7IyMiWPRzKMI78OAAAA4pphWUdrhfFlxIgR6tSpky655BJJUr9+/TR9+nSbUwEAAABHllDrD7p27VpVsBcuXKi8vDybEwEAAABHl1BnuH+6S8mKFStkWZaeeOIJnXzyyXbHAgAAAI4ooQo3AAAAkGgSakkJAAAAkGgo3AAAAEAUUbgBAACAKKJwH6dFixZp8ODBkqT169eroKBAhYWFGj58uMI/PmZ90qRJGjBggAYOHKipU6faGTduHTiPkvTZZ5/pzjvvrPp64cKFuu666zRo0CC98MILdkRMCAfO47Jly1RYWKjBgwfrpptu0s6dOyXxeTyWA+dw1apVKigo0KBBg/Twww8r9OMDqZjDYzv4Z1qS3n//fV1//fVVXzOPx3bgPBYVFalv374aPHiwBg8erA8//FAS81gTB87jrl27dOutt+qGG27QoEGDtGHDBknM47EcOIdDhw6t+hyed955Gjp0qCTmsEYs1NrLL79sXXbZZdZ1111nWZZl/eY3v7G+/vpry7Is68EHH7Q+/fRT64cffrAuu+wyy+fzWfv376/6M/7r4Hl89NFHrf79+1t33HFH1T5XXHGFtX79eiscDls333yztWTJErvixq2D5/GGG26wli5dalmWZU2YMMF64okn+Dwew8FzeOutt1rffPONZVmWNWzYMH6ma+jgebQsy1q6dKn1i1/8omob83hsB8/jpEmTrFdffbXaPszjsR08j8OGDbMmT55sWZZlzZ4925o6dSrzeAyH+5m2LMvau3evdcUVV1jbt29nDmuIM9zHITc3V6NHj676uqioSD179pRU+TCeWbNmafHixerSpYvcbrcyMjKUm5ur5cuX2xU5Lh08j127dtXDDz9c9XVJSYn8fr9yc3NlGIb69Omj2bNn25A0vh08jyNHjlS7du0kSaFQSB6Ph8/jMRw8h6NHj1aPHj3k9/u1Y8cONWjQgDmsgYPncc+ePXrmmWd03333VW1jHo/t4HlcsmSJvvzyS91www267777VFJSwjzWwMHzuGDBAm3fvl033nij3n//ffXs2ZN5PIaD5/Ano0eP1s9//nOddNJJzGENUbiPQ//+/as9Tt6yLBk/Pn7d6/WquLhYJSUlysjIqNrH6/WqpKQk5lnj2cHzeMkll1TNo1RZuNPT06u+/mluUd3B83jSSSdJqvzlMm7cON144418Ho/h4Dk0TVObN2/WZZddpj179qh169bMYQ0cOI+hUEj333+/7rvvPnm93qp9mMdjO/jzmJ+frz/96U96/fXX1aJFC/31r39lHmvg4HncvHmzMjMz9c9//lNNmjTRK6+8wjwew8FzKFUuzZk9e7YGDBggiZ/pmqJwR4DD8d9pLC0tVWZmptLT01VaWlpt+4EfSBzb4eYwMzPTxkSJ48MPP9Tw4cP18ssvKzs7m8/jcWjWrJk+/fRTFRQU6Mknn2QOa6moqEjr16/Xww8/rD/+8Y9atWqVHn/8cebxOFxwwQXq2LFj1Z+XLl3KPB6HrKwsnXfeeZKk8847T0uWLGEej8PHH3+syy67TKZpSjr872rm8FAU7gho37695syZI0maPn26unfvrvz8fM2fP18+n0/FxcVavXo1j6KvpfT0dLlcLm3YsEGWZWnGjBnq3r273bHi3rvvvqtx48Zp7NixatGihSTxeayl3/72t1q3bp2kyrM1DoeDOayl/Px8TZ48WWPHjtXIkSN1yimn6P7772cej8NNN92kxYsXS5Jmz56tDh06MI/HoVu3bpo2bZokae7cuTrllFOYx+Mwe/Zs9evXr+pr5rBmnMfeBccybNgwPfjggxo5cqTatGmj/v37yzRNDR48WIWFhbIsS0OHDpXH47E7asJ55JFHdNdddykUCqlPnz7q1KmT3ZHiWigU0uOPP64mTZpoyJAhkqQePXro9ttv5/NYC7fccovuueceuVwupaam6rHHHlNOTg5zGAHMY+09/PDDevTRR+VyudSwYUM9+uijSk9PZx5radiwYXrggQc0ceJEpaen69lnn1W9evWYx1pau3Zt1ckciZ/pmuLR7gAAAEAUsaQEAAAAiCIKNwAAABBFFG4AAAAgiijcAAAAQBRRuAEAAIAo4raAAFCHPPnkkyoqKtKOHTtUUVGhFi1aqH79+poxY4Y6dOhQbd8xY8ZowIABevLJJ9WtWzdJ0tKlS3XnnXfqzTffrPYESQDAkXFbQACog95++22tWbNGd911lzZt2qQ//vGPmjRp0iH7ffPNNxo+fLjeeecdORwODRo0SMOHD+ee+ABQC5zhBgAcUc+ePXX22Wfrr3/9q1JSUnT++edTtgGglijcAACtWrVKgwcPrvq6Q4cOuueeeyRJQ4cO1fXXX6+srCy9+uqrdkUEgIRF4QYA6JRTTtHYsWMP+z2Px6Pzzz9fDRs2lGmaMU4GAImPu5QAAAAAUcQZbgDAIUtKJOmJJ55QixYtbEoEAMmDu5QAAAAAUcSSEgAAACCKKNwAAABAFFG4AQAAgCiicAMAAABRROEGAAAAoojCDQAAAEQRhRsAAACIIgo3AAAAEEX/D8KUpYNw4HaIAAAAAElFTkSuQmCC\n",
      "text/plain": [
       "<Figure size 864x720 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize = (12,10))\n",
    "sns.histplot(df[\"TEY\"],kde=True,bins=40,color=\"red\",cumulative=True)\n",
    "plt.title(\"Cumulative of the Target Column\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Unsurprisingly, Mostly none of the features are on the same scale as we already saw in the previous section."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "+ ### Multivariate Analysis<a class=\"anchor\" id=\"4.2\"></a>\n",
    "\n",
    "\n",
    "Let's now proceed by drawing a pairplot to visually examine the correlation between the features."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 96,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAXIAAAEFCAYAAAD+A2xwAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjUuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/YYfK9AAAACXBIWXMAAAsTAAALEwEAmpwYAAA9mElEQVR4nO2dfZQU5ZX/v9VdTfcA0wMkwzRHCCMISFwdEwOchB2M8qIDyyoMMgg7JuiJm2DWZQWjcFYYMUFYJZ5jdEwmCZpjWM6uIBIONJGXPyDEIO6PjK7bDgJBAZ1GAZkZmOmZ7q7fH0211dX11FtXdVd13885niPT3fXS/dR97nOfe7+XEwRBAEEQBOFaPIW+AIIgCCI3yJATBEG4HDLkBEEQLocMOUEQhMshQ04QBOFyyJATBEG4HL7QF0AQeujr68Ntt92G66+/Hr/5zW8AAGfOnMH06dMxduzY9PsEQcB9992HefPmWXbulpYW7Ny5EwDw8ccfY/DgwSgvLwcA/OIXv0BzczMOHTqEIUOGZHyuvr4eN910E773ve9h8+bN+PrXv55+bdOmTXj11VexdetWDBgwwLJrJUoUgSBcwM6dO4UHHnhAmDRpknD8+HFBEATh9OnTws0335zxvvb2duFb3/qWEIlEbLmOf/qnfxLC4XDG3x577DHhN7/5DfMzL7zwgjBr1iwhFosJgiAIf/vb34SJEyfado1E6UGhFcIVbN68GVOnTsXMmTPxu9/9jvm+qqoqjBw5EqdOncr4+7Jly7Bx48b0v//zP/8TS5cuxeXLl/Hwww/jrrvuwpw5c/Dv//7vSCaTll77j370IwwaNAjPPfccEokEfvKTn+CRRx7B9ddfb+l5iNKFDDnheI4fP46jR4/izjvvxN13343t27fj4sWLiu89evQoPv74Y9TU1GT8/Z577sG2bdvS/962bRvmz5+PPXv24PLly9i+fTu2bNkCADh9+rTha3zllVdw1113ZfzX1tYGAPB4PHjmmWewbds2rFy5EsOHD0dDQ4PhcxAEC4qRE45n8+bNuO222zB48GAMHjwYw4cPx3//939j1qxZ6OnpwV133QUASCQSGDx4MJ555hkMGzYs4xiTJk1CLBbDe++9h7KyMly4cAHf/va3cebMGTz33HNobGzEd77zHXzve9/DyJEjDV/j97//fTzwwAPM14cNG4bly5fj5z//Od58803DxycINciQE47mypUr2L59O/r164fbb78dANDV1YXf//73uOOOOxAIBLB9+3bN43Ach3nz5mH79u3w+XyYN28eOI7DiBEjsGfPHhw+fBh/+ctfsHjxYqxZsyZ9LisZMWIEhgwZgoEDB1p+bKK0IUNOOJodO3Zg0KBB+OMf/wiv1wsA6OjowG233Ybdu3cbOtacOXPSIY3NmzcDSMXK/+d//gfPPvssamtrcf78efzf//2fLYacIOyCDDnhaDZv3ozFixenjTgABINBNDY24pVXXjF0rMrKSnz9619HPB5HVVUVAODuu+/G22+/jZkzZ6KsrAzDhg1DY2Oj4et85ZVX8Ic//CHjbzU1NVizZo3hYxGEUThBIBlbgiAIN0NZKwRBEC6HDDlBEITLIUNOEAThcsiQEwRBuJy8Z60kk0kkErS/ShAEYQSfz8t8Le+GPJEQ8MUXV/J9WoIgCFdTWVnOfI1CKwRBEC6HDDlBEITLIUNOEAThcsiQEwRBuBwy5ARBEC6HRLMIokgIR6JoPngK0c4Yqsr9WFJbjbrxVYW+LCIP5F00q68vQemHBGERovFu74wpvl5fE8Lj08Yqvka4C7X0Q/LICcJlaBlvKVtb21FzTQV55kUOGXKCcDjSkEm534vuviT6kvoX0s0HT1luyCmM4yzIkBNEAVEyiACYhrsjljB8jqgOz90I4UgUa9/8ED3xJACgvTOGtW9+CABkzAsEGXKCKADhSBTP7jueYZjbO2NYE24Dx3E5GW45VeX+nI8hpfngqbQRF+mJJ23x/Al9kCEnSp58hwnkHq2UuADAwvyDAO9Je/lWwfLwrfb8Cf2QISdKCnm8uS+RRHf8S8MpDRMAsMXAK3m0VhAq92PyqME4dPKirZNSVblfcaPVas+f0A8ZcqJkkHvCrLBFTzyJDftPIBZP2hIHztVz5TlggJ9HR09c01iHI1HMbjlsqWFfUludtaKww/Mn9EOGnCgZjHjCl3riWX/riSfRFG4DkJsxZ3m0QMpIS2Pk4t/0Gm4pdm1Kip/NdbViJKRFWTLqUEEQUTJM2HDAkuMEeA9Wzhhj2pCwYuQVAR7Lbh8NgG0k5QZNLZQy9YVDiquOULkfOx6cpOs65efa2/Z5epIL+r1YPvU6U9+D0nfA+l61vq9SMehqBUFkyImSYdLPD0BP+nWA9wCCgB6VTlZ6jSELIwZZ+hnWJqmIz8OhzOdRzXbhALy9bIrqdekpNgJSq4VVdeMMTzSzWw4rnkPpe2W9V+n8xQwZcoKAPo+8IsBj2riv4o1326HVkXDNTGsMiF6PU82gGYE1CemZKFgEvBw4D4fuPvXPil736l1tYH29oXJ/huFftatN9ZhBvxf7fjw5K6UzV4/daeEcMuREyROORDUNApDyVv1eTtUbF8k1xCKiZqCl55i44QDT+BlhzcxxALLDN0Y88VwIXc1usfJcQb8XXbEElKYRM3ozRkI/+YIMOVEysColzXqaWqh5t3q9Oa2VgocDmurGWWJoy3gO/Xiv4mZuMWN09WQk9JMvyJATJUEuoQGzcACenDkubbSDAR698URGbjqQil17OaQ9fXHZ33r2Era2tmuexwMoepvFQkWAt3VyEQ2w3gmWtfpR21+wGzLkhCMws8FnhGkv/jnvnmbQ70VvQsjr5FGMhFRSMq0i6Pficm8iY+/D5+HwxJ1js8adWrgrVKB4uZohpw5BRF5Yt/cYVu1qQ3tnDAJS8dGtre0Z/1775ocIR6Kmjh+ORAsSLuA4ruiMuBjDzif5KO/viCWyNrD7kgJW72rLGndLaqtT2UsK5DpW7YAMOWE74UhUV/hAFF4yg9nP5Uoxxpp3PDgJR/IcPshrWEDh3HLDXDe+CitnjGFOarmMVTugyk7CdowMeLOeGQk2WYdVaY5uQqreKA8BsnDSmCNDTliOkkSrXjgutdGkVNGoliMctHmzrJQoNSMuEu2MKcoasBCNvBPyzWmzkzANK9VvTbgNcQtGlZi3q3bMoN+L6ddXYvu77ZackyhdjOS3ixWlQHZqq1355pS1QlhOIVL9CMIuxDRStYpTpfezcvvtyDfP2ZDffffdKC9PHWT48OF4+umnM17v7u7G4sWL8bOf/QyjR49WPRYZ8uKgFOOoRHET9HsNhQMDvIfpyNiRb65myDVj5LFY6mF99dVXFV9/7733sHr1akSjzknFIezHSRs9BGEFRvd0euJJeDgoCrHlu8mGZvrhBx98gO7ubtx///2477778Ne//jXj9d7eXrz44osYNWqUXddIOIxwJAqOK/RVEIT9BLypilwWSQFZ+eaFaLKh6ZEHAgE88MADuOeee3Dq1Cn84Ac/wO7du8HzqY/ecssttl8k4RzE2LgeOViCcDuD+vfDld4401sv83mwYvqYgmetaBrya6+9FiNHjgTHcbj22msxaNAgfPbZZxg2bFg+ro9wGHb1myQIJ6IVQuzuS6L17KWCCWmJaBryLVu24NixY2hqakI0GkVXVxcqKyvzcW1EHmHlwkobDbDigQRRrFTpSEnc2tqOmmsqCqpVrmnI582bhxUrVuDee+8Fx3FYu3YtwuEwrly5goaGhnxcI2ExSuJVO98/l9Xbccd7n+LI6Y7058iIE6WENNatlWprVWNus1AeeYlB+d8EoY28H6mexiR2a5XnlH5IFBcU4yYINiyJ2rrxVZqGvJApuWTISwzK/yaIbKQdhMKRKGa3HM7aL9JqfpHv3HEpJGNbYhRysBGEU5Ea8bVvfqiok7/s9tHweZSTyguROy6FDHmJoSSYH+A9qK8JIVTuB4fU8rK+JlSYCySIPFMR+DIwoRR6lErcPnHn2LS4lmjTQ+X+gjZlBii0UnKIg01PAYOeZhAE4WZ8Hg7Lbv9SH4oVehT/Xje+qqAGmwUZ8hJE72DMRx9FgigUShubVYwx7/SQJIVWShBxM2fihgOY3XKY2XtwSW01MyZIEG6GQ6qlndyhYYUeCxn/1gMZ8hJDbTNHjhgTNEIhGvcShFFYHnbd+CrMumFoOv7t4YBZNwx1ZDhFChnyIkPL21bbzFGibnwVynhtr9zLpYooop0xkA9PFBLpJmR9TciQhx2ORLHz/XPpKuakAOx8/xxz1eoUKEZeRCj1G5SXDmtt5ijRo6OHWkIwrudMEHZw+JHMhg4111ToVifUylpxKmTIiwg9g9DMZg5JrBBuIhyJZhhdrc19qfYQa6w7vZCOQitFhB5v28xmDu13Em6CFSZUQr5nxMLpWSuu9shZ0qulSjmj56A4CMXvS9qiiqUtIeWW4cEMFUSCsAueA3RE8lQx4j3r0R5yQ9aKKw15OBLF028eQ7fkF1eKBxcrShMYkBK5l8NzKS9cHj9PCqnXuvsSWL2rDc0HT2HyqME4dPJi1sTYdu5yPm+PKFFEp0JLnEqLYIDP0koBlIvgtOokPFxmMoBTbYvrZGzDkSie2n0MfQxxbA8HCAKK1kNXkqEN8B74eY+ioI8YFeFMNoUo83kUJwiCsJIA70mXuecitezzcBAEIcOr5zmA47gMmyGeryncpvhccAD8vCfrOStkKb6ajK3rYuTNB08xjTiQMlZa+dFuhrWhyVJlE67+Z7YpBBlxwm6kXq+4UblyxhjNmgQOKdVCqUZQmc+TFZqJC8iyGeL5WM+FcPU9Sp9xIq4KrYQjUUMl425IG9KLtOUaQRQTojGVh0frxldh4oYDzE3IqnJ/VkbKxA0HdJ832hkzLENhNnvF7v081xjycCSKNWHjsTOnpw3pQc9SM+j3ojchUNMIwtXoTZcFkLEBGY5EsWH/CUOpsqJBVQpV9vNyqokDauhtpQhYF3N3TWil+eApU7vZTk8b0oPWznqA92D51OvSy1EO7JRBSiUknI5WuiwA1NeEMjTEn9p9jBle5DlkaQaJmSjSMI4Ynlk5YwyWT73OlOaKkgTG1tZ228M0rvHI9XjWAdnmBABMHjXYrkuyHT3hFHn6oFwgX2mzZvWuNiryIWyjjOfQj/eioyeuusnuYbwmdb6kssvtnbH0Zw6dvJiOp6vtm3k4YFXduPQxlEIbagVDRsMhRlopWhktcI0hV1tiASmDNnnU4CwN7Z3vnwMAxbQ6J6N35769M4Zn9x1P/1s68GbdMDR938EAD0EQsHpXG7irmT0sAl4O/XgPldwTphDAYdq4r+LQyYuqz2xSyHa+5F6v3JlRiqerGURBQJaToxcz2uNGjLOV0QLXhFaW1FaDpd3k83BYUluNQycvZr3WE09ia2u7LrU/J2FkZu+IJdC0qw1rwm0Z97nz/XNYUluNJ2eOQyyeREcsoZnBEuA9WHnHWPTv55o5nnAY0mdODTGMIQ9rKLVdY52n+eApVYOY79CqkfNZWWTkqjzycCSKZ/cdz/AUy3we+DwcOq8aKb2Eyv3Y8eAkU9eRD9R2640gpnApPQziMlWpytOq8xOEEnpysme3HNacDDgAc2tCit2s+KthlXyuvo3kwB9ZNkXzPVLU8shd5XbJlzril2YmBOD0bBatUJJetJadrMFk1fkJQg4HfdWSep7Rcr83HT6VUsZzWDFjbN5DqPJWiqw9Aqt1+11lyOUYCT/IcXo2i1JalBmqVDxy6XeglTJFEFYh2rX2zhjWhNuwYf8JdPTEM/avwpGo5l4OkCr0URqjFWX9CrYPJnU4WUkHVmu3uNqQm/Wq3SCCIw4EVgmxHqT3qTaYlHTMd75/DrNuGIo9H3xGm56EbcQFpNMGxf2r1rOXMpo7qMGqPHbKittIs/NccLUh17v8D3g5DOrfz1VZK8CXg0CvZ14R4FHm8zLvU5rCJV3assr+93zwGXoTFCkn8kdPPIlt77abdl5EWCvuQiimmsl+MYquzc67774b5eWpQPvw4cPx9NNPp1/bv38/XnzxRfA8j/r6esyfP1/1WLmKZgHGytULseFhNXrCHkqbR3o/R+ETwg7UKiTtPq/SRqpabYUb7IPaZqemIY/FYmhoaMAbb7yR9VpfXx9mzpyJLVu2oKysDPfeey9++ctforKyknk8M4ZcapCCAR6XY3HFKk8xl9xtOeNm0PIsjOyeswozCMIsHg5oulqIY8Vej3hMpXEa9HvRvx+v+cyzsmCcnsEmklPWygcffIDu7m7cf//9iMfjeOSRR3DzzTcDAE6cOIGvfe1rqKioAADccssteOedd1BXV2fNlSPbILHKcN3yY1iF1nLNyEYwqzCDg5Ch+U4QevAgZcSl4zOXvR4gtbK+66aQ4opy+dTrdDlrZvrVugXNgqBAIIAHHngAv/3tb/Hkk09i+fLliMdTxrSrqysdcgGAAQMGoKury9IL1GuQiuHHsBIj3werMKMf77XvAomiRf601o2vQlPdOEXNFL0M8PN4fNpY1QIiLVhxc6dnsOlB0yO/9tprMXLkSHAch2uvvRaDBg3CZ599hmHDhmHgwIG4fPnL7jGXL1/OMOxWoNcgFcOPYSW6N4Il4kHyB2J1jp1aCPdgVM5VC7l8NEszRS8dV1fiuWwcspQOnZ7BpgfNKXLLli1Yt24dACAajaKrqysdAx89ejQ++ugjfPHFF+jt7cU777yDb3zjG5ZeoB4DXSw/hpWwmizX14Q0PZpwJIrZLYepsrNEEMOSVhapKDlgdeOrsOPBSTiybAoOPzIFR5ZNQX1NSNfxBKRi3HqkNcTxO3HDgYzPsJQOi2EPTXOzs7e3FytWrMAnn3wCjuOwfPlynD17FleuXEFDQ0M6a0UQBNTX12PRokWqJzS62am0aefzcCjzedAZSxT1hmaumEm1CkeieDLcBso6LB04AG8vm5LW/LdiW0Tc7NQab3rK8KVoZZm4PTNFjZyyVqwm16wVMtzGMPrdTXn+T9TercQQPfJ1e48papaw4ADVVZseA2pG00ctsYE1MeidWJyM6w05YQ693onU2JMj7k4mjAji9Bcxw3FucTwAMNS9vr4mhMenjQWQGj9qWSlyzXwpRj1y4MsVhBJqE4PbPfOiar5cqrDifmqwKjabwm3pz8s7mhDu5L1Pu7CkthpHlk3BhBFBXZ+Rxog37D9h6HxSyei68VWqmijtnTGs2tWGaS/+OWvcmmn8Yla21snNk3PF1SX6pYKSFoooqg+wdRxYGT9JAXhq97EsSWDCvYhGqvXsJRw53aH6XqlnKjoIrPoMFvKxpSdL6lJPPKNXZTgSVVQu1Lp2qUaQfOxric0Va5oyeeQugOVZP7vveFZ/QLFphqgex6IvKRgy4tTq0/lEO2PY9i47xi3P1NBq3KCG3PNl9daUo6Xxowbr2qVjHwBWzhjD7E1brGnK5JG7AJYXoWSIRQPfmxAsKbv3cMCcm0KouaYi5+o8wl4EqMu+VpX7Ee2MmTakUuTpvvI8cTXE8azXO64I8Nj70Hcy/sZybpoPnkpvhBZrzrgSZMhdgNEmD1aES5QeHsA63Qwi/4hjSPRezf6OHJSbQYjFOhM2HFD9vOgV6x3Xl3rimN1yWFfYUDxevuRjnULJGHI3pzAuqa02lFFgBR0KMVPx+6LYuvvpiSdNi6UJUFbW1KNdL/WKjTRPkYZO6sZXqU4C4Ug0Pam45RnPlZKIkbPiaU5vwCxSiMHIiiXWja/Cvh9PxpqZ45hxSMIerP66cwmTrdqV2eh7a2u7phGvCPAZ6X/ySsuKAM9ssA5kxtfVsl2KNTNFjZLwyNXiaW6ZsXPVwvB5OAiCoKtqz+fhNGOJRpteELlj1u6qyb/ma2VVEeCx7PbRWc+bUh9etTh7e2cMU184BE5lJ79YM1PUKAlDXgzylbn08BQLMoAvY4blfi84jstKO2M9cEqI79mw/0TWcXgO4DgOfXnYHaXmGOowZYrV0posRkw9bD17SbFfgDxUUxHgmSmRWpNPsWamqFESlZ1uF5QXMVOBma97VNqDAPRlMZhB6k1WBHj0xhOknc6gIpDy10TDGPR7sXzqdVi9q63gRWAB3oNZNwzN0hnPxRFYM9PdpfgsSr5EvxiFdPSUNud6j1ZtEFspxgSkDFFvQrl7OpGJh0tthMm/+/qaEA6dvGjLJGsUVuhHS8tFiQkjgmief7OrkxtYlLwhB9ydtaKE0uTEcykB/o6eeM73aPXkF45EKdvFYZT5PJYIpDmpVSDLw3e74waQIS9a7Jyc7FSRMyOURDiTI1flb3PZ9LZ6ImAdz22hVDk59ewknIudebJqOi3SfF4p6/Yew7Z325EUvqwIFRXyRMKRKBnxImHCiGDamTBrxFkedC6wJgU3JTcYhQw5oYhawYU8dTMcieLpN49lbDYmBaS1raVyp1KxL8Ld/L8zHXjndIfpDVMfh3S4w4gEhJYHz4y5c18WCxUbJVEQRBhHSwRJNPKicWZljEhFnHLx3AjnkRDM57YDQALIKA6ac5N227cA70FT3TjVtnS3DA8qjl1xNemWQkAjkCEnFBGr7liIVZ1axlnqGRXz0rYYsTvLXDo2WJK2E0YEFXtsqo2l9z7twqwbhipWHrM0yc3o/TsJCq0QTOrGVzE1XsSHUMs4Sx8mo+JfRGGxOwtCOjZYDsHpL2KKG5RaoT9xr0YJ+ZhV0/t3SxiGPHJCFdYSNiRRsFNDulxWCtcEeA8mjAiSbkuRwnNgdiySjg2j1ddaoT+1GLp8zKpJeLgFMuSEKizjK1WwU3qgOGT2dQSyRZLEpXLz/Jtx+BHlHoyEe+GQKkQ6/UUMXx2Qvfg/dPJiOoTBcgjUxNvUGkiwUNIkt1PCI18hGwqtEKpo6Tob1X1mpUy6LSZJaCM6xWoCWGIIg6UldKU3npVpIq2fKPd70d2X1FXKz9IRYoVpjGi2sCQq8hWyoYIgl1FsFaqA9SX8hLsQC3XCkaiiAJu8x6haRTOnkprIqu4MR6J4avexjMnA5+HwxJ1jdT1brCpoP+9RFP4yW5ikVhBEoRUX4XZddRbP7jtORryEEUMYdeOrUObzZr3eE0+iKdyGiRsOoCncluW1xwWgN55AVblfNTauFveW+7NG/FtWjJ2l3mhH9haFVlyAmkZzIXTV9a4KpJWeInJJXcpiIcQQhlrVrziGWPa1Oy6gW8dYau+MpVvRiSqQzQdPZTkScSHVPKP54CnNVa9Rw2yHzC6FVhyOXh2LI8uyNwu1pGXFCriQgjFWauXFUssTy6ylOtPxZAKfX1b2SAhCCs8B3xgexJHTHQU5t9ZqUB5mkT8b3X0JRe9bSaUzF/EuEs1yMXoEpjwc0lkfWh1W9OD1AAkqwCSINGU+Dw48/PfMGL1cO1002IB1DaDJkDsYrTDFxA0HCi7+TxBEqmGFHiepjOfQj/daIictJWf1w/Pnz2Pu3LnYuHEjRo8enf77G2+8gd/+9rcoLy/HnDlzcM899+R8sXZS6IwPpXCFVPVNnp7k9k1MgigmxGdXi+64gO54KtSSrypRTY+8r68PS5cuxfHjx9Hc3Jw25BcuXEB9fT22bduGYDCI73//+1i7di2GDx+uesJCeOTr9h7D663tip6tmFsKQDEmrGb01WLQSj0J5SlOBEG4C7NN0K3QQs/JI1+/fj0WLFiAlpaWjL+fOXMG119/PQYNGgQAuPHGG9Ha2qppyPPNur3H0nKqSlzqiWPVrraMtlLtnbGMzyjNqkr6DE/tPpbRqV76uQ37T5ARJwiXI19F68VuwTjVPPLXX38dQ4YMQW1tbdZrI0eOxPHjx/H555+ju7sbb731Fq5ccU7sWyyNVTPiUrRMrDwHVSl3tC8pZO2Ai59j5ZQSBOEeDp28aEoawI6UQymqoZVFixaB4zhwHIdIJILq6mq89NJLqKysBADs378fv/71rxEKhTBw4EDceuutmDZtmuoJ8xFaybX1lBqhcr+hLvaAuSayBEE4Dw7A28umGEpCsKpfqOnQyqZNm9L/39jYiKamprQRj8fjaG1txaZNmxCPx7F48WL827/9W04XahV2NjAwEx+rKvfjSm9csfGwjwP6yMoThCsQPWu9ksxKNRp2YLiyc8eOHbhy5QoaGhrg8/kwd+5c+P1+LF68GEOGDLHjGg1TqAYGPg+XESMHMtXW5HoiPAf09/MUdiGIAtHPy6E3oc+Tkqt+qukD8RywKscm5UYoyjzyfHdp5wDNrBVAOctl9a425hJNLCEGYFuoiCBKlTUzx6UzypTyw8t8Hvg8HDpjCcXMtXAkimf3HU+vtMUQql1eeMkVBNkZI5eTa1oRa9KRH1c62KSl9WZ30QmilBFj3W4i54IgtyHOhEqSmFJYpbUs+Uk5SiL1RlHSYVY6LkvHGwBqrqnQrYtCEIT9WST5pig9cilyEXqO4zJKZ4HsUAiQHcpQEoayavlkV8WpXl0IFmU8h3hSQB85+0QRYVUWSb4pudCKFRS6nN8q9CogKqEWNpr08wOq2s9EaZCv1Fox7sxqBq6FmtKnWyBDTqjCyolViyNqVcwShFVIPehpL/7ZUJZXvrNH7IQ6BBGqGG18CwCPTxuL+pqQ4Qo3gjCC2KAbSCUGGE3VjQtA69lLdlyaoyhJj7xYwiYsjN4fq+egkThivlM+ieJG2szBiiw0MdXQzZRc1ooaSmJXemQm3WL8zdyf+Pdc7m/yqMEUaiEsoy8ppFsYsiq1g34v+vfjdTkQdrdDLLR9KDlDzmqUqvZDmzX+hcDM/QHq6Y1ahCNR7Hz/nKnPEgSL9s4Y1u09xqzU7owlsO/HkzH1hUOK8hdS7Kz2doJ9KDlDzvpB1X5os8Yx36g1rxWbztqxe2+ntg1R2mxtbUfAy6FHoYxe3MPp09GX0M68cSfYh5Lb7DSzsWfG+Ocb0SvQIinTSreiC5HR76GM5+CjXVJCJ0pGHEiF88KRKLo1uif7PFzOhXtqOME+lJwhX1JbjQCfedtaFZpmjH++MeMVyzXWzaLne/B5OAT9XnAAKsr64R9vrELIQd8fUVgCXuMT+6GTFzXHb5nPk940tQsn2IeSM+R146uwcsYYhMr94PBlepPaD23G+Ocbs7O/2c+JjTsmbjiA7r4EeJXnsCLAQxAEdMQSEJBaDex8/xyW1FaTMScQ9HtxcGmt4VTWaGdMdfzW14Rw4OG/V2zRKI7d2S2Hc16VOsE+lFyMHDC+sWdFVofd6NVHVvqcUeSbO5d64imPu59HUSlu6guHmJ2TtORACfupCPAo83kR7YwhGMivrHKA96QVPufcFDKU+SSOXaVxXxHg8fi0sVl/t2Nj0gn2oSQNuRlyyeqwGqVUJyXxLS3Meg2sNndf6cdj348nZ10rK6Mg2hlLf6dSOVAiv1zqiaPM58W3RgTxP2c68npu6WpYNLzb3m1HUkiV1ft5D7oZYj9qukhiQ3U5dm1MFto+kCF3GSyPYuWMMVg5Y0zawAcDPC7H4qqe7qwbhpoafHo2d1gaz1LK/d70//fvx6MzlgCnov1C2Ed7ZyzvBV2hcr+qVn/d+CqmFMSEEcGMsavXG3bCxqQdkCF3GWoexY4HJ2UJ36sZ00MnL5q6BlYYR1zq6q3E64glMGHDgYy/5bfOmCgU0tWgknOyalcbnt13nPn5tnOX0/9vxBvWGrtupeQ2O92OEY+ibnwVdjw4Caw9JLNeiNbmDuWVE2rIEwxY46UjlmCG2zpiiaxNSj2bmEtqq7M25nkOjkpcMAMZcpdhJtXJ6vQorcwfty9TCXuJdsbQfPBU2tCaHS/S1MNwJIqndh9De2csnRn11O5jisac4zjVf7sRCq24DL0dhXL9jBZqy1mzGTREaSAa2rVvfogd731qWs9cOgFs2H8iq1lKX1LA03s+zIqlK73PaVXaRiGP3GWYyYM385lcUAq9+Dycaq45YS9OLKTtiSdx5LT5LBnpipKVMtndl8zwymmzs0QotIqZHsykOhn5TK7fgVJe7eRRg7Hng88oxbBAJIXUKkxt70JsZ+i0Zt5lPg8EAaZXlFJvu1g3O0tSj5yFFbrcboLVBs7q74DVO3SAn8/on7rjvU9z8tAINgEvh5V3jFXN16+vCeHxaWMRjkQ1G5fnk/qaUFaDcblzoaWAGLJxfOcLavWmE1ZzBLXelW6FNWn183KKD0Qu34GR75VayOWGWg/Na4cE8GlHL9Pblv8eTvkt9Iy9cCSq2c9TNNiAs6u0WZAh14mZ3pVuxWhHH/E70Bt2kb5PbYCFri51RXndoN+LzquaLIRxJowI4p3THaa+P6VxrsdA5gM9sst6Jh43O2VkyHVSSh45a9Ji4eFSWhhK8dOg34vlU6/LqNLLtTUXkX+kmivSUJsTDLmIGP5hoRUWssspy8feGjVf1okTVMzyhdHNnaSQEvlnFW5Itc2pIMh9+DwcLsfiGXnYa9/8EBv2nyj0pWWwtbVdU60wpjL27NjUFB0X+Xdnhda/Xsgjl+GGrBUrsMNrFlcuRr19orAEr2reWJ1RpJUlkysVAR7Txn0Vh05eTD+v3X0JpjcuZuVI32/F852vlXzOzZfPnz+PuXPnYuPGjRg9+ktVsT/84Q94+eWX4fF4UF9fj4ULF+Z+tQWm0Cpm+UIpRfBKbzynh1nMxaWCIHex78eTMVGmeZMLHJBOObVzs/RSTzzj+FpjTp5aaVVvTSfkpmuGVvr6+rBq1SoEAoGs1/7jP/4DL7/8MjZv3oyXX34Zly5dsuUiCXsQtVjeXjYFOx6chOVTr8sKLRlBXLYqhagIZyI29rAq5BAq9+PJmeMAQLcRz0etUqjcj0MnLzIF53LBFR2C1q9fjwULFmDo0KFZr40bNw6dnZ3o7e2FIAhFoVlQykgrQJUI8B7U14RQEcheyEn3EsTjKL2PcBbib6YkJmWGyaMGp+PFehANf31NKPeTa1yXXZ6zE/bWVJ+0119/HUOGDEFtbS1aWlqyXh8zZgzq6+tRVlaG6dOnIxgM2nahpUahYvXS0BLrGsSiEa3rU9t0IpyB+JvVja/KuQioIsArer1K+DxcRi/NuvFVOHTyom0huT0ffAaOU5ZJztVzdkKHINXNzkWLFoHjOHAch0gkgurqarz00kuorKzEBx98gKVLl+K1115D//798eijj2L69Omoq6tTPaHTNzudQDFUmBrNUyfyj3wzLpdNanF8rt7VpnkMebqqiFrOeoWORilmcNNzZXqzc9OmTen/b2xsRFNTEyorKwEA5eXlCAQC8Pv98Hq9GDJkCDo6qLzaCuxqR5VP1JaroXI/RgzyUzl+HuAABHgO3TILqLT0N7tJLTXMao1MtIp66sZXofXspazYurR1m1pOe0gja0WOh4NrjLgWhnekduzYgf/6r//CNddcg4aGBixcuBD33nsvOjs7MWfOHDuuseRwwi54rrCWq6IX2Dz/ZtTXhPKy0VXK8B5AKdLBQcDqXW0ZDRgmjxps6hz9+/FpY8iKF4sx8NW72jDtxT9j6guHFBtAPD5tLNbMHKeo1Fk3voq5fyOOq2W3j9a90S4IuWWrOAnKI7cYK2LbRvJSnZr3ric8JDYDkOtDE/klwHtw47CBiiukCSOCOP2Fdj9PqbctH5OTRw1WVVQ0Et5QKsNXGlfS87O8dLdVbFOJfp6wKrbNOo68mEHpAWGdrxAGX+ucFEd3PkG/FxzH6Q5XKJXQ6/md5UZVrzIn65xSimHPCSBDnjesrPAy6tWonc/KgZzLhCD/LBnx4kQuO6vHwEg1UFjj1c97THvWTl25GoEMeZ6wUz3RiPcqP59VE4yZCUF8gMholxY8B0MZJtKxaGalpkcd0e2QaFaesLPCy8hGp/x8Vm2eqmXTKCEVEyJKC6Npgu2dMcxuOYx1e4+ZGi+FEKpyEmTILcTOCi+9kwErrSyXY4oYnRD0qiCW+Tyq/yZKg/bOmGpZf9DvVc1IsaLc3q3QE2OQcCSK2S2HFVOn7GxyrJbWpXU+qyYYoxOCXo9fHtzr7qOKUD24LXWzIsBnjFUjEg4B3oPlU69TlZAAMsec2rNabFCM3ACF3v3OdaNRWoLNqq7TOoaR+9cT61RrTUaoUxHgHdNXU4rPw0EQhKzwilx21sjvvmbmOF0ZT56rZfhBRiWoVoaLk6HNTotwcwchK1Mj9U4mWprnPg9HOeRFhrTJsVqjZ6PHVKqfMKunL58U3ELOeuREilw3DQuZAmVV2b+SqNbqXW2K9yMXEwr4POmwiYcD/vFGe4WSiMIhlutbYciVKk6lY8vo+GkKt2UcoxggQ24AVu6znk1DuQdhlai9Xqwu+zd6PwIyY99JAfjDe1GQbHlxIR0HVklK7Hz/HGquqcgaV+K/jXrmSQF5ffbyAT1GBshl09Bo6p7VWJ0aqed+tNIP+5JClpgT4X7EcWBVYwW158Rsf9ieeBJN4bai2QAlQ26AXLJSCi2EZXVqpJ77oSbMzoTnUnFiUZzKDqKdMUs7RZl5frTOLHrmxWDMKbRiELM9PXMJy1iB1eL3eu7HTWqNpURcAFrPXsLj01KNHezQvAkGeMUxd64zBjNTu9qKkpW90lQ3TlX2FnCfPDQLMuR5YklttWLWSD7bQVnZWFrP/ZCeinMRC28enzYWS2qrNQ2eUcRkOPmYm/bin1VTJieMCOK9T7s0nxM16QdpNpae+yoGh4NCK3mkn/fLEo6KAO869TUpSmGmWTcMRfPBU+kCjMmjBlMTZgeztbUd4UgUdeOrEPR7LT12JyNbpUMj7715/s2a4Uu1vRf5+yeM0G4/mc8myXZBHnkeUMp5NdrP0onqbfJURHkWy873z2HWDUOx54PPLElDI6xHDCssn3qd6bxsJYyGQgCkJxOtFFfW3ktI8rr0/QCY3ajyvSq2CyoIygO5FhLlu6LUzKTBuseKAI9YPEmbnjYT8HKIJQTDVbIcgCdnjkuHKTxcahMwV+prQhna+dKmE6xmIvKGzKxxrzaW5K9rNZ1wgkOkF6rsLDC5ytvms6LU7KSRS+NeonCU8RwEcBm/t1EJWjmsOLc4hsKRKLNJc0WAx96HvgOAHU9nTTasv0uPmQuFngRIxrbA5JrDzVqK2rFJw8oPX7WrDev2HmN+rhjijKVId1zI+r3jQsrDN8uR0x3MGgPRGLLmiUs9cUx94RDW7T3G3BRNClBMpWWtJC71xHNOMZTG5QU4TzaXPPI8oLd1m9IMr9SjUKTM58GBh//e8LUoeRV6G0BMGBFE8/ybs/6udp0EYSVyz9vDAXNuCmFv2+eqGTG5NJ9wgs4Saa3kCZaRVMqnlbduUypxD0eiqsbRqNwrq6y+9ewl3W3kjpzuwIQNB7Ieir1tnxu6FoIwi9zzTgrA9nfbNWP7emQxWM+w3oKkQoVfyCO3CCWv2+fhUObz2JqxwQEov9ogt6Mnzhw84UgUTeE2Q7FFvVi1QUYQ+YAVM1fbH2KtVkWPXC4TLf+8FcacYuR5QCm23JcUbE+7EwB0xBK41BNnxu7EAcoytrkaYTLihJtgxczV9IPUJC7E50sprJMvPSUy5AZhdR1xUnWYfPBoaZ543NZqhiByRMm4qoVP1HSWtJ6vfNgGipEbQE261Wnl6NLBozaQfB4O/3hjle4YOUEUAyydIDX9IJbEhZahzkdGF3nkBjC69LILPR60dPCoDaS+pICtre3o5+UM9VAkCLcjD6+YVQhVe77yVTlKhtwARpZednWC93k4zLkppPoe+eDRM8l0xBKIxZOorwmRPgpREsjDK2ZlqlnPV9DvzZueki4X7Pz585g7dy42btyI0aNHAwA+++wzPPLII+n3RCIRLFu2DPfee689V+oAjCy9ZrccRneftaEWDl+2R2PBaqrM6fDie+JJHDp5UXWXniCKBSXHzIxCqNUS0WbQNOR9fX1YtWoVAoFAxt8rKyvx6quvAgCOHj2K5557DvPnz7fnKh2CESlalvfOwVw8neeAu24KMWPZQUkKouhpaGlbKNHeGUsLDhFEMWPlGLdSItoMmmvo9evXY8GCBRg6dKji64Ig4KmnnkJTUxO8XmulMJ2GkaUXa5CU+7241NNn+Nwcx2Fv2+dMI96bEBRTEJsPnjLcqV48BkEUK8Wieiii6pG//vrrGDJkCGpra9HS0qL4nv3792PMmDEYNWqULRfoNPTOvEreO8+lqjGNGlYgtSnJKj9WylUXN2GdlBZJEE4gl1J9p6Ja2blo0SJwHAeO4xCJRFBdXY2XXnoJlZWV6ff867/+K+677z7ccsstuk5YrJWdSsjLdbv7EqpaEFZjNowjJ1TuR/SqWBBBFAqt8VxfE8Lrre0Z41SU6S0Go22JjG1jYyOamprSm50i06ZNw549e8Dp2U1DaRlyOVZIvSrpLft5j+IEIXoeRmLkSscQRYHs6O1IEHrRciiOLJtScKlZO7FUNGvHjh24cuUKGhoacOHCBQwYMEC3ES91rPCOxYwS6UAFwNyEFQfx03s+NCyypZTGyOoiUxHg0dkTz2is6+VSHlEu2tZEaaDmkIh09yVQ7vcqhhJDGkU7xWzgARLNyivhSDSnJrdqkplaA1WvN+3hAEGAqvgW6zxKrwFg3rPYWCMciaJpV5up7uqE+wmpOCRyfB4OgiBkOAdawlT57rBlF9QhyEFM2HCA+ZqaWmKuA09PWMeuwa1Hy3nd3mNZ8U0tQg6TRXAj/bwcehOFWzLJnROpM8CpdPwp83l1e9dO0BK3AtIjdxAs4+PhkNWv0MqloJ6wjt+Gis5wJIruPuWJSfTCwpEodr5/zpAR54C0fKh8D4BkdbURC8cKWfillAIoDY1MZDg9HT1xQ63b9GqJuxky5HmGVVQk94StLjBYUlutGda51BPXFN43gtKSFsiuPtVSj1Mi4PNg4oYDqCr3p6tdtfYNihEOqbx/I5MXB2DfjycDAFbnEOrTOkcwwDNj3npSALUqqfVi1XGcDIlq5Bmzeg5WnLe+Rl2jBbBWP5lloPv34zPuV60K9siyKaivCWUIhXmv5uOLhUs73z+HJbXVeHvZFOx4cFJ6EpR+z0G/FxUBHhxSS3NeYX9+wogg3LRt7+WQXsUkhVRoTum+5OgVVMuFqnI/OhhGXFxNmdEwMVPIY9VxnAx55AWgUOW8j08bi5prKtIhG5YDZ9WSU++SVstjenzaWDw+bSwA5XhnTzyJDftPqLbZk6PWu/TZfcdtbwgieqwdV6txWe95e9kUxetVqknoSwoZ8eNyvzerAE1PJpK0n6z0uxZ7YwJQbUEonoMVttE7eVilYeIELRS7oc3OEsbuTSC1TJmQzHjqzSrQm4uf68YtKyykhRjqMHJten8HqTFXO8eRq8Zf/hkzmUgsWJvTZn9XQhvKWiEUsftB0zKG0nOpechmq2P1TEji8ds7Y+k4szQdzgrv3McB/f08s6cq63cQvWKWd62EhwOa6sblzfvUmgSKPX87n5AhJxSRN4xlSeDmeg61zAit3HglvRqO4wxVqpbxHHxeDzpjiQxjsm7vMWaIQDSk21rbc85v93DA4UemqL5HbvAmjxpsumuTvPqX54ABKhOJWVgTdX1NKB0Kk7+fjLp5yJATWeR72asWEmFlMLBCDkG/F/378a7Sf5GGO/Qw9YVDtsXp1X5nVlGXkgFWC52tkembUJgld8iQE1nku0hCT2VpRYDHsttHZ+QRKw1OaUVoLpWy+UKPRy4SjkTx9JvH0G2zrgHPAW89MiVrVWYVYojH6D4AwUbNkFP6YYnCyihp74xhdsvhrH6GuaKn3ZyYxy6em5XdIP59w/4Tll6jXWi15hMJR6JYE26z3YgDKf2b+S+/jad2H7NFkTMpIOO3tLsoJxyJYnbLYUzccMCW8et0KP2wiFGLSapVeoqNKVrPXsoqtMlFkMgvi90qIeax142v0uzIlE9J4HzQfPBUXgXG/nahx9bjS39LO4ty5GEbcfwC1hS2uQEKrbgM6eahPNWtviaUzhNXC2ME/V509ybQl+MvzwH41ogg3vu0SzX2aTSVT9SQZmWTiMdV061xKvLwkRQ33k8uWBUjL5WwDcXIiwQ9BlFPHnM+EB+icCSKpnCbIe0TpXuQp+NZIQlcKHweLkNXB8hdGdNtWJkhpbWXUixQjLxI0KNJ4gQjDqRin+LEY1TASuntPfEktra2o/1qpopbjTiQqsCUyyBYJYvgFuQyDbmgtZdSCpAhdxFuUmurKvebEsMqFeS/pRt+26DfizUzx6WbOOSClfdbCloqWpAhdxFu8TB4LvVwOdU46REPsxv5b+mG37YjlkhvantyVBez8n4LJUTnJChrxUWotVoTKUSMXC6hKrb+Y7XlKiRi1aGa6JPd+Dxclreo9Nt6OaCAPR8UEUNaueyseQDLveVCCdE5BfLIXYTU8wCQJblaXxPCk1eXvqJnUl8TSr9f9KKCfq8l18MBKPN5smLgYgzYab1cWaXjduOTPGUVAT5roxNQ9iq1jLjPkzL2LDxITRpKlPEcKgIpP058S6jcjzUzx2HNzHGaOf/pcygcnkP22JSet6lIuto7CcpaKVH09vCUIsqYSo2hWsYAoL46EEvtpSmGdiEvkc811S9U7keZj2PmYlsxaahd4xGZvK08HVVMcwTMybfKawNYY6XYMkOcDLV6I7LQCtNoNWEW0Sr0YBmAAO/JSj8TKxvzURRTweheU+bzoLtPe4O2vTOGAO9BfU1IV9GUGViTm9QL1hNSMHM98uOyJn43xPZLATLkJYr4kCrJtBop1NCqvlSaLFhFMdIGAFIvPZhjrF1pc3PZ7aOzen36PBxWTB+ju49lTzyJQycv2lZ0MuemkGIsX2/Jv5Vo/c5EYSFDXsKIXlcu8qJ6uq8YOTbLwzQTCuEAzGWEOLSuW28lqp2ZOeJ1b3u3HUlBObSVL9zUZacU5XIpRk64Aq2YvrTFmRUPr15VQCeXgZeiQStmuVwq0Sdcj5o8gZ0PqlbpvFx32ykUs0FjoSYHURHgsfeh7+T/oiyESvQJ1yNPvZSmzNlpnOrGVzErGSsC1pWZW41SVa2oRlhshCNRTHvxz1i1i63pc6knXtTSthQjJ1xDoYo+WBt9YnqfE7Fb/9spGFHWFCV1ixEy5AShgZs2+kTs1P92Ekb0fIptEpOiy5CfP38ec+fOxcaNGzF69JdeyLvvvot169ZBEARUVlbimWeegd9fXAOFIAD3lYCXSrqgEeNcbJOYFE1D3tfXh1WrViEQCGT8XRAEPPHEE3j++ecxcuRIvPbaazh79ixGjRpl28USBKEPN64izKBXl74YJzEpmoZ8/fr1WLBgAVpaWjL+/re//Q2DBg3C7373Oxw7dgy33norGXGCcBBuW0WYgVWhXMZz8Hk96IwlinYSk6JqyF9//XUMGTIEtbW1WYb84sWLOHr0KJ544gmMHDkSP/zhD/F3f/d3+Pa3v23rBRMEQYiUyspDC9U88kWLFoHjOHAch0gkgurqarz00kuorKzEiRMnsHTpUuzYsQMA8Morr6Cvrw8/+MEPVE9IeeQEQRDGMS2atWnTpvT/NzY2oqmpCZWVlQCAESNG4PLly/joo48wcuRIvPPOO5g3b55Fl0wQBEHoxXD64Y4dO3DlyhU0NDTgZz/7GZYtWwZBEPCNb3wD3/3ud224RIIgCEINKtEnCIJwAVSiTxAEUcSQIScIgnA5eQ+tEARBENZCHjlBEITLIUNOEAThcsiQEwRBuBwy5ARBEC6HDDlBEITLIUNOEAThcsiQEwRBuBwy5Ab51a9+hYaGBsydOxevvfZaxmtvvPEGZs+ejYULF6ZfSyaTWLVqFRoaGtDY2IiPPvqoEJdtGUbvX6S1tRWNjY35vFTLMXrvfX19ePTRR7Fw4ULMmzcP+/btK8RlW4bR+08kElixYgUWLFiARYsW4eOPPy7EZVuG2bF//vx53HrrrThx4oR9FycQuvnLX/4i/PM//7OQSCSErq4u4fnnn0+/dv78eeG73/2ucPHiRSGRSAiNjY3C6dOnhT/+8Y/CY489JgiCIBw9elT44Q9/WKjLzxkz9y8IgtDS0iL8wz/8g3DPPfcU6tJzxsy9b9myRfjpT38qCIIgXLhwQbj11lsLdPW5Y+b+9+zZIzz++OPpz5fi2O/t7RWWLFkizJgxQzh+/Lht10fNlw3wpz/9CWPHjsVDDz2Erq4u/OQnP0m/dubMGVx//fUYNGgQAODGG29Ea2sr3n33XdTW1gIAbr75Zvzv//5vIS7dEszc//Dhw/G1r30Nv/jFLzLe7zbM3Pudd96JO+64I/0+r9eb78u2DDP3P2vWrLQi6ieffIKvfvWrBbhyazA79lkd1qyGDLkBLl68iE8++QS//OUvcebMGfzoRz/C7t27wXEcRo4ciePHj+Pzzz/HgAED8NZbb6G6uhpdXV0YOHBg+hherxfxeBw8776v3sz9A8Add9yBM2fOFPbic8TMvQ8YMAAA0NXVhYcffhhLly4t7E3kgNnfnud5PPbYY9izZw+ef/75wt5EDpi5f7UOa1bjPmtSQAYNGoRRo0ahX79+GDVqFPx+Py5cuICvfOUrqKiowIoVK/Av//IvCIVCuOGGGzB48GAMHDgQly9fTh8jmUy60ogD5u6/WDB7759++ikeeughLFy4ELNnzy7wXZgnl99+/fr1WL58OebPn4+dO3eif//+BbwTc5i5/5dffhkcx+Gtt95CJBLBY489lu6wZjW02WmAW265BQcPHoQgCIhGo+ju7k4vp+LxOFpbW7Fp0yasX78eJ0+exDe/+U1885vfxIEDBwAAf/3rXzF27NgC3kFumLn/YsHMvX/++ee4//778eijj7q+e5aZ+3/jjTfwq1/9CgBQVlYGjuNcG14yc/+bNm3C73//e7z66qsYP3481q9fb4sRB8gjN8Rtt92GI0eOYN68eRAEAatWrcKuXbvSHZN8Ph/mzp0Lv9+PxYsXY8iQIZg+fToOHTqEBQsWQBAErF27ttC3YRoz918smLn3n/70p+jo6EBzczOam5sBAL/+9a8RCAQKfDfGMXP/M2bMwIoVK7Bo0SLE43GsXLkSfr+/0LdiCqePfZKxJQiCcDkUWiEIgnA5ZMgJgiBcDhlygiAIl0OGnCAIwuWQIScIgnA5ZMgJgiBcDhlygiAIl/P/ASjBI8QGyoTPAAAAAElFTkSuQmCC\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAXIAAAEFCAYAAAD+A2xwAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjUuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/YYfK9AAAACXBIWXMAAAsTAAALEwEAmpwYAAAhZElEQVR4nO3dfXBU5d038O/ZbLLLS17M3CFxbpBAgMi0vhWtQ5mACKIBo0JMwkuhoiMqPqA1tAo1IaC8VRmntsDT/FGqNHVUEqxpWeS26Mg8Um6hSChdEg2NU6hZIihJSLLZzZ7nj7DLbrLv2d1zrnO+nxlH2JOwV87AN1d+53ddlyTLsgwiIhKWQekBEBHR0DDIiYgExyAnIhIcg5yISHAMciIiwTHIiYgEZ1R6AERufX19ePPNN1FfX4++vj44HA7MnDkTzzzzDFJSUvDrX/8aNTU1yM7OBgA4nU7k5+dj9erVyM3NBQDcfffdSE5OhtlshiRJcDgcmDZtGl544QUYDLGZt3z66afYtm0bAOCbb75BX1+fZ0xPPPEEenp6sGnTJowePdrn8/Lz8/GLX/wCDzzwAFasWIElS5Z4rp05cwY//vGPUVNTg/z8/JiMk3REJlKJF198UV61apXc3t4uy7IsX7lyRX7qqafkNWvWyLIsy6+//rq8YcMGn8/Zt2+fPH36dLmjo0OWZVmeOXOm3NDQ4Llut9vl0tJSec+ePXEZs78x1dbWyitWrAj4OUePHpVvvfVW+ezZs54x3n///fLbb78dlzGS9rG0Qqpw7tw51NfXY/PmzUhNTQUADB8+HBs2bMDs2bMDft5DDz2EvLw81NfX+72ekpKCKVOm4OzZsz6vv/3223jyySc9v29ubkZBQQH6+vrw+uuvo6ioCAsWLMBjjz2GCxcuxOArvOaHP/whli1bhueffx4ulwuvvfYaJk2ahNLS0pi+D+kHg5xU4fTp05gwYQJGjhzp83pWVhbuvffeoJ+bn5+PpqYmv9dsNhs++ugj3HnnnT6vz5s3D8ePH0dbWxsAoK6uDgsWLMCFCxfwxhtvoLa2FnV1dZg2bRoaGhoi/nqOHTuGBx980Oe/2tpaz/VVq1YBANatW4ePP/4YGzZsiPg9iNxYIydVMBgMcLlcUX2uJEkwm82e369ZswZmsxkulwvJyckoKSkZ9M1g5MiRuOeee/D+++/jkUceQX19vaf+fuONN2L+/PmYPn06pk+fjqlTp0Y8pttvvx2//e1vA143Go3Yvn07Zs+ejXfeeWfQNzCiSDDISRVuvvlmnD17Fp2dnT6hZrPZUFFRgddffz3g5546dQrFxcWe37/66qu46aabQr5naWkpKioqkJeXh7y8PIwZMwYA8Ic//AGnTp3CkSNHsHnzZhQUFODnP//5EL46/9zv5/4/UbRYWiFVyM7ORlFREdatW4fOzk4AQGdnJ6qqqpCRkeEz4/b27rvv4ty5cygsLIz4PW+99VYAwI4dO1BSUgKgv3vk/vvvR15eHp544gk88sgjOHXqVHRfFFGCcEZOqrF+/Xrs3LkTCxcuRFJSEnp7ezF79mxPPRkA9u/fj+PHj0OSJLhcLowbNw5vvvkmTCZTVO9ZUlKCnTt3eh6o3njjjSgsLERxcTGGDx8Os9mMF198MeI/110j95aUlIS6urqoxkkUjCTL3MaWiEhkLK0QEQmOQU5EJDgGORGR4BjkRESCS3jXisvlQl8fn68SEUUiOTkp4LWEB3lfn4zvvutK9NsSEQktKys14DWWVoiIBMcgJyISHIOciEhwDHIiIsExyImIBMdNs4iI4sxitWHn4RbYOuzITjVhZUEuCidnx+zPZ5ATEcWJxWrDlv/5At2Oa4emtHbYsfngFwAQszBnaYWIKA62ftiEyv2NPiHu1uN0Yefhlpi9F2fkRERDNLB0MibDhM/+3R70c2wd9pi9P4OciGgILFYbNh/8Aj3O/pl3a4cdrWGEdHZqdIeh+MPSChHREGw/1OwJ8UisLMiN2Rg4IycizYtX14jFasPlHmfEn1d8S05Mu1YSftSbw9HHTbOIKGG2ftiE2pOtg14vviUHL8yeFNGfNfAbQlevE+32vrA/f5hRwto5k6IK8WCbZnFGTkSaZbHa/IY4ANSebMUt/50edqj6q4VHIppvHOFikBOR5lisNmw/1Byy7FFlaQTg288dqAyz83BLVLVwIL4hDrC0QkQaY7Ha8NKBJjhckUebdPU/77g2Gw1YN2ci1u9vRKR/YrrZiPK782JSDw9WWmGQE5FmWKw2VO5vjPmfm3O1VTDccspn5dNjPgbWyIlIM7xLH2lmI2RZRoe9D2lmY1QdJOGIpB6eE8P+8HAxyIlIGAMfOHoHd7xCPBJmoyGm/eHh4oIgIhLGUB44JsJIkyGm/eHhYpATkTBiuT9JPHxzRZmfClhaISJVs1htePWvX0a08EZvGOREpFqBVmWSL5ZWiEiVgq3KJF+ckRORqrjbCyNdAq8Gd4xJU+R9GeREpBoilVLuGJPmc3jEHWPSsLP0VkXGwiAnIkWJOAOXAMVC2x8GOREpRqQZuLeE7msShrCC/KGHHkJqav86/9GjR2PLli0+17u7u7F8+XJs2rQJeXl5sR8lEWmOyA8zlViGH0zIILfb+3/c2bNnj9/rp06dwvr162Gz2WI7MiISXqAtYeO1uVUiKLUMP5iQ7YdnzpxBd3c3Hn30USxbtgyff/65z/Xe3l7s2LED48ePj9cYiUhA7n1RWjvskNG/8VTl/kbcsf0T1YS42WjAMKMU8uMMUn9dPCfVhHVzJiqyDD+YkDNys9mMxx57DCUlJWhpacHjjz+OAwcOwGjs/9QpU6bEfZBEJB6174uSc/UnhPUhvqm49yNXW3h7Cxnk48aNw9ixYyFJEsaNG4eMjAy0tbXh+uuvT8T4iEhQat0XZeBpPcE6ZnJieFBzPIUsrezduxdbt24FANhsNnR2diIrKyvuAyMisWWr7IGg276GVlis157prSzIhdnoG4VmowEb5+ajfsWdqg9xIIwZ+cMPP4y1a9di0aJFkCQJmzdvhsViQVdXF8rKyhIxRiJSMe8HmqmmJEiShPYeJ8zJ6twBxCUDmw9+AaD/rE53UPt7KCsKHvVGRFETtQ8c6C+b1K+4U+lhhI1HvRHRkPhrI6w/9bXPEnXRqLWGHw0GOREFNfB4NXcboVrdMSYNp77uDNkxo9YafjTUWcQiIlWwWG2osjSquo1woFNfd2Le90YhJ9UECUC62YiBreJqXNQzFJyRE5Ff7pm4S20bi4TQ43Th/5391qf+HWiFqVYwyInIL7Uv6AlmYP3buztFixjkROThPXMVbCLuQ0v173AwyIkIwOCHmmomoX8rWff/vWmt/h0OBjkRAQA2f9CEnj4x5uEb5uZ7SiVar3+Hg0FOpANbP2zCvoZWuOT+nfymjE7Dv7+ze8JvTIZJmBA3SvAJaq3Xv8PBICfSuIGrL10yfBbytHbYhTpmrbIwX+khqA6DnEjj9jWIuYR+oDRTEtbMmqD72bc/XBBEpHGi9YEHwhAPjEFOpHGG0AfgCGHn4Ralh6BaLK0QCSrQ9rHuzg2gP/y0MiPX0iZXscYgJxLQwJ7vdnuf55raN7WKlt4W+USCpRUiAYm8fD4aelzkEwnOyIkEpKcygyjnZiqJQU4koOxUk1C93+EaZpTQ45R1u0IzWgxyIgGtLMgVZl+USKQPS8EnAh2/phYMciKVCrWHSEqShB6nggOMAz2VjGKJhy8TqVCwnQj97finFaIdiJxIPHyZSCDu49UC9X9rNcTZmRI9th8SqYiox6vFwro5E/lwM0qckRMpzLsWLkna2RslUgzx6DHIiRTgDu+BLYSJfWKlHv81glE0FLx7RAkm0pFqifBfI4ywPPkjpYchNAY5UYLpbXm9N7PRwFp4HIQV5A899BBSU/tbX0aPHo0tW7Z4rh06dAg7duyA0WhEcXExSktL4zNSIkEE6v8OVE7RAwngas04Chnkdnv/X7o9e/YMuuZwOLBlyxbs3bsXw4YNw6JFizBz5kxkZWXFfqREAhhYNmntsGPzwS9w8vxl/OX0BV3OxNkbHn8h2w/PnDmD7u5uPProo1i2bBk+//xzz7Xm5mbccMMNSE9PR0pKCqZMmYJjx47Fc7xEquavbNLjdGFfQ6tmQzzNlISNc/OxcW4+zEbfSGFveGKEnJGbzWY89thjKCkpQUtLCx5//HEcOHAARqMRnZ2dnpILAIwYMQKdnZ1xHTCRmgVaYq7FlsJAuxIG21aA4iNkkI8bNw5jx46FJEkYN24cMjIy0NbWhuuvvx4jR47ElStXPB975coVn2An0hut7ko4UPEtOXhh9qRBrxdOzmZwKyBkaWXv3r3YunUrAMBms6Gzs9NTA8/Ly8NXX32F7777Dr29vTh27Bhuu+22+I6YSMWmjb8u7I8V+SjNv5y+AIvVpvQw6KqQm2b19vZi7dq1+M9//gNJkrBmzRqcP38eXV1dKCsr83StyLKM4uJiLFmyJOgbctMs0qpI+sPdZYmNlkY4BS278CFmYgXbNIu7HxLFSFH10bDLKneMScPxc+1C184lAP9bPl3pYegGdz8kSoBIauOf/bs9jiNJDB6GrB4McqIweS/0STMb0evsQ/fVukiaKUnT+4QPxLZCdWGQE/kxcHXmtPHX+SzouTzgaJ52e58Sw1QED0NWH9bIiQbgplb+bZybz/BWULAaOQ+WIBpg+6FmhrgfOw+3sOVQpRjkRF4sVtugsgn1c+8bwzBXHwY5kZedh1uUHoKq9ThdvEcqxIedpFv+tpsNtFeK3uSkmmDrsPvtwuE9Uh/OyEmX3A80W6+GVWuHHZX7G2FKEnnhfGjhfHXu1sJAfeLsH1cfBjnpUqBTenr6tNkJnpNqwsa5+UE/Rrr6ce4TfFYW5HJbWkGwtEK6pKfyQLrZ6NkTJdAJRf72TXG3GnJbWvVjkJOuuOvi2px3++e9VGRlQe6gHvlgs2xuSysGBjnpxtYPm1B7slXpYSRch9eqU86ytYlBTrpgsdp0GeLA4IeTnGVrD4OcNM1itWH7oWbdLvLhw0l9YJCTpnj3hpuTDeh26HOpvQSwbKIjDHLSjIGbXek1xAOdp0naxT5y0oxAveFadseYNBiurvIxSAxxveKMnDRDD6fXe2NokxuDnIThb28Ud/1XbzvyMcTJG4OchDCw/u3eUhXob6fT0458BgkMcfLBGjkJwd9hDz1OF9bvb8Qd2z/RZFml+JYcv6/Pv9n/66RfnJGT6gU77EGrS+0lXJt172tohUvun4nPv5klFRqMZ3aSqlmsNlTub1R6GAnHGjgNFOzMTs7ISbX0uDcKZ90UDQY5qZJeQjzNlIQ1syZw9SUNCYOcVMNitWHLwSZ0O7Va+e63cW4+g5tiikFOinD3hGux2ySYYUaJIU4xF1b74cWLFzFjxgw0Nzf7vP7ee++hqKgIixcvxrvvvhuXAZL2bP2wCZX7G3UX4hKAtXNY+6bYCzkjdzgcqKyshNls9nn90qVL+NWvfoV9+/YhLS0NjzzyCKZOnYrRo0fHbbAkPr3Uvv1JMxs5G6e4CDkj37ZtGxYuXIhRo0b5vH7u3DnceOONyMjIgMFgwE033YSTJ0/GbaAkPj0f7gAA7TrdE53iL2iQ19XVITMzEwUFBYOujR07Fl9++SW++eYbdHd348iRI+jqYn84Bbblf75QegiKGnhSD1GsBC2t1NbWQpIkHDlyBFarFc8//zx27dqFrKwspKenY+3atVi1ahVycnLwve99D9ddd12ixk2CsVhtut0fHOBJPRRfQYO8pqbG8+ulS5eiqqoKWVlZAACn04mTJ0+ipqYGTqcTy5cvx09/+tP4jpaEpadNrfxZN2ci6+MUNxG3H9bX16OrqwtlZWVITk7GggULYDKZsHz5cmRmZsZjjKQBNp11qHhj3zjFG/daoYQoqj6qu3ZDowRUFjLEKTa41wopbmVBriY3v8rhAcekAgxySojCydmoP/U1Pvt3u9JDGTIJwAaWS0hFGOSUEBarDSfOiR/i6WYjyu/OY4iTqjDIKSF2Hm6BSHthDUs2INkgocPeN+h8UCK1YZBTzHkfkpxmNkKWZbTb+5QeVtjSzUZ8+PSPlB4GUdgY5BQ1f6fanzx/2WcZfqAj2tTKbDSg/O48pYdBFBG2H1JUBp5qDwDJBgkOl0D1k6uGJRvQ43CxhEKqxvZDirmdh1sGnWovWohzoQ5pRVj7kRMNJPpKzeJbchjipBkMcoqKyDv5mZMkHm5MmsIgp4hZrDZc7u5VehhRMUrAunsZ4qQtrJFTREQ+4YfL6UmrGOQUNhFP+OFKTNIDBjmFTYQ9xc1GA/f+Jt1hkFPY1N6pwtIJ6RWDnPxyr9pU+x7iDG8iBrnu+VtmDwAvHWhS7QKfNFMS1syawPAmuopL9HXM3zJ7s9EASYJqD0r+rHy60kMgUgSX6JNf/pbZD/y9muQIvAiJKJ64IEjH1P7wcqBp469TeghEqsQg1zHRltn/5fQFWKw2pYdBpDoMch1bWZCLJEnpUYSvx+kSopedKNFYI9eRgR0qw5Il9KmzMSUg0cpBRInAINeJgR0qau8PD0S0chBRIjDINcJfP7i7z9pitWH9/kaINPlONxtxxe70ObDZbDR4+tyJ6BoGuQb4m21vPviF5/oGixghPnCDq2DfnIjoGi4I0oCi6qN+SyXuvmtRyig5qSbUr7hT6WEQqRIXBGlcoAeAtg67EDNxNz7IJIpOWO2HFy9exIwZM9Dc3Ozz+vvvv4/58+ejuLgYf/zjH+MyQAot0ANAkUIc4INMomiFDHKHw4HKykqYzeZB1375y19i9+7deOutt7B7925cvnw5LoOk4FYW5MIoUD+4P3yQSRS9kEG+bds2LFy4EKNGjRp0LT8/Hx0dHejt7YUsy5AkwdNEUIWTszHCpO4qWbBvNOlmI1KSJKzf34ii6qNcvUkUoaBBXldXh8zMTBQUFPi9PnHiRBQXF2PevHm46667kJaWFpdBUnAWqw2Xe5xKD8MvowRsnJuPI89Nx8a5+chJNUFC/4PNjXPzsXFuPuxOF9rtfZBxreOGYU4UvqBdK0uWLIEkSZAkCVarFbm5udi1axeysrJw5swZPPvss3j33XcxfPhw/OxnP8M999yDwsLCoG/IrpXYsVhtePmDJvSqeHlmskFCxX2TArYNBuu4YQcL0TVRd63U1NR4fr106VJUVVUhKysLAJCamgqz2QyTyYSkpCRkZmaivb09RkOmUCxWG6osjVDp2Q8eDpeMnYdbAgZ5sI4bIgpPxIXV+vp6dHV1oaysDGVlZVi8eDGSk5Nxww03YP78+fEYIw0g2krNYKGcnWryOyNnBwtR+LggSDAWqw0bLY0+S9fVLliZJNApRevmTOQqTiIvXBCkERarDZX7G5UeRkRCtRW6w5pL8Ymixxm5ygTaX8RitWGDpVGIbWcNEiDLYCgTxRBn5IIItvnV9kPNQoQ40B/i/8tDkokShkGuIoEOQxbpwSbAB5VEicaj3lTCYrUF3KVQrSFuQH+fuDcutSdKPM7IFWax2vDqX79Eu71P6aFEZFiyAckGCe32PhgkwCX3d6ewJk6UeJyRK8hdExclxHNSTfisvH+pvSzDM26XfG0mzhAnSjwGuYL81cTVyrtkEqiWzxPuiZTB0oqCRDq5x3u2zWX1ROrCII8jd094a4fdU0dONxshy7IQ5RQJ/tsIuayeSF1YWokTd/3bHXjuza0u9ziFCHEgcDCvLMiF2ej7V4fdKkTK4Yw8TkSqfwcSKJi5rJ5IXRjkMeBvWb3o9eI0U1LQYC6cnM3gJlIJ7rUyRIF27zMZDao9tScU7j5IpD7cayWOArXiiVhWkcCNrohExCAfItFLKN640RWRmNi1MkRaabnL0cjXQaRHDPIhWlmQCyn0h6kaWweJxMbSSoS2ftiEfQ2tqj/0OFzc6IpIfAzyCGz9sAm1J1uVHkZMsDOFSDsY5BHY16CNEOcsnEhbGORBFP7fT/HNFTF7wQORgIAn2hORmPiwMwAthjignS4bIrqGQe6HxWrTZIizO4VIm3RfWvG31awWGSTw4SaRRuk6yAfuk6LVEGeHCpG26TrItbDVbCA8EJlIP8IK8osXL2LBggX43e9+h7y8PABAW1sbnnvuOc/HWK1WlJeXY9GiRfEZaRxoaZ8Ut0Cn+hCRdoUMcofDgcrKSpjNZp/Xs7KysGfPHgDAiRMn8Nprr6G0tDQ+o4yTQEeWiYxdKUT6E7JrZdu2bVi4cCFGjRrl97osy3jppZdQVVWFpKSkmA8wnqaNv07pIcQUu1KI9ClokNfV1SEzMxMFBQUBP+bQoUOYOHEixo8fH/PBxZPFasNfTl9QehhDMswoISfVBAn9tXA+0CTSp6AnBC1ZsgSSJEGSJFitVuTm5mLXrl3IysryfMwzzzyDZcuWYcqUKWG9oRpOCLJYbaiyNArdpcJOFCJ9ifqEoJqaGs+vly5diqqqKp8QB4DTp0/jBz/4wRCHmDgWqw0vHWgSOsTTzUaU353HECciAFG0H9bX16OrqwtlZWW4dOkSRowYAUkSZ0fu7Yea4RAoxVOSJJiNBnTY+3gMGxH5pbvDl+/Y/oli7x2OdLMRHz79I6WHQUQqE6y0wr1WVKa9R3t7vBBRfOkuyNNM6miRNASoRrEPnIgipbsgXzNrguJnbEoAqgrzYTb63n72gRNRNHQX5IWTs5FmVnaLmexUEwonZ2PdnInsAyeiIdPlpllK1qG9Z92Fk7MZ3EQ0ZLqbkQPR16GNUdRkkg0S0kxJnHUTUdwIPyN3Hwxh67AH7bP2PkAi2hq5M8JGTW4hS0SJIHSQDzwYorXDjs0HvwAAn/Ac+HGJapznIcdElAhCl1b8HQzR43Rh5+GWkB+XCBarLeHvSUT6I/SMPNDBEK0ddlWs4PT30wERUawJvUR/9o5PcVnlKyHTTEkYnmIMWcMnIgom6t0P1cxiteGKXd0hDgDt9j602/sABK7hExENhXBBbrHasP1Qs+pn4oG4a/gMciKKFaGC3L2XuEjb0PqjxUOfiUg5QnWt7DzcouoQN0jhbcrFjbGIKJaECnI1z2SHJRtQVZiPv/6faUEXHHFjLCKKNSFKK+5VmWqbi39WPt3v69mpJrT6+aZjkMAl+kQUc6qfkbtXZfoLRiWZkwLPu1cW5PrdoraqMJ8hTkQxp/oZuVKrMoORAKy7d1LA6+6wDmcPGCKioVJ9kKuhLi4BSDMb0d7jZCgTkeqoPsgD1ZsTxSAh4pJIuJt5ERHFgupr5P7qzYmSbJCiqmuHu5kXEVEsqD7IvY9EA64dWpwexXFtBglICfKQ0tswo4SK+yZFHOIWqy3gTxBqKBMRkfaovrQCBD4Sraj6aMiyS06qyWdf8HBXhyYnGaIKcXcJxR8uBCKieFD9jDyYUGUXf4tvCidno+K+SZ5DjwNpt/cN2k/cYrWhqPoofrj9ExRVHx10/dW/fhmww4YLgYgoXoTexhbwPeotzWyELMvosPeF3V0SbFbvPZsf+ADTrfiWHLwwexIsVhsq9zcGfJ+Nc9lDTkTR0+Q2tm5DPYl+ZUFuwAD2rmkH6mevPdmKW/47PeiDzJxUE0OciOJG6NJKLBROzg744NS7ph3sQaX7J4JAWFIhongKK8gvXryIGTNmoLm52ef1hoYGLF68GIsWLcLq1atht4vZlVF+d57fJfXeARzsQaV79aY/6WYjZ+NEFFchg9zhcKCyshJms9nndVmWUVFRgS1btuCtt95CQUEBzp8/H7eBxpN3i6OE/lLIwM2tgs2q3fV4f98Myu/Oi9OoiYj6hayRb9u2DQsXLkR1dbXP6//617+QkZGBN954A01NTZgxYwbGjx8ft4HGW6hae+HkbJw8fxm1J1t9XnfP3Lm/ChEpJWiQ19XVITMzEwUFBYOC/Ntvv8WJEydQUVGBsWPH4sknn8T3v/99TJ06Na4DVtILsyd5Hmz6C+uhPnglIopG0PbDJUuWQJIkSJIEq9WK3Nxc7Nq1C1lZWWhubsazzz6L+vp6AMDvf/97OBwOPP7440HfMNbth0REehB1+2FNTY3n10uXLkVVVRWysrIAAGPGjMGVK1fw1VdfYezYsTh27BgefvjhGA2ZiIjCFXEfeX19Pbq6ulBWVoZNmzahvLwcsizjtttuw1133RWHIRIRUTDCr+wkItKDYKUV3S8IIiISHYOciEhwCS+tEBFRbHFGTkQkOAY5EZHgGORERIJjkBMRCY5BTkQkOAY5EZHgGORERIIT5sxOh8OBdevW4fz58+jt7cVTTz2FWbNmea5/8MEHqK6uhiRJKCsrQ0lJiYKjTbxQ98etoqIC6enpWLNmjQKjVFaoe7R7927s3bsXmZmZAIANGzYIvcd+pELdn4aGBmzduhWyLCMrKwuvvPIKTKbAJ2dpTbD709bWhueee87zsVarFeXl5Vi0aFFiBicLYu/evfLLL78sy7IsX7p0SZ4xY4bnmtPplO+55x65vb1ddjqd8pw5c+SLFy8qNFJlBLs/bm+99ZZcWloqv/LKKwkenTqEukfl5eXyqVOnFBiZOgS7Py6XS37ggQfklpYWWZZl+Z133pGbm5uVGKZiwvk3Jsuy/Pe//11eunSp7HQ6EzY2YWbk9913H+69917P75OSknx+vX//fhiNRly8eBEAMGLEiISPUUnB7g8AnDhxAidPnkRZWRnOnj2b6OGpQqh7dPr0aVRXV6OtrQ133XUXnnjiiUQPUVHB7o/WTgSLRqi/P0D/EZgvvfQSXn31Vb/X40WYGvmIESMwcuRIdHZ2YvXq1Xj22Wd9rhuNRhw8eBAPPvggbr/9dhiNwnyPiolg9+fChQv4zW9+g8rKSuUGqAKh/g7NmzcPVVVVeOONN3D8+HF89NFHygxUIcHuj/tEsMWLF2P37t3429/+hiNHjig3WAWE+vsDAIcOHcLEiRMT/k1OmCAHgK+//hrLli3Dgw8+iKKiokHX58yZg08++QQOhwPvvfde4geosED358CBA/j222+xYsUKVFdX489//jPq6uoUHKlyAt0jWZbxk5/8BJmZmUhJScGMGTPwz3/+U8GRKiPQ/cnIyMDYsWMxYcIEJCcno6CgAP/4xz8UHKkyQmXQ+++/j9LS0sQPLGFFnCFqa2uT77vvPvnTTz8ddK2jo0NesmSJbLfbZVmW5crKSrmuri7RQ1RUsPvjrba2Vrc18mD3qL29XZ4+fbrc2dkpu1wuedWqVfLHH3+swCiVE+z+2O12eebMmZ4a+dNPPy1/9NFHCR6hssL5NzZr1izZ5XIlcFT9hNn98OWXX4bFYvH5kaWkpATd3d0oKyvD22+/jb1798JoNCI/Px8VFRUJrVEpLdT9caurq8PZs2d12bUS6h6999572LNnD1JSUjB16lSsXr1awdEmXqj7c+TIEWzfvt1zItiLL76o4GgTL9T9uXTpEpYvX44//elPCR+bMEFORET+CVUjJyKiwRjkRESCY5ATEQmOQU5EJDgGORGR4BjkRESCY5ATEQnu/wM4odLH/AdfOAAAAABJRU5ErkJggg==\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAXIAAAEFCAYAAAD+A2xwAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjUuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/YYfK9AAAACXBIWXMAAAsTAAALEwEAmpwYAAAzs0lEQVR4nO2de3RU9bn3v3tmwkyETJA2JK7KRW4xrxVUVFZfGi4FqYGiQLiJTY+XI0vpqS3gKYFTYoRTFCt29ZyKXZxW2+VCao/BVpThWMxxoYh4qStaTgglSoueN4EKJANJJjOZ/f4x7GFmz77O7D3X72ctl2TPzJ5nZvZ+9rOf3/N8H0EURRGEEEJyFkemDSCEEJIadOSEEJLj0JETQkiOQ0dOCCE5Dh05IYTkOHTkhBCS49CRk6zgs88+Q2VlJf7zP/8zbvuvfvUr1NfXR/8+c+YMfvSjH+GWW27BbbfdhgULFuCXv/wlBgYGAABHjhzB5MmT8fHHH8e9Zvbs2XjjjTcss/f48eO4/fbbcfvtt2PGjBmYPHly9O9f//rXOHz4MCZOnBjdJv131113ob+/HwsXLsTWrVvj9tnZ2YkpU6bgrbfessxOUiCIhGQBJ0+eFK+++mpx8uTJYnt7e3T7L3/5S3HdunWiKIpiV1eXOGfOHPE//uM/xGAwKIqiKJ47d0588MEHxTVr1kRfs2vXLnHmzJniuXPnxGAwKNbV1Ynbt2+3zfampiZx5cqVcdveeecdcd68eaqvaW9vF6+//nrx3XffFUVRFMPhsHj33XeLTz75pG12kvyFETnJGjweD+6++2489NBD6O/vT3h8165dqKqqwj/+4z/C5XIBAEpLS/H444/j0KFD+OijjwAAy5cvx+TJk7FhwwZs27YNXq8X999/f8L+3nrrLcyfPz/6d3d3N2666SZ0dXXh+eefx2233Yba2lqsWLECx48ft/SzjhkzBvX19Vi3bh16enrw/PPPo6+vDw8++KCl70MKA1emDSAklgceeACHDh3CT3/6U6xbty7usQ8//BDV1dUJr3G73Zg8eTL+9Kc/YeLEiQCARx55BAsWLEBrayv27NkDQRASXjd16lRcuHABH3/8Ma699lq88sormD59OoYMGYItW7agubkZw4cPx+9//3t88MEHGDdunKnP8re//Q2333573LZbb70VDzzwAABg6dKleOutt7BhwwZ89NFH2LVrF5xOp6n3IASgIydZhsPhwE9+8hMsWLAAX//61xMeDwaDiq+TR/CffvopLly4gEAggCNHjuDmm29OeI0gCKitrcVLL72Ea6+9Frt378YPf/hDOJ1O3HrrrVi+fDlmzJiBr3/965g+fbrpzzJy5Ej84Q9/0HzO5s2bMWvWLGzatAnl5eWm34MQgIudJAu54oor8Mgjj2DdunU4e/ZsdPsNN9yAd999N+H5UlR9ww03AIgsbn7ve9/D+vXrsX79eqxZswanT59WfK/Fixdj3759aG1thd/vjzr8J554Ar/4xS8wcuRI7NixA2vWrLHhk0ZSQ16vFyNGjLBl/6QwoCMnWcmtt96KadOm4Te/+U1024oVK9De3o4dO3ZEq1S6urpQX1+PG2+8ERMnTsTAwABWr16NmTNn4lvf+hZqa2tRXV2N1atXR18TS3l5OSZOnIiGhgYsXrwYQORCMH36dAwdOhR33XUXfvCDH8RVwRCSbTC1QrKWH/3oR/jggw+ifw8ZMgQvvPACfvazn2Hu3LkoKiqCIAj41re+hXvuuQcA8Pjjj6O3tzcuv97Q0IClS5fiySefxD//8z8nvM+SJUvw/e9/H08//TQAYNiwYXjggQdw1113wePxwOl04l//9V9N26+UIweAZ555Bl/60pdM748QNQRRpIwtIYTkMkytEEJIjkNHTgghOQ4dOSGE5Dh05IQQkuOkvWolHA5jYIDrq4QQYoaiIvWu37Q78oEBEefO9aT7bQkhJKcpKytRfYypFUIIyXHoyAkhJMehIyeEkByHjpwQQnIcOnJCCMlxKJpFSJrxtXZi+5sn0OkPoLzEjVXVo1FTRS1ykjxpF80KBgdYfkgKDsl5d/gDqs+poFMnGmiVHzIiJ8RmfK2d2PLaX9AXCms+r8MfwJbX/gIAdObEFHTkhNiEkShcTl8ojG3N7XTkxBRMrRBiA0ajcC2YaiGxsLOTkDSz/c0TKTlx4FKqxdfaaZFVJF+hIyfEBjpNpFO06AuFsf3NE5bsi+QvzJETYgPlJW5TuXEtOvwB3LztQFypIksYSSzMkRNiA1bkyJXwuBy49ooheO9kd8L2DXPG05nnMVo5cjpyQpLASEQc+5x0nGQVJW7sWTklDe9EMgEdOSEWkEpTz/wdhy1LtaghAHh37bS4bUzB5A9sCCIkRXytndi87xiCYe24R62pZ+qYy9HU0mGrjeUl7ri/5ekduW108vkDI3JScKg5MKXtAEw39QCA1+3EZYNctkfhsWyaWxnniNXuAqS7BnkOn3n27IapFVJwxDrlErcTgiCguy+EErcTvcGwbmSda9ROqkD97Alx227adkDxuQLUq2qYZ89emFohBYU8pdAdGIg+FvvvfEAtJ6/VRFRe4latc7eq/p2kFzYEkbzDiq7KXKGnP6S4XauJaFX16IR8uoTadpLd0JGTvKOQosruwAA2+doSInCt76CmqhyrqkfD44o//T0uR3RdgOQWdOQk7yi0qDIkJkbgat9BxcXtNVXl2DBnPCpK3BAubudCZ+7CxU6Sd9jVVZnteN1O+AMDKC9xY+qYy/HqkVMJ30Gpx4W13xhLh52DUP2QFBTyaNPrdqK4KP8P9e7AAERE6sVfPXIK864ZDq/bGfecrr4QFRXzEEbkJO9Ri9AFIC2t85lCitCVPiPLDHMPRuSkoFGrYikvceO9tdOieeN8o1vFiQOFtSBcCNCRk7xHr2ZaqYIj30l1QdjX2on5Ow7j5m0HMH/HYaZqMoyhhqAFCxagpCQS1l955ZV49NFH4x7v7e3F3XffjR//+McYO3as9VYSYgJ5q73X40JXX2K9tSBEuh8dAhAWEf1/IZBKmeFj+4/F6cZwaHTm0XXkgUAkannuuecUH//444/x8MMPo7OTV2RiD2bEnZSEolwCUOQQEtrypT/l/yfq+Fo7FcW/+kJhNOxtw/Y3T3D4RQbQdeRHjx5Fb28v7rnnHoRCIaxZswbXXXdd9PH+/n489dRT+OEPf2innaRA0VPwk6OUDw+JgHeQA19Ks4hVNpOsCqLe2LkOfwANe9vQsLctYTujdvvQrVppa2tDS0sLlixZghMnTuC+++7Dvn374HLFXwPq6urQ2Niom1ph1Qoxg5aCn1LVxc3bDigu8Ela3VOePMDI+yJSaaKS/ozHKWDDNyMiXLFOPtULIatlkicl0ayrrroKo0aNgiAIuOqqqzB06FCcPn0aV1xxhaVGktzE7ttns+JOas5GWtyjE7+EloBY34CIhr1tcACQ7m+suJtJd7VMoaR3dB35iy++iGPHjqGxsRGdnZ04f/48ysrK0mEbyULk8rCxkrB23D7rOWY5ajrb1BBJDqt7Y+2UT5A7bXl3az6nd3RrrhYvXgy/34877rgDq1evxpYtW+Dz+fDCCy+kwz6SRUj56o6LMyi7AwMJC4h9obBuHtUMZsWdqCGSHpKpvbfzgio/Njv8ATS1dCSsl1h9fGYL7OwkhjE6d1JpdmQqWHl7nI7ZmfmO1+3E6/801dR3WewSsH7OBNsuqGZssfr4TBccLEEswWh+U7p9tsoB11SVW+YAlFIvxByCIAAwl+8WIdhlDgBztuSjOmZhtbORlDByAki3z0q3utkg1hSbegFgs3vJT7ovNlepHQ8OhS/V7pSGUeecr+sldOTEMEr5apcQkUaV56OV6rnNnsx2tYFLgxXyVWPFbiSnqbZ+oVYZZGfFipottZMqCmK9hKkVoohWWsRIukTtpO3wBzB/x2Hd15ttBDL6eZgfTw2XcKm9X+14UPue7UxpmDk28xEudpIE1GRflSa1qzl8o4tPHpdDMUoy2wgkt2lbc3tUX8XjFDAgIqHChugTK/XrdTvx0Kxxus5R6fhR+52JcbjYSUyhJvsqaWwc/OQsOv0BeIoc6A1eep7Unt3yeZfhRUUp3SI/wZOd8i4XdAIizS3EPMk630KPjjMBI3KSgFqbeyqUqigQSlTITvZkInJfa2eCxgcxTqnHheIiZ1xDzcFPzqLDH4gqQ8p/J5I+GJETU1ihqSEnEAprOnN5DjyZDs18bPRIFx6XI26Wpzw9ImWl8rk7MpdhRE4SsCuy9bqd6B8QNdMtsRG32Tp0O+4kCgUpF+51OyEIgubdE0Dxq0zAiJyYoqaqHC2fdynqTqeCPzCAR+ZWalaPxObAzTQC+Vo7IQhAesOS/EH62rSEtGLhqLjsgnXkRJH62ROwaW5lXA1uqpSXuFFTVY49K6eo7i+ZEjUpDcCilPSRj92RuQwjcqKKPCLWKinUG5Mmz29bqVKoVmVD7CFfuyNzGTpyYhg15xtbohbbeKNV6WBliRpv8+2HVSvZDRc7iSmyUaifiob2smluZcZ/Y8LFTmIhVioRWsXUMZdbvjBLLpFtvzdJhIudJCdQE9DytXbi1SOnMmxdfvPY/mOZNoHowIicZD1aAlpc6LSfppYOTPpKKSPzLIY5cpL1aLXrd17UOyf24rhYo6+nWJlt6yf5hFaOnKkVkvVoCWixnjk9hEVoDgjJ1kEihQIdOcl61Jy1FPWR9KI0IERtkEjD3jZLh4IQZejISdajNv2Ft+6ZQ36XpFXLbzQ6t2siVCHAxU6S9VDfOnPEDpaIRX6XpKeYqaQ7H5tT93pcuBAIIUSVxaSgIyc5QTbWrxcCSk48dsC25IhL3E4UOQTNKUyxUbu8EklJbVFt6AhJhI6c5CScwZkZBEQc7BOvH0dvMBx13N2BAbgE9QgeiI/ijZaNdvoDrIYxAB05ySjJnKRK49xIetCSuw3p1IHGLkwb1cfxuARLh3DnK1zsJBnD19qJzfuOxZWsbd53THORy9faSSeeg5R6XHGO12jZaF8ocRCJUtVMoUNHTjLGtub2hJxqMCxiW3O76mt4Auce0hi5WJQqkZRQC/KpeBmPodTKggULUFIS6Sq68sor8eijj0Yfa25uxlNPPQWXy4Xa2losXbrUHktJ3qE2TkxrzBhz4rmB1HWrli6TVyIJKnr2ajr3bASLR9eRBwKRE+e5555LeCwYDOLRRx/Fiy++iOLiYtxxxx2YOXMmysrKrLeUFCTyEjWSGxhZ64itRJJXsQCRSH7eNcPx6pFTlgwgyWd0z4yjR4+it7cX99xzD0KhENasWYPrrrsOANDe3o6RI0eitLQUADB58mS8//77qKmpsdVokh943U7VGZFSnlyvRI1kJ1IKTGshW77QPe+a4Tj4ydmE50/6SimrVnTQdeQejwf33nsvlixZghMnTuC+++7Dvn374HK5cP78+WjKBQAGDx6M8+fP22owyR8emjUOm3xtitUOW177C9wuB5UNcxSpukSt2kRJ0fLVI6fipk1JsIdAH93Vhquuugq33XYbBEHAVVddhaFDh+L06dMAgCFDhuDChQvR5164cCHOsROihhSNqZWs9YXCjMBzGIcAzWoTNW0WLmYnh64jf/HFF/HYY48BADo7O3H+/PloDnzs2LH461//inPnzqG/vx/vv/8+rr/+enstJjlPrFIeyU/UGjylahMtRUtiHt3UyuLFi7F+/XrccccdEAQBW7Zsgc/nQ09PD5YtW4b6+nrce++9EEURtbW1KC/nLRDRhsMg8h+19Q+p2kRNm4XVKMnBwRIk7dy87QCHQeQxxS4BvQo5M5cANNRUKubIgUg1ilKOnETgYAmSVTDqym+UnDgADHZf6u6sqSrHhjnj4XU7o4+7DTQIEWX4zZG0wxrg7KAozWd/t8Lidf/AJaff1RdK0C2nRrkx6MhJ2uGtc3YQTPMyhfxOTK9yhePjjMNWuQInUxKhFSqLXVpNQiS3kd+J6VWuaDl6BgPxMCIvYDIZ8aiNb3to1ri4vCnJD2onVSQ4X61ZrABLFM1AR17AZLIpQ1rsqihxQ0AkQpcqFh6aNc729yfp46YRXtTPnpCwXWsWK6Dv6MklmFopYDId8ai1XtdUlWNbczs7O/OEDz7rVtyuN4t1VfVoxRJFLpYnQkdewGRzU8bab4xFw962TJtBLEBjjKemjgqHbhuHjryAyeaIp6aqnI48T3AIyb+WglnGoCMvYLI94lGrbCG5xcKJFZk2Ie9hiz7JWvTauKXSSTr77OWmEV5sX3qd6uOZKn/NRdiiT3ISeWVLqceFQU4BD+9tw/wdhwEAe1ZOwXtrp6F2UkX0Ft4hRByIkZmQxF5OnlO/yLLhxzoYkZO0kGrkpRSdA5H6ZKXSttj37PAHorMfK0rc6OkPsekoTQgA3l07TfGx+TsOK95NVZS4sWflFJstyz20InLmyIntKE2DiZ0WYwQ16dumlg5M+kqpahmj0mgxOvH0UV7iVr2IGyl/9bV24onXj0d/s1KPC2u/MZbpFxm89yS2Y0XjkVZtu5H9cJhF+ilyCJg65nLV9Ilew4+vtRObfG1xF96uvhA27zvG9IsMOnJiO1Y0HmnVthvZD4dZpB+XA/jj0dOqF3Gtzk5faycaVea5BsMiR8LJoCMntmNFq7VWbbuR/VCfI/30hkTVNFaHP4Dtb57AvGuGJ8g0AJHh21qNRPw942GOnNiOFY1HNVXlaPm8C00tHXHbje5HrYuVZI4OfwCvHjmVMBVo/o7DundP2dB9nE0wIie2oyWQZYb62ROwaW5lUvtRuo0nmUdprUQv2i5yCFnRfZxNsPyQFAzy6omuviB60z1doUCRyj/VqChxR3+X3uCAqmBaIVetaJUf0pGTgsXX2onN+44hqOVhiCUIMJ7ecgmAIAhxvwsHM7OzkxBFaqrKsfHWCSj1cKnIbqT6cSPprZAIFBc54lJo864Zju1vnuDsThUYkZOCh5G5vSjp40hpFLUIPbYjVKonVypFrCggfRZ2dpKCIFkZgG3N7XTiFlPqcaG7L5TwO8i7bdXa9GOrUp54/biiEweS6xLOR+jISV5gRgYg1uG7nQL6BujErSYQCmPRpAoc/OQsHt7bFm0Akn4LLeXK2JJSX2unrqQCBzIztULyBKMCTGriWyQ9lHpcmF35Zbx65JTibxCbKjHzW2mJc+ULTK2QvEet9rjDH8C0f3uLZYZZQldfKKGpS0J+0TUjq1DoDUKMyElWYjbfrRaRk9xCitgPfnI2qd9TALBIQ9o4l0m5jvyLL77AokWL8Mwzz2Ds2LHR7b///e/xq1/9CiUlJVi4cCGWLFmiawwdOdFD6Za6yCGguMgBf2BA0bE/tv+YaqRHCpN8q2hJKbUSDAbR0NAAj8cTt/3MmTP42c9+hpdeeglerxd33XUXvva1r+HKK69M3WIZHAeV2xj5/WKfIyh0AQbDIoIXF706/AE07G3Dno//H9pOXaC+OFFEWvBu+bwLBz85m9f+Q7c6f+vWrVi+fDmGDx8et/2zzz7D1VdfjaFDh8LhcODaa69FS0uL5QZyHFRuY+T3kz/HaCXgeye76cSJJn2hMJpaOvLef2g68t27d2PYsGGorq5OeGzUqFE4fvw4/v73v6O3txeHDh1CT4/1KRMrhhKQzGHk96NWOEkn+eg/NFMrTU1NEAQBhw4dQmtrK9atW4enn34aZWVlKC0txfr16/G9730PFRUVuOaaa3D55ZdbbqAVQwlI5jDy+/G3JOkm3445TUe+c+fO6L/r6urQ2NiIsrIyAEAoFEJLSwt27tyJUCiEu+++G6tXr7bcQLU23kIvN0qVdK07GPn91J6jp5hHSLLkm/8wLZq1Z88evPDCC3C5XCgqKsKiRYtQV1eHuro6DBs2zHIDtcZBkeRI57qDkd9P7TmNNZXYNLeSOuIkAY/LgWKXoPiYcPG/ihI3aidVFIT/yIk6clatWIvRLkgrMDoFXes3jm3nlqL0ihI3Rgx1472T3ZbaS7ITlwAMdsfrtzy8tw1Kzkve5Zkv/oN65CSOm7cdMHQCpIpSPbgVutJaOh0k/1CrB09nQJINUI+cxGHFMGQj2FFxFJsWIrlJcZE5t6MWQTPtegk68gIkXSeAHRVHLFXMfczq3qhd+K2aBZsPUDSrAJEOdLvzhnZUHOVb2RjRR+s3l+ubFyp05AVKOk6AVdWjFXPkZiJ/+UJVidup2s3pdTsRHAijV20KAclJ8q1U0A7oyIltpBr5Kw2LKHIIcAmImxijtICqtqBLsg+HAIgiFId8yIdMSMeS1+OCKIqqImqFBh05sZVUIn+lfHgwLKLU40JxkVPz4mB0YjvJLPKLsFqpoPyi3tUXiu6D495YfkiymFTKJDlQOftxCEBjTaUh52tEb77U48L+7/5fq8zLOjghiOQkZhdL5dGcUwCCdhtJksIlAA0XnbiRhh0ji9xdfSE8tv9YXg6V0IPlhyRrUSqTBCK30vN3HNaUwu3wBzhUOYsRhEh7vVG5CKMLnk0tHXknUWsEOnKStcTWCcuRn/CsL88tgmER2988YbhpTO2irkS+SdQagTlykhPotWOzSiX3kCSv1H63ihJ3XLoFuFQBpfU6q6UmsgXmyIltpEuQSC1HKjl3VqnkHlK6RO13k7ZLd18b5oyPaqj4WjvRsLdNc7+FBFMrJGnSKYerdXL6Wjuxqno0ihzKsqaxeN1OK80iSSLVhxtNmcjTLTVV5aidVKG630KDjpwkTTrH8GmdnNua21FTVW5ajIlYh1P/Ghql1OOK1o4r6aWoIb8rq589AZvmVqLUcymxMEjBEF9rJ+bvOIybtx1IWCTPF5haIUmTjjF8sakbNaTmEL+BQcwc1qyPpPluZkKT0QKhm0Z4sX3pdXHb5E1jaushghA5HuSpu0BMMNEdGIhrDlLqDs7H5iGGMCRp7JbDladu9J4rmIgKiTphEXhv7TQcXjMNm+ZWQmUQT1K8d7Ibj+0/pvmcVdWjFd8zLCIhdad1V+hr7USjr60ghrfTkZOksVsO12hJoccpYJOvjfM9LWTKkwfw2P5jqKkqx2C3tTfuTS0duikOQeWqLHfCWovgW177i+ox0eEP5FWKhakVkjR2y+EaSdG4hMgtNwUPrSUsRhzun06ei9M1sQqtFMf2N09oSivEpl20BnfrBQH5lGKhIycpYaccrpGSwuuv9HJup418eqbPtn33hcJo9EVKCGOPISMXcClXriaVbOROToru88GRsyGIZBVyqVJ/Xwjs18xvihwCioscUUnanv6Q7qJ07FxOpV4GozNdpeahXBjQzJmdJCeQL2520YkXBMGwiO7AQLQXoTcY1l1g1Yvajdanez2utPZD2AUjcpI1GJEqJYWBpDmvdjxIwyi8HhcuBEIJayRetxOVwwcbSrsVFzkU54jGRv3ZACNykhNwHieR6O4LYc/KKdg0t1Ixsg6LiN61KS10dwcGDK+dqA2DzqXjkY6cZA2FqJFBlJGOBXnnpwEVBsttyAXoyEnWYFQvheQ3HpcDU8dcHm2r3/7mCayqHo13105DuhLBRQ4hpzRbmCMnGUVeLdDV249eFoUXLBUlbkwdczlePXIqoaRww5zxhqtRrLQnWypYmCMnWYlStQCdeGFS7BKwaW4l9qycgoOfnFVtqzcTJXtcDtROqogT5LpphNeUXVZUsKRDtMtQRP7FF19g0aJFeOaZZzB27Njo9pdffhnPPvssHA4HamtrsWLFCt03ZEROJFilQswg1XzftO2A6nPkwyiUImlfayeeeP24KQG1ZCtY5KJdwKW7C7NRfkoReTAYRENDAzweT8Jjjz/+OJ599lns2rULzz77LLq6ukwZRgobo06ceXMCXFp8VJO6dQgw1NBTU1WOywaZa2pPtoIlXVLPuo5869atWL58OYYPH57wWGVlJfx+P/r7+yGKoqrQDSFyNAWTYv7tdTux8dYJKOKhVdDEirGpNftIJYlG0iFmHbO8gsVouiQdUs+AjiPfvXs3hg0bhurqasXHx48fj9raWsybNw8zZsyA12su/0QKF62IJDbX1x0YQMPeNhQP4mQfLQQAtZMq0lqelwoeg5MopNx2bCrCSEmiXtRrprRQruhpphPUbqlnCU1H3tTUhLfffht1dXVobW3FunXrcPr0aQDA0aNH8cYbb+D1119Hc3Mzzpw5A5/PZ6lxJH8xG5F0BwbgEhA3DYZcQgQw6SulOLxmGt5bO01z0k6mKXYJ6DMwiaKixI13107DnpVTEtIkNVXl2LNyimZJolbqTquFv9TjgtftVLyIAObSJXZLPUtonhU7d+6M/ruurg6NjY0oKysDAJSUlMDj8cDtdsPpdGLYsGHo7qYKHTFGMsOSQyJQXOS0RVY1H2j0teHhvW0oVynhyxaMVCZJzs6ImJXWsaQ0UQhITYLZTLrEbqlnCcN15JIj/5//+R/09PRg2bJl2LVrF5qamlBUVISRI0di8+bNGDRokOZ+WLVCAO0p6FoIgO60IJIaXrcTvcGwpia4lUi12lKNuDRirlRBR0Wp4kPrWLJSL0W6qKhdNGI/hx1OW6tqhQ1BJGNolZGp4XU7c3buZpFDSJtzTBaPywG3y5HWux6prPCx/cfQ1NKh+3wlCVs15yrtO1WUyghj8bgcmHfNcNVGJiucuZYjZ8KRZIxSj8u0wzjfn5tOXE1hL5sodglYP2c8Hk7iTikVykvc8LV2GnLiwKXct55zlfYdi16qRu1xrbGDsZG4Wu7c7s5QOnKSMQJBbaeslEbJ8oBWlWx34kAkd51MuisVpFy42bpqyeFqOXG1ahPpNfJxc1qPq+XFBSB6d6B2AezwB3DztgO2Dqxgiz7JGHqVCznqs4kGHqeAUo8roSLEbBWTFDWrYbbaxNfaiUZfm+rjRsoItUoK7R5YwYicZC3SohfJPWonVeDVP3dGL9YCgEWTKlA/e4Li881WMUmpD6XXmMmhA5ccrNqx1ukP4JG5lYqt9rERv9L8UDl2pVoYkZOMoVcTvnBihaFxXST7+OPR00BMp7cI4NUjp1SjUbW67uIi5d9fKrFUQtoe27ijhUOAbp5d3oSkFPHLn6OGHQMrWLVCbMPIwtIjvjYoZVhqL0ZvyZYpkuxFTRpW6XgBoCo6pRZpSxG5EVE2lwDFCUPy90omglZ7/2RLIiljS9KOkTbmmqpyPFxTGReZe91ObJpbGXcLnitt51aSz59ZLVdcU1WOVdWjoymTRl8bGva2QYAYjXAdAjDvmuGaefUOfwCzn3rbUKpmsNul2QUrvVcypKurE2BETmwimWhEHpFlc3ciSR35sWCknBC4VLP90kcdlqyhbFLIf8e+Vyp14EY6U43ChiCSdm7edkCx6kStQcNoM0ihUEgLvQ4hsh5y8JOzadendwjA4TXTolUrSt+5QwBEEbaWDxqBDUEkrfhaOyFcPPjlKJVomWkGKRTcTqFgpiWFRWTs95ccd01VuWoduPQcKSXU8nkXDn5y1lbtFLMwR04sRbo9Vops1PKDVovs5wO55sRzNadfYbAOXKIvFEZTS4chCdt0QkdOLEWr226Qiga1HeVYJH04EEmNWIHLxguC3sKjlrStFnZM/DELHTmxFC2n3B0YUIxerBbZzzWKDXqvm0Z4s3LsXRjA/ra/W7IvQRDgdVs/RKTU4zJVB26WTAcjdOTEUvScslL0kmwklC8YTaO8d7Ib132lJCuHa1illhgMi7hskAub5lZaekx09YXQ8nlXdBiF0rAK4NLACrOXy0wHI4V79hBbMOKU5dGLFAnZEYnlG++d7Mbab4w1HMXnIh3+AGqqyjHvmsQ5wanQ1NJhOJet5piLixxpqw03A8sPieWkIsAvr7tNdzkasZZiV3LVN163E4IgWK6LrqTD0ukPoOTi+3X3hVDidiIYFhMUK6WackB94o+VdeNyWEdOMoJSg4eaAL9LiHTZdfeF4k6A2U+9zdFuOYreEJDaSRUZKTv0up3oC4XRb2BuaOxrHpo1TtMpqx3v6RgsQUdObEUpQtFTowMi03SKixw5Ow2IaCPdlSWro6OnkWI1RjqS1Y5pqxqK6MhJxvG1duKJ14/TMRMAkZTLIFduDdJ+72JHsq+1E4++diyplFEqETo7O0lG8bV2YpOvLa0RFMluekMiekO548QBRBdKU1HjtEuPnBE5sRR5KmXEUDfeO9mdabMISRmHAAxyCLqTrYywaW6laWfOiJykBXnk3eEPsOqE5A1hUX88oVEaL0b1VkXmrCMnliANgGD6hBB9wrBWY4iOnKSMNOmHEGIcK9v6mVohmsTmvL0eF0RRRHdgIKqXXUi62YRYiZVt/XTkRBV5g0NsqZjkvOnECUkOteHRycDUClFFS5KWEJIaBz85a9m+DDnyL774AtOnT0d7e3t02+nTp1FXVxf978Ybb8SuXbssM4xkHlacEGIfac2RB4NBNDQ0wOPxxG0vKyvDc889BwD48MMP8dOf/hRLly61zDCSeZj/JsQ+rMyR60bkW7duxfLlyzF8uLKkpCiK2Lx5MxobG+F0UoY0n6ATJ8QeXAIslb7VdOS7d+/GsGHDUF1drfqc5uZmjB8/HmPGjLHMKJIdJDMphRAST3GRI04/3ut2oqHGfGenFpot+nfeeScEQYAgCGhtbcXo0aPx9NNPo6ysLPqc73//+/jOd76DyZMnG3pDtujnDkqynBIelwPXXjEEH3zWzcid5DWpltpqKSeaIekW/Z07d0b/XVdXh8bGxjgnDgBHjhzBDTfckKKJJBuRIgZJolM6kCtK3Jg65nK8euSU4QPb43JAEJAg1k9INiMAOLwmonp487YDSe0jHfM8TdeR79mzBz09PVi2bBnOnDmDwYMHQxDyd+xUoVNTVa54Czh/x2HDpYmS9jQAbN53DEGG8CRHiF2Q1JpY5XE5IEBUlLZNxzxPw45cqlAZO3ZsdNuwYcPwhz/8wXqrSNajFWUI0BbRl4/XyiVNapJ7CACSDR1iFyRXVY9WTDV63U7ccnUZXv44cR6oA9YuaqpBGVuSFPN3HFaMTpLJB6rtixCJVJxxqScyQjCZ18uDErWZnFrHsPT+dk4IYos+SQql6CTZaeKp5BClWYoP721L+kQnyjgADNGZu5kO1Oa8GsEB4EIgOScORC4eHf4Atrz2FwDqqUatY1i645Tvx0rYok+SoqaqHBvmjEdFiRsCIpF4siOskskhVpS4sWluJV7/p6moqSpPSx4yl3GYXMaqKHGj8eL3m0mk46p+9gRsmDMepZ7E2NPjcmDT3EpsmlsZ97jX7cQQt9MSaWVpso8aRo8/vf0kC1MrJONIWuZq6OXcpX0o5S+LixyY+3+G4+AnZ9HpD8DjElRnLRYJEZ1oI7MDaidVoH72BPhaO7GtuT2pPP+muZUAUhsdpkaF7PvSKiVVem1semzKkwdMl905BKDxYq30Y/uPoamlQ/W5xUWOaDWTQwAWTox8t2qopTeUuHnbAcvu1AQA716c26lkk9HfUWs/WjC1QrKamqpy1cHMRnPusaWSRk7wx/Yfw0sfdUTrgyXnoeQktPardKutN1UdiFwI5OWdZnEKyhcd6SITi9L3o/aeHf4A5u84HP2cyRQZieKl95RsUfq+k0EtvSHH19oJ4eIEeyO4BKChplL199CKumuqytHyeZfmBUvCq3BXkSqMyElWoBQxmp04biZSSwda0WBFiTtqp1STrxct3zTCi5PnAnGfr+XzrqQcpK+1E42+Nk0nLX3/ao5N6vxVeszrdmY0LWP0DiT2d4hd0Ez2WDRyEZcuGFbO7KQjJ1lDKo7YiguBWZv0HjO6ACst5knpH8m5x/6td2dg9jOZSbOsqh6dUP9f5BCw8dbIBaNxbxuU9lTqcWHtN8ZaZrcZjFRCxaZ/5Ogdi3q/vd73m0x1F1MrJCcwc8ssP4mUtNOlhaVk9wkg7oTs8AfQsLcN25rbMbvyy3FRtPRYw942lHpcpiol+kJhHPzkbPRzdPoD0b/VnENsBQRgzlGa0ZmXqjHk8V4wLKLl8y5M+kqpohMHItUam/cdgyiKcQO5zVRuJHtxN1IJFRaRlC3yC4T8Mxn5fq3u9mRETnICrVtWj8uheeJIkaFaJK2W2vA4BcumphtB/jli7yjUIkyv24n+AdHUnYiZBcBSjwvFRU7V6NabZHmiPCI1ciEF1D+b/PU9/SHDdinZIn/fIocQd0HS2o+R79fqiJyOnGQ9Rm5VjQgaFbsEDIhAfxqds1HU7Jdu/5Opk1fLmZtpwErWUeshVW74WjuxeV8blCR4XAJUHWfpxfmx/sAAStxO9AbDCakfPccbS6yMhN7agRVsmsscOSkwjDoevci8UPE4BWz45oSo49ArB0wHkuNUy69bgXQ3IUXpvcEBzTJRs84/Fd6zuPyQDUEk6zGST5QaR3KR2kkVtmq/9w2IaNjbhpu2HcD8HYfxx6OnLdu3gEjkbJbe4AC2Nbfb5sQBoLsvhKljLocgRPLYXX0hODVsDYbT48SVmppShYudJGPE5jUlAS0lTQqtmmfgkjSAtNCUjbotRQ4Bt11bjv1tf0+ICqXywXRg9XcjQj39oUU6hNI8RY6EO48BMb4BKRPYkQRhRE4ygpT37vAHIALoDkRue2O1LXytETU5Pf2W2MWvdCjNed3OuIkvRgiGRTS1dCg6sGxR9bUjUswUHpe6sw6EwqYlC6zEb8OaAx05yQh6JVqxmhQ1VeXwupXnwVaUuOMWjWqqylE7qcJSW+VcNsiFuddkrtHILiaUXZZpE1Ki1OOK0/1RIyxm9uJphy5Q/lyCSU5hJO8d+5yHZo0zrLZYP3sCJn2l1LY0S4c/gJc+yuxioR28d7I77SWXVqEkS5CO6hOzJKsQqgcjcpIRjEQlsc8xq7ZYU1WOPSun2LaIqOUgbhrhteU900Egx5y4Q4iU8inJEiycaO+dWTKk2mmsBiNykhHUpq1IKEUuRjs/zbyPHZw8F7Ct/tpucsuNRy6osSm4WORiXZlGnga0EkbkJCPII2yv25mQ47TioI99n3TR6Q/goVnj0vZ+hY58cTyW+tkTcHjNtIwv5NqVUpFgQxApKIy256eC1H6dDY03qVChU/YJROrIH7moqy59r1oOJVZbXqtbNZnRblpt77N+flDxDsnjFDD0skGqYmVKx4ckE6ClPGmHEic7OwnRQE2Ey6i2SSxyLZBkB084BWDwICf8gQHDsyH19mcm/S05Ra33kw+vkFAbROEQgMNrLnU06s19NftZtQY2qOmf6A15sHI2bapQ/ZAQDdRy70pVMlLKxEgjk7Tv7W+eMO3IB8RImaNc01tJUlZOkUNAcZEj7iIg2dzhD+hGu7FpALXZrFqpr4UTKxTvROSLj3pzX9Ued7scit+n1gK6WlOZ3qK7WnWVfHumtfDpyAlRQG/ikJmTNFnJUqXXSe8bG+UXuwQUOR0J0bvaawHjXbVmJy8BxicCGf2OjaoiauWgkx0WbuQCYKXEcLIwtUKIzSSbDsnE7XuukEwEnOxrlKqeYqWRrZQY1oI5ckIyiJ4Mr5LqnhXTjYh51HTRlWbKSr+RWYnhZC/QVD8kJIPolVpuvHUCGmoqDTc7EXuQ6//EpkguG5SYhZZkJMy23NvRbcyInBBCoF2holZWKZVfmmk6k1fvGIUROSGE6KBVoaIWdZdf7NaMvePSU1a0o8vUkCP/4osvMH36dLS3t8dt/+ijj7BixQrccccdePDBBxEIZJ8ONCGEGEHLWa+qHg2PK95dxla9SNo+766dBr0chx1dxrqOPBgMoqGhAR6PJ267KIrYuHEjHn30UezatQvV1dX4/PPPLTeQEELSgZazNiPappUzt6tVX7eOfOvWrVi+fDl27NgRt/3TTz/F0KFD8Zvf/AbHjh3D9OnTMWbMGMsNJISQdGCkrt3IArSaUJvX7cRDs8alX/1w9+7dGDZsGKqrqxMc+dmzZ/Hhhx9i48aNGDVqFO6//3589atfxde+9jXLjSSEkHSQjMKm0j6A9DQCSWhWrdx5550QBAGCIKC1tRWjR4/G008/jbKyMrS3t+MHP/gB9uzZAwD49a9/jWAwiPvuu0/zDVm1Qggh5klaa2Xnzp3Rf9fV1aGxsRFlZWUAgBEjRuDChQv461//ilGjRuH999/H4sWLLTKZEEKIUUxrrezZswc9PT1YtmwZfvzjH2Pt2rUQRRHXX389ZsyYYYOJhBBCtGBDECGE5ABsCCKEkDyGjpwQQnKctKdWCCGEWAsjckIIyXHoyAkhJMehIyeEkByHjpwQQnIcOnJCCMlx6MgJISTHoSMnhJAcJy8d+cDAANavX4/ly5fjzjvvxN/+9jfF523cuBFPPPFEmq2LoGdjtkxf0rPz5ZdfxsKFC1FbW4vnn38+IzZKqE2yam5uRm1tLZYtW4bf/e53GbLuEmp2vvLKK1iyZAmWL1+OhoYGhMPGZkDagZqNEpk8d2LJlellanZadv6Iecgf//hHsb6+XhRFUXznnXfE+++/P+E5u3btEpcuXSr+5Cc/Sbd5oihq2xgOh8XbbrtNPHHihCiKovi73/1ObG9vzzo7RVEUp06dKp49e1YMBALi7NmzxXPnzmXCTLG/v19ctWqVOGfOHPH48eNx2yW7AoGAuGjRIvHUqVMZsVHLzt7eXnHWrFliT0+PKIqiuHr1anH//v1ZZaNEps8dCTU7s+n8EUXt79Oq8ycvI/LZs2dj8+bNAID//d//xZe//OW4xz/88EO0tLRg2bJlmTAPgLaNsdOXvv3tb+PcuXMZm76k911WVlbC7/ejv78foihCEHQmz9qENMlq+PDhcdvb29sxcuRIlJaWYtCgQZg8eTLef//9jNgIqNs5aNAg/Pa3v0VxcTEAIBQKwe22frajEdRsBLLj3JFQszObzh9A+/u06vzJS0cOAC6XC+vWrcPmzZvxzW9+M7r91KlT+PnPf46GhoYMWhdBzUZp+tKKFSvw7LPP4p133sGhQ4eyzk4AGD9+PGprazFv3jzMmDEDXq837fbFTrKSc/78eZSUXFKNGzx4MM6fP59O86Jo2elwOKIXyeeeew49PT2YOnVquk3UtDGbzh0tO7Pp/NGyE7Du/MlbRw5EroT/9V//hY0bN6KnJyKdu2/fPpw9exYrV67Ejh078Morr2D37t1ZZePQoUMxatQojBs3DkVFRaiursaf//znjNmoZufRo0fxxhtv4PXXX0dzczPOnDkDn8+Xdtuamprw9ttvo66uDq2trVi3bh1Onz4NABgyZAguXLgQfe6FCxfiHHu22AkA4XAYW7duxcGDB/Hv//7vGbm70bIxm84dLTuz6fzRstPS8yf1DFD28dJLL4m/+MUvRFEURb/fL86cOVPs6+tLeF5TU1PG8nxaNgYCAXHmzJnRHN93v/td8b//+7+zzs7PPvtMvP3228VAICCKoihu3rxZ/O1vf5sROyW+/e1vJ+TIb7nllmgecuHChWJHR0cGLYwgt1MURfFf/uVfxEceeUQcGBjIkFXxKNkokclzR47czmw6f2KR22nl+WN6QlAuMGfOHKxfvx533nknQqEQNmzYgNdeey062Sgb0LMxW6Yv6dm5bNkyrFixAkVFRRg5ciQWLlyYETvlxE6yqq+vx7333gtRFFFbW4vycvuG4JpFsvOrX/0qXnzxRdx44434h3/4BwDAd77zHdxyyy0ZtjD+u8xmcmV6WaydVp0/lLElhJAcJ69z5IQQUgjQkRNCSI5DR04IITkOHTkhhOQ4dOSEEJLj0JETQkiOQ0dOCCE5zv8HoaQv8Dp6G6gAAAAASUVORK5CYII=\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "for i in df.columns:\n",
    "    if i!=\"TEY\": \n",
    "        plt.scatter(np.log(df[i]), np.log(df['TEY']))\n",
    "        plt.title(i+ ' vs TEY')\n",
    "        plt.grid()\n",
    "        plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<seaborn.axisgrid.PairGrid at 0x14b9aced310>"
      ]
     },
     "execution_count": 39,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<Figure size 1980x1980 with 132 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.pairplot(df,\n",
    "                 markers=\"+\",\n",
    "                 kind='reg',\n",
    "                 diag_kind=\"auto\",\n",
    "                 plot_kws={'line_kws':{'color':'#aec6cf'},\n",
    "                           'scatter_kws': {'alpha': 0.5,\n",
    "                                           'color': '#82ad32'}},\n",
    "               \n",
    "                 diag_kws= {'color': '#82ad32'})"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<seaborn.axisgrid.PairGrid at 0x14b921e7e20>"
      ]
     },
     "execution_count": 40,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<Figure size 1440x1440 with 0 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<Figure size 1980x1980 with 77 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=(20,20))\n",
    "sns.pairplot(df,\n",
    "                 markers=\"+\",\n",
    "                 kind='reg',\n",
    "                 diag_kind=\"kde\",\n",
    "                 plot_kws={'line_kws':{'color':'#aec6cf'},\n",
    "                           'scatter_kws': {'alpha': 0.5,\n",
    "                                           'color': '#82ad32'}},\n",
    "               corner=True,\n",
    "                 diag_kws= {'color': '#82ad32'})"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "***Pearson's Correlation Coefficient***: helps you find out the relationship between two quantities. It gives you the measure of the strength of association between two variables. The value of Pearson's Correlation Coefficient can be between -1 to +1. 1 means that they are highly correlated and 0 means no correlation.\n",
    "\n",
    "A heat map is a two-dimensional representation of information with the help of colors. Heat maps can help the user visualize simple or complex information."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:>"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAA1AAAAKqCAYAAAA0Zb/6AAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjUuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/YYfK9AAAACXBIWXMAAAsTAAALEwEAmpwYAACTSUlEQVR4nOzdeXhU5dn48e9MJiF7ZAmgICgoCriwiCugqNS+SvuqaAErWIu1rUt/dYX6utYN60LdtVIU96hVC66tilKidUERcAHBDUHZAyRkm5nz+2NsQgoho0Jmot/Pdc3lOed5ZuY+D0mce+7nPCcUBEGAJEmSJKlJ4VQHIEmSJEkthQmUJEmSJCXJBEqSJEmSkmQCJUmSJElJMoGSJEmSpCRFUh2AJEmSpPR3WSiU6hC4JA0WEN/mCdSKFeu39VukteLiAsfAMXAMcAzAMQDHABwDcAzAMQDHoLi4INUh6FtyCp8kSZIkJckpfJIkSZKaZOKQYAVKkiRJkpJkAiVJkiRJSbISJ0mSJKlJmakOIE1YgZIkSZKkJFmBkiRJktQkE4cEK1CSJEmSlCQTKEmSJElKkpU4SZIkSU1yEYkEK1CSJEmSlCQrUJIkSZKaZOKQYAVKkiRJkpJkAiVJkiRJSbISJ0mSJKlJLiKRYAVKkiRJkpJkBUqSJElSk0wcEqxASZIkSVKSTKAkSZIkKUlW4iRJkiQ1yUUkEqxASZIkSVKSrEBJkiRJapKJQ4IVKEmSJElKkgmUJEmSJCWp0QTqiiuuaM44JEmSJKWxzDR4pINGE6gFCxY0ZxySJEmSlPYavRZs2bJllJSUbLZtxIgR2ywgSZIkSUpXjSZQtbW1rFixojljkSRJkpSm0mUKXao1mkB16tSJM844o8Gx1atX89hjj23zoCRJkiQpHTWaQHXo0KFue86cOTzwwAOUlpbyox/9qFkCkyRJkpQ+vA9UQqPjcNVVV/HEE0/wwAMPkJWVRXl5OS+88ALZ2dnNGZ8kSZIkpY1GV+E77LDDmD9/Ptdddx0PPvgg7du3N3mSJEmS9IPWaAVq9OjRPPXUUyxZsoTjjjuOIAiaMy5JkiRJacRFJBIarUCdeuqpTJ06tS6RmjdvHtdee633h5IkSZL0g9XktWD77rsv++67L+vWrePvf/87559/Pk8++WQzhCZJkiQpXbiIREKjFaj/VlhYyOjRo02eJEmSJP1gJZ1ASZIkSdIPnZU4SZIkSU1yEYkEK1CSJEmSlCQrUJIkSZKaZOKQYAVKkiRJkpJkAiVJkiRJSbISJ0mSJKlJLiKRYAVKkiRJkpJkBUqSJElSk0wcEqxASZIkSVKSTKAkSZIkKUlW4iRJkiQ1yUUkEqxASZIkSVKSrEBJkiRJapKJQ4IVKEmSJElKkgmUJEmSJCXJSpwkSZKkJrmIRIIVKEmSJElKkgmUJEmSJCXJKXySJEmSmuQUvgQrUJIkSZKUJCtQkiRJkppk4pAQCoIgSHUQkiRJktLbslAo1SHQIQ1Sl22eSIaO3NbvkN6CZ2DFivWpDiOliosLHAPHwDHAMQDHABwDcAzAMQDHoLi4INUh6FuyEidJkiSpSZlmDoCLSEiSJElS0swjJUmSJDUpYuYAWIGSJEmSpKSZQEmSJElSkizESZIkSWpSZkaqI0gPVqAkSZIkKUlWoCRJkiQ1yUUkEqxASZIkSWrx4vE4F198MSNGjGD06NF89tlnDdqnTp3KMcccw/Dhw3nwwQe/9fuYR0qSJElq8V544QVqamooKSlh9uzZTJgwgdtvv72u/U9/+hNPPfUUubm5HHXUURx11FEUFRV94/cxgZIkSZLUpMw0zxxmzZrFoEGDAOjTpw/z5s1r0L7bbruxfv16IpEIQRAQCoW+1fuk+TBIkiRJUkJJSQklJSV1+yNGjGDEiBEAlJeXk5+fX9eWkZFBNBol8vXFW7vuuivDhw8nJyeHoUOHUlhY+K1iMIGSJEmS1LQ0WMZ844Tpv+Xn51NRUVG3H4/H65KnDz/8kJdffpkXX3yR3NxczjvvPJ599ln+53/+5xvH4CISkiRJklq8fv36MWPGDABmz55Njx496toKCgrIzs6mVatWZGRk0KZNG9atW/et3scKlCRJkqQWb+jQoZSWljJy5EiCIOCqq65i2rRpbNiwoa5ydcIJJ5CZmUmXLl045phjvtX7mEBJkiRJalqaZw7hcJg//vGPDY517969bnvUqFGMGjXqu7/Pd34FSZIkSfqBSPM8UpIkSVJaMHMArEBJkiRJUtJMoCRJkiQpSRbiJEmSJDXNzAGwAiVJkiRJSTOBkiRJkqQkNVqIW7RoEX/+85/Jy8vj3HPPpV27ds0ZlyRJkqR0kpHqANJDoxWoSy65hOOPP56DDjqIa6+9tjljStqwfeGNP8Or18MpRzTeb9Ae8PmUhsdyWsHM62C3zts0REmSJEnfI41WoMLhMIMHDwbgb3/7W7MFlKxIBkw8FQb8HiqqoPQ6mPYGLFvTsF/ndnDOMZC5Ucbcf1e44wzo3LZZQ5YkSZJaLheRAJK8Bioej2/rOL6xnjvCwqVQVg61UZj5Hgzq3bBPq8xEonTabZseP+Zy+PCL5otXkiRJUsvXaB5ZVlbGzJkzCYKgbvs/Bg4c2CzBbUlhLqzdUL+/vhKK8hr2ueU0uO5xWLqq4fFX39/28UmSJEn6/mk0gerduzdPP/10g+01a9ZQWlrK3Llzmy3A/3b5GBjYC/baGV6fX3+8ICdRjfqP7dskKlK7bA+XnABtCuChcTDqmuaPWZIkSWrxnMIHbGEYrr766rrtOXPmcP/99zN37lyOO+64ZgmsMRfdm/hvJAPevwNa50N5FQzeI1Ft+o8vV8Pup260f7/JkyRJkqTvptEEqqamhqeffpoHH3yQzMxMysvLefHFF8nOzm7O+BoVjcHZd8HzV0A4BJP/mZiq1zofJv0/GH5lqiOUJEmSvkdcxhzYQgJ16KGHMmzYMK699lp22mknTjnllLRJnv7jqTcSj42tKd988rT9iZseGzJ+28QlSZIk6fup0QRqzJgxPPXUUyxZsoTjjjuOIAiaMy5JkiRJSjuNLmN+6qmnMnXqVEaPHs1TTz3FvHnzuPbaa1mwYEFzxidJkiQpHUTS4JEGmrwP1L777su1117LP//5Tzp27Mj555/fHHFJkiRJUtpJOo8rLCxk9OjRjB49elvGI0mSJCkdpUkFKNWarEBJkiRJkhJMoCRJkiQpSRbiJEmSJDXN+0ABVqAkSZIkKWlWoCRJkiQ1zcwBsAIlSZIkSUkzgZIkSZKkJFmIkyRJktQ0MwfACpQkSZIkJc0ESpIkSZKSZCFOkiRJUtPMHAArUJIkSZKUNPNISZIkSU3LSHUA6cEKlCRJkiQlyQRKkiRJkpLkFD5JkiRJTTNzAKxASZIkSVLSzCMlSZIkNc3MAbACJUmSJElJM4GSJEmSpCRZiJMkSZLUNO8DBUAoCIIg1UFIkiRJSnM/C6U6Angk9anLNq9AhdJgnFMpCGAEU1IdRkqVcBIrVqxPdRgpVVxc4Bg4Bo4BjgE4BuAYgGMAjkFxcUGqQ/jmnLsGeA2UJEmSJCXNBEqSJEmSkmQhTpIkSVLTzBwAK1CSJEmSlDTzSEmSJElNcxlzwAqUJEmSJCXNBEqSJEmSkuQUPkmSJElNM3MArEBJkiRJUtLMIyVJkiQ1zcwBsAIlSZIkSUkzgZIkSZKkJFmIkyRJktQ0MwfACpQkSZIkJc0ESpIkSZKSZCFOkiRJUtMyUh1AerACJUmSJElJsgIlSZIkqWlmDoAVKEmSJElKmgmUJEmSJCXJQpwkSZKkppk5AFagJEmSJClp5pGSJEmSmuYy5sAWEqiBAwfWba9du5aioqK6/ZkzZ27bqCRJkiTpG4jH41x66aXMnz+frKwsrrjiCrp27VrXPmfOHCZMmEAQBBQXF3PttdfSqlWrb/w+jSZQGydJo0eP5r777vvGL76tDRsGF18M0ShMngyTJjVsb90aFiyAefMS+088ATfdBGedBWPHwooVieO//nWiX0sQxAO+uPQ5KucvJ5SVQZcrjqRV1zZ17Wtf+oivbp1JKBKm7fC9aPuzvnVttasqWHDsZLpPHkV293ZULVzB4ouehQCyd29P54t+RCjDWZ2SJElqeV544QVqamooKSlh9uzZTJgwgdtvvx2AIAi46KKLuOmmm+jatSuPPvooS5YsoVu3bt/4fZKawhcKhb7xC29rkQhMnAgDBkBFBZSWwrRpsGxZfZ9+/eChh+B3v2v43H79YMwYePvt5o15a1j7wnziNVF6lJxExewlLJnwIt1uPx6AoDbGkqtfoMdjvyCck8VHo+6lcMiuZBbnE9TG+OLiZwlnZ9a91tIbXmH7sw8hf0AXPhs/jbUvfcR2Q3dL1alJkiQpnaX5xT+zZs1i0KBBAPTp04d5/6miAJ988gnbbbcdU6ZMYcGCBRx88MHfKnmCFryIRM+esHAhlJVBbS3MnAlfj1ed/v0TydLLL8Mjj0DHjvXH//AH+Ne/YPz45o78u6mY9QWFgxL/2Hl9OlE578u6tqpFq2jVpTWRohzCWRnk9+9MxVuLAVhyzYu0HdmPSPv8uv4733ws+QO6EK+JEV1RQWbbvOY9GUmSJOkbKCkp4dhjj617lJSU1LWVl5eTn1//WTcjI4NoNArAmjVreOeddzjhhBO4++67+fe//81rr732rWJIagpfWVlZg/2Nr49KlcJCWLu2fn/9etjoMi0APvwQZs2CF1+EE06Am2+G44+Hhx+GW2+FdesS0/qOOgqefrp54/+2YuXVZORn1x/ICBNE44Qi4URbQf08znBeFrHyalY9PodIm1wKB3Vj2V9erWsPZYSpWbKWhSc/SEZ+K1rt3AZJkiRps9KgAjVixAhGjBix2bb8/HwqKirq9uPxOJFIIujtttuOrl27sssuuwAwaNAg5s2bxwEHHPCNY2i0AvX000/XPXr37s3TTz/N/fffz29/+9tv/CZb0+WXw/TpMHVqIon6j4KCRDVqYy+9lOgLiUSp79eXA/35z7BqVaJy9fTT9cdbgoz8VsQqqusPxANCkfBGbTX1TRU1ZBS0YvXf3mX9q5/w0ej7qfxgGZ+Pm0btinIAsjoV0esfv6XdqH4smfBis56LJEmStLX069ePGTNmADB79mx69OhR17bjjjtSUVHBZ599BsBbb73Frrvu+q3ep9E88uqrr67bnjNnDvfffz9z587luOOO+1ZvtLVcdFHiv5EIvP9+YqGI8nIYPBiuu65h30mT4G9/g0cfhcMOS1SjCgsTi0r07Jm4durQQxMLULQUef06s3b6R7Q+shcVs5eQ3aO4ri27e1uqP1tNtKyScG4W5W8tpnjs/uz64551fT4afT87XvpjMovz+fg3j9Jp/GG02qkN4bwsQuH0u9ZNkiRJSsbQoUMpLS1l5MiRBEHAVVddxbRp09iwYQMjRozgyiuv5JxzziEIAvr27cshhxzyrd6n0QSqpqaGp59+mgcffJDMzEzKy8t58cUXyc7ObuwpzSoahbPPhuefh3A4kQQtXZpIqCZNguHDE9c3TZ4Mp52WSJZOOSUxbe+CCxKVqerqxPS+Z59N9dkkr2jobqwv/YQFI6dAAF2uOoo1094jtqGGdiP60mn84Swa+zAEAW2G70VWh4JGX6vDqQfw+finCGWGCedksuMVRzXjmUiSJKlFSYMpfFsSDof54x//2OBY9+7d67YPOOAAHnvsse/8PqEgCILNNQwcOJBhw4YxcuRIdtppJ0455RQm/fc64cm8wQ+8qBEEMIIpqQ4jpUo4iRUr1qc6jJQqLi5wDBwDxwDHABwDcAzAMQDHoLi48S+509bVafDB/g+bTV2aVaN55JgxY3jqqadYsmQJxx13HI3kWZIkSZJ+CDJSHUB6aHQRiVNPPZWpU6cyevRonnrqKebNm8e1117LgpZyx1lJkiRJ2sqavA/Uvvvuy7XXXss///lPOnbsyPnnn98ccUmSJElS2kn6UrDCwkJGjx7N6NGjt2U8kiRJktJRmi8i0VyarEBJkiRJkhLMIyVJkiQ1zcwBsAIlSZIkSUkzgZIkSZKkJFmIkyRJktQ07wMFWIGSJEmSpKSZQEmSJElSkpzCJ0mSJKlpZg6AFShJkiRJSpp5pCRJkqSmmTkAVqAkSZIkKWkmUJIkSZKUJAtxkiRJkprmfaAAK1CSJEmSlDQrUJIkSZKaZuYAWIGSJEmSpKSZQEmSJElSkizESZIkSWqamQNgBUqSJEmSkmYeKUmSJKlpZg6AFShJkiRJSlooCIIg1UFIkiRJSnOPhVIdARyX+tRlmxfiOvD5tn6LtLaMLoxgSqrDSKkSTqJgw8pUh5FS63PbsWLF+lSHkVLFxQWOgWPgGOAYgGMAjgE4BsXFBakO4ZvLSHUA6cEpfJIkSZKUJC8FkyRJktQ0MwfACpQkSZIkJc0ESpIkSZKSZCFOkiRJUtPMHAArUJIkSZKUNPNISZIkSU1zGXPACpQkSZIkJc0ESpIkSZKS5BQ+SZIkSU0zcwCsQEmSJElS0kygJEmSJClJFuIkSZIkNc3MAbACJUmSJElJM4+UJEmS1DTvAwVYgZIkSZKkpJlASZIkSVKSGk2gnn32WQ4++GCOOOII5syZ05wxSZIkSUo3kTR4pIFGE6gpU6YwdepUJk2axG233dacMUmSJElSWmo0j8vKyqKoqIiioiIqKyubMyZJkiRJ6SZNKkCpltQ1UEEQbOs4JEmSJCntNZpHLl68mBtuuIEgCOq2/+Pss89uluAkSZIkKZ00mkD97ne/2+y2JEmSpB8gp/ABWxiGY445hpqaGmbNmsWaNWvo2LEjffr0IRx25XNJkiRJP0yNJlAffPABZ599Nr1796Zt27Y8++yzLFq0iJtvvpnu3bs3Z4ybiLz0Gtm33g+RDGqG/5ianx3ZoD20ei25515FqKqGePu2bLj6XMjJJvPJf5L910cJCvKoOeZH1Bz/P1BTQ+4friO8+EuC/FwqLz6T+E6dU3Rm30wQD/ji0ueonL+cUFYGXa44klZd29S1r33pI766dSahSJi2w/ei7c/61rXVrqpgwbGT6T55FNnd26Ui/K2m1SszKfjL3ZCRwYajh7Hh2J9utl/eAyWEV65m/f/7LQA5z/6TvAcfgXCY2l27s/aCc8EvCCRJkjYvI9UBpIdGE6jrrruOW2+9lW7dutUdW7BgARMmTOCuu+5qluA2qzZKztV3UP7YLQQ52eSP+j21Q/YnKK5PHLJvu5/aYYdSc+wRtPrLw7QqeZqanx5Ozo33sP6J2wkK88n7xTiiB/Ql8vK/CXJzKH/kZsIfLybn8luo+OuE1J3fN7D2hfnEa6L0KDmJitlLWDLhRbrdfjwAQW2MJVe/QI/HfkE4J4uPRt1L4ZBdySzOJ6iN8cXFzxLOzkzxGWwFtVGKrr+JFfdPIsjJod0vfkPV4IOIt2tb36eqmu0un0DW3PepPOyQumMFt/2FFY/cR5CTzXbjL6HVjFKqDxmUktOQJElSy9Do1+1VVVUNkieAHj16UFtbu82D2pLwos+Jd9mBoKgAsjKJ9t+DyFtzG/TJmDWP2kEDAKgdPIDIq28T/uJLYrt3J9iuEMJhYnv2IOPdD8hY+DnRwYm+8W47krHo82Y/p2+rYtYXFA5K/Bvl9elE5bwv69qqFq2iVZfWRIpyCGdlkN+/MxVvLQZgyTUv0nZkPyLt81MS99YU+eRTojt2JigshMxMavruRdY77zboE6qppnLYj1l/ypj6g1mZrLznToKc7ESfWAxaZTVn6JIkSWqBGk2gMjI2X6OLx+PbLJhkhMo3EBTk1R/IyyFUXtF4n7xcQusriHftRHjhZ4RWroHKKiKvvUNoQxWxnt2JTH8dgoCM2e8TWrYKYrFmPKNvL1ZeTUZ+dv2BjDBBNF7fVtCqrimcl0WsvJpVj88h0ia3LvFq6cIVFQT59T8PQW4u4fXlDfoEhYVUH7Dffz0xTLxtomqZ99CjhDZUUr3/vts8XkmSpBYrkgaPNNBoGMuWLaOkpKTBsSAIWL58+TYPanOyJ95N5O15hOd/Qmyv3esbKioJChpWUoL8XEIVGwiyW0HFBoLCfIKiAir/8BvyzryMeMdiYr13Jd66iOgh+5G96HPyx5xLtF9vYr13hUaSx3STkd+KWEV1/YF4QCgS3qitpr6pooaMglasuO8tCMH61z6l8oNlfD5uGjvffjyZxS2rGlVw61/IemcOmR8tpGaP3nXHQxs2EC8oSO5F4nEK/3wbkc8+Z811V0IotI2ilSRJ0vdFownUT37yE1asWLHJ8WHDhm3TgBpTddbJiY3aKAVHjSVUto4gN4fIW3OpHnt8g76xfr3JfOUNao49gswZbxLtvwdEY0Te/YDyB26AaIz8k8dRddYvyZg7n1j/Pai64LdkzJ1P+PMvN/Pu6SmvX2fWTv+I1kf2omL2ErJ7FNe1ZXdvS/Vnq4mWVRLOzaL8rcUUj92fXX/cs67PR6PvZ8dLf9zikieA9aefmtiojdJ++M8JrU38PGS9/S7lY05I6jWKrvgTZGWyeuIEF4+QJElqSppUgFKt0WE4+OCD2XPPPZszluRkRqgc/xvyxv4BgoCa4UcQdGhHqGwdORfewIZbLqXqtz8nd9yfyHrkGYLWRVRc/weIZBBkZpJ/7GnQKovqk48jaFNEHMi+8R5aTX6UoCCfDVe2nJsEFw3djfWln7Bg5BQIoMtVR7Fm2nvENtTQbkRfOo0/nEVjH4YgoM3wvcjqkGRlpiXJjLD2nDNpe9pZEARs+N+jiLcvJrR2Hdv98WrWXH/15p/2wXxyn3yKmr570/bUxH3OKk44nqpDD27O6CVJktTChIIgCDbXMGbMGO69914ArrjiCi688MJv9QYdaDmLMmwLy+jCCKakOoyUKuEkCjasTHUYKbU+tx0rVqxPdRgpVVxc4Bg4Bo4BjgE4BuAYgGNQXNwCv9hemAaXO+yy2dSlWTVagdo4r1qwYEGzBCNJkiQpTbWMZQK2uUYv/Ah5Qb0kSZIkNdDkKnxBEGyyIt+IESOaJThJkiRJSidJrcL3n+3Vq1fz6KOPmkBJkiRJPzSuwgdsYRjOOOOMuu05c+Zw//33U1paynHHHdcsgUmSJElSumk0gaqpqeHpp5/mgQceICsri/Lycl588UWys7ObMz5JkiRJ6cAKFLCFRSQOPfRQ5s+fz3XXXceDDz5I+/btTZ4kSZIk/aA1mkeOGTOGp556iiVLlnDcccfRyO2iJEmSJCnl4vE4l156KfPnzycrK4srrriCrl27btLvoosuoqioiHPPPfdbvU+jFahTTz2VqVOnMnr0aJ566inmzZvHtdde6z2hJEmSpB+iSBo8tuCFF16gpqaGkpISzjnnHCZMmLBJn4cffvg75zONJlD/se+++3Lttdfyz3/+k44dO3L++ed/pzeUJEmSpK1t1qxZDBo0CIA+ffowb968Bu3vvPMO77777ndeUTzpS8EKCwsZPXo0o0eP/k5vKEmSJKnlCTJSHQE8UlKyyf1p/5MQlZeXk5+fX9eWkZFBNBolEomwfPlybrnlFm655RaeffbZ7xSDa2lIkiRJahE2Tpj+W35+PhUVFXX78XicSCSR7jz33HOsWbOGU089lRUrVlBVVUW3bt049thjv3EMJlCSJEmSWrx+/foxffp0jjzySGbPnk2PHj3q2saMGcOYMWMAePzxx/n444+/VfIEJlCSJEmSkhBLg8xhSyEMHTqU0tJSRo4cSRAEXHXVVUybNo0NGzZ85+ueko1BkiRJklqEcDjMH//4xwbHunfvvkm/b1t5+g8TKEmSJElNSvcKVHNpchlzSZIkSVKCCZQkSZIkJSkdqmCSJEmS0lw0I/W1l1apDgArUJIkSZKUNCtQkiRJkpoUi5g6gBUoSZIkSUqaCZQkSZIkJck6nCRJkqQmxTIyUh1CWrACJUmSJElJsgIlSZIkqUkxrEABhIIgCFIdhCRJkqT0tpKCVIdAO9anOoRtX4Hakze39VuktbkM4HRuSHUYKXUrZxN6NdVRpFZwINA3lOowUuudgBUrUv9HL5WKiwscA8fAMcAxAMcAHIPi4tQnI/p2nMInSZIkqUlRp/ABLiIhSZIkSUkzgZIkSZKkJDmFT5IkSVKTYqYOgBUoSZIkSUqaaaQkSZKkJnkfqAQrUJIkSZKUJBMoSZIkSUqSU/gkSZIkNckpfAlWoCRJkiQpSVagJEmSJDXJClSCFShJkiRJSpIJlCRJkiQlySl8kiRJkpoUdQofYAVKkiRJkpJmBUqSJElSk2KmDoAVKEmSJElK2hbTyBUrVjB58mRyc3MZO3Ysubm5zRWXJEmSJKWdLVagxo0bR5cuXcjMzOTaa69trpgkSZIkpZkYGSl/pIMtVqCi0SijRo0C4Be/+EVzxCNJkiRJaWuLCVQoFKrbjsfj2zwYSZIkSekpXSpAqbbFBKqyspJPP/2UeDxOVVUVn376KUEQALDzzjs3S4CSJEmSlC62mEC1atWKiy66aJPtUCjEvffeu+2jkyRJkqQ0ssUE6r777muuOCRJkiSlsahT+IAmEqgrrriCCy+8EICpU6fy05/+FIDTTz+dW2+9ddtH14j4S28Tu/UJiGQQHn4wGT8b0qA9WL2e6Lm3QlUNofatybj6VEI5reraoxf9FYryiJw7kqCmltgf/kKweDnk5xC5+BeEdurY3KeUtCAe8OGl/6J8/ipCWRn0uuJgcrsW1bWveOlTPrn1bUKREDsM351OP+vJ0sfns/SJ+QDEq2OUf7CKQaWjqVy8jg8v+RfhrAzye7Zlt/87iFA41Nhbtwh5s1+i7d9vJciIsG7QcNYe/LMG7ZFVS+kw+QJCsRgQsOykP1K7fbfUBLu1hEJwwW3QY2+oqYY/ngKLFyXa2naACQ/X992tD9w0Hh67M7HfuhgenAW/HQqfzm/20CVJklqaLS5jPn9+/Qeqxx57rG573bp12y6iJgS1UaJX309k8ngi911IvOQlghVlDfrEbnuC8LADyXzwYkK9uhIveam+7eEXCRYsrtuPPzIdcrPJfOQyIheOIXr5lOY6lW9lxQufEK+JMaDkGHY9Zz8WTHitri1eG2PB1a/Rd/JR9L/vpywp+YDqFRvY4djd2Oe+n7LPfT+lsHc7elx4IJmFrfjgohn0uOBA9nnwf4nkZ/HVtI9SeGZbQbSW4oeu5otzJ7N4/H0UvVxCxtoVDbq0ffxGyg47kS/G38fqo35Nu8duSFGwW9GQoyErG046MJEcnX19fduqZfCrIYnHzX+AD96Gx+9KtEUicOGdUF2ZkrAlSVLLEiOS8kc62GICtbH/LB4BDVfna27BoqWEunQgVJRHKCtCqP9uBG81/OY8mDWf8KC9AAgN3pv4q/MAiL/zEcG7CwmPOLS+78IlhAd/3bfbDgSLljbTmXw7ZbO+ou2gHQEo6tOB9fPqE4SKRWXkdikks6gV4awMivp3pOytL+va181dQfnCNXQe0QuA6mUVbNcvUW3brl9HymZ91YxnsvVlfbmI2vZdiOcVQSSLyh79yVnwVoM+K0aOo2KvgwEIxWMEma0291ItS9+B8Opzie25r0OvfTbfb9zNcNVv4T8rap51HTx2B6xI7595SZKkdLLFBGrjRCmVSVMD5ZWECnLr9/OyCco3NOgSlFfC131CeTmwvpJg+RpitzxOxsW/aNA31LMr8emzCYKA+OyFsGw1QSx9l2yPltcSyc+qP5ARJh5NxBsrryFSUN8WycskWl5Tt//JnW/T7fT+dfs5Oxay5o3Eh+eV0z8jVhndxtFvW+HKcuI5BXX78ew8whvKG/SJF7SBSCaZX35Mu5JrWPW/pzd3mFtfXiGUr63fj8Ug47/mKB/8E1j0Hny2ILH/k5NgzQp47R/NF6ckSdL3wBbrYIsXL+aGG24gCIIG21988UVzxVcnOvFRgrfnE8xfTGiv7vUNFVWECvIa9A3l50BFJWRnEVRUQmEu8efegDXriZ56LcGKtVBVQ6zbDoSHH0xs0VKiY64k1K8Hod47E8pIujDX7CL5mcQqausPxAPCkUS8GflZRDdqi1bUEilIVFhq11Wz4eMy2uzfqa6911WHsODKUj6d9C6FexYTymqZFwa2/dtEcj56m1ZfzKeq2151x8NVFcRzCzbpn/PBv2l/32V89as/tfzrnwAq1sHG5xkOJ5KojR15Ijx4Y/3+0b+EIID9Dk9cF3X5vfD7nyam/EmSJG2G94FK2GIC9bvf/W6z22eeeea2i6gRkbOOBxLXQNUeNY6grBxyswne+pDQ2CMb9A3160H8lXfJOHYwwYx3CfffjYwxR5Ax5ggAYo/PIPh4KRnHDib+zkeE+/cgfMGJxOd+TPzz5c1+bt9EUb+OrJz+GR2O7M7a2cvI79Gmri2v+3Zs+GwttWVVZORmUvbWl3QduzcAZW9+SZsDOzd4rZWvfE6vqw6hVYc8Prx8Ju0Gd2nWc9laVg0/K7ERrWWn/zuKcHkZ8exccua/xZofj23QN+eDf9P+wStZcvYkou06bebVWqDZpTD4J/DPR2HP/WDh3E379OwP775avz/24Prtu6bDlb8xeZIkSUrCFhOo5cuX8+tf/7q5YklKKDNCZPzPiY69BoKA8PCDCXVoQ1BWTvTCSWTe8nsyfns00XF3JBaIaF1A5PrTGn+9rh2J3vgYscnPQEEukSt/1Yxn8821H7ozq0u/4M2RT0IQ0OuqQ/hq2kdEN9TSeUQveow/gHfGPkMQBOwwfDeyOySqcxWflJHTuWE1JrdrIe+c+iwZORFa77cD7Q5umQlUnUgmK0aNp/P1YyEIWDtoONHWHQiXl9Hh7gv58sxbaP/QVYSitXScNB6Amo47s/wXf0xx4N/RS0/A/kPhntLEinyXnAw/HgW5+YkFI1q3gw3rUx2lJEnS90Io2Hh1iP8yZsyY73zD3D158zs9v6WbywBO53uw0tt3cCtnE3q16X7fZ8GBQN80uY4wVd4JWLHih53IFRcXOAaOgWOAYwCOATgGxcWbXmaQ7mawX6pDYDCvpzqELVegysrKmDlz5mbbBg4cuE0CkiRJkqR0tcUEavXq1Tz99NObbTOBkiRJkn44XEQiYYsJ1M4778zVV1/dXLFIkiRJUlrbYgKVsdG9ZJYtW0YsFiMUCrH99ttv88AkSZIkKd1sMYG68MIL6xaSOOmkkygqKmLZsmVcfvnlDBo0qLlilCRJkpRiUafwAbDFO8Zed911nHfeeQAUFxdTUlLClClT+Mtf/tIswUmSJElSOtliBaqyspI999wTgIKCxFKLXbt2JRqNbvvIJEmSJKWN2JZThx+MLVagqqur67Zvu+22uu1IxMGTJEmS9MOzxQSqffv2zJkzp8GxOXPmUFxcvE2DkiRJkqR0tMVS0nnnncdpp53G/vvvT9euXVm8eDGvvfYad9xxR3PFJ0mSJCkNeB+ohC1WoHbccUceffRR+vbty4YNG9hjjz14+OGH2WGHHZorPkmSJElKG01ezJSdnc2RRx7ZHLFIkiRJSlNWoBK2WIGSJEmSJNUzgZIkSZKkJLkeuSRJkqQmRZ3CB1iBkiRJkqSkWYGSJEmS1KSYqQNgBUqSJEmSkmYCJUmSJElJsg4nSZIkqUneByrBCpQkSZIkJckKlCRJkqQWLx6Pc+mllzJ//nyysrK44oor6Nq1a137U089xZQpU8jIyKBHjx5ceumlhMPfvJ5kBUqSJElSk2JkpPyxJS+88AI1NTWUlJRwzjnnMGHChLq2qqoq/vznP3Pvvffy8MMPU15ezvTp07/VOJhASZIkSWrxZs2axaBBgwDo06cP8+bNq2vLysri4YcfJicnB4BoNEqrVq2+1fs4hU+SJElSk6JpsIhESUkJJSUldfsjRoxgxIgRAJSXl5Ofn1/XlpGRQTQaJRKJEA6HadeuHQD33XcfGzZs4KCDDvpWMWzzBGouA7b1W6S9Wzk71SGkXHBgqiNIA+8EqY4g5YqLC1IdQso5Bo4BOAbgGIBjAI6BvrmNE6b/lp+fT0VFRd1+PB4nEok02L/22mv55JNPuPnmmwmFQt8qhm2eQJ3FhKY7fY9NZDzH8UCqw0ipx/g5oZKm+32fBSOA33+7X9LvjT8HvP0t/1B9X/QLAlasWJ/qMFKquLjAMXAMHAMcA3AMTB63vn79+jF9+nSOPPJIZs+eTY8ePRq0X3zxxWRlZXHbbbd9q8Uj/sMpfJIkSZKaFEvz1GHo0KGUlpYycuRIgiDgqquuYtq0aWzYsIE99tiDxx57jH322YeTTjoJgDFjxjB06NBv/D7pPQqSJEmSlIRwOMwf//jHBse6d+9et/3hhx9ulfcxgZIkSZLUpKaWEf+hcBlzSZIkSUqSCZQkSZIkJckpfJIkSZKa5BS+BCtQkiRJkpQkK1CSJEmSmmQFKsEKlCRJkiQlyQRKkiRJkpLkFD5JkiRJTYo6hQ+wAiVJkiRJSbMCJUmSJKlJMVMHwAqUJEmSJCXNBEqSJEmSkmQdTpIkSVKTvA9UghUoSZIkSUqSFShJkiRJTbIClWAFSpIkSZKSZAIlSZIkSUlyCp8kSZKkJkWdwgdYgZIkSZKkpJlASZIkSVKSGk2g7rzzzuaMQ5IkSVIaixFJ+SMdNBpFaWkpv/71r5szliYF8YB5l77G+vmrCWdlsOcVB5HXtbCufdlLn7Pw1ncJRUJ0Hr4rXX62G7GaGHP+MJPKxeuJ5GfS++L9ydupiPULy5h3USkEULB7G3pftB+hjJZVkAviAUsvfYbK+V8RzorQ6Yqf0Kprm7r2dS/NZ/mt/4JIiDbD+9LmZ/0IYnGWXPgU1Z+shIwwna/+Ka26tNnCu6S/YTvAxb0hGofJn8Ckjxu275gLkwdAJAwh4NS3YMF6OKsHjO0GK6oT/X799fEWJxSC426DTntDtBoePgVWLqpv33EfOPqGRL91X8H9J0KsFkbeBcW7QRCDB0+GVR83/h7pLhRix9tuI2fvvQmqq/n8lFOoXlQ/Bq1POIEO55xDEIuxavJkVt5xB21OOom2v/gFAOHsbHL69GFux47E1q5N0UlIkqSWoNEEqqysjJkzZ262beDAgdssoC1Z9sJnxGtiHFgyjDWzl/PBhDfY5/bDAYjXxvng6jc46LGfkJET4bVRz9BhyI58+fynRHIjHPjIMMo/Xst7l/+bff96BAtumMVuZ/enzYCOvDv+Xyx7aTEdh3ZNyXl9W+te+JB4TZRdSsayYfYXfDnhH+x0+0gAgtoYX179D3Z57BRCOVl8POpuCob0YMO7XwDQ/eFfUv76p3x5df1zWqJICCb2gQH/hIoYlB4G05bCsqr6PpfvAbcshL8vgR91hKv3guGl0K81jHkd3l6TsvC3jj2Phsxs+POB0HU/+N/r4a9H17ePvAvuPi6RVO0/Ftp0hQ49E203DYRdDk4kWBs/p4XZ7uijCWdns+DAA8ndbz86XX89Hx99dF175+uu4/3evYmXl9Pz/fdZ8/DDrJ4yhdVTpgCw4y23sHLyZJMnSZK2wPtAJTSaQK1evZqnn356s22pSqBWz1pO8aBOALTu056181bVtZUvKiO3SyGZRa0S7f3bs/qtZZQvXEvx4M4A5HcronxR4gNSv5uHEMoIE6+JUb2iklZts5v5bL67ilmfUzCoOwC5fTpTOe/LuraqRSvJ6tKGjKKcRHv/Hdnw1ucU/U8vCg/pAUDt0rVE2uU3f+BbUc9CWFgOZbWJ/ZkrYFA7eOyL+j7nzIa1X7dHQlAVS2z3bwN/6Akds+HpL2HCB80a+tbTbSB88Fxi+7PXExWn/2jfAypWwcG/h+33hPefhuULEo/3nkr0ad0V1i9r9rC3pryBA1n3XGIMNrz+Orn77NOgvXLOHDKKigiiUUKhEARBXVtu//5k9+7N4jPOaNaYJUlSy9RoArXzzjtz9dVXN2csTYqW1xDJz6rbD2WEiEfjhCNhouW1ZBZk1rVF8jKJltdS2LMNy6cvpsPhXSh7dwVVyzYQxOKEMsJULinn9ZOfI5KfRd7ORak4pe8kXl5DRn6ruv1QRoggGicUCRMvryajoL4tIy+LWHmiLBOKhFk87knW/fNDutx0fLPHvTUVZtYnRwDro1CU1bDPqprEf3sUwHV94OivC6sPfw63fgTrovDEQXDU9olEqsVpVQhVG1VOghiEMyAeg7x2sNOB8LczYcVHcOpTsHgWfPRSov2Ee2CvYxIVqhYso7CwYfUoFoOMjMR/gcp589h91iziFRWUPf54g74dL7iALy+7rLlDliRJLVSjF/1kZKRfiS6Sn0WsYqNPy/GAcCT8dVsm0Y3aohW1RAqy6Dx8VyL5mbw+5jmWT19MUe+2ddc65XTK55B/HEfXUbvzwYQ3mvVctoZwfhaxipq6/SAeEPp6PML5rRq0xSpqyCior7LteM3R9Hj+DJZc9BTxDfX9WorL94DpQ2DqwEQS9R8FESjbzOkc0h6eHAij/11/ndOfFySSq9p4InHq27p5Yt/qqtdBq4L6/VA4kRxBovq0ciEs+wDi0USlasf+9X0f/AVc2QNG3AVZuc0a9tYUW7eOcMFGYxAO1yVPOXvuSdFRRzFv552Zt9NORNq3Z7vjEgljRlERrXbfnfKXX05B1JIktSwxMlL+SAeNJlD33HNPM4aRnNb92rN8RmJu1prZyynoUf+JN7/7dlR8to6asmriNTFWv7WM1n2LWTt3Ja37d2D/+/6HDod3JXfHxIest37zAhWfJr6FzsiLQDjU/Cf0HeX168L6GQsB2DD7C7J7tK9ry+7ejprPVhMtqyReE6Pirc/J7duZNU/OYfmdiRJMOCczsbBAC1s8A+CieTBkOnT4O+ySD62zIDMMg4vhtVUN+x7SHm7sCz9+BWZ9fb1TYSbM+zHkfV2DPbR9fVuL83Ep9Doysd11P/hybn3bqo+hVT60S0z1pNsg+Oo92OdEOHx84ljNBgji9UlXC1RRWkrRkYkxyN1vPyrn1o9BbO1a4pWVBJWVEI8TXb6cjNaJvx35gwez/oUXUhKzJElqmRqdwrel65waW1xiW+s4tCsrS5fy6sinIIC9rhrIkmmLiG2I0mXEbvQcvy9vjv0HQRDQefiuZHfII5yZwYIb3+aTyfPILMhizysT59X91D2ZM34mocwwGTkR9rzioJSc03dROHR3yks/ZtHIyYlzvup/KZs2l/iGGtqM6M/244fy6dgHCIKANsP7kNmhkKIf7c4Xf5jKop/fQxCNscMFRxBulR5LQn4b0QDOng3PH5z4NmDyJ7C0MpFQTRqQWCziz30hKwxT9ks8Z/56+M1bcMGcRBWrOgYvLoNnW+L0PYC5T8BuQ+H/lSYS4gdPhn6jEonTa3fBw2Nh9IOJtk9ehfefSVSbRt0NZ74CGZnwxO8TK/i1UGVPPEHB0KH0KE2MwWcnn0zrUaMI5+ez6q67WHnnnfSYOZOgpobqRYtY/fUXRK12243qj1vw6oOSJDWjdKkApVooCDa6mnoj/+///T9uvPHG7/wGZzHhO79GSzaR8RzHA6kOI6Ue4+eESlIdRWoFI4Dft7wq51b154C3Qz/sMegXBKxY0RLXyt96iosLHAPHwDHAMQDHoLi4oOlOaWY8qb9meAKXpDqExqfwrVnTUuczSZIkSdK20ejcrcWLF3PDDTdstu3ss8/eZgFJkiRJSj9Rp/ABW0igsrOz2XnnnZszFkmSJElKa40mUO3ateOYY45pzlgkSZIkpalY46nDD0qj10DtsccezRmHJEmSJKW9RhOocePGNWcckiRJkpT2rMNJkiRJapL3gUpotAIlSZIkSWrICpQkSZKkJlmBSrACJUmSJElJMoGSJEmSpCQ5hU+SJElSk6JO4QOsQEmSJElS0qxASZIkSWpSzNQBsAIlSZIkSUkzgZIkSZKkJFmHkyRJktQk7wOVYAVKkiRJkpJkAiVJkiRJSXIKnyRJkqQmOYUvwQqUJEmSJCXJCpQkSZKkJlmBSrACJUmSJElJMoGSJEmSpCQ5hU+SJElSk6JO4QMgFARBkOogJEmSJKW3EUxJdQiUcFKqQ9j2FagltNvWb5HWOrGS0DOpjiK1giPhSB5PdRgp9QzHEgrVpjqMlAqCTEKhy1IdRkoFwSWETkt1FKkV3AYrVqxPdRgpVVxc4Bg4Bo4BjkFxcUGqQ/jGYk5eA7wGSpIkSZKSZgIlSZIkSUmyDidJkiSpSd4HKsEKlCRJkiQlyQqUJEmSpCZZgUqwAiVJkiRJSTKBkiRJkqQkOYVPkiRJUpOiTuEDrEBJkiRJUtKsQEmSJElqUszUAbACJUmSJOl7IB6Pc/HFFzNixAhGjx7NZ5991qD9pZdeYvjw4YwYMYJHHnnkW7+PCZQkSZKkFu+FF16gpqaGkpISzjnnHCZMmFDXVltby9VXX83kyZO57777KCkpYcWKFd/qfUygJEmSJDUpRkbKH1sya9YsBg0aBECfPn2YN29eXduiRYvo0qULRUVFZGVl0b9/f956661vNQ5OZJQkSZLUIpSUlFBSUlK3P2LECEaMGAFAeXk5+fn5dW0ZGRlEo1EikQjl5eUUFBTUteXl5VFeXv6tYjCBkiRJktQibJww/bf8/HwqKirq9uPxOJFIZLNtFRUVDRKqb8IpfJIkSZKalOrpe01N4evXrx8zZswAYPbs2fTo0aOurXv37nz22WeUlZVRU1PDW2+9Rd++fb/VOFiBkiRJktTiDR06lNLSUkaOHEkQBFx11VVMmzaNDRs2MGLECMaPH8/YsWMJgoDhw4fToUOHb/U+JlCSJEmSmhRtogKUauFwmD/+8Y8NjnXv3r1u+9BDD+XQQw/97u/znV9BkiRJkn4gTKAkSZIkKUlO4ZMkSZLUpJipA7CFCtTy5cubMw5JkiRJSnuNJlDnnntuc8YhSZIkKY2legnzppYxby5eAyVJkiRJSWp0IuN7773HyJEjGxwLgoBQKMTDDz+8zQNrSjwON16ax6L5GWRmwblXlNOpa7yufcbzWTz0lxwIwbARVRx1fHVd25pVIX5zbBHXTl5Hl+7xzb18izGsPVy8C0QDmPwFTFrcsH3HbJi8F0RCEAJOnQcLKmDk9vD7nSEWwJz1cNo8CFJyBt9OEI+z8tK/Uz3/S0JZEdpfcSyZXdvVtVe89AFrbn0RImEKh+9D4c/2rWurevdzVl33HJ3uO7XuWPk/36Piubl0uL7hz3xLMWxYiIsvDhONwuTJcSZNaviv2bo1LFgQYd68xPEnnggoKYnz8MP13+T06RNi/Pg4d97ZMn8nhg3rwcUXDyYajTN58mwmTXq7QfvEiUfQp09HADp2zKesrIoDDvgr++yzAzfc8CNCoRBffVXOiSc+TnV1LBWn8J0N2xMuPhKiMZj8GkwqbdjeoRAe+AVkReDLtfCLe6GyFvbpCjcMT/yN+GodnHgPVEdTcAKSJLUAjSZQu+yyC9dff31zxvKNzHwhi5oauKVkHe/PjnD7hDyuuH09ALEY3HV9Lrf/bS05uQEnH7kdAw+roahNQLQWbrg4n1bZKT6BrSASgok9YUApVMSg9ACYtgyW1dT3ubwH3PIZ/H0Z/KgdXL0b/Hw2XNED9vwXVMbhwT6JRGxaC7rsreKF9wlqonQuOY2q2Z+zcsIzbH/7GACC2hgrr36Kzo+dQTgnkyWj7iB3SE8ixQWsuesVyqe+Qygnq+61Vl4xjQ0zP6JVz+1TdTrfSSQCEydmMGBAlIoKKC3NYNq0GMuW1ffp1y/EQw/F+d3vGiZHQ4YkEoX99w9x5ZVh7rqrZSZPkUiYiROPYMCAu6ioqKG09JdMmzafZcsq6vqcddbzdX1nzjyZX/1qGgB33fUTjjvuERYtWsPYsX3p2nU7FixYlZLz+C4iYZg4HAZcAxU1UHouTJsLy9bV9xn/I5jyOtz3OlxyFPx6EPz5JbjrBDhuEixaAWMPhK5tYEEL+nsgSWoe6TKFLtUancKXlZVFp06dNvtIB/NmRRgwqBaAXn2izJ9XnwtmZMA9z5SRXxCwriwEQE5e4pv3O67J5acjq2jbvmV+UNxYz3xYuAHKolAbwMw1MKhNwz7nfABPf/1BKBKCqjhUx+HA1xLJ08bHW5KqWZ+SM6gHANl9ulA9b0ldW82i5WR2aUtGUQ6hrAjZ/Xei6q1PAcjs0pYON5/Y4LWy+3Wl+NL/bbbYt7aePWHhwoCyMqithZkzAwYNCjXo079/iH79Qrz8cgaPPJJBx44NX+Pmm8P89rcx4i3s5+A/evZsx8KFqykrq6K2Ns7MmYsZNKjrZvueeea+/OMfHzNv3nJ69GjLqlUb+P3v9+fll0+iTZucFpk8AfTcHhaugLJKqI3BzIUwqHvDPmc9Bve/AaEQ7Ng6kVz1aA+rKuD3Q+Dls6BNnsmTJElb0mgCddxxxzXYLy8v57777uPII4/c5kElY0N5iLz8+mlKGRkBsY2mnGREYMY/svjV/27HXvvUkhGB5x5vRVGboC7xaukKI7B2o1NZH4Wi/6oprqpNTO/rkQfX9YTLPkpM1Vv+dZXqjK6QnwH/XNlsYW8V8fJqwvn1ZcRQRoggGqtvK6hvC+e1Il5eBUD+EXsQijT89iT/yL0SnyhbqMLCEGvX1u+vXw9FRQ37fPhhwCWXxDnkkBhPPhnn5pvrx+AnPwnx3nuwYEEzBbwNFBa2Yu3aqrr99eurKSpqtUm/zMwwv/51f6677lUA2rXL5cADd+S2297k8MPv47DDdubQQ3dutri3psJs2GgIWF8NRTmb9ssIwbwLYUgPKP0Y2uXDgd3gthlw+I1w2G5w6G7NF7ckqeVI9QIS6VIBazSB+t//TXwjv3DhQi699FIOO+wwPvroIyZMmNBswW1Jbn5AZUX9h954PJE0bWzwj2p4ZMYaamtD/OPJVjz7t1bMejWTs0YXsvCDDK4eV8DqFS3vg/PlPWD6fjC1fyKJ+o+CSKIa9d8OaQNP9ofR7yauf4LEtQ7X7g5D28Hwtzd9TroL57ciqKi/ri2IB3WJ0X+3xSsaJlTfF5dfHmb69AymTs2gsLD+eEEBlJU17PvSSwHTp9df/9S3b/3P/YknhvnLX1pm6enyy4cwffpJTJ06isLC+oSpoKAVZWVVm/Q//PBuzJjxOevWJX4+Vq3awMKFq/ngg5VEo3Gee24R/fu3rKmcl/8Epv8epv4mkUT9R0GrRDXqv0Xj0PtyOPUBuPekRPVp4Qr44KtE23PvQ/8uzRa+JEktTqMJ1PPPP8+YMWO46KKL6N27N7vvvjt//OMf2WuvvZozvkbt0S/K6zMyAXh/doRuPeov+q4oD/H7EwupqYFwGLJzAsJhuPGBdfz5/nVMvG8du/SM8Ydr1tOmuCUtnZBw0QIY8jp0eBF2yYPWmZAZgsFt4LU1Dfse0gZu7AU/fgNmbVSluHMPyA7D0bPqp/K1JNn9dmLDjPkAVM3+nKwe9XPSsrq3p/azVcTKNhDURKl86xOy+37/PhFedFGcIUNidOgQZZddQrRuDZmZMHhwmNdea/hzPWlSBsOHJ5Kmww4LMWtWfXv//iFefbXl/R4AXHTRdIYMmUKHDtexyy5taN06m8zMMIMHd+G1177YpP/hh3fj2Wc/qtv/+OM15Odn0b17awAGDerCe++taLb4t4aLpsGQP0OHcbBLMbTOhcwMGLwrvPZxw763joRDEjNfWV8N8QA+Xgn5raB7ceL4oF3gvS+b9RQkSWpRGl1EYty4cZx00kn84he/oHXr1vzjH/9ozriaNHBoDbNKMzljZCEEcP5V5bw4LYvKDSGGjajm8J9U8/ufFxKJQLfdYhz+0+qmX7SFiQZw9gfw/AAIhxKr8C2tTiRUk/ZMVJb+3AuywjBl78Rz5lfAnZ/D2B3hX6vhpf0Sx2/8FJ5c1uhbpZ28ob2oLP2IL0beDkFA+6uOY/202QQbaigcsS9txx/Fl2MnEwQBhcP3IdKhqOkXbaGiUTj77BjPP59BOBxi8uQ4S5cmVt5LJE4xxo+PMXlyBqedBhUVcMopiS8c2rVLTPlr6aLROGef/Q+ef/7Er8dgNkuXrqd162wmTfopw4c/AsBuu7Xl3nvfrXtebW2csWOn8uCDwwmF4NVXv+CZZz5q7G3SWjQOZ/8Nnj/z678Hr8LStYmEatKJMPwvcNN0uGNUYqW+eBxOezhxvdTY++HBkxMzWV/9GJ6Zl+qzkSSlo2iaTKFLtVAQBJv96nn58uU8/vjjTJ06lR49evDFF1/w2GOPfeM3WEK7pjt9j3ViJaFnUh1FagVHwpE8nuowUuoZjiUU+n5ce/dtBUEmodBlqQ4jpYLgEkKnpTqK1ApugxUrvgdZ+3dQXFzgGDgGjgGOQXFxQapD+Mb25M1Uh8BcBqQ6hMan8F111VX85je/4ZlnnmHEiBHsuOOOHHrooVxzzTXNGZ8kSZKkNBAjkvJHOmg0itWrV9dtH3DAARxwwAGsWbOGv//9780SmCRJkiSlm0YTqMWLF3PDDTc0ZyySJEmSlNYaTaCys7PZeeeWeT8USZIkSVtXutyHKdUaTaDatWvHMccc05yxSJIkSVJaazSB2mOPPZozDkmSJElpzApUQqOr8I0bN64545AkSZKktNdoAiVJkiRJaig9FlOXJEmSlNaiTuEDrEBJkiRJUtJMoCRJkiQpSU7hkyRJktSkmKkDYAVKkiRJkpJmGilJkiSpSd4HKsEKlCRJkiQlyQRKkiRJkpLkFD5JkiRJTXIKX4IVKEmSJElKkhUoSZIkSU2KxdOgApUG5Z80CEGSJEmSWgYTKEmSJElKklP4JEmSJDUpGk2DKXxZqQ7ACpQkSZIkJc0KlCRJkqQmxaJpkDpYgZIkSZKklsMESpIkSZKSFAqCIEh1EJIkSZLSW+66NakOgQ2FrVMdwra/Bio0Y1u/Q3oLBsO1/C7VYaTUedzEdjVfpTqMlCrL6giHhFIdRmq9HPCn0A97DM4PAl7+gY/BIUFAKHR7qsNIqSD4LStWrE91GClVXFzgGDgGP/gxKC4uSHUI+pbS4EowSZIkSekulg7LmKcBr4GSJEmSpCSZQEmSJElSkpzCJ0mSJKlJ0Vqn8IEVKEmSJElKmhUoSZIkSU2Kx0wdwAqUJEmSJCXNBEqSJEmSkmQdTpIkSVLTvA8UYAVKkiRJkpJmAiVJkiRJSXIKnyRJkqSmOYUPsAIlSZIkSUmzAiVJkiSpadFQqiNIC1agJEmSJClJJlCSJEmSlCSn8EmSJElqWjTVAaQHK1CSJEmSlCQrUJIkSZKaZgUKsAIlSZIkSUlrNIGqrKxszjgkSZIkKe01mkANHz6cuXPnNmcskiRJktJVNA0eaaDRBOq6667jkksu4fbbbycIguaMSZIkSZLSUqOLSPTq1YuSkhLuuusuxo4dyxFHHFHXNmLEiGYJril5775E22m3EmREWHfQcNYO/lmD9siqpXSYcgGhWAwIWDb6j9R27AZAqLqSzhNP5quTrqR2++4piP67C+IBpZe+z+r56wlnhRl0RW+KuuY16BOtjPHMyW8x+MrebNc9H4Anjn6VzILEP31B5xwOvnrPZo99a8t6uZT8O6YQZGRQdcyRVB73k832y73vUcIrV1N+1q/rjuVfcwuxnXek8mf/21zhbl2hEJx1G3TfG2qr4dpTYMmiRFubDnDxw/V9d+kDfxkPT/8Vxk2GjjtBZiu47wp4dVoqot86QiF+dNttFO+9N7Hqap475RTKFi2qa+514onse955VK9dy7x77mHu5MmEIxGOnDKFop12Ih6L8fyvfsXq+fNTeBLfUShEj9tuI2/vvQmqq5l/yilUbjQG7U84gR3POYcgFuOryZNZescdhLKy2P3uu8np1o3ounV8dPrpVC5cmMKT+O6GDevKxRfvQzQaZ/LkD5k06YMG7RMnHkSfPm0B6Ngxl7KyGg444HFGjtyF3/9+L2KxgDlzVnHaaTPwu0NJ+i+1qQ4gPWxxFb4gCKiqqmLNmjWsWLGiuWJKTrSW4pKr+fz/HiPeKocuE0ZRvvcQYkXFdV3a/v1GyoacSEXfw8md9y/aPX4DX552C60+nUuH+y8hsmZZCk/gu/v0heXEauL8tGR/ls8u4/UJ8/nR7f3q2lfMXUvpJe9Tsayq7li0OgbAsPv2bfZ4t5naKAV/upXVD91JkJtNm9GnU33IgcTbta3vU1VN4aXXkjn3faoPPxiA0Ooyii64kozPvmDDziNTFPxWMPBoyMqG0w+EXvvBb6+HC49OtK1eBr8fktjutT+cciU8dRf8aAysWwVXjYHCNnDXOy06gdr16KPJyM7mgQMPZPv99mPI9dfzxNFHA5DTti2DrriCKX37UlVWxogXXuCzF1+k/d57E45EeOCgg+h6+OEMuvJK/n7ccak9ke+g3dFHE87O5p0DD6Rwv/3ofv31zPt6DAC6X3cdb/buTay8nH3ff5/lDz9Mh5//nFh5OW8fcAA5PXqw6y23MOfHP07dSXxHkUiYiRMPYsCAx6ioiFJaegzTpn3KsmX11/SedVZpXd+ZM4/mV796mezsDK64Yl/23PMRKiujPPjg4QwbthPTpn2aojORJKWzRhOod955h//7v//j4IMPpqSkhKysrOaMq0lZXy2itn0X4nlFAFTu2p+cj96ifJ//qeuz4vhxxHMKAAjFYwSZrRLb0RqWnnYrHf96fvMHvhUtm7WGzoPaAdC+z3asnLeuQXusJs7ht/bh5fPrr2Vb/eF6opUxnv3lW8SjAQPO3pX2fbZrzrC3usjHnxHr0omgKPFvXdN3LzJnzaH6iCF1fULVNVT99Ahq9u9P5JPPE8c2VFJ+2sm0mvl6SuLeavYcCG88l9h+/3XYbZ/N9/t/N8MVP4d4HF55FF55rL4tliaTir+lzgMH8slziTH48vXX6bhP/RgUdevG8tmzqVqzBoCv3nyTHfbfn+Xvvks4EoFQiFaFhcRrW/bXakUDB7L66zFY9/rrFOzT8OegYs4cIkVFBNFoomoZBOT26sXqZ58FoHLBAnJ79mz2uLemnj23Y+HCtZSV1QAwc+aXDBq0PY899vEmfc88cw/+8Y/FzJu3mlAIDjzwCSorE78HkUiYqqqW/TshSUqoqqrivPPOY9WqVeTl5XHNNdfQpk2bBn3uuecenn76aQAOPvhgzjjjjC2+ZqPXQI0bN47LLruMcePGpV3yBBCuLK9LjgDi2XmEK8sb9IkXtIFIJplffUy7R69h1U9OB6Bql/5E22zfrPFuCzXlUbLy63PgUEaIeDRet9+xf2vyt89p8JxIdgZ7jt2JH/+1PwMv68X0c+c0eE5LFKqoIMivn7oY5OUQLq9o0CcoKqDmwAENjsU7b090r17NEuM2lVcI5Wvr9+MxyMho2OfAn8An78HiBYn9ygqoLIecfLjsMfjrhc0X7zbQqrCQ6rX1YxDEYoS+HoM1H31Eu969yW3fnkhODl0PO4zMvDxqy8sp2mknTvnwQ4646y5m3XRTqsLfKiKFhUQbGQOAinnz6D9rFgPee49VTz1FdO1aymfPpu2wYQAU7rcfrTp1gnDLvbtFYWEWa9fW1O2vX19LUVGrTfplZob59a97c9117wIQBLB8eaJKdcYZe5Cfn8k///lF8wQtSS1JLA0e39BDDz1Ejx49ePDBBzn66KO57bbbGrQvXryYqVOn8vDDD1NSUsLMmTP58MMPt/iajVagHn/8cfLz8zc5vmTJEjp16vTNo99K2j4xkZyFb9Pqi/lUddur7ni4qoJ4bsEm/XM+/DftH7iMr8b+qe76p++LrPwItRX1P0lBPCAc2fKHn6Kd8yjsmksoFKJo5zyyt8tkw4rqTRKtliDvpklkvTOXyIJF1O5Z/815qKKSeMGmP7vfWxXrYOOf/XAYYv/1F2boifC3GxseK+4MVzwBT94GLz607ePchqrXrSOroH4MQuEwwddjUF1WxktnncXRf/sb67/4gmVvv03lypXsc9ZZfPL888y44AIKOndm5EsvMXnPPYlVV6fqNL6T6Lp1ZDQyBnl77knbo47i3zvvTKy8nJ7330/xccfx1eTJ5PbsSZ/p01lbWsr6WbMSFcoW5vLL92XgwI7stVdbXn99ed3xgoJMyso2/fc8/PDOzJixlHXr6pOtUAj+9KcD6NFjO4YPf75Z4pYkfXMlJSWUlJTU7Y8YMWKL6zPMmjWLU045BYDBgwdvkkB17NiRSZMmkfH1l47RaJRWrTb98m1jjSZQp512Gvfeey8A11xzDePGjQPgD3/4Q93xVFh1zFmJjWgtO11yFOGKMuKtcslZ8BZrfjS2Qd+cD/9N+4evZMnvJxFtm7qkb1vp0G87Pp++gm5HdmT57DLa9Ng0gfxv8x/7gjULyjno0l5ULKuipjxGbvGWf0jSVcXvTqECoDZK26PHEFq7jiA3h6xZ77LhF+mx0EmzmFeaqDC9/GjiGqiPN3P7gR79Yd6r9fut28N1/4Abz4C3X2q+WLeRJaWldP/JT5j/6KNsv99+rNjoFgyhjAx22H9/Hhw8mHAkwogXXmDGBRdQvNdeddP2qlavJpyZSTgj49t8uZUW1paW0u4nP2HFo49SuN9+lG80BtG1a4lVVhKvrIR4nNrly4m0bk3BgAGsnTmTRWefTUH//uR0b5kL6lx00RtAYurd+++PoHXrVpSX1zJ48A51VaaNHX54Z5599vMGx+6882Cqq2McffSzLh4hSY1Jg9nNW0qYHn30UaZMmdLgWNu2bSn4+gvGvLw81q9f36A9MzOTNm3aEAQBf/rTn+jVqxc777zzFmNoNIHaeOny9957b7PHUyqSyYqfjafzxLEQBKwdOJxo6w6EK8roMOVCvjztFtqXXEUoVkvHyeMBqOm4M8tH/zHFgW89Ow3twJLSVUwd+ToEAYOv2oOF05YS3RBj9xE7bvY5ux3XmRl/mMu0Ua9DCAZf1bvJqlXay4yw/rzTaf3rcyEeUHnMkcQ7FBNau47CS/7E2j9fkeoIt61/PQH7DIVbShNfo19zMhw2KjE976m7oKgdVDb8Y8HPL4CC1jDmosQD4Pz/gZqqTV+/BVjwxBPsNHQoPy9NjMGzJ59Mz1GjyMrP59277iJWU8NJs2YRrarizeuvp3LVKt6aOJH/mTyZUTNmkJGVxYwLLqB2w4ZUn8q3tvKJJ2gzdCh9vx6D+SefTPtRo8jIz+fLu+7iyzvvpO/MmQQ1NVQuWsRX99xDpLCQnS+/nB3PPZdoWRnzx45t+o3SWDQa5+yzX+X554cRDoeYPPkDli6toHXrVkyadEhdZWm33bbj3nvrV1zs27cdY8f25F//+pKXXvopADfeOJcnn/wkJechSfp2jj/+eI4//vgGx8444wwqKhKXdlRUVFBYWLjJ86qrq7ngggvIy8vjkksuafJ9QkEjGdHo0aO57777NtkeM2bMN6pAhWYk3fV7KRgM1/K7VIeRUudxE9vVfJXqMFKqLKsjHBJKdRip9XLAn0I/7DE4Pwh4+Qc+BocEAaHQ7akOI6WC4LesWLG+6Y7fY8XFBY6BY/CDH4Pi4qZnDqWbdPhcHwz+Zv0nT55MRUUFZ555Jk8//TRvvPEGl112Wf3rBQGnnHIK++23H6eeempSr9loBSq00f/kQz/w/+FLkiRJP3hpMIXvmxo1ahTjxo1j1KhRZGZmcv311wNw991306VLF+LxOG+88QY1NTX861//AuDss8+mb9++jb5mownUe++9x8iRIwmCgIULF9ZtL9roxoySJEmSlK5ycnK4aTMr7Z588sl123Pnbub68S1oNIEaOHAg55/fsu+TJEmSJGkraYEVqG2h0QRqzZo1KV2uXJIkSZLSTaMJ1OLFi7nhhhs223b22Wdvs4AkSZIkKV01mkBlZ2c3uQa6JEmSpB8Ip/ABW0ig2rVrxzHHHNOcsUiSJElSWmv0Dqp77LFHc8YhSZIkSWmv0QrUuHHjmjMOSZIkSenMKXzAFipQkiRJkqSGGq1ASZIkSVIdK1CAFShJkiRJSpoJlCRJkiQlySl8kiRJkppWm+oA0oMVKEmSJElKkhUoSZIkSU2LpTqA9GAFSpIkSZKSZAIlSZIkSUlyCp8kSZKkpnkfKMAKlCRJkiQlzQqUJEmSpKZZgQKsQEmSJElS0kygJEmSJClJTuGTJEmS1DSn8AFWoCRJkiQpaVagJEmSJDXNChQAoSAIglQHIUmSJCm9hW5LdQQQnJbqCJqhAtUuvmRbv0VaWxnuxGWMT3UYKXUJEwjNSXUUqRXsBRwQSnUYqfVawGWhH/YYXBIElP7Ax+CgICAUujLVYaRUEPyfvwtBwIoV61MdRkoVFxc4Bj/wMSguLkh1CPqWnMInSZIkqWlO4QNcREKSJEmSkmYCJUmSJElJcgqfJEmSpKY5hQ+wAiVJkiRJSbMCJUmSJKlptakOID1YgZIkSZKkJJlASZIkSVKSnMInSZIkqWmxVAeQHqxASZIkSVKSrEBJkiRJaprLmANWoCRJkiQpaSZQkiRJkpQkp/BJkiRJappT+AArUJIkSZKUNCtQkiRJkppmBQqwAiVJkiRJSTOBkiRJkqQkOYVPkiRJUtNqUx1AerACJUmSJElJ2mIFKh6P88orr5Cbm8t+++3XXDFJkiRJSjexVAeQHraYQF166aWsX7+eDRs28N577/HLX/6yueKSJEmSpLSzxSl8CxcuZOLEidxyyy3MmDGjuWKSJEmSpLS0xQpUJJJozszMJB6PN0tAkiRJktKQ94ECvker8GVOf5Xc2+4jyMig+tgfU/2zYZvtlz3lMcIrV7PhnFObOcKtL4gHvHnpbNbMX0tGVgb7XdGXgq75DfpEK6O8dHIp+13Zj6LuBcRjAW9c+DbrPiknlBFi/6v7UdAlv5F3aBny3nqJto/dShCOsO7Q4aw9/Geb7bfd0/cQWbOSlSeeC0DBv6bSetrdEA6zdshw1h5xQnOGvfWEQnDebbDL3lBbDVefAl8sSrS16QCXP1zfd9c+cPt4+Ptd8Ie7oMtuEIvBlSfDko9TEv5WEQpx1G230WHvvYlVVzP1lFNYs2hRXfNeJ57IgeedR9Xatbx7zz28M3kyGVlZ/O/dd9O6Wzeq163jmdNPZ/XChSk8ie8oFKL7bbeRu/feBNXVLDzlFKo2GoPiE05gh3POIYjFWD55Ml/dcQehrCx2vftusrt1I7puHR+ffjpVLXkMgGHDduXiiwcSjcaZPPldJk2a3aB94sSh9OnTAYCOHfMoK6vmgAPuqWu/884jWb26kj/8YXozRr0V+bsgSdvcFhOot99+m4EDBwJQVlZWtw0wc+bMbRvZN1EbJW/Cbax95HaCnGyKfv47aoYcSFDcpr5PVTX5F19P5N0PqPnRoNTFuhV98cJSYjVxjig5hJWzV/P2hLkcfPsBde2r5q7hzUtms2FZZd2xJdO/BOBHDx/MstdX8PbVDZ/T4kRrKb7naj6f8BjxVjl0uWgU5f2HEGtdXNclVF1FhzsvJPujOZTv96O648X3/olPb3iKeHYuO511FOsPOop4flEqzuK7GXw0ZGXDqQdC7/3gzOth3NGJttXL4PQhie099odfX5lIngb+JHHs1wOh78Hwuxvqn9MC7X700USys5l84IF02m8/fnT99ZQcfTQAOW3bMuSKK7izb1+qysoY88ILfPzii/QYNoya8nL+esABtO3Rg/+55RYe+PGPU3si30Gbo48mlJ3N3AMPJH+//djp+uv58OsxANjpuut4p3dvYuXl9H3/fVY8/DDFP/85sfJy5hxwADk9etDtllt4vwWPQSQSZuLEwxkw4G4qKmooLT2JadM+Ytmyiro+Z531z7q+M2eO4Ve/erqu7dRT+7LnnsW88srnzR771uLvgqRtygoU0EQCNW/evOaK4zvJ+PgzYl06ERQVAFDbbw8yZ82h5seH1PUJVddQ/b9DqT2gHxkft9z/OW5s+axVbD8o8U1quz5tWD2vrEF7vCbOoFv347XzZ9Ud2/HwHeh0SEcAKpZuILtddrPFuy1kLVlEbccudYlP5e79yfnwLcoP+J+6PqHaatYdfDQb9jyQrI2qLNVddyO8YT1kRIAgUclpifYeCP9+LrH93uvQc5/N9zv7Zrj05xCPw4y/Q+lTiePbd00kWi1Yl4EDWfhcYgyWvP46O+xTPwatu3Xjq9mzqVqzBoClb75J5/33p7hXLxY++ywAqxYsoLhnz+YPfCsqHDiQsq/HoPz118nfp+HPQcWcOWQUFRFEo4mf9SAgt1cv1nw9BpULFpDbwsegZ892LFy4hrKyKgBmzlzMoEE78thjH27S98wz9+Ef//iEefNWALD//p3Yf/9O3HnnO+y+e9tmjXtr8ndBkra9Ju8D9fLLL3P++efzq1/9iosuuoh///vfzRHXNxIq30CQn1e3H+TlElpf0aBPUFRA7UEDmju0bSpaHiUrvz4HDmWEiEfrr1Ur7t+WvO1zN3leOBLmtXFv8dblc+hyxA7NEuu2Et5QTjy3oG4/np1HeEN5gz7x/CI27D3wv59K9Y670nXccLqefRQV/Q4hnle4zePdJvIKoXxt/X4sBhkZDfsM/Al88h58vqBhv4vuSSRW0x9rllC3lVaFhVSvrR+DIBYj9PUYrP7oI9r37k1e+/ZEcnLY+bDDyMrL46vZs+kxLDHVt9N++1HQqROhcMu9NV6ksJDo2sZ/DjbMm8fes2bR9733WPPUU8TWrqVi9mzafD0G+fvtR1anTtCCx6CwMIu1a6vr9tevr6GoaNMviTIzw/z61/247rrE/886dszn0ksHcfrpzzVbrNuKvwuStO1tsQL1wAMPMGPGDMaMGUPbtm1ZunQpd9xxB5999hkjRoxorhgblfPnv5L59jwyFnxMdK/6b8xCFRsIClv2dT3JiORHqK2or6UG8YBwJLn/6R1wzT70ObeK53/2MsOePpxIbsu6HK7tQxPJ+fBtWn02n6pd96o7Hq6qaJBQNSbrsw/Jf/tlPrn1ReLZuXS8+TzyX3u2QeWqxahYB3kbnXM4nPjwvLEfnwiP3Ljpcy//Bdw6Dia9Dif0gqoN2zTUbaV63TqyCurHIBQOE3w9BlVlZTx/1ln87G9/Y90XX/Dl22+zYeVKFjz9NMU9e3LS9OksLi3ly1mzCFrwYjnRdevIKNj8z0HunnvS+qijmLXzzsTKy+lx//20Pe44lk2eTG7PnuwxfTrrSkspnzUrUaFsYS6//GAGDtyRvfZqz+uvL607XlCQVVeN2tjhh+/MjBmfs25dItk6/vjdadcul2eeGUnHjnnk5mby4YermDJlTrOdw9bi74Kkbao21QGkhy1+2p42bRq33XYbBx10ELvvvjuHHnoof/nLX5g2bVpzxbdFlb8fy7p7J7LmX38j47MlhMrWQU0tmW/NIdqnV6rD2+aK+7Vl6YzE1KuVs1ezXY+mr9/55MnPee/O+QBEcjIIhUKEMlre1LVVo87ii8vuY9GkUjK/+pzw+jKorSHn/beo6tG3yefHcwuIZ2UTz2oFGRnECtuQUb5u2we+LcwphQOOTGz33g8Wzd20z279Yc6r9fs/PhHGjE9sV22AIA7xlnt3vMWlpex6ZGIMOu23H8vm1o9BKCODzvvvz92DB/PEmDG02313Pi8tpdOAAXw+cyZThgzhgyeeYM3HLXgRDWB9aSmtvx6D/P32Y8NGYxBbu5Z4ZSXxykqIx6ldvpxI69YUDBjAupkzmTdkCKufeIKqFjoGF130CkOG3E+HDn9ml11a07p1NpmZYQYP7sJrry3ZpP/hh+/Ms8/WL6xw881vsc8+kxky5H4mTHiNBx98r0UmT+DvgiQ1hy2WHTIzM8n4r6lAWVlZmxxLucwIFeN/S+GvxkE8TtWx/0O8QzGhsnXkXXQd5Tf/MdURbhM7Dt2Br0qX84+RrxAEAftf1Z9Ppy0muiHKLiN23vxzfrQD//7D2/zz5zOIR+P0v2BPMlql2b/nNxHJZMVJ4+l85ViIB6w9dDjRth0Iry+jwx0X8uV5t2z2adHiTqwdOoIuF51AEMmkpmMX1h5yTDMHv5W88gTsOxT+UgqEEivq/WgU5OQnFozYrh1sWN/wOS8/DhfeDbe9ApFM+PPvoaZ6c6/eInzwxBN0GzqUX5aWQijE308+mT1GjSIrP5+377qLWE0Np86aRbSqiteuv57KVatYBQy5/HIOOPdcqsrKmDp2bKpP4ztZ9cQTbDd0KHt+PQYLTz6ZdqNGkZGfz7K77mLZnXey58yZBDU1VC1axPJ77iGjsJAul1/ODueeS6ysjIUtfAyi0Thnn/0Czz8/inA4xOTJ77J06Xpat85m0qSjGD78bwDstltb7r13M180fA/4uyBJ214oCIKgscYxY8Zw7733Jn18c9rFN/3274dkZbgTlzE+1WGk1CVMINQyv8zdaoK9gANaXqVvq3ot4LKWulDHVnJJEFD6Ax+Dg4KAUOjKVIeRUkHwf/4uBAErVqxvuuP3WHFxgWPwAx+D4uKmLzlIN6FfpjoCCCanOoImKlDvvfceI0eObHAsCAIWbXRPCUmSJEn6odhiAvX3v/+dmTNnsv/++5OZmcmXX37JggULGDx4cHPFJ0mSJCkdeB8ooIlFJJ588kleffVVOnToQKdOnejYsSOvvvoqf//735srPkmSJElKG1tMoF555RVuvPFGcnJyAOjcuTMTJ07kpZdeapbgJEmSJCmdbHEKX05ODqH/utA1MzOTvLy8Rp4hSZIk6XvJKXxAExWonJwcFi9e3ODY4sWLN0mqJEmSJOmHYIsVqHPPPZfTTjuNAw44gB133JGlS5cyc+ZMrrnmmuaKT5IkSVI6qE11AOlhixWoXXfdlQcffJBevXpRWVlJ7969eeihh+jVq1dzxSdJkiRJaWOLFSiAgoICjj766GYIRZIkSZK2nqqqKs477zxWrVpFXl4e11xzDW3atNmkXzwe59RTT+Wwww5j1KhRW3zNLVagJEmSJAmAWBo8vqGHHnqIHj168OCDD3L00Udz2223bbbfn//8Z9auXZvUa5pASZIkSfpemjVrFoMGDQJg8ODBvPbaa5v0ee655wiFQgwePDip12xyCp8kSZIkpcMy5iUlJZSUlNTtjxgxghEjRgDw6KOPMmXKlAb927ZtS0FBAQB5eXmsX7++QfuCBQt46qmnuOmmm7j11luTisEESpIkSVKLsHHC9N+OP/54jj/++AbHzjjjDCoqKgCoqKigsLCwQfuTTz7JsmXLOOmkk1iyZAmZmZl06tRpi9UoEyhJkiRJ30v9+vXjlVdeYa+99mLGjBn079+/Qfv5559ft33zzTfTrl27JqfymUBJkiRJaloaTOH7pkaNGsW4ceMYNWoUmZmZXH/99QDcfffddOnShcMOO+wbv6YJlCRJkqTvpZycHG666aZNjp988smbHDvzzDOTek0TKEmSJElNq011AOnBZcwlSZIkKUkmUJIkSZKUJKfwSZIkSWpaLNUBpAcrUJIkSZKUJCtQkiRJkprWApcx3xasQEmSJElSkkygJEmSJClJTuGTJEmS1DSn8AEQCoIgSHUQkiRJktJb6MBURwDBq6mOoBkqUDvwybZ+i7S2lJ0ZwZRUh5FSJZzEjfw61WGk1P/jTkLnpTqK1AquhVCbVEeRWsFqCIUeT3UYKRUEx3Iyd6Q6jJS6m98QCqU6itQKAgjlpjqK1Ao2wIoV61MdRkoVFxf8oMeguLgg1SHoW3IKnyRJkqSm1aY6gPTgIhKSJEmSlCQrUJIkSZKaFkt1AOnBCpQkSZIkJckESpIkSZKS5BQ+SZIkSU3zPlCAFShJkiRJSpoVKEmSJElNswIFWIGSJEmSpKSZQEmSJElSkpzCJ0mSJKlptakOID1YgZIkSZKkJFmBkiRJktS0WKoDSA9WoCRJkiQpSSZQkiRJkpQkp/BJkiRJapr3gQKsQEmSJElS0qxASZIkSWqaFSjACpQkSZIkJc0ESpIkSZKS5BQ+SZIkSU2rTXUA6cEKlCRJkiQlyQRKkiRJkpLU5BS+IAh48803Wbp0Kdtvvz377rsvoVCoOWKTJEmSlC5iqQ4gPWyxArVy5UqOO+44Hn74YT7++GMeeOABjj32WJYvX95c8W1Wxkv/Jnf4meSO+D2Zjzy7SXto9VpyfnkBOSecQ/bvr4LKKgAiU18i95jTyR1+JpkPPpXoXFND9jkTyP3Z78n55QWEPl3SnKfyjQXxgMUXP8uCEVP4aPT9VH+2ukH72pc+Yv7wu1kwYgqrHnmnQVvtqgreO/hmqhatBGDDB8v46IR7+Wj0/Swa+xC1K8ub7Ty2liAe8PLF8/nbiLd5cvQ7rP1swyZ9aitjPD7ybdYsqgDgw8e/5MnR7/Dk6Hf4289mceeer1C9rmVP6h3WE974Hbx6Bpyy76btHQrghVNhxm+h5ETIyUwcP3bPxPNePxPGbuZ5LcmwI+CNF+DV5+GUMZu2t94OVnwE06cmHr/7dX1bTg7MfBZ227XZwt0mhg3ryBtvDOHVVw/mlFN22qQ9NzeDKVP6M2PGYP7970MYMKA1ACeeuCPvvnsYM2YM5pe/7NrMUX83QTxg0cUvMGfEw8wb/SiVn5U1aF/90iLeHf4gc0Y8zLJH5tYd/+LON5gz4mHePfYBlj06D4ANC1cxd1QJc0c+zKJLXySIxZvzVLaaYcPgjTfg1VfhlFM2bW/dGlasgOnTE4/f/a5h+513wtVXN0+s28qwI+GNf8Gr0+GUkzdtb90aVnwO059LPH53WuL4yOPh369A6Utw+03g98WSNmeLFagJEyZw7rnncsABB9QdmzFjBldffTUTJ07c5sFtVm2U7KvvpOKxmyAnm9xRZxMdsh9BcZu6Llm3PUDtsEOIHvsjsv5SQmbJM9T+4lha/ekuKp76C+Rmk3fUqdQedTCZ06YT5Oaw4ZE/E/p4MdmX30rlX69KzbklYe0L84nXROlRchIVs5ewZMKLdLv9eACC2hhLrn6BHo/9gnBOFh+NupfCIbuSWZxPUBvji4ufJZydWfdaS678B50uOoLcnh1Y+fDbLL/r33T6w+GpOrVv5eMXVhKriTO8pB9fzV5L6YRFHHn7nnXty+eu45VLFlC+rLru2O7Hbs/ux24PwIzLFrD78O1pVZi5yWu3FJEwTPwpDLgJKmqg9HSY9gEsW1/fZ/wQmDIL7psFlwyFX+8PN82ECf8D+9wE5dXw/rnw5DxYtWkOmvYiEZh4JQw4DCo2QOmzMO05WLbRdz399oaH/ga/G9/wuf37wB3XQ+cdmjXkrS4SCTFx4l4MGDCdiooopaWHMG3alyzb6Gf/vPN6MG/eOk46aRZ77lnI3nsX8fHHFVxxRW/69n2RsrJaXnhhIC++uILPNvNlRDpa/cJC4jUx9ioZyfrZX/LphFfoefv/AhCvjfHp1a+w12MnEM7JZO6oEloP6Ublx6tZ/85S9nxoBPHKWpZMngXA5zeU0uXsgyga0JmPxj/P6pc+pu3QXVJ5et9YJAITJ8KAAVBRAaWlMG0aLFtW36dfP3jooU0TJ4BTT4U994RXXmm+mLe2SAQmXgMDBn09Bi/BtGf+awz6wEOPwu/OqT+WnQ1XXAJ7DoDKSnjwnkQiNu3p5j4DKY15HyigiQrUV1991SB5Ahg8eDBffvnlNg1qS8KLPifeZQcoKoCsTGL99yDjrXkN+mTMeo/YoH0AiA4eQOTVRCUmvtvOhNZXQE0NBEAoRHjh58QGJ/oG3XYkvGhxs57PN1Ux6wsKB3UDIK9PJyrn1f9bVC1aRasurYkU5RDOyiC/f2cq3kqcz5JrXqTtyH5E2ufX9d/phmPI7dkBgCAWJ9Sq5S3K+NWstXQZlEieO/YpYsW89Q3aYzUBP751D1p3y93kucvnrmP1wgp6j2jZn5x7doCFq6CsEmpjMPMTGLRzwz5nTYX73058m7rjdonkKh5Az+tgXRW0zU20ldek5BS+s549YOEnULYWamth5uswqOGfLvrvnUiiXp4Gj9wNHRM/+rRqBceMgQ8/av64t6aePQtYuLCCsrJaamsDZs5cyaBB7Rr0OeKI9tTUxHnuuYO46KLdef755XTrlsfs2WWsWVNLEMCbb65h//3bNPIu6WfdrKW0HrQTAAV9tqdiXv2n5MpFq8nush2RomzCWRkU9t+BdW8toWzmZ+T2aMeHp0/lg9/8nTaHJH5hdrt5GEUDOhOviVG7ooLMtpv+3Uh3PXvCwoVQVvb178JMGDSoYZ/+/RNJ1MsvwyOPQMeOieP775943Hlnc0e9dfXcHRZ+vNEYvAqDDmzYp3/fRBL18vPwyP2JMaiuhgMPTSRPkEjEqqqaO3pJLcEWE6hwOP3WmAiVbyAoyKvbD/JyCJVXNNonyMuB9Yn2+K47kTf8DPKO+jXRQ/aFwnziPbuRMf11CALCsz8gtGwVxNJ3gmesvJqM/Oz6Axlhgmi8vq2gVV1TOC+LWHk1qx6fQ6RNbl3i9R+ZXydTFW9/wcr7Z9H+FwO2/QlsZTXlUbLy6xO/UEaIeLR+2s32/Yso2D57c09l1p2fM+D0nbZ1iNtcYStYW1m/v74aijZzyhkhmHcODOkOpZ8mjsXicMwe8O7ZMOPjRALWEhUWwNp19fvry6GosGGfDz+CSybAIT+BJ5+Gm69JHH/1dfgivWfuJqWwMJO1a+unoq5fH6WoqGFltV27VrRuncWPf1zKtGlfcd11e/DRR+X07l1I+/atyMnJ4LDD2pOXl9Hc4X9rsfIaMvKz6g80+JtY0+BvYsbXfxNr11RSPm8Zu904jO6XHcaCc58jCAJCGWGqlqxj9rB7qV1TSc7OrZv7dL6zwkJYu7Z+f/16KCpq2OfDD+GSS+CQQ+DJJ+HmmxMJxKWXwumnN2Ow20hhwX+NQflmxmABXHIFHHIEPDkNbr4eggD+c4XCGb+B/Hz454vNF7eklmOLGdIOO+zASy+91ODYyy+/TKdOnbZpUJuTNfEeckafR85plxIqr59aEqqoJCjIb9A3yM8lVFFZ105hPuEPPybj5Tcof3EKFS9NIbS6jMizM6gdfgTk55Ez5nwi018n3nsXyEjfDw8Z+a2IVdRPySEeEIqEN2qrLyHEKxIfHlb/7V3Wv/oJH42+n8oPlvH5uGnUrkhc77TmmfdZfMmzdPvLz4i0yaOlycqPUFtR/6k/iAeEI00n/tXrain7eAOd9m95H5D+4/IjYPpvYOrJULhRwlTQKlGN+m/ROPS+Dk79G9w7sv74E/Og0xWQFYEx/bd93FvT5Rckrmea+mDiQ9N/FOQnqlEbe+lfMP1fie0nnoa+e/K9cPnlvZg+fRBTpx5AYWH9lwkFBRHKyhqWFFetqmHq1ETVetq0L9lnn9aUldVy1llz+Nvf9mPy5H68/XYZK1e2nFJkRn5Wg797Df8mNmyLff03MXO7bLYbuBPhrAxyurUh3CqD2tWJX5rsToX0+8fJdBy1F59OaDnz2C6/PHE909SpiSTqPwoKEpWYjb30UqIvwBNPQN++cPzx0K4dPPMMjB8PJ5wAJ53UbOFvFZdfkrieaeqj/zUG+ZsZg5dh+tf/vE9Mhb57J7ZDIbj2Khh6GAwf1RxRSy1MNA0eaWCLnzTPP/98/vrXv/Kb3/yGq666itNOO43bb7+d//u//2uu+OrUnPULKu+7lvLShwl/vhTK1kNNLRlvzSXWt2eDvrF+vcl45Q0AIjPeJNZ/j0RFKrsVtMqCjAyCNtsRWldOeO58Yv17U3nftUQPP5D4jts3+7l9E3n9OrNuxiIAKmYvIbtHcV1bdve2VH+2mmhZJfGaGOVvLSa3b2d2fWA0u94/ml3vO5Gcnh3ocs1PyCzOZ/Xf57Hy/rfY5b4TabVjy0wkOvYr4rMZqwD4avZa2vbIb+IZCUvfXEvnA1vmOf/HRc/DkDugw2WwS1tonQOZGTC4G7z2WcO+tx4Dh3RPbK+vSkzfK2gFL/8GsjIS37xW1CSOtyQXXQVDfgoddoNddk4sFJGZCYMPgNfebNh30o0w/KeJ7cMGw6x3mz3cbeKii95nyJB/0aHD0+yySz6tW2eSmRli8OB2vPZaw0VmZs5cyZFHJuZrDR7cjvfeW0dGRoj992/D4MEzGDPmLXbfvYDS0lWpOJVvpbDfDqyZ8SkA62d/SW6P+mmLOd3bUPVZGbVlVcRrYqx7awkFfXegoH8nyv71KUEQULOsnHhlLZnbZfPBb/5O5adrgES1inDLWUHgootgyBDo0AF22SWxSEJmJgweDK+91rDvpEkwfHhi+7DDYNasRBVqn30SrzFhAjz4IEyZ0vzn8V1cdBkM+TF02Al26bbRGAyE195o2HfSbTD86MT2YUNg1tdrLt15S+JaqKN/Vj+VT5L+2xYvemnTpg3Dhg2jd+/eLFmyhKFDh/LRRx/Rpk0K58dnRqgafyq5Yy+AIKB2+I8IOrSDsvVkXziRqlsupua3o8gedx1ZjzxHvHUhVdePh9xsakccSe4J5xBkRgi67EDtMUMJlW8g88Z7yZr8GEFBPlVXnpW6c0tC0dDdWF/6CQtGToEAulx1FGumvUdsQw3tRvSl0/jDWTT2YQgC2gzfi6wOBZt9nSAWZ8mV/yBz+0I+OfNvAOQP6ML2vxvcnKfznXUb2o7Fpav528i3IYBDr9qNBdOWUbshtsVrm8o+2UBh581P7WtponE4exo8/6vE573Jb8LSdYmEatLxMPzexIIRdwyHiw9PJEmnPZGY6vfAOzDjtMTUvTlfJq6TaomiUTj7Qnj+MQiHYfIDsPTLREI16UYYfhKMvwwm3wyn/TKx0MQp/y/VUW9d0WjA2WfP4fnnBybGYPJnLF1aRevWmUya1I/hw1/nqqvmM2lSP1599WBqawPGjHmLWCygpibOrFmHUlUV4/rrP2LVqpZTgWozdBfKSj9j7siHCQLY5aofsWLah8Q21NBxxF7sNH4w7499HIKA9sN706pDPq065LPuzS+Yc9xDEATsfPGhhDLCdDp1AAvHP08oM4NwTia7XNGyFtWBr38Xzobnn//6d2EyLF2aSCb+kziNH584ftppiUUWNrdSX0sWjcLZ4+H5qV+Pwb0bjcFticrS+Itg8h1w2qlf/z04Dfr2gbEnwb9K4aWvF/i98TZ4cmpKT0dKLy170eKtJhQEQaPfOd9888189NFHXHPNNeTk5PDFF18wYcIEevbsyelJTpTegU+2WrAt0VJ2ZgQt7Gu8rayEk7iRXzfd8Xvs/3EnofNSHUVqBddCqOWsTbBNBKshFHo81WGkVBAcy8nckeowUupufvODXx47CCDU8tbo2KqCDbBixfqmO36PFRcX/KDHoLh4819yp7NQGixaHKRBErfFKXwzZszgxhtvJCcnB4DOnTszceLETa6LkiRJkqQfgi1O4cvNzSX0X1+TZWZmkpfX8hYbkCRJkvQdtNDVere2LVagsrOzWby44X2RFi9evElSJUmSJEk/BFusQJ177rmcdtppHHDAAey4444sXbqUmTNncs011zRXfJIkSZLSQQtbrXdb2WIFatddd+XBBx+kV69eVFZW0rt3bx566CF69erVXPFJkiRJUtrYYgUKoKCggKOPProZQpEkSZKk9LbFCpQkSZIkqZ4JlCRJkiQlyQRKkiRJkpJkAiVJkiRJSTKBkiRJkqQkmUBJkiRJUpJMoCRJkiQpSU3eB0qSJEmSoDbVAQCZqQ7ACpQkSZIkJcsESpIkSZKS5BQ+SZIkSUmIpjoAnMInSZIkSS2ICZQkSZIkJckpfJIkSZKSkA6r8OWkOgArUJIkSZKULCtQkiRJkpKQDotIpJ4VKEmSJElKkgmUJEmSJCXJKXySJEmSkpAOi0h8M1VVVZx33nmsWrWKvLw8rrnmGtq0adOgzyuvvMKtt94KQK9evbjkkksIhUKNvmYoCIJgm0YtSZIkqcULhZalOgSCoMM36n/33XdTXl7OmWeeydNPP80777zDhRdeWNdeXl7OyJEjuffee2nTpg133XUXw4cP3yTJ2tg2r0CFQl9s67dIa0HQmekcmOowUmoIrzKBs1IdRkqNZyL7UJrqMFLqLQ4iNCPVUaRWMBhCU1MdRWoFP4UjeTzVYaTUMxxLu/iSVIeRUivDnQi9n+ooUivoBR+yU6rDSKnd+ZQVK9anOoyUKS4uSHUI30LqK1AlJSWUlJTU7Y8YMYIRI0Y02n/WrFmccsopAAwePJjbbrutQfs777xDjx49uOaaa1i8eDHHH3/8FpMncAqfJEmSpBZiSwnTo48+ypQpUxoca9u2LQUFiWQ1Ly+P9esbJu1r1qzh9ddf58knnyQ3N5ef//zn9OnTh5133rnRGEygJEmSJLV4xx9/PMcff3yDY2eccQYVFRUAVFRUUFhY2KB9u+22Y88996S4uBiAffbZhw8++GCLCZSr8EmSJElKQjQNHt9Mv379eOWVVwCYMWMG/fv3b9C+xx57sGDBAlavXk00GuXdd99ll1122eJrWoGSJEmS9L00atQoxo0bx6hRo8jMzOT6668HEotLdOnShcMOO4xzzjmn7jqpH//4x/To0WOLr2kCJUmSJCkJqV9E4pvKycnhpptu2uT4ySefXLd91FFHcdRRRyX9mk7hkyRJkqQkmUBJkiRJUpKcwidJkiQpCd98EYfvIytQkiRJkpQkK1CSJEmSktDyFpHYFqxASZIkSVKSTKAkSZIkKUlO4ZMkSZKUBBeRACtQkiRJkpQ0K1CSJEmSkuAiEmAFSpIkSZKSZgIlSZIkSUlyCp8kSZKkJLiIBFiBkiRJkqSkmUBJkiRJUpK2mEAtX758s8fffPPNbRKMJEmS9P/bu/P4KKpsgeO/6i3ddBYCISEiQUARDI4sDsawGZGIiAvEsIws+piZpzCLCJHIoiiKCctjBllGBJ4scZRVwfGpIzAgARUYyQyLQgADhB2SQCfp9HbfHy0dG0jISJLu4Pl+PvmYrnurPafoqq5T91ZFBCtnEPwEXqUF1PDhw9m2bZvvtVKKWbNmMWHChBoPTAghhBBCCCGCTaUPkViwYAHPP/8833zzDf369WPMmDHcdNNNrFq1qrbiE0IIIYQQQgQFeYgEXGMEKiYmhsWLF7Nr1y4eeOABevbsydSpUwkNDa2t+IQQQgghhBAiaFQ6AuVwOMjIyKCwsJCXX36ZhQsX0qJFC7p27Vpb8VVZnz5mXnopHJdLsWhRCQsWFPu116unMW9efZo3N2Ayafz+9wVs3x4c8yh/Ko9H8ddJxRz7zo3BBENeCyW6md7Xvv2jMtYvtqPTQ5NWegZNsqLTaXzyVik5Gxy4nYrug8x0TjUHMIvrozyKbZN2c/67i+hNOjq/difhzax+fVylbj59+ms6v34n9VuWF/+l58pY2y+bBxd18lteV7g27KJszlow6DGmdMHUv7tfu+f8Rexj5oPdgRZdH/Mb/4VmCcH56Q4c8z8GTcM4oDum1G44V2/BuSYbAFXmxLPvCKHZf0ILrxeI1H4Sa84GGq6bg9IbuNA5haJu/f3aDeeOE7N4HJrbDShODXkVZ+MWAGhlpdw882lODnsdZ2zLAERfPfrEwEutwKVg0RFYcMS/vakFFrUDgwYa8Nsc2F8M/WIh/VZQwPw8WHjkKm8eRJTHw9lJH1L23Qk0k4Ho1/phbBblay/esI+COevBoCM85W7C+3eqcJ2yPfmcefkDNJMBU5tYosb3QdPpKJi/CdvfctCFhlD/192wJrUJYMY/nXHjVurNXYrS6ynr14uy/n2u2s+8eCW6s+cpGf3bWo6wZli3b6Dh8jkonYELPVIoSu5/1X71172DoeAsZ4eOQV9whtgZz/vaQg7v4+yQ0RT1GlRbYVcbjwf+MsnA999pGE3wu9ecxDYrb9/6qY5V8/VoGiQPcJOc6sHlhDfHGTidr+F0QOqzbu7p4QlcEkIEsUoLqNTUVLp27cq7776LwWAgMTGRUaNGsW3bNl544YXaivGaDAaYObM+v/zlKYqLFdnZ0axbV8qpU+U7flpaGLt3Oxk2rIA77zRy113GOl9A5XzuwOmAse9HcGiXk5UZxYyYFw6Aw6748E8lvLSuPiaLxoLnL/LvjU7MoRoHv3GS9tdwHKXw90WlAc7i+uR9fgq3w0Of9xM5vauArzP28cC8u33tZ/9dyNaXd1Nyyu63nsfpYetLuzGY6+aDKJXThf2N97CunAiWEEoGTcGQ1A5dowhfH8fctRj73IOxXxfK5v8N5/ubMA55gLIZK7GuegnqmSnuPR5Dj/YY+3XB2K8LAPZXlmJM6VKniidcThq9/wZHxq/EE2IhLmMQtruScEc08nVp+OGfKUwaTHH7B6i3+wuiVv8PJ0bMJuT7fxOz7GUMBacCmMD1M2gwsy38cjMUuyC7C6w7BafKyvtMbg2zD8OHJyG5EbxxB6Ruh4w2cPdmsLlg7/3wwUk45whcLtdS/PlelMPFze+PwL7rCGczPiZ23lAAlNPN2Tc+4uaVv0NnMZI/6C/US2qD/Zu8q65zZuIaoiY8grlDM87N/AzbuhxMrWOxfbSLJitGAJA/8C9YElqis5gCmfZ/zunCmjGXouXzUBYzEU/+AUdSIqpRg/I+9jJCX5qBIWcfjuTguzj6k7icNFr0Bkem/XA8GDcI2y+TcEeWHw+0MjsxcydgPvAvbAnJALgjG3HstaUAmL/9hqismRT1vHrhFey++lyH0wFT33fy3S6NRRkGxs/zTr1yu2HJDD0zVjkx14Pf9TaS0MPD1xt1hNVXjJrm4kIBjOpr4p4eQXwgEAFSt8+dq0ulZ49jx45lzJgxGAzeOqtJkyYsW7YMhyO4dqg2bYzk5rooLFQ4nbBlSxldu4b49XnwwRAcDvjkkygmTgzj00/tFbxb3ZG700V8VyMALdoZydtdPi/VYIIX3ovAZNEA8LjAGAJ7tzho0srAX0ZeZO4zF/jFfXXshOAyp3eep0lX75didLtIzu0u8mt3OzzcP6cjES38R5i+zvyW1gPjqBddN0ffPAdPoIuLRouwopkM6DvehnvHfr8+7p0H0He9EwBDtztxbd2Lptdh/fh1tLB6qEIbAJq1fBu4/30YT+5xTAPuq7VcqoPp5EGc0XF4rBFgMFF6W0csB3b49TmTOpbiO72jdJrHjTJ6jxGay8HxEXNw/DAaVVe1CYPcYih0glPBlvPQtYF/n9F74G8/1IkGDexu8ABtNsIFFzQ0eUembEE+xd2+83ssXVsBYG4XR9nufF+b4+BpjHEN0UdY0EwGzB1vwb7j+wrXcZ0qwtzBe2ne3KEZ9p3f4zx4GkunFuhCjOhCjBibNcTx3clazvL66Q/l4Y5rgooIA5MRZ4e2GHf+y6+PVuag7LGelD7zZICirH6mYwdxxsbhCY0Ao4nSNh2x7PU/HmjOMi4kPc75J5658g2UInrBZE49Mwn0+ivb64C9OzXad/VeRL69nSJ3d/npnl4Pcz52Yg2Di4XeZWYrdO7l4Vd/dPv1E0JcXaUFVGJiIjabjTVr1jB79mxWr16Nw+EIuqfwhYdrFBWVjzZdvKiIiND8+kRF6YiM1NGr11nWrbMzfXrE5W9T59htCktoeZ46vYbbpby/6zTCo7z/vBuXllJWomjT2YitQJG328Vv/xzGr14JZdGYiyilAhJ/dXDYXJhCywdSNb2Gx1X+WYjp2IDQWIvfOgdWH8PcwOQrvOoiZStFCyvPS7OaUbbSy/rYfX00qxkulnh/N+hxfraTksdexnB3KzCUf0s63vobppGP1kIG1UtXasNjCfO99pit6Eptfn08YQ3AYMR48hBRKzI598hIAOy3dsTVILZW460J4QYo+tGFwYsuiDD69znn8E7va2WF6fHwynfe5W4FfWMh5z7YfA6cQT5rx2MrQxdaXvhreg3lcpe3hZW36awheGz2CtcxNG1A6deHACjZuA9PqQPT7Y0p3XEYj60Md0Ex9m+O4CkNrguHVaHZSlCh5VOalbUe2kX/6e0qIgxn51/Wdmg1Sldiw1PvsuNByWXHg9AIStp1uer61u0bKGt6G84mdfeiSolNw/qj64Y6Pbh/dGFEb4Btn+l47jETd9yt0BvAYoV6oVBig8w/GHnyuSC/kiICxBUEP4FXaQH1/fffM2DAAA4dOkSjRo04cOAAqampHD58uLbiq9TkyeFs3NiItWujCA8vTyUsTKOw0L8oOHfOw9q13hPMdetKufvuuj3yAmAO1bAXl+epPKA3lBdUHo9iZWYx+7Kd/PebYWiaRmh9jTu6GDGYNBq30GMI0bh4vu4WUKZQA87i8itmygM6Q+XT8g6sOsrxrWf5vyFfcn7fBb4Ym0PJmbJK1wkWZTNXUzIkk9IRb/oVTKrYjhbmP+VOCzWjiu2+dn40Jc+Y3BHr5hkopwvXBz/c+3ShBM+hExgS6s69Hg3XzOTmaUNoMnsEOnv5CZLOXux3AnWJ5dsvuWnOSE4On+q7/6mum9waNibC2k7eIuqSMIN3NOpy9zWEDzrBkH9673+6ZM0JaPIZmHQwtGnNx309dKEhqOLyfVZ5FNoPFwIub/MUewuqitaJnvIEBW/9gxO/fQd9w1D0kVZMLaOJePJeTvzmfzmX+THmu5qij/S/tzKYWf60kPChowgbOQGtuMS3XCsuQYXXvfs9q6ph1kxunjCEJm+M8CuYdPZiPNYrjwcVCd+0tsJ7puqKeqGK0h/t397zA/8+9yZ7WLTZgcsJGz/wfm+eOQEThhpJesxN90eC/EqKEAFU6ZlmZmYmM2bMYPTo0QwYMICxY8cyffp0MjMzayu+Sk2ceIGkpDPExBzn1lv1REZqGI3QrVsI27b5nxBv2eKgd2/v1cdu3ULYs6fuz+Fs2cHA7s3ePA7tctKklf94e9ZLxTjLFM/MDfNN5WvZ0cieL5wopSg85cFRqgitr13x3nVFdIdIjm32/sHn07sKiGx17S/J3ln30ntZAg8tTaBBm3C6Zt5FvUYh11wvGISM6ke9pWMJzZ6J58hpVKEN5XDh3rEffXv/hx/oO9yGe5N3uo5r87/Rd2yFspVSMjgD5XCi6XRolhDQeQ8D7u3foU+8o9Zzuh7n+o7iWNpSDs7Ixnj6CLriQnA5sOzfgb1Fe7++lm+/JPq918l/bgFlt9wZmIBrwMRvIWkrxHwKt1oh0ghGDbo1hG0F/n3vawh/vhN6fQk7f5jtGmaAfyR6CycFFP8wrS+YmTvcQslm7/CZfdcRTK0a+9pMLaNx5p3DXViCcrgo3XEYc/u4Ctcp2fQt0VOeIHb+U7gLS7B0vhX3eRvughKa/PUZGo5/BNeJQky3xdR+oj9R6XPDubBkJgVfrEKfl49WeAEcTow7/oWrXd3ax/8T554cxbHXlnLwf7MxnjyC7mIhOB1Y9uzAfnv7a65/ScjBPdhbd6i5QGtBmw6KnZu9x/bvdmk0a1V+obTEBuMGG3E6vId/s8X738KzMOm/jAxLc/HAE8F+FBAisCp9iITNZqN169Z+y+Lj4ykqKqpgjcBwueD554v49NNG6HSwaFExx497iIzUWLCgASkp55gy5QILFjRg69ZGOJ0wdOj5QId93dr1NLEv28nUgUUoBcOmhPL1ujLKShTN2hrYurKMW+82MHPYBQDuH2qmfc8QDmx3kvGEd52BL1nR6etuAdWsZ2OOZ5/lo4FbQUGXKb/g4Lp8XCVubh8QF+jwaoxmNGBOH0jJ8P8BpTCmdEEXE4kqtGGf8A6W2b/D9Gwf7GMX4li+GV1kKOYZ/41WLwTDIwmUPJmBZtCju70phkfvBcBz+CS6m+votEaDkTP907l55nBQiqIuKbgiY9AVFxKzeAInRswm+v0paG4njRelA+Bo3JzTQ14NcODVx6Xg+T3waQLoNO9T+I7bvQXVgnaQsh3+1BZMGiz+4VzyOxs88y/IyofNnb1T9/51AZYdDWgq12TteQel2Qc4NnCe936VKU9wcd0uVImD8AGdaJj+MCeGL0IpRXjK3RhiIrD2DLtiHQBjsyhO/PYdNIsRyz0tsHZvjVIK17HzHEuZjWY00PCF3mj6OvjAGaOB4vRnCf/NWPB4sPd7CE9MI7TCC1gnTsf25o3z+fdjMHLm6XRufnU4eBRFPVJwNYxBd7GQmDkTOJE+u8JV9UXn8VisoNXd70WAhJ4edmXreGGgERT8YYqLTet02EvgwQEeuj/i5sUnjRgMcMvtiu6Pelj0hh7bBY3lcw0sn+t9n5fedhJSN28VFjWm7g9AVAdNVXIDzODBg1m2bNkVy/v378/y5cur9j/Qjv306G4ASt3MRhIDHUZAJbGVDEYFOoyASmcmd5Md6DACaged0TYHOorAUt1AWxvoKAJLPQq9WR3oMALqY/oR5cm/dscb2FldE7S9gY4isNQd8C23BDqMgGrN95w5czHQYQRMo0ZVn1oaLDQt8MdvpfoFOoTKp/C1adOGrKwsv2VZWVnEx8fXaFBCCCGEEEKIYOMMgp/Aq3QK36hRo5g4cSLvvfcecXFxHD9+nKZNmzJ16tTaik8IIYQQQgghgkalBdRnn31Gly5diI+Px+Px0LZtW2JjY/nkk094/PHHaylEIYQQQgghhAgOlRZQBw8e9HutlGLq1KmYzWYpoIQQQgghhPhZCY6/wxRolRZQo0eP9v2el5dHeno69913H+PGjavxwIQQQgghhBAi2FRaQF2SlZXF4sWLefHFF0lKSqrpmIQQQgghhBBBJzge4hBolRZQp06d4sUXXyQiIoIVK1YQERFRW3EJIYQQQgghRNCptIDq06cPRqORhIQEXn3V/w/uzZgxo0YDE0IIIYQQQohgU2kBNWfOnNqKQwghhBBCCBHU5CEScI0CqlOnTrUVhxBCCCGEEEIEPV2gAxBCCCGEEEKIuqJKT+ETQgghhBBC/NzJU/hARqCEEEIIIYQQospkBEoIIYQQQghRBfIQCZARKCGEEEIIIYSoMimghBBCCCGEEKKKZAqfEEIIIYQQogrkIRIgI1BCCCGEEEIIUWUyAiWEEEIIIYSoAnmIBMgIlBBCCCGEEEJUmRRQQgghhBBCCFFFMoVPCCGEEEIIUQXyEAmQESghhBBCCCGEqDIZgRJCCCGEEEJUgTxEAmQESgghhBBCCCGqTAooIYQQQgghhKgiTSmlAh2EEEIIIYQQQtQFMgIlhBBCCCGEEFUkBZQQQgghhBBCVJEUUEIIIYQQQghRRVJACSGEEEIIIUQVSQElhBBCCCGEEFUkBZQQQgghhBBCVJEUUEIIIYQQQghRRYZAB1AT5s+fz5IlS1i/fj3jxo3j9OnT5OfnYzQaiY6OplWrVkycODHQYdaYH+cfEhJCeno6e/bsoX79+gC43W5eeeUVbrvttsAGWoOutg169+5Nt27dfH06d+5MdnZ2AKO8fpfn+eabb/LRRx8RHR3t65OWlsamTZt8y91uN2azmTFjxnDHHXewevVqZs2aRdOmTQFwOBwMGzaM3r17ByqtKjl69CjTpk3j5MmTmM1mzGYzaWlpvPbaa3g8Hg4dOkSDBg2oX78+iYmJxMTE+OUJ+I4FQ4YMobS0FIvFAoBeryczM5OYmJhApfeTZWRksGfPHs6cOYPdbqdp06ZERkayc+dOsrOzGTZs2FW3z7PPPhvo0KtFRfmPHz+e5ORkMjIyeOihh/jggw9YtWoVZWVl5ObmEh8fD8D06dPr5L/7JRXlv2XLFl+Ol8ydO5d+/fqRkZFBx44dAdi7dy+jR49m5cqVWK3WQKRw3Q4cOMC0adMoLS2lpKSE7t2707dvXx577DHi4+NRSuFwOHj00UcZPHgwAG3btqV9+/YAuFwuWrZsyaRJkzAY6v5p0tW2x+9//3sKCgrIzMzk+PHjuN1uYmNjSU9Pp1GjRoEO+bp99dVXjBw5knXr1hEbGwt49+0WLVrw4IMPMnPmTPbt24dOp8NqtTJ27FiaN29OdnY2mZmZLF++HLPZzKlTp/j1r3/NggUL6vRxQdQAdQPq06ePev3119WqVat8y2bNmqXefffdAEZVey7Pf+zYsWrTpk2+9n/84x9q5MiRgQqvVlxrGyilVGJiYiBCq1aX51nR5/zy5bm5uerBBx9UdrtdrVq1Sk2bNs3XVlBQoLp27ao8Hk/NJ/ATlZSUqIcfflj985//9C3LyclRgwcP9r2+/N/88jx/bPDgwSo3N9f3OisrS02ZMqUGIq89l+d7+ef9avvEjeTy/OfOnaumT5/u9xlRSqmjR4+q1NTU2g6vxv04/8py/Oqrr1SvXr1UaWmpKisrU3379lW7du2qzVCrVVFRkerTp486fPiwUkopl8ulRo4cqd59912/beBwONRvfvMbtX79eqXUlfvHH//4R/X555/XWtw1paLtkZWVpQYNGqT+/ve/+/pmZ2ervn37KpfLFaBoq8+XX36pEhIS1LBhw3zfZdOmTVOrVq1So0aNUkuWLPH13bdvn3rooYfUhQsXlFJKZWZmqpdfflk5HA41aNAgtWXLloDkIILbDTeF76uvviIuLo6BAweSlZUV6HBqXVXyLyoqol69erUcWe35uXwGrifPli1bEh8fz86dO69ou3jxImazGU3TqivUardx40YSEhJ8V4wBfvGLX7BkyZJqef8bfR/5uVFK8eGHH/L000/jdDrZv39/oEMKGp06daJ79+7MmTOHt99+mx49enDXXXcFOqyfbP369dxzzz3ccsstQPlockJCgl8/o9HI0KFD+fjjj694D6fTSUlJyQ1xDKhoe7Rt25awsDAeeOABX9/ExETi4uLYvn17gKKtXgkJCURERPh9PxYUFLB//36GDBniW9a6dWuSkpL47LPPABg1ahR79uxhxIgRJCYm0rlz51qPXQS/uj82fZkVK1aQmppKixYtMJlM5OTk1Okvg//U1fIHmDZtGm+//TY6nY7o6GjS0tICHGnNudY2uKSoqChQIVaLivJ85513fCcFlU1XbdiwIQUFBQB89NFH5OTkoGkaFouFqVOn1k4SP9GxY8eIi4vzvX722Wex2WycPn2axYsX07hx46uudynPS1JSUnj88ccBGDt2LBaLBU3TaN68+Q29j/zcbNu2jVatWtGgQQNSUlLIysrilVdeCXRYtSo3N9fvpDE+Pp709HTAe8I4YMAA6tevz8KFCwMVYrU4ffq03zRdAKvVitFovKJvVFSU7xhYVFTk2z6aptGtWzfuvffemg+4hlW0PY4dO3bFcoCmTZty/Pjx2gqvxk2aNInU1FS6dOkCgMfjuWbeRqOR/v37M2nSpJ/dcUJU3Q1VQBUVFbF582bOnz/P0qVLsdlsLFu27GdTQFWUv16vJy0tze/+nxvVf7IN6vJVpYryjIuL46mnnmLQoEHXfI/jx4+TnJzMkSNH6NOnD2PGjKmFyKtH48aN2b17t+/1vHnzAOjfvz8ul6vC9SrLMzMzk5YtW1ZvoCIoLF++nGPHjjF8+HCcTifffvstY8aMISwsLNCh1Zpbb72VpUuXXrUtJCSEHj16EBUVhV6vr+XIqtdNN93E3r17/ZYdPXqUkydPXtE3Pz/fd7ElIiKiwu1Tl1W0PaKiosjPz7+if15eHomJibUVXo2LjIxk3LhxpKen06FDB5xO51ULxLy8PN/xPz8/nwULFpCWlkZaWhpLliyp8/uFqH431BS+tWvXkpKSwqJFi1i4cCHLly8nOzub8+fPBzq0WvFzzx9+PtvgevPcv38/ubm5tGvXrmYDrSE9evRg27Zt7Nq1y7csLy+PkydPBvXUQ1H7zp8/T05ODitWrGDhwoUsWbKE5ORk1qxZE+jQRA1ISkriiy++4MiRI4B3Ol5GRsYV0zYdDgdLlizh4YcfDkSYtaai7XHgwAHOnj3Lhg0bfH03b95MXl4enTp1ClS4NeL++++nefPmrFmzhsaNGxMXF+c3rW/Pnj1s2LCB5ORkHA4Hzz33HOPGjeOpp54iNjaW2bNnBzB6EaxuqBGoFStW+E09slgsJCcns3z58gBGVXsqyn/lypW+Jw3d6H4u26CiPFesWMH48eOvus6lqX06nQ6DwcCsWbPq7BOmrFYr8+bNY8aMGUyfPh2Xy4XBYGDy5Mk0adKkwvUun8IXGhrqG70SN6YPP/yQ5ORkvyvI/fv354UXXvCb0naju3wKH8CUKVOuOp2pLgsNDSUjI4MJEyaglKK4uJikpCS6devGjBkzGDJkCJqm4XK5eOSRR26o0ZarqWh7/OpXv6JXr15MmTKFt956C/CO7M+fP/+GHG0ZP348X375JeCdbTB16lRSU1PR6/WEh4czd+5cwsPDmTx5Mh07dqR79+6Adwpgv379SEhI4J577glkCiLIaEopFegghBBCCCGEEKIuuKGm8AkhhBBCCCFETZICSgghhBBCCCGqSAooIYQQQgghhKgiKaCEEEIIIYQQooqkgBJCCCGEEEKIKpICSgghhBBCCCGqSAooIYQQQgghhKii/wc8HTKwk/zblgAAAABJRU5ErkJggg==\n",
      "text/plain": [
       "<Figure size 1152x864 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Having a look at the correlation matrix\n",
    "\n",
    "plt.figure(figsize=(16,12))\n",
    "mask = np.zeros_like(df.corr(), dtype=np.bool)\n",
    "mask[np.triu_indices_from(mask)] = True\n",
    "sns.heatmap(data=df.corr(), cmap=\"jet\", annot=True,linewidths=1, linecolor='white',mask=mask)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "metadata": {},
   "outputs": [],
   "source": [
    "x = df.drop('TEY', axis=1)\n",
    "y = df[[\"TEY\"]]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Text(0.5, 1.0, 'Correlation with Turbine energy yield \\n')"
      ]
     },
     "execution_count": 47,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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XpOfvi6zoecPI/2X/5IlS2ry5MmPtI9z585p2rRp8vLy0sKFC1W5cmWr9T/99NMj1+fl5SU3NzfFxMQoMTEx09ET6b14lPk97qdmzZry8PDQr7/+qnr16unGjRsKDg6WdHtI+syZM7Vnzx7LB8cGDRrYvIasll773UKIR72U5G4jI/78809JUtGiRR9pv5lJP47r1aun999//7H38zjvh/spXLiwLly4oAEDBjx2D9JHodxtPpe7jRRJDw7379+vixcv6ttvv5Wnp6dlLpUHkT9/fjVq1Ehff/21IiMj9c033zzQZSHp7+mUlBSFhIRYzQ/yoAoUKKBcuXLp8uXLmZ5fH2RUDgAgc8w5AQA5TOXKlVWtWjVFRUVpwYIFd93u4sWL+uyzzyRJHTp0kHR71EXu3Lm1f/9+ywe5O509e1aHDx9Wvnz57nnN/4OoVKmSPDw8dODAgUwvQTl27JiaNGmiXr163fWvsAcOHFBaWppq166dIZhIS0uzzNNw5193H/SvzG5ubgoMDFRKSoq+++67DOtTUlK0fft2SZlfv/64PDw8FBwcrAMHDlhClvRwokqVKvLw8NDevXu1a9cueXl5PdBf27PbyIj0iQuvXr2aIaA4ffq0/vrrr0fa76+//prhmLl586Z27twpZ2dnVatW7dEKzkT6vnbu3Jnpcfr555+refPmmj179j33Y4v3w4PWumPHjkzXv/XWW3r99de1f//+++4rODhYzs7O+uWXXzKdPPKHH36462PbtGkjwzC0aNEinT59+r6TWN5tH5K0adMm7dmzR9WrV7/vZSHu7u6qUqWKkpOTLeeGOyUlJal169bq0KFDprcIlm6/h2rWrKnk5ORM+5g+/wsA4OERTgBADjR8+HDlypVLM2fO1OTJkzP8BfrkyZPq3r27Ll++rICAAL322muSbg93btu2rVJSUtS3b1+ru0rExsaqb9++SktLU9u2be85IuNBeHp66rXXXlNCQoI++ugjq+919epVffzxxzp79qyKFSt21w/V6de779u3T3FxcZbliYmJ+uSTTywTYt55Z5H0ERAP8lf5rl27SpLGjRtnNflecnKyRo0apbNnz6pSpUoZghFbqV+/vlJTU7VixQrlzZvXcjcUd3d3BQQE6PDhwzp69Kjq1KnzQH8Ffpjnbg9eXl4qUqSIbty4YXU3i4SEBI0aNeqR93vy5ElNnz7d8nVSUpKGDRumuLg4vfDCC5a/+ttCzZo15e/vr8jISE2bNs1qItrDhw9r2rRpOnHixD3nkZBs8364n86dO8vZ2VlTp061mvTSMAzNmjVLu3bt0vnz5x8oeCxWrJiaNGmimJgYjRkzxup5r127Vj/++KOkzAOxf/3rX5a7Ykj3nsTybmrXrq2iRYtqzZo1llDhQaS/p0eMGKHjx49blqekpGj06NE6dOiQbty4cc+gI/3OQ+PGjbMaZXbgwAHNnz//oZ8LAOA2LusAgBzIz89PS5cu1XvvvaeFCxdqxYoVqlixogoWLKgLFy7owIEDMgxDgYGBmjNnjtV10n379tXhw4f122+/qXHjxpYJ5vbu3avr16+rTp066t27t03q7Nevnw4dOqSff/5ZTZo0UeXKleXq6qrw8HBdv35dAQEB9/xelStXVmBgoCIjI9WsWTMFBQUpLS1NkZGRunbtmp599lmdPHnSasLK4sWLy9XVVUeOHFG3bt1UvXp1vffee5nuv3HjxurWrZuWLFmiNm3aqFq1avLy8tL//d//6eLFiypRooSmTp1qk15kpl69epJuhwn169e3CiCCg4Mttxp90Es60ucSmD17tiIiItS1a9f7TgyY1d58801NmDBBw4cP15dffqn8+fMrPDxcnp6eCg4Ovu9dLjJTpUoVzZs3T9u3b1eZMmV08OBBXbhwQWXKlNGwYcNsWr+Tk5OmTp2qrl27av78+dq0aZPljg/h4eFKS0tT586d7zvHgfT474f7qVSpkgYOHKgJEyaoU6dOKl++vIoXL67jx4/rjz/+kIeHh2bMmPHAweOQIUO0f/9+ffHFF/rll19UsWJFnTt3TocOHVKpUqV09uzZTOdgyJcvn5o0aaLNmzffdxLLu3F2dtbLL7+sefPmPdRlIU2bNlXXrl21bNkytW7dWhUrVlShQoV08OBBRUVFydvb+77v6Tp16ljOC82bN1etWrWUlJSkPXv2qGLFivr9998f+vkAABg5AQA5VtWqVfXVV1/pvffe0zPPPKPDhw/r22+/1dmzZ1WrVi2FhIRo5cqVGSbq8/Dw0JIlS/Txxx+rdOnS+vXXXxUeHi5fX1+NGTNGCxcufOxRE+ly586tZcuW6eOPP1apUqW0b98+RUREqHTp0ho4cKBCQ0PvOdzbxcVF8+bNU+fOnZUvXz7t2rVLx44dk7+/vyZPnqywsDA5OTlp586dlr/sFihQQKNHj1bx4sW1d+9e/ec//7lnjQMHDtTs2bMVHBysw4cPa8eOHcqbN6969uypDRs2qFSpUjbpRWZKlCihZ599VtL/LulIl34piYuLi+V2lffTtm1btWzZUikpKfr555914sQJ2xb8CN58802NGzdO5cuX1/79+xUZGalGjRppzZo1jzyvSePGjTVz5ky5urrqxx9/VGpqqt544w19/vnn952Y8lGUKVNGGzdu1BtvvKFcuXJp165dOnXqlKpVq6ZPP/1UQ4YMeaD9PO774UG88cYbWr58uRo0aKA///xTP/30k9LS0vTKK69o48aND3XJS5EiRbRmzRq1bdtWiYmJ+v7773Xr1i2NGjVKnTp1kpT5nUQkKSgoSJLuO4nlvaTv42EvCxk8eLBmz56t6tWr69SpU9q5c6c8PDzUuXNnbdy4Uc8888x99zFw4EBNnjxZZcuW1e7du3X8+HF16tRJISEhj/RcAACSk/GoFy4CAADgHykxMVGnT5/Wk08+memtRseMGaPPPvtMCxcu1PPPP59hfdeuXbV37159//33evLJJx+phsGDB2vdunVauXKl6SOAAACPj5ETAAAAeCjJycl69dVX1aRJE0VHR1utO3TokDZs2CAvLy+rSzZu3bolSVq/fr12796t+vXrP3Qwkb6Pn3/+WZs3b5afnx/BBADkEMw5AQAAgIeSN29etWvXTmFhYWrSpImqVq2qJ554QtHR0fq///s/ubi4KCQkRLlz57Y85o033tDhw4eVmJgoNzc3ffjhhw/9fYcMGaLvvvvOMsltv379bPacAADmIpwAAADAQxs6dKgCAwO1evVqnThxQnFxcSpYsKCaN2+ut956K8NdPwICAnTs2DE988wzGjhw4CPdjrhSpUr66aefVKhQIfXs2dMyaSwAwPEx5wQAAAAAADAVc04AAAAAAABTEU4AAAAAAABTEU4AAAAAAABTEU4AAAAAAABTEU4AAAAAAABTEU4AAAAAAABTEU4AAAAAAABTEU4AAAAAAABTEU4AAAAAAABTEU4AAAAAAABTEU4AAAAAAABTEU4AAAAAAABTEU4AAAAAAABTEU4AAAAAAABTEU4AAAAAAABTEU4AAAAAAABTEU4AAAAAAABTEU4AAAAAAABTEU4AAAAAAABTEU4A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qkNOCuKok5k4InptP/Q6Ix+ffNJKJ7PLeHgdDMXExJtdxUOh1/blkP120F5nJc7b1hzyuJYc9th2yH47aK+zEucR+3HUXmfphJgAAAAAAACPinACAAAAAACYinACAAAAAACYinACAAAAAACYinACAAAAAACYinACAAAAAACYinACAAAAAACYinACAAAAAACYinACAAAAAACYinACAAAAAACYinACAAAAAACYinACAAAAAACYinACAAAAAACYinACAAAAAACYinACAAAAAACYinACAAAAAACYinACAAAAAACYinACAAAAAACYinACAAAAAACYinACAAAAAACYinACAAAAAACYinACAAAAAACYinACAAAAAACYinACAAAAAACYinACAAAAAACYinACAAAAAACYinACAAAAAACYinACAAAAAACYinACAAAAAACYinACAAAAAACYinACAAAAAACYinACAAAAAACYinACAAAAAACYinACAAAAAACYinACAAAAAACYinACAAAAAACYinACAAAAAACYinACAAAAAACYinACAAAAAACYinACAAAAAACYinACAAAAAACYinACAAAAAACYinACAAAAAACYinACAAAAAACYinACAAAAAACYinACAAAAAACYinACAAAAAACYinACAAAAAACYinACAAAAAACYinACAAAAAACYinACAAAAAACYinACAAAAAACYinACAAAAAACYinACAAAAAACYinACAAAAAACYinACAAAAAACYinACAAAAAACYinACAAAAAACYinACAAAAAACYinACAAAAAACYymbhRGpqqqZMmaI6deooMDBQH3zwgS5fvnzX7b/66iu1atVKAQEBatKkiRYsWKDU1FTL+p9++kl+fn4Z/l28eNFWJQMAAAAAgGzA1VY7mjlzpjZs2KCQkBB5eXlp1KhR6tWrl1atWpVh2x07dqh///4aPHiwnn/+eR0+fFjDhg1TcnKyevbsKUk6fvy4ypcvrwULFlg9tmDBgrYqGQAAAAAAZAM2CSeSkpK0fPlyDR06VLVr15YkTZ06VY0aNdK+ffsUFBRktf3nn3+upk2bqlOnTpKkUqVK6dSpU1q/fr0lnDhx4oR8fX3l4+NjixIBAAAAAEA2ZZNw4ujRo7p+/bqCg4Mty0qUKKHixYsrPDw8Qzjx3nvvydPT02qZs7Oz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7dq1ZpeWI23fvl2nTp1SdHS0JQBq1qyZihYtaplEmmPctj799FMtW7ZM69evt/yhyjAMubu7y9PTU+7u7pZleHgbN27U119/rdmzZ6tz585q2LChAgMDNWbMGH344YcyDENjx47VH3/8QQiURZycnFSkSBGVLl1aR48elSTLH2QdCX8ytpP0vyS7ublp5cqV8vPzkyR16dJFTk5Omjx5siTpnXfeMbPMHKd27doKCAjQ999/r2LFiulf//qXnJycNH/+fK1cuVKhoaHy8vJi6KoN/PDDDxozZozatWunbt26ydXV1TJ8b/78+Vq7dq0mTJigH3/8UZs2bZIkLvHIAvXr11eDBg20dOlSFStWTE2aNJG7u7sWLFigsLAwLV26VN7e3hzzNrJ582aFhoZq2bJllmAiPQwKDw+Xl5eXvLy8zC3SwXz//ffatm2bFixYoEqVKmnq1Knq1auXFi9eLEl66aWX5O7uzvBgG0lNTVVqaqpWrVolX19fSdK7774rJycnDRs2TJL06quvmllijrJt2zb17t1bTk5OeuKJJxQUFKSnn35aNWrU0LVr1ywjrTi2bSchIUHnzp2Ts7Oz/vrrL0myjJqdO3eu9uzZo82bNzOS9hGkn4cjIiLUoEEDVahQwbIuJSVFrq6uevnll5WcnKwZM2Zo586deuqppzh/28iqVauUO3du+fj4qEyZMvL29paPj4/++9//SvrfcZ5+SaQj9N3JICK0i3PnzqlkyZKWr+/8BTYxMVFhYWGaNGmS+vXrR0DxmNLfeOk9jouL0/vvv69bt25pwIAB2r9/v5YsWaLJkyerbt26DvFGdQR//fWXRo0apTNnzqhp06aWgGL+/PlasmSJJk2apOeff17R0dEaN26coqOj1bhxY7399ttml54jvfvuu9q3b59mzJih33//XaGhoZoyZYrq1KnDMW9DBw8e1MCBA5UnTx7NnDlTRYoUkSQtWLBAc+fO1YoVK1S+fHnCoAeUkJAgd3d3xcbGqmjRopbz+K1bt9SzZ09dunRJb731FgFFFrnzOE1ISND8+fO1aNEijR49moDCBn777TdVr15dV65cUWRkpE6fPq29e/fq6NGjKliwoI4dO6YGDRpo4MCBeuqpp8wuN0dIP0dER0crJCREP/74oxYsWKDq1atrwYIFWrx4saZPn65atWoxp80jSEtLU2Jiopo3b6433nhDnTt3tjov33lOef/99xUVFaW1a9fSZxvYt2+f+vfvr6tXr+rWrVvKnz+/ChcubAnhevfurYCAAOXPn1/58+c3u9wHRjhhBzNmzNDcuXPVtWtXBQcHq1GjRhm2uXXrllasWEFAYQPnz59XiRIlrJZdvXpV7733ns6fP6+bN29q2rRpev755/nA8JjSfwAlJyfLzc1NCQkJ+uSTT3TixAm1a9dOsbGxCg0NtQQT6T/4o6OjNXjwYKWkpOjTTz91qJNmdnT+/HnduHFDLi4uyps3r+UDcvfu3bVz5055enpq6tSpql+/vrmF5iCrV6+Wr6+vAgICdPToUfXv319ubm7asGGDli5dqnnz5mnq1KmqXbs255kHtH37do0fP16hoaEqWbKk5a9u6f9NTEzU+++/T0BhIzt37lRQUJDy5Mlj1b87+0lAYTtjxoxRWFiYfvnlFxUsWNBqXWpqqk6cOKGIiAjNmzdPDRo0UNeuXS2X5sE2rl69qhEjRug///mPmjVrph9//FETJ05kgmgbaN26tQICAjRs2LAM5+P03xHXr1+vpUuXasWKFXriiSdMqjTniYmJ0dWrV3XgwAEdPXpUBw4c0KFDhyw/P9PvBuTr66v8+fOrVatWKlu2rNll3xXjl+zgxo0byp07t7Zt26affvpJc+fOVY8ePVS+fHnLfAceHh5q166dDMPQjBkzdOvWLfXq1cvkyh1PehD0xhtvKDg4WA0bNpQkFShQQHPnzlXv3r31559/KikpyfJBmQ8Oj+7ChQsqUaKE3NzcJEl58+bV8OHDNWrUKM2fP1+XL1/WzJkzLUFQer+LFCmiCRMmKC0tjWDiMc2dO1fffPONzp07p5s3b6pcuXJq1qyZunfvrgULFqh///7atm2bUlJSlJiYqFy5cpldssMbPXq0tmzZonXr1kmS/P39NXnyZPXp00fBwcGSpFmzZln+n/PL/c2aNUvXr19X0aJF1bdvX02dOtUy67irq6tSU1OVK1cuzZkzR++//74WL14sZ2dnvfDCC5ZRiHhwK1eu1CeffKJSpUqpefPmqlmzpuV4dXJysgRCefPmVY8ePSTdngckMTFRHTt2NLN0hzRu3Dht3rxZGzduVMGCBa0CoPQPbv7+/vL391euXLk0ffp0ubi4qH379pZLbfBwvvnmGx06dEh//PGH/P399fzzz6tixYoKCQnRiBEjtG7dOvXr14/RhI8oJSVF8fHxypUrlzw9PVWmTBn99ttvio6OVtGiRa22Tf8dMSoqSrly5VKePHnMKDnH8vHxkY+Pj+VcsXfvXr377ruaMmWKnJ2dderUKR09elTh4eFKSEhQ69atTa743hg5YQe7du3SZ599psaNG8vFxUWff/65Dhw4oGLFiqlLly4KCgpS5cqVJd3+IZU+FHjbtm0qUKCAydU7lvHjx2v16tV64okn5OHhoXz58qlHjx7y9fVV6dKlFRcXp169eumvv/7Su+++q6ZNmzrMNVjZzd2CIOn2JR7jx4/Xvn371L59e3Xs2FFubm5Wd/HgA9vjS5+IcciQISpRooQuX76sTZs26bvvvtMrr7yi8ePHS5J69OihyMhIjRo1Sg0bNiSgeAzpt6FbtmxZhlstHjlyROPGjdP58+e1efNm5c2b1+RqHcPYsWO1adMmrVq1SufPn9eCBQt08+ZNzZgxwxJQuLi4WP6bmJioXr166dixYxo4cKBeeukls5+Cw4mIiNDo0aPl7u6utLQ0HT16VP/6179Uu3ZttWzZ0rJd+s/GhIQETZ8+XZs3b9b27duVL18+E6t3LOPGjdOGDRsstzJPD37+7s5zycaNGzV06FB17NhR/fr1I4B7SJMmTdLXX3+tp556Sk5OTtq3b5+8vb31wgsvaMCAAUpISNCIESP0448/as6cOapZsya/Bz6EsLAw/fbbb9qzZ49y586tf//73woMDFSbNm30wgsvaPTo0Zke40OGDNETTzyhjz76iF4/opMnT+r8+fOKiYlR/vz5FRQUpEKFCkn635QBCQkJatasmYYMGWL18zE1NVXXr1/P9qNWCCfspH379sqTJ48WLVokSVq7dq127typb7/9Vk8++aSqV6+url27qnjx4sqfP7+uXr1KMPEI7hUEtW/fXi+++KJ8fHzUo0cP3bp1S6+//rpatWrFB+VHcLcgyM/PT6VKlVJCQoLGjh2rQ4cOqWXLluratatVQIHHk/6BbsmSJapYsaJleVRUlDZt2qRZs2bplVde0ejRoyXdnoNi//79GjhwoF588UV+2X0E48aN0/r16xUWFpbphwzDMHT06FHLh4klS5bI29vbxIqzv5CQEK1du9bywU26Pbnu4sWL7xlQpM8h9NFHH1nN54R7mzVrlurUqaNixYpp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P3689u3bp/bt26tjx45yc3PjQ/JjmjVrln766SfFxsbq8uXLatCgQaZzptwZUEyZMkV+fn70HsADYQ4V2zt06JDat2+vf/3rX+rXr58KFSokyXoerA8++ECVK1dWz549Ta4WuLf9+/frzTffVMWKFVWmTBl5enrql19+0R9//KHU1FR5enoqKChIv/zyiwoUKKCAgAB99NFHevLJJ80uPUsQTgBANjB+/HitXr1aTzzxhDw8PJQvXz716NFDfn5+KlWqlBISEjR27FgdOnRILVu2VNeuXQkoHsOUKVO0du1aTZw4UU899ZSKFSum2NhYFS5cWL///rtKly5tNUxy165dCg0N1bFjx7RkyRLm9ADwwBihYntr167VyJEj1axZM3Xp0kVVqlSRJN28eVPz58/Xxo0btWzZMnoOh7Bv3z4NGDBALVq00FtvvaU8efLor7/+0pEjR3Tw4EGdO3dOe/bs0aVLl+Tm5qYtW7bk2MmLCScAIBvYtWuXPvvsMzVu3FguLi76/PPPdeDAARUrVkwdOnRQ06ZN9eSTT2rkyJE6efKkatWqpffff98yaRIe3FdffaUZM2Zo4sSJqlKlitVw1NDQUM2fP1//+te/9P7778vb29vyuJ9++klr167VwIEDc/SQSgDI7rhrBHKa8PBwffjhh2rVqpXat2+f4feM5ORkxcfHKyUlJccGExLhBABkG+3bt1eePHm0aNEiSbf/MrRz5059++23KlKkiJo1a6aGDRtq+vTp8vHx0SeffJKjJkHKaukhxLhx43Tjxg0NHz7cavLQBQsWaPHixQoODtaff/6poKAgvf/++1Y9vnnzpnLnzm1G+QCAv+GuEchJwsPD1bt3b7Vq1UodOnRQ8eLFJd2e5PWfckkY4QQAmCz9vtbh4eF666231KdPH73xxhuSpB49eujYsWMqU6aMIiMj5eTkpLJly2ry5MncIeIhGYah1NRUvfLKK3rppZf03nvvWS6L2bx5s0aOHKn58+erWrVqmj59ur788kt169ZN7du3t5okEwAAICvcGVB07Ngxx84tcTf8pgUAJku/hV/p0qVVrlw57dmzR5I0aNAgHTp0SMuWLdOsWbP01VdfqX379ho/fjzBxCNwcnKSq6ur8uTJoxMnTkiSJXCoVauWQkNDVa1aNUlS7969dfnyZV25ckUuLi4EEwCQTd35d1b+5gpHl/4Hkq+++koLFy5UVFSU2SXZFb9tAUA24ePjo06dOmnnzp1q1aqVdu/erXnz5ql06dLKnTu3ihYtqv79++vpp582u1SHZBiG0tLSVKlSJR0/flxHjhyxLC9UqJAqVaok6fa1zL/88ouKFi2qRo0amVkyAOA+7ryFJbezRE5QrVo1jR8/Xrt377a6/PSfgHACALKR5557TlWqVFF0dLRCQkJUsWJFs0vKMdIvzXj99dd14cIFhYaG6vLlyxl+mXV2dtZPP/2kIkWKWK73BAAAsJeaNWtq/fr1KliwoNml2BXhBABkI97e3nruueeUkJBgmXgxLS3N5KpylmeffVbDhw/X1q1bNX78eEVGRlrWXbx4UdOmTdP69es1dOhQJhwFAACm+CdOwM2EmACQTaTfTSIxMVHNmzdX+fLlNWPGDLPLypHS0tK0detWjRw5Uvnz51fx4sXl5OQkNzc3XblyRRMmTJC/v7/ZZQIAAPxj/DPuSQIADsDJyUmGYcjNzU1BQUE6evSo4uPjlS9fPrNLy3GcnZ3VokULValSRT/88INOnDghNzc31a5dW5UrV1aRIkXMLhEAAOAfhZETAJANnTt3ToZhqFSpUmaXAgAAAGQ5wgkAwD9a+uU0f/9/AAAA2A/hBAAAAAAAMBV36wAAAAAAAKYinAAAAAAAAKYinAAAAAAAAKYinAAAAAAAAKYinAAAAAAAAKYinAAAAAAAAKYinAAAAAAAAKb6fzsIpGwIfBJgAAAAAElFTkSuQmCC\n",
      "text/plain": [
       "<Figure size 1296x720 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "data2 = df.copy()\n",
    "\n",
    "correlations = data2.corrwith(df.TEY)\n",
    "correlations = correlations[correlations!=1]\n",
    "positive_correlations = correlations[correlations >0].sort_values(ascending = False)\n",
    "negative_correlations =correlations[correlations<0].sort_values(ascending = False)\n",
    "\n",
    "correlations.plot.bar(\n",
    "        figsize = (18, 10), \n",
    "        fontsize = 15, \n",
    "        color = 'orange',\n",
    "        rot = 45, grid = True)\n",
    "plt.title('Correlation with Turbine energy yield \\n',\n",
    "horizontalalignment=\"center\", fontstyle = \"normal\", \n",
    "fontsize = \"22\", fontfamily = \"sans-serif\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Feature Selection Technique<a class=\"anchor\" id=\"5\"></a>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Feature importance\n",
    "from numpy import set_printoptions"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "metadata": {},
   "outputs": [],
   "source": [
    "# feature extraction\n",
    "test = SelectKBest(score_func=mutual_info_regression, k='all')\n",
    "fit = test.fit(x, y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "metadata": {},
   "outputs": [],
   "source": [
    "# summarize scores\n",
    "scores = fit.scores_\n",
    "\n",
    "features = fit.transform(x)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Score</th>\n",
       "      <th>Feature</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>1.712704</td>\n",
       "      <td>CDP</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>1.599840</td>\n",
       "      <td>GTEP</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>1.325077</td>\n",
       "      <td>TIT</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>0.891890</td>\n",
       "      <td>TAT</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>0.660760</td>\n",
       "      <td>AFDP</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>0.512258</td>\n",
       "      <td>CO</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0.412339</td>\n",
       "      <td>AT</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>0.301021</td>\n",
       "      <td>NOX</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0.140308</td>\n",
       "      <td>AP</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>0.097477</td>\n",
       "      <td>AH</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "      Score Feature\n",
       "7  1.712704     CDP\n",
       "4  1.599840    GTEP\n",
       "5  1.325077     TIT\n",
       "6  0.891890     TAT\n",
       "3  0.660760    AFDP\n",
       "8  0.512258      CO\n",
       "0  0.412339      AT\n",
       "9  0.301021     NOX\n",
       "1  0.140308      AP\n",
       "2  0.097477      AH"
      ]
     },
     "execution_count": 51,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "score_df = pd.DataFrame(list(zip(scores, x.columns)),\n",
    "               columns =['Score', 'Feature'])\n",
    "score_df.sort_values(by=\"Score\", ascending=False, inplace=True)\n",
    "score_df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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nDnZDcd+D8B/+MMfxpw5c2Lq1KkOwLNozpw5UVhYGL/5zW/iC1/4wj7r33nnnbj88svjV7/6VVx77bVZSJjbHn744Zg8efJe5c/o0aNj+fLlzTfbbGhoaC5TaT/f+ta3sh2BFjguSLbdu3fHj370o1i5cmUcf/zxcfLJJ0ffvn2jpqYmfve730VFRUX88z//c8yaNctxdhb8/ve/j5tvvjluv/32+Md//Mfm5ZMnT46tW7fGrFmzoqysLIsJef3116OioiLWr18f77//fhx99NFRVlYWY8aMiT59+ih+skz508H16NEjJk6cGBMnToxNmzbF0qVLo7KyMh5++OGIiFi0aFFMnDgxjjvuuCwnzU1btmyJPn36xIUXXhhnnXVW8wEe2ecAPNlWr14dM2bM2G/xExFx7LHHxpQpU+KOO+5Q/mTB/k4adiJxMixdujSWLFkSRx11VLajsB+OC5Lt/vvvjzVr1sS8efPin/7pn/ZZ/+yzz8b1118fjzzySIwbNy4LCXPX5s2b48orr4xzzz03Pve5z+217oorroh77703rrjiili8eHGccsopWUqZ25YuXRrTp0+PTp06xcCBA+P000+PHTt2xJIlS+LBBx+M6dOnx8iRI7MdM6cpf1LkzDPPjDPPPDP+8z//M1atWhXLli2LRx55JB555JG48MILY968edmOmHOcmZVcDsCT7b333ouTTz65xW1OPfXU+Pvf/95OiaBjeOutt/a6FwbJ4rgg2SorK+O6667bb/ETEXHRRRfFNddcE48//rjyp53Nnz8/vvnNb+73ZsHnnHNOnHPOOfH9738/5s2bF3Pnzs1Cwty2YcOGmDZtWnz3u9+NyZMn73XT7d27d8f9998f06ZNi379+u1zQ2jaT362A9D6OnXqFMOGDYt77703fve738W1114b27Zty3asnPTxmVmVlZWxePHiGDJkSFRWVsaECRNiz549sWjRIj+8Zsnq1atjzJgx8cwzz8T48ePjq1/9atx8883x4osvOgBPgIaGhujSpUuL23Tp0iV27tzZTokADp/jgmR78803M36K11e/+tV444032ikRH9u0aVNcfvnlLW7zne98JzZs2NA+gdjLggULYuTIkfGDH/xgn09b69SpU0yePDnGjh0bCxYsyFJCIpz5k3o9evSISZMmxaRJk7IdJec5MytZXDKZfEo4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ExERc+bMyW444sknn4xrr732Mz8Rp3PnzvHd73435s+f387JAID29txzz2U7AgfA3ZcASJwtW7ZEnz594sILL4yzzjqrufghGV5//fUoKyvba1nv3r2jU6dOzY8HDhwYf//739s7GgAA++HMHwASZ/Xq1bF8+fJYtmxZ3HXXXXHMMcfEN77xjRg6dKh7LiRAly5d9rq5c0TE8uXL93pcX1/vpqgAAAnhzB8AEqdHjx4xceLEqKysjMWLF8eQIUOisrIyJkyYEHv27IlFixY5qySL+vbtG//zP//T4jbPP/98nHLKKe2UCACAlrjhMwAdwu7du2PVqlWxbNmyeP755yMi4sILL4x58+ZlOVnueeSRR2LevHnxwAMP7PfG26+++mqMGzcubrjhhvjWt76VhYQAAHyS8geADufdd9+NZcuWxfLly6OysjLbcXJOU1NTTJo0KV544YUYOXJkfPnLX46jjjoqqqqqYt26dfHYY4/FV77yFTd8BgBICOUPAHDQ9uzZEwsWLIhHHnlkr0vwevToEd/+9rfjyiuvjIKCgiwmBADgY8ofAOCwvPHGG/Hee+9FSUlJ9O7dO/Lz3VIQACBJlD8AAAAAKeZXcwAAAAAppvwBAAAASDHlDwAAAECKKX8AAAAAUkz5AwAAAJBi/w9osFFL6is1DgAAAABJRU5ErkJggg==\n",
      "text/plain": [
       "<Figure size 1440x432 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "fig, axes = plt.subplots(figsize=(20, 6))\n",
    "plt.bar([i for i in range(len(scores))],scores)\n",
    "axes.set_xticks([0,1,2,3,4,5,6,7,8,9])\n",
    "axes.set_xticklabels(x.columns.values)\n",
    "plt.xticks(rotation = 90, size = 15)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAABJYAAAIaCAYAAAByNcfqAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjUuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/YYfK9AAAACXBIWXMAAAsTAAALEwEAmpwYAABa/UlEQVR4nO3dfXyN9ePH8fex+1juNvq6l9sxtzNyH0aamxCmL0alVO6KkptWSISkJKRQbr4YxpIJIUqappV8ixqF3DVmMbOdbc7vD1/n55jdXdk5Z/N6Ph4eD+dzPudc73Mus533Ptd1mSwWi0UAAAAAAABAHhVxdAAAAAAAAAAUTBRLAAAAAAAAMIRiCQAAAAAAAIZQLAEAAAAAAMAQiiUAAAAAAAAYQrEEAAAAAAAAQyiWAAAoYN577z3VqlUr2z+//PJLvm3/woULSk5Ozrfnz6tr165p1apV6tu3rxo3bqyGDRuqS5cumjNnji5fvuzoeE7NbDbr3LlzOc67du2a/vzzT+vtiIgI1apVS9HR0fkZT5L0008/aeTIkWrZsqX8/f3VqlUrvfDCC/rpp58MPV90dLRq1aqliIiIO5wUAIC7k6ujAwAAAGOeeeYZ3X///be9r1y5cvmyzd27d+vFF1/Uhg0bdM899+TLNvJq7NixioqK0sMPP6xu3bqpSJEiOnTokD766CN9/vnnWr16tUqWLOnomE7n1KlTeuKJJzR06FD16tUry3lJSUkaPHiw2rZtqxEjRtgxobRnzx4988wzql69ukJDQ1WqVCmdOXNG69ev1+eff6733ntPQUFBds0EAABsUSwBAFBAtWjRQs2aNbPrNg8ePKhLly7ZdZvZ+f7777Vp0yaNGzdOjz/+uM19bdq00fPPP6+PPvpIL730koMSOq8///xTf/zxR47zEhMT9dNPP6lt27b5H+oWb7zxhmrXrq01a9bIzc3NOh4aGqpHHnlEU6ZM0YMPPihXV36kBQDAUTgUDgAAFFixsbGSpJYtW2a67+GHH1bZsmX1ww8/2DkV7oSEhAT98ccfatasmU2pJEklSpRQjx49dP78eZtD9AAAgP1RLAEAUMjFxsbq8ccfV6NGjdSoUSM98cQTOnjwoM0ci8WiVatWqXfv3mrUqJHq1aunzp07a9GiRbJYLJKkcePGad68eZKkDh06aODAgZKkgQMHqn379pm2e+v4wIED9eSTT2rOnDlq1KiRmjdvriNHjkiS4uLiNGzYMDVp0kQNGjRQv3799NVXX+X42ooWLSpJCg8P17Vr1zLd/8UXX2jlypU2Y+fOndOECRPUqlUrNWrUSI8++qi++OILmzmnTp3SSy+9pAceeED16tVT9+7dFR4ebjNn3Lhx6ty5s1auXKnAwEAFBgZqz549kqSzZ89q7Nix1sf36NFDn376abav5Y033pCfn58SExOtY0eOHFGtWrX0zDPPZJobEBCgtLS0TM+TXa4bIiIiFBoaKkkaP368atWqddtM0dHR6tChgyRp3rx5qlWrlk2Rc+HCBb344otq0qSJGjdurGHDhun06dM2z5Gamqo5c+aoffv28vf3V4cOHfTuu+/KbDZn+354eXnJxcVFO3bsUHx8fKb7R4wYof/+97+qUqWKdSw+Pl5TpkxRhw4d5O/vr4CAAIWGhurAgQPZbuvatWtasmSJOnfuLH9/f7Vu3VpTp05VUlKSzbz9+/erf//+atKkiRo1aqR+/fpp586d2T43AACFHeuGAQAooC5fvqyEhIRM497e3tYVHnv37tXQoUNVu3ZtjRo1SmazWREREerfv7+WLl2qJk2aSJLeeecdLVy4UD179lTfvn115coVbdy4UbNnz5avr6969uypkJAQJSUlafv27Ro/frxq1KiR58zff/+9jh8/rpdeekl//vmnqlevriNHjujf//63fHx8NHToULm5uemzzz7T008/rdmzZys4ODjL5+vUqZPefvttLV++XDt37tRDDz2k5s2bq0mTJrrnnnvk7u5uMz8xMVF9+/ZVYmKi+vfvr4oVK+qzzz7T8OHDNW/ePAUFBenkyZPq27evUlNTNWDAAPn6+mrbtm0KCwvTH3/8obFjx1qf78yZM3r//fc1fPhw/fXXX2rQoIHOnTunPn36yGKxaODAgSpevLh27Nihl156SX/99ZeGDBly29fStm1bLVu2TPv371enTp0kXS8ybrxvFotFJpNJkvT111+rZcuWmVbyZJfrZoGBgXrmmWe0cOFChYSEKCAg4LbPU61aNY0fP17Tp09Xx44d1bFjR5UqVcp6/4QJE9SkSRO9+OKLiouL03/+8x/9+eefioyMlCRlZGRo6NCh+v7779W3b19Vq1ZNhw4d0sKFC/XLL79owYIF1td0Ky8vLwUHB2vTpk0KCgpS+/bt1apVKz3wwAMqX758psPfUlJS1L9/f12+fFn9+/dX2bJl9ccff2jVqlV6+umntXv3bhUrVuy225o4caI2btyonj17avDgwTp69KhWrVql77//XqtWrZKHh4eOHTumoUOHys/PTy+88IKk64Xmc889pxUrVli/lgAAuOtYAABAgTJ37lxLzZo1s/zz7bffWiwWiyUjI8PSoUMHS79+/Szp6enWx1+5csXSsWNHyyOPPGKxWCwWs9lsady4seWFF16w2c7ly5ct/v7+lqFDh2ba9smTJ61jAwYMsLRr1y5TzlvHBwwYYJPv5vGgoCDLlStXrGNpaWmWf//735YWLVpYUlNTs30/fvjhB0uHDh1s3oO6detahg4davnxxx9t5s6cOdNSs2ZNS0xMjHUsJSXFEhQUZHn00UctFovF8vzzz1tq165tOXTokHVORkaGZejQoZZatWpZfv31V4vFYrG8/PLLlpo1a1rWr19vs42XX37Z0rRpU8u5c+dsxkePHm3x9/e3nD9//ravIzU11dKgQQPL5MmTrWPDhg2ztG7d2lKzZk3L4cOHLRaLxXLq1Knbbvfm7Wd3/w3ffvttruadPHnSUrNmTcvcuXOtY+vXr7fUrFnT8uyzz9rMHTdunKVmzZqWEydO2Mzbs2ePzbzVq1dbatasadm+fXu2205KSrI8//zzmf6Nd+nSxbJixQpLRkaGde7mzZtvu61Vq1ZZatasadm6dettX/eN26tWrbJ53FdffWWpWbOm5eOPP7ZYLBbLokWLLDVr1rRcuHDBOichIcHSqVMny7Jly7J9HQAAFGYcCgcAQAH18ssva+nSpZn+1K5dW5L0888/6+TJkwoKCtLff/+thIQEJSQkKCUlRe3atdMvv/yis2fPys3NTd98842mTJli8/wXL15UsWLFlJycfMcye3p6KjAw0GYb+/fvV9u2bZWSkmLNeOnSJXXs2FHnz5/P8bLyDRo00Oeff64PPvhAISEhqlChgtLS0rRr1y6FhIRo06ZN1rlffvml6tata7NCx8PDQ4sWLdLcuXOVkZGhL7/8Uq1atVLdunWtc4oUKaJnnnlGFosl06FPrVq1sv792rVr+uKLL9SkSRO5urpaX09CQoI6deoks9msvXv33vZ1uLu7q1mzZvr2228lXT888bvvvtPAgQNVpEgRxcTESJK++uormUwmtWnTJtv35eZc+aVLly42t+vVqydJ1kPXtm3bplKlSqlu3bo270Xbtm3l4uKiL7/8MtvnL1q0qObMmaOoqCiNGDFCjRo1kqurq3777TdNmTJFzz33nDIyMiRJwcHB2rdvn83rvvlwu6z+HW/btk0mk0lt27a1yVinTh35+vpaM953332SpNdff12HDh2SJJUsWVJbt261HhYKAMDdiEPhAAAooOrWrZvtVeFOnDghSZo5c6Zmzpx52zlnzpzRfffdJzc3N3355ZfasWOHfv/9dx0/flx///23JFnPsXQnlChRQkWK/P/vtU6ePClJWr58uZYvX55lxpy4urrqwQcf1IMPPihJOnbsmP7zn/9o+fLlmjp1qjp27ChPT0+dOnXqtueDqlq1qiTp/PnzSk5Ott6+WbVq1SRdP//SzUqXLm39+8WLF3X58mV98cUXmc7blJvX06ZNG02ZMkXx8fE6f/68EhMT1b59e23atEkxMTHq37+/vv76a/n7+8vHxyfb9+TmXPnl5sPipOvFoSTruZ9OnDihhIQENW/e/LaPz82+la6/98OHD9fw4cN1+fJlff7553r33Xe1a9cubd261Xq4pMlk0qJFixQbG6sTJ07oxIkT1iy3OwfXjYwWi8X6b+dWN87j1blzZ23fvl1RUVGKioqSr6+v2rZtq549e3IYHADgrkaxBABAIXXjg/SoUaPUsGHD2865//77ZbFY9NJLL+mzzz5TQECAGjVqpJCQEAUGBmrQoEGGt39jJcnNXFxcbjunf//+CgoKuu3zVK9ePcttzJs3T2XLllWfPn1sxu+//3698sorSktL0+rVqxUXFyd/f39lZGRkeU4fKfsS7cb7eet5m25+TTdez0MPPaR+/frd9nkqVqyY5TZurEL69ttvdeHCBZUqVUrVqlVT06ZNtXXrVqWnp2vfvn252i+3vtf54eaS8HYyMjJUpUoVvfbaa7e9/957783ysV9++aX27t2rl156yeY99/b2Vp8+fVSzZk317dtXBw4cUHBwsE6dOqWQkBAlJyerVatWCg4Olp+fnywWi4YNG5bldq5du6aiRYtaT0x/Kw8PD0mSm5ub5s6dqyNHjmj79u3as2ePIiIitG7dOo0ZM0ZPP/10tu8FAACFFcUSAACFVPny5SVJ99xzj1q0aGFz38GDB/X333/L09NTMTEx+uyzz/Tcc89p1KhR1jnp6elKTEzMtgiRrpcLt7vC1/nz53Od0cXFJVPGuLg4/fnnn/Ly8sry8Rs3bpQk9e7d+7aFUc2aNSXJ+hzlypWzruS62YYNG3TgwAG9+uqruueee3Ts2LFMc37//XdJ/39I1O2UKlVKXl5eSk9Pz/R6Tp8+rZ9//jnb11OxYkVVrVpV3377rRITE62HDTZt2lTLly/XZ599psuXL6tt27ZZPoczqVChgg4dOqQHHnjApoRKS0vT9u3bs30v//vf/2rZsmXq2LGjmjZtmun+GyePv7FKat68ebpw4YK2bNlic6W4mw+FvJ3y5ctbV4HdWnRt3bpVJUqUkHR9/50+fVpNmjRRrVq1NHz4cJ09e1aDBg3S4sWLKZYAAHctzrEEAEAh5e/vL19fXy1fvlxXrlyxjiclJen555/X+PHj5eLiYr28/a0rg8LDw3X16lWlp6dbx26UAzev7PHx8dGFCxd07tw569ihQ4d0/PjxHDOWKVNG/v7+2rBhg83j09LSNGHCBI0cOdJm+7fq1q2bTp48qYULF2a6LzU1VRs3blSVKlV0//33S7q+Iuinn36yniPnxrYWL16sQ4cOyd3dXa1bt9bevXv13//+1zrHYrHoww8/lMlkyvKQKen6IXlt2rTR7t27dfjwYZv73nzzTQ0bNkwXL17M9j1p06aN9u3bp++//95aqDRp0kQmk0nz5s1T6dKlrecy+idurGjK6hCxvM67nfbt2ysxMVGrVq2yGV+9erVeeOEF7du3L8vHdunSRUWKFNGMGTN06dKlTPevWbNGktShQwdJ16/45+XlpXLlylnnmM1mrV69WtLtV9DdyChJCxYssBnfuXOnRo4caS2mFi5cqMGDB9v8O73vvvtUtmzZHFduAQBQmLFiCQCAQsrNzU1hYWF6/vnn1atXL/Xu3VseHh5au3atTp8+rbfeekuurq5q1KiRihUrpunTp+v06dO69957FR0draioKHl4eNiUUjfOqfPRRx+pTZs26tChg7p27arPPvtMTz31lB577DFduHBBy5cvV5UqVaznt8nOK6+8okGDBunRRx/VY489phIlSmjz5s368ccfNWbMGJUsWTLLxw4dOlTR0dF65513tHv3bnXo0EGlSpXSmTNntGnTJp09e1ZLliyxrmYaOnSoPv/8cw0aNEgDBgxQmTJltHnzZh09elSLFy+WJL344ouKjo7WwIEDNXDgQPn6+mr79u369ttv9fjjj2d7aN7Nj+/fv7/69++vcuXK6csvv7SeTPzGSpustGnTRp988okkWVcslSpVSjVq1NCvv/6qXr16WV9PcnKytm/frkqVKqlRo0ZZPuft5t14Xz/99FNZLBb17NlTrq6ZfzS8cV6snTt3qly5curUqVO2+W/Wp08fbdiwQa+//rr++9//qn79+vr111+1Zs0a1a1bV7169crysVWqVNH48eM1bdo0Pfzww+revbvuv/9+paSkaO/evdq1a5cGDhyoxo0bW9+3nTt3aujQoercubMuX76sjRs3Wleo3fzv+GZt27ZVhw4dtGTJEv35559q0aKFTp06pZUrV6pcuXJ68sknJV0/XDMyMlL9+/dXSEiIihcvrm+//VbR0dEaOXJkrt8TAAAKG4olAAAKsYceekhLlizRggULNH/+fBUpUkQ1atTQggUL1K5dO0nXVxwtWrRIb731lubPny93d3dVrVpVb7/9tg4ePKhly5bp/Pnz8vHxUZcuXbRt2zZFRERo//796tChg9q1a6dXX31Vy5Yt0xtvvKGqVatq0qRJ+u6773K86pckNWrUSKtWrdJ7772npUuXKj09XVWrVtWbb76pnj17ZvtYT09PLVu2TKtWrdKWLVv00Ucf6cqVKypVqpRatGihoUOH2pyI28fHR+Hh4Zo9e7ZWr14ts9ms2rVra8mSJdYTTFeqVEnh4eF65513tHr1aqWkpKhatWp644031Lt37xxfz43Hz507V+Hh4UpOTlbFihU1fvz4XF09rGnTpvLy8pKHh4f1UD7pesn066+/2lwNLiEhQWPHjlXPnj2zLZZuN69atWoaOHCgIiIi9NNPP6lZs2aqVKlSpsd6eXnphRde0OLFizV16tTbzsmKu7u7Pv74Y73//vvaunWrPv30U5UpU0aPPfaYhg0blu1hgZIUGhqqOnXqaOXKlYqKilJCQoI8PT1Vu3Ztvf322zZXpevXr58uXbqktWvXaurUqfLx8VHDhg01b9489evXT99++60GDx6caRsmk0nvvvuuPvroI23cuFG7du1SqVKl1KlTJ40aNcp6kvRatWpp6dKlev/997VkyRIlJSWpSpUqCgsLU//+/XP9ngAAUNiYLHfyUi8AAAAAAAC4a3BAOAAAAAAAAAyhWAIAAAAAAIAhFEsAAAAAAAAwhGIJAAAAAAAAhlAsAQAAAAAAwBBXRwe4065du6aMDC50BwAAAAAAcKe4ubncdrzQFUsZGRYlJiY7OgYAAAAAAECh4evrfdtxDoUDAAAAAACAIRRLAAAAAAAAMIRiCQAAAAAAAIZQLAEAAAAAAMAQiiUAAAAAAAAYQrEEAAAAAAAAQyiWAAAAAAAAYAjFEgAAAAAAAAyhWAIAAAAAAIAhFEsAAAAAAAAwhGIJAAAAAAAAhlAsAQAAAAAAwBCKJQAAAAAAABhCsQQAAAAAAABDKJYAAAAAAABgCMUSAAAAAAAADKFYAgAAAAAAgCEUSwAAAAAAADCEYgkAAAAAAACGuDo6AAAAAAAAwM2KlfCSlxuVRX66mpaupMSr//h52EsAAAAAAMCpeLm5qsG6rY6OUaj92PshJd2B5+FQOAAAAAAAABhCsQQAAAAAAABDKJYAAAAAAABgCMUSAAAAAAAADKFYAgAAAAAAgCEUSwAAAAAAADCEYgkAAAAAAACGUCwBAAAAAADAEIolAAAAAAAAGEKxBAAAAAAAAEMolgAAAAAAAGAIxRIAAAAAAAAMoVgCAAAAAACAIRRLAAAAAAAAMIRiCQAAAAAAAIZQLAEAAAAAAMAQiiUAAAAAAAAYQrEEAAAAAAAAQyiWAAAAAAAAYAjFEgAAAAAAAAxxSLFkNpvVtWtXffPNN1nOOXr0qEJDQ9WgQQM99NBD2rp1qx0TAgAAAAAAICd2L5ZSU1M1evRo/fbbb1nOuXLlih5//HHdd999ioyMVP/+/TVmzBjFxcXZMSkAAAAAAACy42rPjcXFxWnMmDGyWCzZztu4caNcXV31xhtvyM3NTVWqVNHevXsVGxur6tWr2yktAAAAAAAAsmPXYikmJkYtW7bUiBEj1LBhwyznRUdHq3379nJzc7OOffDBB3ZICAAAAAAAgNyya7HUr1+/XM07ceKE/Pz8NGnSJH3xxRfy9fXVyJEj1a5du3xOCAAAAAAAgNyya7GUW1euXNHixYv173//W4sWLdLXX3+tYcOGKTw8XP7+/tk+1sXFpBIl7rFTUgAAAACAszKZ0uXq6uHoGIVaenqqLBanrBaQC3eiP3HKve/i4qKaNWtq9OjRkqQ6derowIEDuSqWMjIsSkxMtkdMAAAAAIAT8/X1Vvjapo6OUaj17bNf8fGX7/jz+vp63/HnRGZ56U+y2id2vypcbpQpU0b333+/zVjVqlV1+vRpByUCAAAAAADArZyyWGrUqJF+/vlnm7G4uDiVL1/eQYkAAAAAAABwK6cpluLj45WSkiJJCgkJ0e+//65Zs2bpxIkT+vjjj7Vv3z6FhIQ4OCUAAAAAAABucJpiqVWrVoqKipIklStXTkuXLlV0dLS6dOmi8PBwzZ07V3Xq1HFwSgAAAAAAANzgsJN3HzlyJNvbDRs21Lp16+wZCQAAAAAAAHngNCuWAAAAAAAAULBQLAEAAAAAAMAQiiUAAAAAAAAYQrEEAAAAAAAAQyiWAAAAAAAAYAjFEgAAAAAAAAyhWAIAAAAAAIAhFEsAAAAAAAAwhGIJAAAAAAAAhlAsAQAAAAAAwBCKJQAAAAAAABhCsQQAAAAAAABDKJYAAAAAAABgCMUSAAAAAAAADKFYAgAAAAAAgCEUSwAAAAAAADCEYgkAAAAAAACGUCwBAAAAAADAEIolAAAAAAAAGEKxBAAAAAAAAEMolgAAAAAAAGAIxRIAAAAAAAAMoVgCAAAAAACAIRRLAAAAAAAAMIRiCQAAAAAAAIZQLAEAAAAAAMAQiiUAAAAAAAAYQrEEAAAAAAAAQyiWAAAAAAAAYAjFEgAAAAAAAAyhWAIAAAAAAIAhFEsAAAAAAAAwhGIJAAAAAAAAhlAsAQAAAAAAwBCKJQAAAAAAABhCsQQAAAAAAABDHFIsmc1mde3aVd98802OcxMTE9WiRQtFRETYIRkAAAAAAAByy+7FUmpqqkaPHq3ffvstV/OnTZumCxcu5HMqAAAAAAAA5JVdi6W4uDj17dtXJ06cyNX83bt36+DBgypVqlQ+JwMAAAAAAEBe2bVYiomJUcuWLbVmzZoc5yYlJWnSpEl6/fXX5ebmZod0AAAAAAAAyAtXe26sX79+uZ47a9YstW7dWoGBgfmYCAAAAAAAAEbZtVjKrf3792vXrl3avHlznh/r4mJSiRL35EMqAAAAAABwKz6DF1x3Yt85XbGUkpKiV155RWFhYfL29s7z4zMyLEpMTM6HZAAAAACAgsTXN++fKZF3+fEZnH1nH3nZd1ntE6crlg4ePKjjx49r7Nix1rGrV6/qtdde0w8//KApU6Y4MB0AAAAAAABucLpiqX79+tq2bZvNWP/+/TVo0CD16tXLQakAAAAAAABwK6cpluLj4+Xt7S1PT09VrlzZ5r4iRYqodOnSKl26tIPSAQAAAAAA4FZFHB3ghlatWikqKsrRMQAAAAAAAJBLDluxdOTIkWxv32zPnj35HQcAAAAAAAB55DQrlgAAAAAAAFCwUCwBAAAAAADAEIolAAAAAAAAGEKxBAAAAAAAAEMolgAAAAAAAGAIxRIAAAAAAAAMoVgCAAAAAACAIRRLAAAAAAAAMIRiCQAAAAAAAIZQLAEAAAAAAMAQiiUAAAAAAAAYQrEEAAAAAAAAQ1wdHQAAAAAAnFXx4h5yd3d3dIxCzWw26++/Ux0dA4BBFEsAAAAAkAV3d3dNmjTJ0TEKtevvL8USUFBxKBwAAAAAAAAMoVgCAAAAAACAIRRLAAAAAAAAMIRiCQAAAAAAAIZQLAEAAAAAAMAQiiUAAAAAAAAYQrEEAAAAAAAAQyiWAAAAAAAAYAjFEgAAAAAAAAyhWAIAAAAAAIAhFEsAAAAAAAAwhGIJAAAAAAAAhlAsAQAAAAAAwBCKJQAAAAAAABhCsQQAAAAAAABDKJYAAAAAAABgCMUSAAAAAAAADKFYAgAAAAAAgCEUSwAAAAAAADCEYgkAAAAAAACGUCwBAAAAAADAEIolAAAAAAAAGOKQYslsNqtr16765ptvspwTFRWlrl27qmHDhurevbt27txpx4QAAAAAAADIid2LpdTUVI0ePVq//fZblnNiYmI0duxYhYaGKjIyUr1799aIESP0888/2zEpAAAAAAAAsmPXYikuLk59+/bViRMnsp23YcMGderUSX379lXlypUVGhqqZs2aKSoqyk5JAQAAAAAAkBNXe24sJiZGLVu21IgRI9SwYcMs5w0cOFCurrbRTCaTUlNT8zkhAAAAAAAAcsuuxVK/fv1yNa927do2t3/77Tft27dPISEh+RELAAAAAAAABjj9VeEuXLig4cOHKyAgQEFBQY6OAwAAAAAAgP+x64qlvDp79qyeeOIJFSlSRHPnzlWRIjn3YC4uJpUocY8d0gEAAAAA7gQ+wxVs7L+C607sO6ctlk6ePKlBgwbJy8tLy5YtU8mSJXP1uIwMixITk/M5HQAAAIC7ga+vt6Mj3BXy6zMc+88+8mP/se/sIy/7Lqt94pSHwiUmJurxxx+Xt7e3li9fLh8fH0dHAgAAAAAAwC2cZsVSfHy8vL295enpqTlz5ujixYt67733lJGRofj4eEmSp6envL1pLQEAAAAAAJyB06xYatWqlaKioiRJn3/+uZKSktSjRw+1atXK+mfy5MkOTgkAAAAAAIAbHLZi6ciRI1nejo6OtnccAAAAAAAA5JHTrFgCAAAAAABAwUKxBAAAAAAAAEMolgAAAAAAAGAIxRIAAAAAAAAMoVgCAAAAAACAIRRLAAAAAAAAMIRiCQAAAAAAAIZQLAEAAAAAAMAQiiUAAAAAAAAYQrEEAAAAAAAAQyiWAAAAAAAAYAjFEgAAAAAAAAyhWAIAAAAAAIAhFEsAAAAAAAAwhGIJAAAAAAAAhlAsAQAAAAAAwBCKJQAAAAAAABhCsQQAAAAAAABDKJYAAAAAAABgCMUSAAAAAAAADKFYAgAAAAAAgCEUSwAAAAAAADCEYgkAAAAAAACGUCwBAAAAAADAEIolAAAAAAAAGEKxBAAAAAAAAEMolgAAAAAAAGAIxRIAAAAAAAAMoVgCAAAAAACAIRRLAAAAAAAAMIRiCQAAAAAAAIZQLAEAAAAAAMAQiiUAAAAAAAAYQrEEAAAAAAAAQyiWAAAAAAAAYAjFEgAAAAAAAAyhWAIAAAAAAIAhDimWzGazunbtqm+++SbLOYcPH1ZISIgaNGigXr166eDBg3ZMCAAAAAAAgJzYvVhKTU3V6NGj9dtvv2U5Jzk5WUOGDFGDBg0UERGhgIAADR06VElJSXZMCgAAAAAAgOzYtViKi4tT3759deLEiWznRUVFyc3NTePGjVO1atU0YcIEeXt7a8uWLXZKCgAAAAAAgJzYtViKiYlRy5YttWbNmmzn/fjjj2rcuLGKFLkez2QyqXHjxoqNjbVHTAAAAAAAAOSCqz031q9fv1zNi4+PV9WqVW3GSpcurcOHD+dHLAAAAAAAABhg12Ipt65evSp3d3ebMXd3d5nN5hwf6+JiUokS9+RXNAAAAADAHcZnuIKN/Vdw3Yl955TFkoeHR6YSyWw2y9PTM8fHZmRYlJiYnF/RAAAAANxFfH29HR3hrpBfn+HYf/aRH/uPfWcfedl3We0Tu18VLjfKli2r+Ph4m7Hz58/L19fXQYkAAAAAAABwK6cslho0aKDY2FhZLBZJksViUWxsrBo2bOjYYAAAAAAAALDKdbFkNps1f/58HT9+XJI0ZcoUNWrUSIMHD9aFCxf+cZD4+HilpKRIkjp37qzk5GS9/vrriouL0/Tp05WUlKTg4OB/vB0AAAAAAADcGbkulmbOnKmVK1cqOTlZX375pcLDwzVs2DClpaVp+vTp/zhIq1atFBUVJUkqVqyYPvjgA8XGxqpnz576/vvvtWjRIhUrVuwfbwcAAAAAAAB3Rq5P3v3555/r7bfflp+fn1asWKFmzZppyJAhatmypQYPHpznDR85ciTb2/Xr19eGDRvy/LwAAAAAAACwj1yvWLp8+bIqV64sSdq7d69atWol6frqoluv4AYAAAAAAIDCL9crlu6//3599dVXKlOmjM6ePas2bdpIktatW6fq1avnW0AAAAAAAAA4p1wXS6NGjdKIESOUnp6u4OBgVatWTW+++aZWrlyp+fPn52dGAAAAAAAAOKFcF0sPPvig9uzZo7Nnz8rPz0+S1LVrVz322GPWQ+QAAAAAAABw98j1OZYkqWTJkjKbzYqIiFBSUpI8PT1Vvnz5/MoGAAAAAAAAJ5brFUuXL1/W8OHDFR0dLZPJpMDAQL311ls6fvy4li5dqvvuuy8/cwIAAAAAAMDJ5HrF0qxZs5SRkaHdu3fL09NTkjRx4kR5e3trxowZ+RYQAAAAAAAAzinXxdLu3bs1duxYlS1b1jpWsWJFhYWFad++ffkSDgAAAAAAAM4r18XS33//reLFi2ca9/DwUGpq6h0NBQAAAAAAAOeX62IpICBAa9eutRnLyMjQokWL1LBhwzudCwAAAAAAAE4u1yfvfvnllzVw4EBFR0crLS1NU6dO1dGjR3Xp0iUtXbo0PzMCAAAAAADACeW6WKpZs6Y+/fRTrVq1SqVKlZKbm5u6du2qAQMGyMfHJz8zAgAAAAAAwAnlulgaNGiQXnnlFT3//PP5GAcAAAAAAAAFRa7PsXT48GF5enrmZxYAAAAAAAAUILlesTR48GC9+uqreuqpp1ShQgV5eHjY3F+2bNk7Hg4AAAAAAADOK9fF0oIFC2Q2m7Vv3z6ZTCbruMVikclk0i+//JIvAQEAAAAAAOCccl0sffTRR/mZAwAAAAAAAAVMroulpk2bSpKSkpJ07Ngxubm5qWLFiipWrFi+hQMAAAAAAIDzynWxlJGRoenTp2v16tXKyMiQxWKRu7u7+vbtqwkTJqhIkVyfBxwAAAAAAACFQJ7OsbRp0yZNnDhRgYGBysjIUExMjN577z35+PjomWeeyc+cAAAAQIFUsriXXN1z/WM3DEo3p+vi31cdHQMA7jq5/g63fv16TZo0SQ8//LB1rFatWipVqpRmz55NsQQAAADchqu7q355Y6ejYxR6fhPbOzoCANyVcn382sWLF1WnTp1M43Xq1NG5c+fuaCgAAAAAAAA4v1wXS9WqVdOOHTsyjW/fvl1VqlS5k5kAAAAAAABQAOT6ULjnnntOI0eO1C+//KJGjRpJkg4cOKDPP/9cM2bMyLeAAAAAAAAAcE65LpY6dOigOXPm6MMPP9T27dvl4eGh6tWr64MPPlCrVq3yMyMAAAAAAACcUJ4uT9GhQwc1btxYPj4+kqTvv/9e9erVy5dgAAAAAAAAcG65PsfS77//rk6dOmnx4sXWseHDh6tbt246efJkvoQDAAAAAACA88p1sTR16lTVrVtXQ4cOtY5t27ZNNWrU0LRp0/IlHAAAAAAAAJxXroul2NhYjR49WiVKlLCOFStWTM8//7xiYmLyIxsAAAAAAACcWK6LJS8vL/3111+Zxi9evKgiRXL9NAAAAAAAACgkct0IderUSZMmTVJMTIxSU1OVmpqqmJgYTZ48WUFBQfmZEQAAAAAAAE4o11eFe/HFFzVq1CgNGDBAJpNJkmSxWNShQweNHz8+3wICAAAAAADAOeW6WCpatKg++ugj/f777zp8+LCOHj2qBx54QE2aNMnPfAAAAAAAAHBSOR4Kt3HjRvXq1UunT5+WJGVkZGjWrFl6//33FRoaqokTJyojIyPfgwIAAAAAAMC5ZFssRUVFafz48apZs6a8vLwkSWPHjlVSUpI++ugjrV69Wj/++KM++eQTu4QFAAAAAACA88i2WFq+fLmef/55vfnmmypZsqQOHz6sn3/+WQMGDFDLli1Vv359jRo1ShEREfbKCwAAAAAAACeRbbF05MgRmyu+ffPNNzKZTGrXrp11rFatWjpx4kSuNmY2mxUWFqbAwEC1bNlSH374YZZzY2Ji1KtXLzVs2FCPPPKIvv7661xtAwAAAAAAAPaRbbFksVjk7u5uvf3dd9/J29tb/v7+1rGUlBR5eHjkamMzZ85UbGysli5dqsmTJ2vBggXavHlzpnkXLlzQM888o86dO+vTTz/Vww8/rGHDhunUqVO5fV0AAAAAAADIZ9kWS9WrV9eBAwckSUlJSfr222/VsmVLmUwm65xt27apRo0aOW4oOTlZ4eHhmjBhgvz9/RUUFKQhQ4ZoxYoVmeZ+//33kqSnn35alSpV0jPPPCNPT0/9+OOPeXpxAAAAAAAAyD/ZFkv9+/fX1KlTNWPGDA0ZMkQpKSkaNGiQpOurij7++GMtWrRIffv2zXFDhw8fltlsVkBAgHUsICBAP/30k9LT023mlihRQpcvX9aWLVtksVj0xRdf6MqVK6pVq5aR1wgAAAAAAIB84JrdnT169FBqaqrWrFkjFxcXzZkzRw0bNpQkzZs3T2vXrtWQIUPUo0ePHDcUHx+v4sWL2xw25+Pjo7S0NCUkJKhMmTLW8SZNmmjAgAF64YUXNGbMGGVkZGjq1KmqVq2asVcJAAAAAACAOy7bYkmSQkJCFBISkml86NChGjlypEqWLJmrDV29etXmfE2SrLfNZrPNeHJysv788089++yz6tixo/bu3atp06apRo0a1mILAAAAAAAAjpVjsZSV++67L0/zPTw8MhVIN257eXnZjC9evFhms1mjRo2SJNWpU0dxcXFasGCBPvjgg2y34+JiUokS9+QpGwAAAICCj88BBRf7rmBj/xVcd2LfGS6W8qps2bK6dOmSzGazdaVSfHy83N3dVbx4cZu5P/30U6YTgtetW1erV6/OcTsZGRYlJibfueAAAADAP+Dr6+3oCHeN/PgcwP6zj/z6DMf+sw++9gquvOy7rPZJtifvvpP8/Pzk5uam2NhY69iBAwdUt25dubra9ltlypTRkSNHbMaOHj2qSpUq2SUrAAAAAAAAcma3YsnLy0s9evTQ5MmTdfDgQe3YsUNLlixRaGiopOurl1JSUiRdP6/Td999pw8//FAnT57U2rVrFRERYb0iHQAAAAAAABzPbsWSJI0fP1716tXToEGD9Nprr2nYsGEKDg6WJLVq1UpRUVGSpPr162vBggXasmWLunfvrmXLlumtt95S8+bN7RkXAAAAAAAA2bDbOZak66uWZsyYoRkzZmS679ZD39q2bau2bdvaKxoAAAAAAADyyK4rlgAAAAAAAFB4UCwBAAAAAADAEIolAAAAAAAAGEKxBAAAAAAAAEMolgAAAAAAAGAIxRIAAAAAAAAMoVgCAAAAAACAIRRLAAAAAAAAMIRiCQAAAAAAAIZQLAEAAAAAAMAQiiUAAAAAAAAYQrEEAAAAAAAAQyiWAAAAAAAAYAjFEgAAAAAAAAyhWAIAAAAAAIAhFEsAAAAAAAAwhGIJAAAAAAAAhlAsAQAAAAAAwBCKJQAAAAAAABhCsQQAAAAAAABDKJYAAAAAAABgCMUSAAAAAAAADKFYAgAAAAAAgCEUSwAAAAAAADCEYgkAAAAAAACGUCwBAAAAAADAEIolAAAAAAAAGOLq6AAAAADIWXFvd7l7ejg6RqFmTknV35fNjo4BAECBQrEEAABQALh7euiNAb0dHaNQm7hinUSxBABAnnAoHAAAAAAAAAyhWAIAAAAAAIAhFEsAAAAAAAAwhGIJAAAAAAAAhlAsAQAAAAAAwBCKJQAAAAAAABhCsQQAAAAAAABD7Fosmc1mhYWFKTAwUC1bttSHH36Y5dyjR48qNDRUDRo00EMPPaStW7faMSkAAAAAAAByYtdiaebMmYqNjdXSpUs1efJkLViwQJs3b84078qVK3r88cd13333KTIyUv3799eYMWMUFxdnz7gAAAAAAADIhqu9NpScnKzw8HAtXLhQ/v7+8vf315AhQ7RixQp16dLFZu7GjRvl6uqqN954Q25ubqpSpYr27t2r2NhYVa9e3V6RAQAAAAAAkA27FUuHDx+W2WxWQECAdSwgIEDz589Xenq6XF3/P0p0dLTat28vNzc369gHH3xgr6gAAAAAAADIBbsdChcfH6/ixYvLw8PDOubj46O0tDQlJCTYzD1x4oRKly6tSZMmqVWrVurZs6d27dplr6gAAAAAAADIBbsVS1evXpW7u7vN2I3bZrPZZvzKlStavHix7r33Xi1atEgPP/ywhg0bpkOHDtkrLgAAAAAAAHJgt0PhPDw8MhVIN257eXnZjLu4uKhmzZoaPXq0JKlOnTo6cOCAwsPD5e/vn+12XFxMKlHinjuYHAAAAHcLfo4s2Nh/BRf7rmBj/xVcd2Lf2a1YKlu2rC5duiSz2WxdqRQfHy93d3cVL17cZm6ZMmVUqVIlm7GqVavm6qpwGRkWJSYm37ngAAAATsDX19vREe4K+fFzJPvOfth/BVd+fYZj/9kHX3sFV172XVb7xG6Hwvn5+cnNzU2xsbHWsQMHDqhu3bo2J+6WpEaNGunnn3+2GYuLi1P58uXtkhUAAAAAAAA5s1ux5OXlpR49emjy5Mk6ePCgduzYoSVLlig0NFTS9dVLKSkpkqSQkBD9/vvvmjVrlk6cOKGPP/5Y+/btU0hIiL3iAgAAAAAAIAd2K5Ykafz48apXr54GDRqk1157TcOGDVNwcLAkqVWrVoqKipIklStXTkuXLlV0dLS6dOmi8PBwzZ07V3Xq1LFnXAAAAAAAAGTDbudYkq6vWpoxY4ZmzJiR6b4jR47Y3G7YsKHWrVtnr2gAAAAAAADII7uuWAIAAAAAAEDhQbEEAAAAAAAAQyiWAAAAAAAAYAjFEgAAAAAAAAyhWAIAAAAAAIAhFEsAAAAAAAAwhGIJAAAAAAAAhlAsAQAAAAAAwBCKJQAAAAAAABhCsQQAAAAAAABDKJYAAAAAAABgCMUSAAAAAAAADKFYAgAAAAAAgCEUSwAAAAAAADCEYgkAAAAAAACGuDo6AAAAsI/i93rJ3YNv/fnJnJquvy9ddXQMAAAAu+GnSwAA7hLuHq6aN2aTo2MUasNnd3N0BAAAALviUDgAAAAAAAAYQrEEAAAAAAAAQyiWAAAAAAAAYAjFEgAAAAAAAAyhWAIAAAAAAIAhFEsAAAAAAAAwhGIJAAAAAAAAhlAsAQAAAAAAwBCKJQAAAAAAABhCsQQAAAAAAABDKJYAAAAAAABgCMUSAAAAAAAADKFYAgAAAAAAgCEUSwAAAAAAADCEYgkAAAAAAACGUCwBAAAAAADAEIolAAAAAAAAGEKxBAAAAAAAAEMolgAAAAAAAGAIxRIAAAAAAAAMsWuxZDabFRYWpsDAQLVs2VIffvhhjo9JTExUixYtFBERYYeEAAAAAAAAyC1Xe25s5syZio2N1dKlS3X27FmNHTtW5cqVU5cuXbJ8zLRp03ThwgU7pgQAAAAAAEBu2K1YSk5OVnh4uBYuXCh/f3/5+/tryJAhWrFiRZbF0u7du3Xw4EGVKlXKXjEBADkoWcxNrl6ejo5RqKVfTdHFpDRHxwAAAAByZLdi6fDhwzKbzQoICLCOBQQEaP78+UpPT5erq22UpKQkTZo0STNnztSYMWPsFRMAkANXL0/tbtPW0TEKtbZ7dksUSwAAACgA7HaOpfj4eBUvXlweHh7WMR8fH6WlpSkhISHT/FmzZql169YKDAy0V0QAAAAAAADkgd1WLF29elXu7u42Yzdum81mm/H9+/dr165d2rx5c5634+JiUokS9xgPCgCAE+B7WcHFvivY2H8FG/uv4GLfFWzsv4LrTuw7uxVLHh4emQqkG7e9vLysYykpKXrllVcUFhYmb2/vPG8nI8OixMTkfxYWAJAlX9+8/9+MvMuP72XsO/vIr59D2H/2wddewcb+K7j4v7Ng42uv4MrLvstqn9itWCpbtqwuXboks9lsXakUHx8vd3d3FS9e3Drv4MGDOn78uMaOHWsdu3r1ql577TX98MMPmjJlir0iAwAAAAAAIBt2K5b8/Pzk5uam2NhYNWvWTJJ04MAB1a1b1+bE3fXr19e2bdtsHtu/f38NGjRIvXr1sldcAAAAAAAA5MBuxZKXl5d69OihyZMn680331R8fLyWLFmi119/XdL11Uve3t7y9PRU5cqVbR5bpEgRlS5dWqVLl7ZXXAAAAAAAAOTAbleFk6Tx48erXr16GjRokF577TUNGzZMwcHBkqRWrVopKirKnnEAAAAAAADwD9htxZJ0fdXSjBkzNGPGjEz3HTlyJMvH7dmzJz9jAQAAAAAAwAC7rlgCAAAAAABA4UGxBAAAAAAAAEMolgAAAAAAAGAIxRIAAAAAAAAMoVgCAAAAAACAIRRLAAAAAAAAMIRiCQAAAAAAAIZQLAEAAAAAAMAQiiUAAAAAAAAYQrEEAAAAAAAAQyiWAAAAAAAAYAjFEgAAAAAAAAyhWAIAAAAAAIAhFEsAAAAAAAAwhGIJAAAAAAAAhlAsAQAAAAAAwBCKJQAAAAAAABji6ugAAO4+xYq7ycvd09ExCrWr5hQl/Z3m6BgAAAAACjmKJQB25+XuqZbvtXR0jEJt74i9ShLFEgAAAID8xaFwAAAAAAAAMIRiCQAAAAAAAIZQLAEAAAAAAMAQiiUAAAAAAAAYQrEEAAAAAAAAQyiWAAAAAAAAYAjFEgAAAAAAAAyhWAIAAAAAAIAhFEsAAAAAAAAwhGIJAAAAAAAAhlAsAQAAAAAAwBCKJQAAAAAAABhCsQQAAAAAAABDKJYAAAAAAABgCMUSAAAAAAAADKFYAgAAAAAAgCEUSwAAAAAAADDErsWS2WxWWFiYAgMD1bJlS3344YdZzo2KilLXrl3VsGFDde/eXTt37rRjUgAAAAAAAOTErsXSzJkzFRsbq6VLl2ry5MlasGCBNm/enGleTEyMxo4dq9DQUEVGRqp3794aMWKEfv75Z3vGBQAAAAAAQDbsViwlJycrPDxcEyZMkL+/v4KCgjRkyBCtWLEi09wNGzaoU6dO6tu3rypXrqzQ0FA1a9ZMUVFR9ooLAAAAAACAHLjaa0OHDx+W2WxWQECAdSwgIEDz589Xenq6XF3/P8rAgQNtbkuSyWRSamqqveICAAAAAAAgB3ZbsRQfH6/ixYvLw8PDOubj46O0tDQlJCTYzK1du7aqV69uvf3bb79p3759CgwMtFdcAAAAAAAA5MBuxdLVq1fl7u5uM3bjttlszvJxFy5c0PDhwxUQEKCgoKB8zQgAAAAAAIDcs9uhcB4eHpkKpBu3vby8bvuYs2fP6oknnlCRIkU0d+5cFSmScw/m4mJSiRL3/PPAAFDA8X9hwcb+K7jYdwUb+69gY/8VXOy7go39V3DdiX1nt2KpbNmyunTpksxms3WlUnx8vNzd3VW8ePFM80+ePKlBgwbJy8tLy5YtU8mSJXO1nYwMixITk+9odgB3lq+vt6Mj3BXy6/9C9p995Mf+Y9/ZB197BRtfewUb+6/g4v/Ogo2vvYIrL/suq31it0Ph/Pz85ObmptjYWOvYgQMHVLdu3Uwn6k5MTNTjjz8ub29vLV++XD4+PvaKCQAAAAAAgFyyW7Hk5eWlHj16aPLkyTp48KB27NihJUuWKDQ0VNL11UspKSmSpDlz5ujixYt68803lZGRofj4eMXHx+vy5cv2igsAAAAAAIAc2O1QOEkaP368Jk2apEGDBqlo0aIaNmyYgoODJUmtWrXS9OnT1atXL33++edKSkpSjx49bB7frVs3vfXWW/aMDAAAAAAAgCzYtVjy8vLSjBkzNGPGjEz3HTlyxPr36Ohoe8YCAAAAAACAAXY7FA4AAAAAAACFC8USAAAAAAAADKFYAgAAAAAAgCEUSwAAAAAAADCEYgkAAAAAAACGUCwBAAAAAADAEIolAAAAAAAAGEKxBAAAAAAAAENcHR0AMKJUcTe5uHs6Okahl2FOUcLfaY6OAQAAAABwUhRLKJBc3D11Yko9R8co9Cq9+pMkiiUAAAAAwO1xKBwAAAAAAAAMoVgCAAAAAACAIRRLAAAAAAAAMIRiCQAAAAAAAIZQLAEAAAAAAMAQiiUAAAAAAAAYQrEEAAAAAAAAQyiWAAAAAAAAYIirowM4UrF7PeXl4eboGIXa1dQ0JV1KcXQMAAAAAACQD+7qYsnLw00BLy1zdIxC7cCsUCWJYgkAAAAAgMKIQ+EAAAAAAABgCMUSAAAAAAAADKFYAgAAAAAAgCEUSwAAAAAAADCEYgkAAAAAAACGUCwBAAAAAADAEIolAAAAAAAAGEKxBAAAAAAAAEMolgAAAAAAAGAIxRIAAAAAAAAMoVgCAAAAAACAIRRLAAAAAAAAMIRiCQAAAAAAAIZQLAEAAAAAAMAQiiUAAAAAAAAYQrEEAAAAAAAAQyiWAAAAAAAAYIhdiyWz2aywsDAFBgaqZcuW+vDDD7Oce/jwYYWEhKhBgwbq1auXDh48aMekAAAAAAAAyIldi6WZM2cqNjZWS5cu1eTJk7VgwQJt3rw507zk5GQNGTJEDRo0UEREhAICAjR06FAlJSXZMy4AAAAAAACyYbdiKTk5WeHh4ZowYYL8/f0VFBSkIUOGaMWKFZnmRkVFyc3NTePGjVO1atU0YcIEeXt7a8uWLfaKCwAAAAAAgBzYrVg6fPiwzGazAgICrGMBAQH66aeflJ6ebjP3xx9/VOPGjVWkyPV4JpNJjRs3VmxsrL3iAgAAAAAAIAd2K5bi4+NVvHhxeXh4WMd8fHyUlpamhISETHPLlCljM1a6dGmdO3fOLlkBAAAAAACQM5PFYrHYY0MbN27U7Nmz9dVXX1nHTp48qaCgIO3YsUMVKlSwjg8aNEgNGzbUCy+8YB179913FRMTo+XLl9sjLgAAAAAAAHJgtxVLHh4eMpvNNmM3bnt5eeVqrqenZ/6GBAAAAAAAQK7ZrVgqW7asLl26ZFMYxcfHy93dXcWLF880Nz4+3mbs/Pnz8vX1tUtWAAAAAAAA5MxuxZKfn5/c3NxsTsB94MAB1a1bV66urjZzGzRooNjYWN04Ss9isSg2NlYNGza0V1wAAAAAAADkwG7FkpeXl3r06KHJkyfr4MGD2rFjh5YsWaLQ0FBJ11cvpaSkSJI6d+6s5ORkvf7664qLi9P06dOVlJSk4OBge8UFAAAAAABADux28m5Junr1qiZNmqRt27apaNGieuKJJ/TEE09IkmrVqqXp06erV69ekqSDBw/qtddeU1xcnGrVqqVJkybJ39/fXlEBAAAAAACQA7sWSwAAAAAAACg87HYoHAAAAAAAAAoX15ynAMiN+Ph4bdy4UadPn1bFihXVtWtXlSlTxtGxgELPz89Pa9asUf369R0dBQAAALjrUCw5kW7duuV6rslk0qeffpqPaZAXP//8swYNGqTLly9bx+bNm6d33nlHbdq0cWAyoPDjiO7C5eeff9bp06clSffdd5/q1q0rk8nk4FTIypdffqmVK1fq9OnTqlChgkJCQtS+fXtHx0IuhIaG6rXXXlO1atUcHQV5xL4DHC8+Pl5nzpxRhQoVVKpUKUfHcTiKJSdy6w/PFotFGzdu1IMPPqiSJUs6MBly8u6776pUqVL64IMPVLduXf3+++8KCwvT1KlTtW3bNkfHQy789ddfuV5htmnTpjwVwQBytnLlSi1atEh//fWXpOvfA00mk3x8fPTUU09ZryIL5xEVFaXRo0fr3nvvVZUqVXTo0CHt2bNHY8aM0ZAhQxwdDznYv3+/rly54ugYMIB9VzisXLkyUzHfv39/FSnC2Wqc2dWrVzVu3Dht377d+svNoKAgTZ48+a4umDh5txNLT0+Xv7+/1q9fr7p16zo6DrLRrFkzTZkyRQ899JB17ODBgwoJCdHu3bs5JK4AeOCBBzR16lQFBQVlOScxMVGvvvqqtm/frl9++cWO6ZCd2rVr6/3335efn1+u5pcrVy6fEyGvJk6cqPXr1yswMFAPPfSQKlSoIBcXF506dUrbt2/X3r171bNnT02fPt3RUXGTvn37qnjx4po7d668vLyUnp6uV155Rbt27VJ0dLSj4yEHtWvXVnh4OIcRF0Dsu4Jv5cqVev3111W1alXVrl1bx48f1y+//KLBgwfr5ZdfdnQ8ZGP69Olavny5Hn30UdWpU0d//PGHwsPD1bx5c82fP9/R8RyGFUtOjKX/Bcfly5dVtmxZm7EaNWrIYrEoISGBYqkAaNu2rUaMGKE+ffpowoQJ8vT0tLl/165dCgsLU2Jiop599lkHpURWhg8fnuOcGytgKAWdy7Zt2xQREaE33nhDjz76aKb7+/Xrp02bNmncuHFq3769Onbs6ICUuJ24uDhrqSRJrq6uevbZZ7Vx40adPHlSFStWdHBCAHBO4eHh6t69u2bMmGH9zDd79mytWLFCL774olxcXBycEFnZsWOHRo0apaFDh1rH6tSpowkTJig5OVn33HOPA9M5DsUScAdcu3Yt07JVd3d3SVJGRoYjIiGPZsyYoTZt2mjSpEmKiYnRW2+9pTp16ujKlSuaNm2aIiIiVL16dS1cuFD+/v6OjotbjBkzRlWqVHF0DBjwn//8Rz169LhtqXRDt27d9N1332nVqlUUS07k6tWr8vb2thn717/+JUlKSkpyRCTk0bBhw6w/r2THZDLpiy++sEMi5Bb7rmA7fvy4xo0bZ7OQ4N///rc+/PBDnTx5kp9pnNi5c+cUEBBgM9a6dWtlZGTozJkzd+25zyiWAOB/unTpogYNGujll19WSEiIBgwYoK1bt+rs2bN68sknNXLkyFz9EAf7a9q0KYcEFFBHjhzR4MGDc5wXFBSksWPH5n8g5NqNVYA3u/Fb9mvXrjkiEvIoMDBQPj4+jo4BA9h3BVtKSoqKFi1qM+br6ytJSk5OdkQk5FJaWlqmzwPFixeXJKWmpjoiklOgWALukOjoaJ09e9Z6+9q1azKZTIqOjtapU6ds5nbq1Mne8ZBLFSpU0OLFi/XYY49p6dKlMplMmjNnjjp37uzoaPiHzpw5o/Xr1+fqsDnYz9WrV62HUmXHw8NDaWlpdkgE3D0GDx5MKV9Ase8KnxtFPadALrju5n1HseREbr16WHbFhEQ54Wxmz5592/GZM2fa3OYcL87thx9+0Pjx43XixAk99thj+uabbzR+/HhdvHhRjz32mKPjIY/S09O1Y8cOrV27Vvv27dO1a9colpxMpUqVdODAATVr1izbeTExMapcubKdUiG3lixZYrNq4sYP1YsXL850dZxXXnnFrtkAoCDiPLsF19287yiWnMjIkSNvO35rMSFRTjibHTt25Gre6dOnFRERkc9pYERaWpreffddLV26VOXLl9eyZcsUEBCg5ORkTZ06VZMnT9aOHTs0bdo0TsbuZHbs2GFdPn7DsWPHtHbtWkVGRurixYsqXbq0+vfvr27dujkoJbISHByspUuXqlu3blme7PnYsWP65JNP9Nxzz9k5HbJTrlw5HTx48LbjP/zwg82YyWSiWCrA9u/fr6ZNmzo6BlCozJgxI9N56iRp2rRpKlasmPW2yWTSggUL7BkNORg0aNBtS6T+/fvbjJtMJh04cMCe0RzGZLmb12s5mdutSrqdG+UEl10uGG63YoJS0Pl07dpVR48eVb9+/TR27NhMh+Zs27ZNr776qiwWi1599VV16dLFQUmRlZSUFG3ZskVr165VbGysPD09lZKSorCwMPXr1y/TCfbhHFJSUtS3b1/Fx8frmWeeUevWrVWhQgW5uLjo9OnT+uKLL7Rw4UJVqlRJK1eu5DxnwB2yYcMGPfjggypZsuRt7z9//rwiIiK0fv16nThxgp9dnEhO+w7Ob+DAgXmav3z58nxKgryaN29enubfLSvlKZYKCMqJgud2KyaCg4PVrVs31atXz9HxcIt27dpp2rRpat68eZZz/vrrL40bN0779u3j68+JHDp0SGvXrtVnn32mq1evqnnz5nrkkUfUrFkztW3bVsuXL1dgYKCjYyIbCQkJGjdunPbs2ZPpN4AWi0WdOnXS5MmT+RAF5DOLxaLdu3dr7dq12r17t9LT01W7dm1169ZNTz75pKPjAQCcFMWSk6OcKFhYMVFwJSUl2Sw7zs7HH3+cq6tYwT5q166tGjVqqHfv3nr44YethypevnxZgYGBFEsFyOHDh7V3716dOnVKFotF5cuXV7t27e7aS/cWFNHR0dbvexcuXJAklS1bVgEBAerTp48aNWrk4ITIyZ9//ql169Zpw4YNOnfunEwmk3r37q1BgwapevXqjo6HW+TlsG6TyaRPP/00H9MgP6Wnp+vChQsqW7aso6PgNlJSUvTTTz/p/Pnzkq5/76tbt648PDwcnMz+OMeSE6KcKHhut2JixowZ1hUTNWrUYL85uVtLpZ9//lmnT5+WJN13332qW7eudSUFpZJzqVWrln799VdFRkYqISFB3bt3p4goYCwWizZt2qQSJUrYrIqwWCx64okn1L17d/Xs2dOBCZGVSZMmac2aNfL09FS9evWsV6k6d+6cNm/erA0bNmjgwIGaMGGCg5PiVmlpadq+fbvWrl2r6Oho3XPPPercubM6dOigZ599Vt27d6dUclL+/v6OjoB/KDg4WHPmzFGtWrWsY2vWrFGnTp1sVuf+97//Vb9+/Vgp72QuXLigWbNmafPmzUpPT7deuMJkMsnT01PdunXTCy+8cFettKZYciKUEwVX7969VaNGDY0cOTLTigkULCtXrtSiRYv0119/Sbr+wdZkMsnHx0dPPfWUQkNDHZwQt4qMjNSvv/6qDRs2KCIiQosWLZKfn586deokk8l0V1+hoyBIT0/X6NGjtX37dj3++ONq06aN9b7z58/rr7/+0oQJE/T1119r1qxZfB90ImvWrNGaNWv07LPPasiQIbrnnnts7k9KStLHH3+s+fPnq1GjRnr44YcdlBS307p1a6WkpKhly5Z666231KFDB3l4ePCzSwFQvnx59enTh1UsBdixY8eUmppqvZ2RkaFJkybJ39//riojCqKEhAT16dNHCQkJCg4OVvPmza1XQT137py+/fZbbdy4UdHR0QoPD1fx4sUdnNg+KJacCOVEwcWKicJh4sSJWr9+vQIDA/XUU09ZTyB86tQpbd++XdOmTdMvv/zCifOdUM2aNfXyyy/rpZde0tdff63IyEh98MEHslgsmjVrlh555BF16tTJ5rLocA5r1qzR7t27NXv2bAUHB9vc5+vrq82bNysyMlITJ07UAw88oD59+jgoKW61fv169enTJ8ur2hYrVkzDhw/XX3/9pVWrVlEsOZmkpCSVK1dOFStWVPHixeXm5uboSMil999/X23atKFYKmQ4Q03BMG/ePKWkpGjdunW3XdXZp08fHTt2TKGhofrkk0+y/B5Z2FAsORHKiYKLFRMF37Zt2xQREaE33nhDjz76aKb7+/Xrp02bNmncuHFq3769Onbs6ICUyEmRIkXUpk0btWnTRleuXNGWLVsUGRmp119/XW+88YYaN27MlVWczLp16/Tkk09mKpVu9sgjj+jQoUNavXo1xZITiYuL07Bhw3Kc165dO4WFhdkhEfLiq6++0meffaaNGzfq448/lo+Pj7p166b27ds7OhpyQAEBOM7u3bv17LPPZnuo8P33368nn3xSERERd02xxHpyJxIZGanIyEg1a9ZMERER6tq1q3r16qWVK1dSThQAN1ZM7N69Wx988IGqVq1qs2LiP//5j/XEbnA+//nPf9SjR4/blko3dOvWTY8++qhWrVplx2QwqmjRourdu7eWL1+unTt3atiwYXwNOqHjx4/n6uTqrVu31h9//JH/gZBrV69ezdUS/xIlSighIcEOiZAXJUuW1MCBA7V+/Xpt3rxZPXr00JYtWzRw4ECZTCZt3rxZR48edXRMAHAqf/31l2rWrJnjPD8/P505c8YOiZwDK5acDIdzFHysmCiYjhw5kquTcgcFBWns2LH5Hwh31L/+9S8999xzeu655xwdBbfw9PRUcnJyjvMsFguH6jgZi8WSq3NeFSlShBUWTmjevHnW8/RUq1ZNL774osaMGaN9+/YpMjJSn376qdasWaP7779fDz/8sIYPH+7oyLjJ/Pnzc3UuHpPJpGnTptkhEXB3SEtLk5eXV47zvLy8dOXKFTskcg4US06KcqJwuLFionfv3jpz5ow2bNigTZs2OToWbuPq1au5+ibh4eGhtLQ0OyQC7g5+fn7auXOnOnTo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\n",
      "text/plain": [
       "<Figure size 1440x576 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=(20,8))\n",
    "# make barplot and sort bars\n",
    "sns.barplot(x='Feature',\n",
    "            y=\"Score\", \n",
    "            data=score_df, \n",
    "            order=score_df.sort_values('Score').Feature)\n",
    "# set labels\n",
    "plt.xlabel(\"Features\", size=15)\n",
    "plt.ylabel(\"Scores\", size=15)\n",
    "plt.yticks(rotation = 0, fontsize = 14)\n",
    "plt.xticks(rotation = 90, fontsize = 16)\n",
    "plt.title(\"Feature Score w.r.t the Sales\", size=18)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 68,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Score</th>\n",
       "      <th>Feature</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>1.712704</td>\n",
       "      <td>CDP</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>1.599840</td>\n",
       "      <td>GTEP</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>1.325077</td>\n",
       "      <td>TIT</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>0.891890</td>\n",
       "      <td>TAT</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>0.660760</td>\n",
       "      <td>AFDP</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>0.512258</td>\n",
       "      <td>CO</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0.412339</td>\n",
       "      <td>AT</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>0.301021</td>\n",
       "      <td>NOX</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0.140308</td>\n",
       "      <td>AP</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>0.097477</td>\n",
       "      <td>AH</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "      Score Feature\n",
       "7  1.712704     CDP\n",
       "4  1.599840    GTEP\n",
       "5  1.325077     TIT\n",
       "6  0.891890     TAT\n",
       "3  0.660760    AFDP\n",
       "8  0.512258      CO\n",
       "0  0.412339      AT\n",
       "9  0.301021     NOX\n",
       "1  0.140308      AP\n",
       "2  0.097477      AH"
      ]
     },
     "execution_count": 68,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "score_df.sort_values('Score',ascending=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 71,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>CDP</th>\n",
       "      <th>GTEP</th>\n",
       "      <th>TIT</th>\n",
       "      <th>TAT</th>\n",
       "      <th>AFDP</th>\n",
       "      <th>CO</th>\n",
       "      <th>AT</th>\n",
       "      <th>TEY</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>10.605</td>\n",
       "      <td>19.663</td>\n",
       "      <td>1059.2</td>\n",
       "      <td>550.00</td>\n",
       "      <td>3.5000</td>\n",
       "      <td>3.1547</td>\n",
       "      <td>6.8594</td>\n",
       "      <td>114.70</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>10.598</td>\n",
       "      <td>19.728</td>\n",
       "      <td>1059.3</td>\n",
       "      <td>550.00</td>\n",
       "      <td>3.4998</td>\n",
       "      <td>3.2363</td>\n",
       "      <td>6.7850</td>\n",
       "      <td>114.72</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>10.601</td>\n",
       "      <td>19.779</td>\n",
       "      <td>1059.4</td>\n",
       "      <td>549.87</td>\n",
       "      <td>3.4824</td>\n",
       "      <td>3.2012</td>\n",
       "      <td>6.8977</td>\n",
       "      <td>114.71</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>10.606</td>\n",
       "      <td>19.792</td>\n",
       "      <td>1059.6</td>\n",
       "      <td>549.99</td>\n",
       "      <td>3.4805</td>\n",
       "      <td>3.1923</td>\n",
       "      <td>7.0569</td>\n",
       "      <td>114.72</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>10.612</td>\n",
       "      <td>19.765</td>\n",
       "      <td>1059.7</td>\n",
       "      <td>549.98</td>\n",
       "      <td>3.4976</td>\n",
       "      <td>3.2484</td>\n",
       "      <td>7.3978</td>\n",
       "      <td>114.72</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "      CDP    GTEP     TIT     TAT    AFDP      CO      AT     TEY\n",
       "0  10.605  19.663  1059.2  550.00  3.5000  3.1547  6.8594  114.70\n",
       "1  10.598  19.728  1059.3  550.00  3.4998  3.2363  6.7850  114.72\n",
       "2  10.601  19.779  1059.4  549.87  3.4824  3.2012  6.8977  114.71\n",
       "3  10.606  19.792  1059.6  549.99  3.4805  3.1923  7.0569  114.72\n",
       "4  10.612  19.765  1059.7  549.98  3.4976  3.2484  7.3978  114.72"
      ]
     },
     "execution_count": 71,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model_data = df[['CDP', 'GTEP','TIT', 'TAT', 'AFDP', 'CO', 'AT',\"TEY\"]]\n",
    "model_data.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 5.1. Data Pre-Processing<a class=\"anchor\" id=\"5.1\"></a>\n",
    "#### Deal with missing data \n",
    "There is no missing value in this dataset. Neither, there are values like “unknown”, “others”, which are helpless just like missing values. Thus, these ambiguous values are removed from the dataset."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "![FilterMethods](https://www.geeksforgeeks.org/wp-content/uploads/ml.png)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Feature Engineering"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Applying some Data Transformation to increase the linear realtionship and improve our model prediction as well it scores"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Applying Standard Scaler\n",
    "+ For each feature, the Standard Scaler scales the values such that the mean is 0 and the standard deviation is 1(or the variance).\n",
    "+ x_scaled = x – mean/std_dev\n",
    "+ However, Standard Scaler assumes that the distribution of the variable is normal. Thus, in case, the variables are not normally distributed, we either choose a different scaler or first, convert the variables to a normal distribution and then apply this scaler"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 131,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Continuous Feature Count 8\n"
     ]
    }
   ],
   "source": [
    "continuous_feature=[feature for feature in model_data.columns if model_data[feature].dtype!='O']\n",
    "print('Continuous Feature Count {}'.format(len(continuous_feature)))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 132,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_standard_scaled = model_data.copy()\n",
    "features = df_standard_scaled[continuous_feature]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 133,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>CDP</th>\n",
       "      <th>GTEP</th>\n",
       "      <th>TIT</th>\n",
       "      <th>TAT</th>\n",
       "      <th>AFDP</th>\n",
       "      <th>CO</th>\n",
       "      <th>AT</th>\n",
       "      <th>TEY</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>-1.357331</td>\n",
       "      <td>-1.379101</td>\n",
       "      <td>-1.488376</td>\n",
       "      <td>0.585240</td>\n",
       "      <td>-0.921232</td>\n",
       "      <td>0.532012</td>\n",
       "      <td>-1.439778</td>\n",
       "      <td>-1.231172</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>-1.363676</td>\n",
       "      <td>-1.363528</td>\n",
       "      <td>-1.482325</td>\n",
       "      <td>0.585240</td>\n",
       "      <td>-0.921495</td>\n",
       "      <td>0.568733</td>\n",
       "      <td>-1.449601</td>\n",
       "      <td>-1.229909</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>-1.360957</td>\n",
       "      <td>-1.351309</td>\n",
       "      <td>-1.476275</td>\n",
       "      <td>0.568715</td>\n",
       "      <td>-0.944385</td>\n",
       "      <td>0.552938</td>\n",
       "      <td>-1.434721</td>\n",
       "      <td>-1.230541</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>-1.356424</td>\n",
       "      <td>-1.348194</td>\n",
       "      <td>-1.464173</td>\n",
       "      <td>0.583969</td>\n",
       "      <td>-0.946884</td>\n",
       "      <td>0.548933</td>\n",
       "      <td>-1.413702</td>\n",
       "      <td>-1.229909</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>-1.350985</td>\n",
       "      <td>-1.354663</td>\n",
       "      <td>-1.458123</td>\n",
       "      <td>0.582698</td>\n",
       "      <td>-0.924389</td>\n",
       "      <td>0.574179</td>\n",
       "      <td>-1.368693</td>\n",
       "      <td>-1.229909</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "        CDP      GTEP       TIT       TAT      AFDP        CO        AT  \\\n",
       "0 -1.357331 -1.379101 -1.488376  0.585240 -0.921232  0.532012 -1.439778   \n",
       "1 -1.363676 -1.363528 -1.482325  0.585240 -0.921495  0.568733 -1.449601   \n",
       "2 -1.360957 -1.351309 -1.476275  0.568715 -0.944385  0.552938 -1.434721   \n",
       "3 -1.356424 -1.348194 -1.464173  0.583969 -0.946884  0.548933 -1.413702   \n",
       "4 -1.350985 -1.354663 -1.458123  0.582698 -0.924389  0.574179 -1.368693   \n",
       "\n",
       "        TEY  \n",
       "0 -1.231172  \n",
       "1 -1.229909  \n",
       "2 -1.230541  \n",
       "3 -1.229909  \n",
       "4 -1.229909  "
      ]
     },
     "execution_count": 133,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.preprocessing import StandardScaler\n",
    "scaler = StandardScaler()\n",
    "\n",
    "df_standard_scaled[continuous_feature] = scaler.fit_transform(features.values)\n",
    "df_standard_scaled.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Now if we check the mean and standard deviation of our scaled data it should have a Mean '0' and Standard deviation '1'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 91,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Mean\n",
      " CDP       0.0\n",
      "GTEP      0.0\n",
      "TIT      -0.0\n",
      "TAT       0.0\n",
      "AFDP      0.0\n",
      "CO        0.0\n",
      "AT       -0.0\n",
      "TEY     134.2\n",
      "dtype: float64 \n",
      "Standard Devaition \n",
      " CDP      1.0\n",
      "GTEP     1.0\n",
      "TIT      1.0\n",
      "TAT      1.0\n",
      "AFDP     1.0\n",
      "CO       1.0\n",
      "AT       1.0\n",
      "TEY     16.0\n",
      "dtype: float64 1\n"
     ]
    }
   ],
   "source": [
    "print('Mean' '\\n',np.round(df_standard_scaled.mean(),1),'\\n' 'Standard Devaition','\\n',np.round(df_standard_scaled.std()),1)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": []
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 5.2 Test Train Split With Imbalanced Dataset<a class=\"anchor\" id=\"5.2\"></a>\n",
    "\n",
    "\n",
    "\n",
    "***Train Test Split*** : To have unknown datapoints to test the data rather than testing with the same points with which the model was trained. This helps capture the model performance much better.\n",
    "\n",
    "![](https://cdn-images-1.medium.com/max/1600/1*-8_kogvwmL1H6ooN1A1tsQ.png)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 92,
   "metadata": {},
   "outputs": [],
   "source": [
    "x = df_standard_scaled.drop('TEY',axis=1)\n",
    "y = df_standard_scaled[['TEY']]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 93,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Splitting data into test data and train data\n",
    "\n",
    "x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=3)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6. Hyperparameter Tuning<a class=\"anchor\" id=\"6\"></a>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 6.1 **`Hyperparameter Tuning`**: Optimal Learning rate ,Number of Layers and Neurons<a class=\"anchor\" id=\"6.1\"></a>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 97,
   "metadata": {},
   "outputs": [],
   "source": [
    "def build_model(hp):\n",
    "    model =Sequential()\n",
    "    for i in range(hp.Int('num_layers', 2, 20)):\n",
    "        model.add(layers.Dense(units=hp.Int('units_' + str(i),\n",
    "                                            min_value=32,\n",
    "                                            max_value=100,\n",
    "                                            step=32),\n",
    "                               activation='relu'))\n",
    "    model.add(layers.Dense(1, activation='linear'))\n",
    "    model.compile(\n",
    "        optimizer=keras.optimizers.Adam(\n",
    "            hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),\n",
    "        loss='mean_absolute_error',\n",
    "        metrics=['mean_absolute_error'])\n",
    "    return model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 98,
   "metadata": {},
   "outputs": [],
   "source": [
    "tuner = RandomSearch(build_model,objective='val_mean_absolute_error',max_trials=5,  executions_per_trial=3,directory='project',project_name='Gas Turbine')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 99,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Search space summary\n",
      "Default search space size: 4\n",
      "num_layers (Int)\n",
      "{'default': None, 'conditions': [], 'min_value': 2, 'max_value': 20, 'step': 1, 'sampling': None}\n",
      "units_0 (Int)\n",
      "{'default': None, 'conditions': [], 'min_value': 32, 'max_value': 100, 'step': 32, 'sampling': None}\n",
      "units_1 (Int)\n",
      "{'default': None, 'conditions': [], 'min_value': 32, 'max_value': 100, 'step': 32, 'sampling': None}\n",
      "learning_rate (Choice)\n",
      "{'default': 0.01, 'conditions': [], 'values': [0.01, 0.001, 0.0001], 'ordered': True}\n"
     ]
    }
   ],
   "source": [
    "tuner.search_space_summary()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 101,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Oracle triggered exit\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Trial 5 Complete [00h 05m 01s]\n",
      "val_mean_absolute_error: 0.4440057973066966\n",
      "\n",
      "Best val_mean_absolute_error So Far: 0.4288428723812103\n",
      "Total elapsed time: 00h 29m 24s\n"
     ]
    }
   ],
   "source": [
    "tuner.search(x_train, y_train,epochs=100,validation_data=(x_test, y_test))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "tuner.results_summary()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 6.2 **`Hyperparameter Tuning`**: Optimal Batch_size, Number of Epochs<a class=\"anchor\" id=\"6.2\"></a>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 115,
   "metadata": {},
   "outputs": [],
   "source": [
    "def create_model():\n",
    "    model1 = Sequential()\n",
    "    model1.add(Dense(32,input_dim=7,activation='relu'))\n",
    "    model1.add(Dense(64,activation='relu'))\n",
    "    model1.add(Dense(96,activation=\"relu\"))\n",
    "    model1.add(Dense(32,activation=\"relu\"))\n",
    "    model1.add(Dense(64,activation=\"relu\"))\n",
    "    model1.add(Dense(32,activation=\"relu\"))\n",
    "    model1.add(Dense(96,activation=\"relu\"))\n",
    "    model1.add(Dense(96,activation=\"relu\"))\n",
    "    model1.add(Dense(32,activation=\"relu\"))\n",
    "    model1.add(Dense(64,activation=\"relu\"))\n",
    "    model1.add(Dense(64,activation=\"relu\"))\n",
    "    model1.add(Dense(units=1,activation=\"linear\"))\n",
    "    \n",
    "    adam=Adam(learning_rate=0.001)\n",
    "    model1.compile(loss='mean_absolute_error',optimizer = adam,metrics=[\"mean_absolute_error\"])\n",
    "    return model1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 116,
   "metadata": {},
   "outputs": [],
   "source": [
    "model1 =KerasRegressor(build_fn=create_model,verbose=0)\n",
    "batch_size=[10,20,40,50]\n",
    "epochs=[10,50,100,200]\n",
    "param_grid=dict(batch_size=batch_size,epochs=epochs)\n",
    "grid = GridSearchCV(estimator=model1,param_grid=param_grid,cv=KFold(),verbose=10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "grid_result = grid.fit(x_test,y_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 118,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Best -0.5385425746440887, using {'batch_size': 40, 'epochs': 200}\n",
      "-1.3277084469795226,0.4157314466582501 with {'batch_size': 10, 'epochs': 10}\n",
      "-1.2819799065589905,0.6388210326784041 with {'batch_size': 10, 'epochs': 50}\n",
      "-0.6195487558841706,0.22372498685599693 with {'batch_size': 10, 'epochs': 100}\n",
      "-0.6067143261432648,0.14244373523566725 with {'batch_size': 10, 'epochs': 200}\n",
      "-1.6372950077056885,0.312710777433406 with {'batch_size': 20, 'epochs': 10}\n",
      "-0.797945237159729,0.24094883948455134 with {'batch_size': 20, 'epochs': 50}\n",
      "-0.9796943306922913,0.3860892553758134 with {'batch_size': 20, 'epochs': 100}\n",
      "-0.5884879648685455,0.13458080593764546 with {'batch_size': 20, 'epochs': 200}\n",
      "-2.6895342111587524,1.1914054795904954 with {'batch_size': 40, 'epochs': 10}\n",
      "-1.5166579246520997,0.5587834119397824 with {'batch_size': 40, 'epochs': 50}\n",
      "-0.9263822197914123,0.28557928597207294 with {'batch_size': 40, 'epochs': 100}\n",
      "-0.5385425746440887,0.09872459202891593 with {'batch_size': 40, 'epochs': 200}\n",
      "-2.8348670959472657,0.776276398121674 with {'batch_size': 50, 'epochs': 10}\n",
      "-1.0364789128303529,0.27062329238132626 with {'batch_size': 50, 'epochs': 50}\n",
      "-1.3295377373695374,0.6363427964769303 with {'batch_size': 50, 'epochs': 100}\n",
      "-0.8493212640285492,0.28902519768644597 with {'batch_size': 50, 'epochs': 200}\n"
     ]
    }
   ],
   "source": [
    "print('Best {}, using {}'.format(grid_result.best_score_,grid_result.best_params_))\n",
    "means = grid_result.cv_results_[\"mean_test_score\"]\n",
    "stds = grid_result.cv_results_[\"std_test_score\"]\n",
    "params = grid_result.cv_results_[\"params\"]\n",
    "for mean,stdev,param in zip(means,stds,params):\n",
    "    print(\"{},{} with {}\".format(mean,stdev,param))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 6.3 **`Hyperparameter Tuning`**: Optimal Droupout rate<a class=\"anchor\" id=\"6.3\"></a>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 119,
   "metadata": {},
   "outputs": [],
   "source": [
    "def create_model(dropout_rate):\n",
    "    model2 = Sequential()\n",
    "    model2.add(Dense(32,input_dim=7,activation='relu'))\n",
    "    model2.add(Dense(64,activation='relu'))\n",
    "    model2.add(Dense(96,activation=\"relu\"))\n",
    "    model2.add(Dense(32,activation=\"relu\"))\n",
    "    model2.add(Dense(64,activation=\"relu\"))\n",
    "    model2.add(Dense(32,activation=\"relu\"))\n",
    "    model2.add(Dense(96,activation=\"relu\"))\n",
    "    model2.add(Dense(96,activation=\"relu\"))\n",
    "    model2.add(Dense(32,activation=\"relu\"))\n",
    "    model2.add(Dense(64,activation=\"relu\"))\n",
    "    model2.add(Dense(64,activation=\"relu\"))\n",
    "    model2.add(Dense(units=1,activation=\"linear\"))\n",
    "    \n",
    "    adam=Adam(lr=0.001)\n",
    "    model2.compile(loss='mean_absolute_error',optimizer = adam,metrics=[\"mean_absolute_error\"])\n",
    "    return model2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 120,
   "metadata": {},
   "outputs": [],
   "source": [
    "model2=KerasRegressor(build_fn=create_model,batch_size=40,epochs=200,verbose=0)\n",
    "dropout_rate=[0.0,0.1,0.2,0.3,0.4,0.5]\n",
    "param_grid=dict(dropout_rate=dropout_rate)\n",
    "grid2 = GridSearchCV(estimator=model2,param_grid=param_grid,cv=KFold(),verbose=10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "grid_result2 = grid2.fit(x_test,y_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 122,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Best -0.6679447650909424, using {'dropout_rate': 0.3}\n",
      "-0.8323201537132263,0.26923293832543493 with {'dropout_rate': 0.0}\n",
      "-1.1670883774757386,0.5794670480557291 with {'dropout_rate': 0.1}\n",
      "-1.182572340965271,0.5096832458496641 with {'dropout_rate': 0.2}\n",
      "-0.6679447650909424,0.27758018642107785 with {'dropout_rate': 0.3}\n",
      "-1.0636109471321107,0.20700826606234907 with {'dropout_rate': 0.4}\n",
      "-0.9452173233032226,0.20756484640550968 with {'dropout_rate': 0.5}\n"
     ]
    }
   ],
   "source": [
    "print('Best {}, using {}'.format(grid_result2.best_score_,grid_result2.best_params_))\n",
    "means = grid_result2.cv_results_[\"mean_test_score\"]\n",
    "stds = grid_result2.cv_results_[\"std_test_score\"]\n",
    "params = grid_result2.cv_results_[\"params\"]\n",
    "for mean,stdev,param in zip(means,stds,params):\n",
    "    print(\"{},{} with {}\".format(mean,stdev,param))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 6.4 **`Hyperparameter Tuning`**: Optimal Activation Function and Kernel Initializer<a class=\"anchor\" id=\"6.4\"></a>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 123,
   "metadata": {},
   "outputs": [],
   "source": [
    "def create_model(activation_function,init):\n",
    "    model3 = Sequential()\n",
    "    model3.add(Dense(32,input_dim=7,activation='relu'))\n",
    "    model3.add(Dropout(0.3))\n",
    "    model3.add(Dense(64,activation='relu'))\n",
    "    model3.add(Dropout(0.3))\n",
    "    model3.add(Dense(96,activation=\"relu\"))\n",
    "    model3.add(Dropout(0.3))\n",
    "    model3.add(Dense(32,activation=\"relu\"))\n",
    "    model3.add(Dropout(0.3))\n",
    "    model3.add(Dense(64,activation=\"relu\"))\n",
    "    model3.add(Dropout(0.3))\n",
    "    model3.add(Dense(32,activation=\"relu\"))\n",
    "    model3.add(Dropout(0.3))\n",
    "    model3.add(Dense(96,activation=\"relu\"))\n",
    "    model3.add(Dropout(0.3))\n",
    "    model3.add(Dense(96,activation=\"relu\"))\n",
    "    model3.add(Dropout(0.3))\n",
    "    model3.add(Dense(32,activation=\"relu\"))\n",
    "    model3.add(Dropout(0.3))\n",
    "    model3.add(Dense(64,activation=\"relu\"))\n",
    "    model3.add(Dropout(0.3))\n",
    "    model3.add(Dense(64,activation=\"relu\"))\n",
    "    model3.add(Dropout(0.3))\n",
    "    model3.add(Dense(units=1,activation=\"linear\"))\n",
    "    \n",
    "    adam=Adam(lr=0.001)\n",
    "    model3.compile(loss='mean_absolute_error',optimizer = adam,metrics=[\"mean_absolute_error\"])\n",
    "    return model3"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 124,
   "metadata": {},
   "outputs": [],
   "source": [
    "model3=KerasRegressor(build_fn=create_model,batch_size=40,epochs=200,verbose=0)\n",
    "activation_function=['softmax','tanh','relu']\n",
    "init=['normal','uniform','zero']\n",
    "param_grid=dict(activation_function=activation_function,init=init)\n",
    "grid3 = GridSearchCV(estimator=model3,param_grid=param_grid,cv=KFold(),verbose=10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "grid_result3 = grid3.fit(x_test,y_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 126,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Best -3.122679662704468, using {'activation_function': 'softmax', 'init': 'uniform'}\n",
      "-3.5212940216064452,0.8786527863154662 with {'activation_function': 'softmax', 'init': 'normal'}\n",
      "-3.122679662704468,0.6747694999995387 with {'activation_function': 'softmax', 'init': 'uniform'}\n",
      "-3.262042427062988,1.3589116630891251 with {'activation_function': 'softmax', 'init': 'zero'}\n",
      "-3.5312024116516114,0.7796613422378148 with {'activation_function': 'tanh', 'init': 'normal'}\n",
      "-4.61785249710083,1.342868088610206 with {'activation_function': 'tanh', 'init': 'uniform'}\n",
      "-3.2921382427215575,0.5384224892388144 with {'activation_function': 'tanh', 'init': 'zero'}\n",
      "-4.91969141960144,1.3631206250623549 with {'activation_function': 'relu', 'init': 'normal'}\n",
      "-3.7872642040252686,1.9546784298801612 with {'activation_function': 'relu', 'init': 'uniform'}\n",
      "-3.934886360168457,1.6337526760723777 with {'activation_function': 'relu', 'init': 'zero'}\n"
     ]
    }
   ],
   "source": [
    "print('Best {}, using {}'.format(grid_result3.best_score_,grid_result3.best_params_))\n",
    "means = grid_result3.cv_results_[\"mean_test_score\"]\n",
    "stds = grid_result3.cv_results_[\"std_test_score\"]\n",
    "params = grid_result3.cv_results_[\"params\"]\n",
    "for mean,stdev,param in zip(means,stds,params):\n",
    "    print(\"{},{} with {}\".format(mean,stdev,param))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 7. Model Building Neural Networks<a class=\"anchor\" id=\"7\"></a>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Standardizing only predictor variable - after train test split"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 147,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>CDP</th>\n",
       "      <th>GTEP</th>\n",
       "      <th>TIT</th>\n",
       "      <th>TAT</th>\n",
       "      <th>AFDP</th>\n",
       "      <th>CO</th>\n",
       "      <th>AT</th>\n",
       "      <th>TEY</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>10.605</td>\n",
       "      <td>19.663</td>\n",
       "      <td>1059.2</td>\n",
       "      <td>550.00</td>\n",
       "      <td>3.5000</td>\n",
       "      <td>3.1547</td>\n",
       "      <td>6.8594</td>\n",
       "      <td>114.70</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>10.598</td>\n",
       "      <td>19.728</td>\n",
       "      <td>1059.3</td>\n",
       "      <td>550.00</td>\n",
       "      <td>3.4998</td>\n",
       "      <td>3.2363</td>\n",
       "      <td>6.7850</td>\n",
       "      <td>114.72</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>10.601</td>\n",
       "      <td>19.779</td>\n",
       "      <td>1059.4</td>\n",
       "      <td>549.87</td>\n",
       "      <td>3.4824</td>\n",
       "      <td>3.2012</td>\n",
       "      <td>6.8977</td>\n",
       "      <td>114.71</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>10.606</td>\n",
       "      <td>19.792</td>\n",
       "      <td>1059.6</td>\n",
       "      <td>549.99</td>\n",
       "      <td>3.4805</td>\n",
       "      <td>3.1923</td>\n",
       "      <td>7.0569</td>\n",
       "      <td>114.72</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>10.612</td>\n",
       "      <td>19.765</td>\n",
       "      <td>1059.7</td>\n",
       "      <td>549.98</td>\n",
       "      <td>3.4976</td>\n",
       "      <td>3.2484</td>\n",
       "      <td>7.3978</td>\n",
       "      <td>114.72</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>15034</th>\n",
       "      <td>10.400</td>\n",
       "      <td>19.164</td>\n",
       "      <td>1049.7</td>\n",
       "      <td>546.21</td>\n",
       "      <td>3.5421</td>\n",
       "      <td>4.5186</td>\n",
       "      <td>9.0301</td>\n",
       "      <td>111.61</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>15035</th>\n",
       "      <td>10.433</td>\n",
       "      <td>19.414</td>\n",
       "      <td>1046.3</td>\n",
       "      <td>543.22</td>\n",
       "      <td>3.5059</td>\n",
       "      <td>4.8470</td>\n",
       "      <td>7.8879</td>\n",
       "      <td>111.78</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>15036</th>\n",
       "      <td>10.483</td>\n",
       "      <td>19.530</td>\n",
       "      <td>1037.7</td>\n",
       "      <td>537.32</td>\n",
       "      <td>3.4770</td>\n",
       "      <td>7.9632</td>\n",
       "      <td>7.2647</td>\n",
       "      <td>110.19</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>15037</th>\n",
       "      <td>10.533</td>\n",
       "      <td>19.377</td>\n",
       "      <td>1043.2</td>\n",
       "      <td>541.24</td>\n",
       "      <td>3.4486</td>\n",
       "      <td>6.2494</td>\n",
       "      <td>7.0060</td>\n",
       "      <td>110.74</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>15038</th>\n",
       "      <td>10.583</td>\n",
       "      <td>19.306</td>\n",
       "      <td>1049.9</td>\n",
       "      <td>545.85</td>\n",
       "      <td>3.4275</td>\n",
       "      <td>4.9816</td>\n",
       "      <td>6.9279</td>\n",
       "      <td>111.58</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>15039 rows × 8 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "          CDP    GTEP     TIT     TAT    AFDP      CO      AT     TEY\n",
       "0      10.605  19.663  1059.2  550.00  3.5000  3.1547  6.8594  114.70\n",
       "1      10.598  19.728  1059.3  550.00  3.4998  3.2363  6.7850  114.72\n",
       "2      10.601  19.779  1059.4  549.87  3.4824  3.2012  6.8977  114.71\n",
       "3      10.606  19.792  1059.6  549.99  3.4805  3.1923  7.0569  114.72\n",
       "4      10.612  19.765  1059.7  549.98  3.4976  3.2484  7.3978  114.72\n",
       "...       ...     ...     ...     ...     ...     ...     ...     ...\n",
       "15034  10.400  19.164  1049.7  546.21  3.5421  4.5186  9.0301  111.61\n",
       "15035  10.433  19.414  1046.3  543.22  3.5059  4.8470  7.8879  111.78\n",
       "15036  10.483  19.530  1037.7  537.32  3.4770  7.9632  7.2647  110.19\n",
       "15037  10.533  19.377  1043.2  541.24  3.4486  6.2494  7.0060  110.74\n",
       "15038  10.583  19.306  1049.9  545.85  3.4275  4.9816  6.9279  111.58\n",
       "\n",
       "[15039 rows x 8 columns]"
      ]
     },
     "execution_count": 147,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model_data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 161,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(12031, 7)\n",
      "(3008, 7)\n",
      "(12031, 1)\n",
      "(3008, 1)\n"
     ]
    }
   ],
   "source": [
    "#assigning predictor variables to x and response variable to y\n",
    "x = model_data.drop('TEY', axis=1)\n",
    "y = model_data[[\"TEY\"]]\n",
    "\n",
    "x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.20, random_state=42)\n",
    "\n",
    "scaler_train = StandardScaler()\n",
    "scaler_test = StandardScaler()\n",
    "\n",
    "x_train_scaled = scaler_train.fit_transform(x_train) # scaling train data -- predictor\n",
    "x_test_scaled  = scaler_test.fit_transform(x_test) # scaling test data -- predictor\n",
    "\n",
    "print(x_train_scaled.shape)\n",
    "print(x_test_scaled.shape)\n",
    "print(y_train.shape)\n",
    "print(y_test.shape)\n",
    "\n",
    "#for removing heading from y_test\n",
    "#y_test = y_test.values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# since we have continuous ouput, AF is not required in the o/p layer\n",
    "model = Sequential()\n",
    "model.add( Dense( units = 50 , activation = 'relu' , kernel_initializer = 'normal', input_dim = 7)) # input layer\n",
    "model.add( Dense( units = 20 , activation = 'tanh' , kernel_initializer = 'normal' )) # hidden layer\n",
    "model.add( Dense( units = 1  , kernel_initializer = 'normal' )) # o/p layer\n",
    "\n",
    "model.compile(optimizer= \"adam\", loss=\"mse\", metrics= [\"mae\", \"mse\"])\n",
    "model.fit(x_train_scaled, y_train , batch_size=50, validation_split=0.3, epochs=100,  verbose=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 151,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "94/94 [==============================] - 0s 2ms/step\n",
      "batch_size: 5 - epochs: 5 Accuracy: 97.73075369067175\n",
      "94/94 [==============================] - 0s 2ms/step\n",
      "batch_size: 5 - epochs: 10 Accuracy: 99.5585905304457\n",
      "94/94 [==============================] - 0s 2ms/step\n",
      "batch_size: 5 - epochs: 50 Accuracy: 99.65107608927266\n",
      "94/94 [==============================] - 0s 2ms/step\n",
      "batch_size: 5 - epochs: 100 Accuracy: 99.64736833004822\n",
      "94/94 [==============================] - 0s 2ms/step\n",
      "batch_size: 10 - epochs: 5 Accuracy: 80.90613588380201\n",
      "94/94 [==============================] - 0s 3ms/step\n",
      "batch_size: 10 - epochs: 10 Accuracy: 98.2573511745601\n",
      "94/94 [==============================] - 0s 2ms/step\n",
      "batch_size: 10 - epochs: 50 Accuracy: 99.65084677862158\n",
      "94/94 [==============================] - 0s 2ms/step\n",
      "batch_size: 10 - epochs: 100 Accuracy: 99.58161139512534\n",
      "94/94 [==============================] - 0s 2ms/step\n",
      "batch_size: 15 - epochs: 5 Accuracy: 58.854512990805404\n",
      "94/94 [==============================] - 0s 2ms/step\n",
      "batch_size: 15 - epochs: 10 Accuracy: 89.82417947308262\n",
      "94/94 [==============================] - 0s 2ms/step\n",
      "batch_size: 15 - epochs: 50 Accuracy: 99.6253504445297\n",
      "94/94 [==============================] - 0s 2ms/step\n",
      "batch_size: 15 - epochs: 100 Accuracy: 99.63428305186946\n",
      "94/94 [==============================] - 0s 2ms/step\n",
      "batch_size: 20 - epochs: 5 Accuracy: 46.13226823645566\n",
      "94/94 [==============================] - 0s 2ms/step\n",
      "batch_size: 20 - epochs: 10 Accuracy: 80.97753229527885\n",
      "94/94 [==============================] - 0s 2ms/step\n",
      "batch_size: 20 - epochs: 50 Accuracy: 99.66061724651402\n",
      "94/94 [==============================] - 0s 2ms/step\n",
      "batch_size: 20 - epochs: 100 Accuracy: 99.68478888871317\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>batchsize</th>\n",
       "      <th>epochs</th>\n",
       "      <th>Accuracy</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>5</td>\n",
       "      <td>5</td>\n",
       "      <td>97.730754</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>5</td>\n",
       "      <td>10</td>\n",
       "      <td>99.558591</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>5</td>\n",
       "      <td>50</td>\n",
       "      <td>99.651076</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>5</td>\n",
       "      <td>100</td>\n",
       "      <td>99.647368</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>10</td>\n",
       "      <td>5</td>\n",
       "      <td>80.906136</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>10</td>\n",
       "      <td>10</td>\n",
       "      <td>98.257351</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>10</td>\n",
       "      <td>50</td>\n",
       "      <td>99.650847</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>10</td>\n",
       "      <td>100</td>\n",
       "      <td>99.581611</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>15</td>\n",
       "      <td>5</td>\n",
       "      <td>58.854513</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>15</td>\n",
       "      <td>10</td>\n",
       "      <td>89.824179</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>15</td>\n",
       "      <td>50</td>\n",
       "      <td>99.625350</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>15</td>\n",
       "      <td>100</td>\n",
       "      <td>99.634283</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>20</td>\n",
       "      <td>5</td>\n",
       "      <td>46.132268</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>20</td>\n",
       "      <td>10</td>\n",
       "      <td>80.977532</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>20</td>\n",
       "      <td>50</td>\n",
       "      <td>99.660617</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>20</td>\n",
       "      <td>100</td>\n",
       "      <td>99.684789</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   batchsize  epochs   Accuracy\n",
       "0          5       5  97.730754\n",
       "0          5      10  99.558591\n",
       "0          5      50  99.651076\n",
       "0          5     100  99.647368\n",
       "0         10       5  80.906136\n",
       "0         10      10  98.257351\n",
       "0         10      50  99.650847\n",
       "0         10     100  99.581611\n",
       "0         15       5  58.854513\n",
       "0         15      10  89.824179\n",
       "0         15      50  99.625350\n",
       "0         15     100  99.634283\n",
       "0         20       5  46.132268\n",
       "0         20      10  80.977532\n",
       "0         20      50  99.660617\n",
       "0         20     100  99.684789"
      ]
     },
     "execution_count": 151,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "def toFindBestParams(x_train_scaled, y_train, x_test_scaled, y_test):\n",
    "        \n",
    "    #defining list of hyperparameters\n",
    "    batch_size_list = [5 , 10 , 15 , 20]\n",
    "    epoch_list      = [5 , 10 , 50 , 100]\n",
    "     \n",
    "    bestParamTable = pd.DataFrame()\n",
    "    \n",
    "    for batch_trial in batch_size_list:\n",
    "        for epochs_trial in epoch_list:\n",
    "            \n",
    "            # create ANN model\n",
    "            model = Sequential()\n",
    "            # Defining the first layer of the model\n",
    "            model.add(Dense(units=50, input_dim=x_train_scaled.shape[1], kernel_initializer='normal', activation='relu'))\n",
    "            \n",
    "            # Defining the Second layer of the model\n",
    "            model.add(Dense(units=20, kernel_initializer='normal', activation='tanh'))\n",
    " \n",
    "            # The output neuron is a single fully connected node \n",
    "            # Since we will be predicting a single number\n",
    "            model.add(Dense(1, kernel_initializer='normal'))\n",
    " \n",
    "            # Compiling the model\n",
    "            model.compile(optimizer ='adam', loss = 'mean_squared_error')\n",
    "            \n",
    "            # Fitting the ANN to the Training set\n",
    "            model.fit(x_train_scaled, y_train , batch_size=batch_trial, epochs=epochs_trial,  verbose=0)\n",
    "                        \n",
    "            MAPE = np.mean(100 * (np.abs(y_test-model.predict(x_test_scaled))/y_test))  \n",
    "                        \n",
    "            bestParamTable=bestParamTable.append(pd.DataFrame(data=[[batch_trial, epochs_trial, 100-MAPE]],\n",
    "                                                        columns=['batchsize','epochs','Accuracy'] ))\n",
    "            \n",
    "            # printing the results of the current iteration\n",
    "            print('batch_size:', batch_trial,'-', 'epochs:',epochs_trial, 'Accuracy:',100-MAPE)\n",
    "\n",
    "    return bestParamTable\n",
    "\n",
    "# Calling the function\n",
    "finalParamTable = toFindBestParams(x_train_scaled, y_train, x_test_scaled, y_test)\n",
    "finalParamTable"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 152,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "index          0.000000\n",
       "batchsize     20.000000\n",
       "epochs       100.000000\n",
       "Accuracy      99.684789\n",
       "Name: 15, dtype: float64"
      ]
     },
     "execution_count": 152,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# getting corresponding row values of the maximum value of 'Accuracy' column\n",
    "finalParamTable = finalParamTable.reset_index()\n",
    "#print(finalParamTable_1)\n",
    "#print(finalParamTable['Accuracy'].idxmax())\n",
    "finalParamTable.iloc[finalParamTable['Accuracy'].idxmax()]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Training Model - using best params"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 153,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<keras.callbacks.History at 0x14bf1f099a0>"
      ]
     },
     "execution_count": 153,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model.compile(optimizer ='adam', loss = 'mean_squared_error')\n",
    "# fitting the model to best params\n",
    "model.fit(x_train_scaled,y_train, batch_size=20 , epochs = 100, verbose=0)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 7.1 Model Evaluation Train and Test Error<a class=\"anchor\" id=\"7.1\"></a>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 189,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAA7wAAAImCAYAAABwyYamAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjUuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/YYfK9AAAACXBIWXMAAAsTAAALEwEAmpwYAAB1C0lEQVR4nO3deVyU5f7/8fc9CzCDrCIgCiJiqClqdlxKyyUtS1tss2+nxZ/l8WRli9lqtp60rE6adTJLW6xTHW0/llpqZSfL3M0d3EBUVEBk2Gbm9wcygaKZAvcAr+fjwYO51/ncwwX6vu/rvm7D6/V6BQAAAABAPWMxuwAAAAAAAGoCgRcAAAAAUC8ReAEAAAAA9RKBFwAAAABQLxF4AQAAAAD1EoEXAAAAAFAvEXgBoA7p27evUlJS9PXXXx+zLCcnR2eeeaa6det2yvtfuHChUlJSTmrdXbt2KSUlRZs2bfK9njNnzp9+zxtuuEEpKSl68803j1nmdrvVo0cPpaSk6PDhw39639Xpggsu0F/+8hcVFRUdsywlJUULFy6skfft27ev3n333ZNat6SkRLNmzaqROk7VkCFDNGXKlOMu37Bhg0aOHKm//OUvSk1N1WWXXeZ3x3AyJk6cqBtuuKHKZeW/H8f7+vHHH2u5WgBoOAi8AFDH2O12LViw4Jj53377rdxutwkVSYZhVPr+Zx3vmJYtW6aDBw+eVm3V4ddff1VOTo7sdnuVJxv8xRdffKHJkyebXcZJ27Nnj2644Qa1aNFCs2bN0hdffKEbb7xRkyZN0rRp08wur9q98847+uGHH475Ovvss80uDQDqLZvZBQAA/pyuXbtq8eLFKi0tlc32+5/xefPmqVOnTkpPT6/1mho3biyLxaImTZqc0vZdu3bV//73Px04cECRkZG++fPnz1enTp20YsWK6ir1lHz22Wfq0qWLwsLCNHv2bF166aWm1nM8Xq/X7BL+lHnz5iksLEwPPvigb15CQoJ2796tf//73xoxYoSJ1VW/8PDwU/4dAQCcGq7wAkAdc+6556qkpES//PKLb97hw4e1dOlSXXDBBZXW3bt3r+677z716NFDZ511lkaPHq29e/f6lm/fvl0333yzOnbsqMsuu0xbtmyptP2+ffs0evRode7cWT179tTDDz+sQ4cOHVNTUFCQmjVrpuTkZEnSBx98oP79+6t9+/a66KKL9Mknn5zwmFJSUtS8eXN9++23vnler1fz58/XhRdeWGnd/Px8jRs3Tl27dlW3bt105513as+ePb7l27Zt08iRI3X22Werffv2GjRokL755hvf8r59++qtt97SDTfcoI4dO2rw4MGV3vdoJSUl+uqrr9S9e3f1799fS5cu1a5du45Zb/Xq1Ro8eLA6dOigm2++WZmZmb5l8+fP16BBg9ShQwf17dtX06dP9y0rLi7W5MmT1bdvX3Xo0EHXXXedVq1aVWUtDzzwgO68885K88q7PC9dulQPPvigcnJylJKSoqVLl0qSPvnkE1144YXq2LGjrrjiCi1atOi4x1paWqoXX3xRffv21ZlnnqlzzjlHTz31lK/nwJQpU3T77bdr4sSJ6tq1q3r27FlpuSS9+eabOu+883TWWWfpxRdfPO57SZLFYtG+ffu0YcOGSvNvvPFGzZgxwzednp6um266yffzmjNnjq/rfcWu9eXmzJlTqWv/6tWrddNNN6lz587q0KGDrrrqKi1fvrzS9q+88oq6deumW2+9VZK0atUqDR06VB06dNCAAQP0+uuvy+Px+Pb53XffafDgwUpNTdWoUaOq/L34s/r27atnn31WvXv31vnnn6/ffvutytrWrFmjG2+80fd7+dxzz6mkpMR37FdccYXGjBmjs846S6+88or27NmjkSNHqkuXLjr77LN15513av/+/addLwDUBQReAKhjAgMD1atXr0ohbvHixWrfvn2lq6MlJSW6+eabtXv3bk2bNk1vvfWW9uzZo1GjRsnr9aqkpEQjRoyQw+HQ7NmzNWrUKL3++uuV3uuOO+6Q1+vVBx98oFdffVU7duzQ3XffXWVdCxYsUGxsrNatW6cnn3xSDzzwgL7++mvdcMMNeuCBB7Rt27YTHtcFF1xQ6ZhWr14tq9Wqdu3aVVrv0UcfVXp6uqZPn6533nlHhmHolltuUWlpqbxer0aOHKng4GB98MEH+vTTT3XGGWfooYceUnFxsW8fkydP1tChQzV79mwlJibqwQcfrLS8osWLFysnJ0f9+vVTz549FRQUVOW9yu+8845GjRqlOXPmKCAgQMOGDZPH41F2drbuvvtu3XTTTfrqq680duxY/fOf/9T//vc/SdITTzyh2bNna/z48frkk0/UunVrDRs2rNKJiZPRuXNnPfTQQwoPD9cPP/ygzp076/vvv9dTTz2lO++8U59//rmuvfZa3Xnnnce9Yv7GG2/o008/1YQJE/T1119r7Nixeu+99yp1N1+0aJHy8vL073//W3feeadmzZrlWz5nzhxNmTJF999/vz788EPt3LlT69atO27NAwcOVFhYmK644gr99a9/1SuvvKKVK1cqJCRELVq0kFR2QuDWW2+Vw+HQf/7zH/3973/Xs88+e9Kfy+HDh3Xrrbeqbdu2+vTTT/Xhhx8qODhY48ePr7TeokWL9O9//1tjx47V/v37NXz4cPXq1Uuff/65Hn74Yb333nu+ExXp6em67bbbNGDAAH3yySfq2LGj/vOf/5x0TSfy0UcfafLkyXr55ZcVGhp6TG3p6em64YYblJycrP/85z968skn9emnn+qFF17w7eO3335TcHCwPv74Y11++eV67LHHZLFY9J///EfvvvuuMjIyNGHChGqpFwD8HV2aAaAOGjBggF544QU98sgjksq6hh59JfSHH37Qjh07NGPGDMXExEiS/vnPf6pfv3768ccfVVpaqszMTH3wwQcKDw9XcnKytm3bpueff16S9NNPP2njxo16++23FRAQIEmaNGmSzjvvPG3atElOp7PK2jIzM2WxWNSsWTM1a9ZM119/vRITEyuF8eMd08033yyXyyWHw1HlMe3cuVNffvmlvvvuO98xPffcc+rWrZu+//57de/eXVdffbWuvPJKhYeHS5L+3//7f/ryyy+1f/9+NW3aVJJ0ySWX6JJLLpEkjRo1SvPmzVNGRoZatmx5TF2fffaZ2rRpo4SEBElSr1699PHHH+v222+XxfL7eeNbbrlFF110kSRpwoQJOu+88/S///1PERERKikpUdOmTX2fSVRUlBITE5WXl6c5c+bohRde0Pnnny9Jeuyxx/Trr79q1qxZxz25UJWAgACFhIRIkq/b7Guvvabhw4f7jjUhIUHr1q3TjBkz1Llz52P2kZycrGeeeUZdu3aVJDVv3lxvvPGGtmzZ4vtZBAYGavz48QoICFBSUpLef/99rVu3ThdeeKHef/99DR061Pd+Tz/9tJYsWXLcmiMjIzV79my9/vrrmjdvnl566SW99NJLatmypZ599lmlpqZqyZIlysrK0ocffqjIyEi1bt1a27Zt00svvXRSn0thYaFuueUWDRs2zHcLwP/93//prrvuqrTeTTfd5Pv5T548WR06dNCoUaMkSYmJibrnnnv01FNPacSIEfrPf/6jlJQU3XHHHZKkESNGaOnSpcc9aVLu6quvrtRmJCkuLk5ffvmlb3rgwIFKTU2VJF9Pgoq1TZw4UfHx8Ro3bpwMw1CrVq300EMPaezYsb56JOn222/3tYOMjAy1a9dOzZo1U0BAgF544QXTB4EDgNpC4AWAOqh379564IEHtH79eiUlJen777/Xgw8+WClcbN68WXFxcb5gKEmxsbFq1qyZNm/erNLSUsXFxfmCoSR16NDB93rLli1yuVxVjvqcnp6uM888s8raevXqpbPOOkuXXXaZWrdurd69e2vIkCG+q1XH06lTJ4WGhmrJkiW64IILNH/+fE2cOLFSiCjvcl0eLMu5XC6lp6erT58+uu666/TFF19o7dq1Sk9P12+//SZJlbrdVgy2jRo1klTWnfdo+fn5WrRoUaV7Sfv376958+bpxx9/VM+ePSvVXy4yMtL3Od90000aPHiwhg8frvj4ePXu3VuXXXaZoqKitGrVKrnd7krh02KxqHPnztq8efMJP6+TsXnzZq1atarSAFAlJSVVBntJ6tevn3766Sc9++yz2rZtmzZu3KiMjAwNGDDAt05cXJzvBIhU9vmVd6ctP95yDofD1839eJo0aaKHHnpIDz30kLZs2aLFixdr5syZGjFihL755htt2bJFzZo1q3TC5M8M8tS4cWNdffXVmjVrljZs2KBt27Zp/fr1lbonS1J8fLzv9ZYtW/Tzzz9X+rl4PB4VFhbq4MGD2rx58zHtPzU1VcuWLTthLZMnT1ZiYmKleRXvw5fkO7FyvNo2b96sjh07VhogrkuXLiopKdH27dsllX3uFe8VHj16tO6++27NmzdPPXr0UP/+/TVo0KAT1goA9QWBFwDqoEaNGqlHjx5asGCBzjzzTLVu3bpSsJXK7qutitfrPe7gRna73fe6PBBXvJeyXOPGjZWTk1PlPoKCgjRjxgwtX75cCxcu1KJFi/T222/rtddeU48ePY57TIZhqF+/flqwYIHi4+PlcrnUqVMn/fzzz7513G637Ha7Pv7442NGhA4LC1NBQYGuueYaBQYGqn///urbt6+cTucxj4upeJwVP5ejzZ07V0VFRZo6dapeeeWVSstmz55dKfBardZKyz0ej+x2uwzD0KRJkzR8+HB98803Wrx4sd577z394x//UJs2bar8LLxe7zGBrPwzOlpVQb2c2+3Wvffeqz59+lSaf3TIKjdlyhS9/fbbuvLKKzVgwACNGTNG99xzT6V1qvrsKtZ39Od4ovWnTZumlJQU39Xt5ORkJScnq3fv3rr44ou1du1aORyOY7arGLir+kwqntzYu3evhgwZolatWum8887T4MGDtX//fo0ZM6bSNhV/X0pLSzVgwIBjrgJL8l1F/zPHWa5p06a+rtrHU9XvbcV5VS0vr6W8zQQGBlZa3q9fPy1atEjffPONvv/+ez322GP67LPPqnwUGADUN9zDCwB11IABA7RgwQLNnz+/0hW4cq1atVJmZmale0H37NmjzMxMJSUlKSUlRRkZGcrOzvYtL78aWr793r17FRwcrBYtWqhFixay2+2aMGGCDhw4cNy6li5dqldffVVdunTRmDFj9MUXX6hdu3Yn9TifAQMGaOHChZo7d64GDBhwTJhJSkpSSUmJXC6Xr6YmTZr4rkj+/PPP2rZtm9577z2NHDlSvXv39h3fqYxg/Nlnn/nu/fzkk098XwMHDtSCBQuUm5vrW3fjxo2+13v37tXu3buVnJysDRs26JlnnlHbtm11++2366OPPtLAgQP15Zdf+j7TivfUer1erVq1SklJScfUY7fbKw2OdPjw4Uo/i6M/r1atWikjI8P3WbVo0UJz586t1IW2olmzZun+++/XAw88oMsvv1zNmzdXZmbmSX92Z5xxRqUBt4qLi094pXrFihWVBvAqV37VvbwLc0ZGhvbt2+dbvnbtWt/r8qBZ8XPZuXOn7/X8+fMVEBCgmTNnavjw4TrnnHOUlZUl6fhtolWrVkpLS6v0uW3dulVTp06VxWJRSkqKVq9eXWmbir87NalVq1ZatWpVpdqXL18uu91e5dVhr9eriRMnau/evbr66qt99wcvWbKEgasANAgEXgCoo/r166fNmzfr66+/Vv/+/Y9Zfs455yglJUX33HOP1q5dq7Vr1+ree+9VYmKievTooR49eigpKUn333+/Nm7cqMWLF1fq+nruueeqdevWuvvuu7V27Vpt2LBB9913n3bu3KlmzZodty6Hw6GpU6fqww8/VEZGhr7//nulpaVV6i59PF27dpXX69Xbb79dZYhPSkpS3759NXbsWC1btkxbt27V/fff7wuI4eHhKikp0X//+19lZGRo/vz5+sc//iFJf3h/5dGysrK0bNkyXXfddTrjjDMqfd1yyy0qLi7W559/7lv/5Zdf1sKFC7Vx40bdf//9ateunbp166awsDC9//77mjp1qnbu3Klff/1VK1euVIcOHeRwOPTXv/5VzzzzjBYvXqytW7fqySef1M6dO3XNNdccU1OHDh20dOlSLViwQGlpaXrkkUcq3RPqdDpVUFCgLVu2qKioSLfccov+/e9/6/3339eOHTv04YcfasqUKcf9+YWHh2vx4sXavn271q1bpzvvvFO5ubkn/dndfPPN+vDDD/Xxxx8rLS1Njz/++AlPjvz973/XqlWrdNddd2n58uXauXOnvv/+e40ePVq9e/dW69at1a1bN7Vv31733XefNmzYoMWLF2vq1Km+fURFRalp06a+QdXmz59faVCx8PBwZWdna9GiRdq1a5fmzJmjV199VdLx28T111+v7du366mnnlJaWpp+/PFHPfroo3I4HLJYLLr22mu1fft2Pfvss0pPT9e77757wtGvy+Xk5Gjfvn3HfP2Z+2n/7//+T7t27dKTTz6prVu3avHixZo4caIuv/xy39XnigzD0NatW/XEE0/ot99+0/bt2/Xll1+qWbNmioiIOOn3BYC6isALAHVUZGSkzjrrLCUmJla6x6+cYRh65ZVXFBkZqRtuuEE333yzoqOjNXPmTAUEBMhms+n111+XzWbTtddeq3/84x8aNmyYb3uLxaJXX31V4eHhuvHGG/XXv/5V4eHhmjZt2jHddytKTU3V008/rRkzZuiiiy7SuHHjNGzYMF155ZV/eEw2m019+vRRYGDgce/TnDhxotq3b69Ro0bpqquu0qFDh/Tmm28qNDRUnTp10t13363nn39el1xyiaZOnar7779fYWFhJxwtuCqff/65goKCqrzXsX379urcubNmz57tmzdy5EhNmDBBV199tQICAjRlyhRJZd1Yp0yZogULFmjQoEG68847dcEFF2jkyJGSpHvuuUcXX3yxHnzwQQ0ZMkSbNm3S22+/fcy9npJ02WWX6YorrtDYsWN1/fXXq23btjrrrLN8y3v06KF27drp8ssv16JFi9S/f3+NGzdOM2fO1MUXX6w33nhD48eP9w0qdbQJEyZo165dGjx4sEaNGqXmzZvrqquuOunPbuDAgXr44Yc1ZcoUDRkyRBaLRd27dz/u+qmpqZo1a5aKioo0atQoDRw4UOPHj1fXrl31z3/+07fe5MmT5XA4dM011+iZZ57Rdddd51tmsVj0zDPPaPfu3br44os1c+bMSoN9DRw4UNdcc40eeOABXXrppfrggw/01FNPyTCMSleKK4qNjdX06dO1du1aXXbZZRo7dqwuvvhiPfzww5LKBvOaPn26fvrpJ1166aX6+uuvNXTo0D/8fG644Qb17NnzmK8/07U4JiZG06dP12+//abLLrtMjzzyiC6//HI9+uijx93m6aefVlRUlIYNG6ZLL73UN3L70QNoAUB9ZHjr2lPqAQBAg7Zw4UKNHDmyUjdyAACqwqk9AAAAAEC9ROAFAAAAANRLdGkGAAAAANRLXOEFAAAAANRLBF4AAAAAQL1E4AUAAAAA1Es2swuoDZmZmWaXcEJRUVHKzs42uwxAEu0R/oX2CH9Ce4Q/oT3Cn5jdHuPi4o67jCu8AAAAAIB6icALAAAAAKiXCLwAAAAAgHqpQdzDezSv16vCwkJ5PB4ZhmF2OdqzZ4+KiorMLqPB8Hq9slgsCgoK8oufPwAAAICa0SADb2Fhoex2u2w2/zh8m80mq9VqdhkNSmlpqQoLC+VwOMwuBQAAAEANaZBdmj0ej9+EXZjDZrPJ4/GYXQYAAACAGtQgAy/dWCHRDgAAAID6rkEGXrOtXLlSffr00bfffltp/vDhwzVhwoQaec/s7GxddNFFWrRokW/eV199pWnTpp3yPrOysnTbbbedcJ2PP/74pPfXv39/3XXXXZW+9u3bd8r1AQAAAGjY6NdrkoSEBH377bfq27evJCktLU2FhYU19n5fffWVrrzySn3yySfq3bt3jb3P0d555x1dccUVJ7VuSEiI/vnPf9ZsQQAAAAAajAYfeD0/fivvkgXVuk/j3AtkOafvCddp1aqVdu3apfz8fIWHh2v+/Pnq16+f9u7dK0latGiRPvroI1ksFnXo0EEjRozQvn379OKLL6q4uFh5eXm68cYb1bNnTw0fPlwdO3bU1q1bZRiGnnrqKTVq1Mj3Xl6vV/PmzdPkyZO1atUqpaenq2XLlpKkdevW6Z577lFBQYFuuukm9ejRQ9OnT9eKFSvk8XjUr18/XXXVVbrrrrt0zz33KCEhQZ999pkOHDigiy66yPcey5Yt0xtvvKGAgACFhYVp7Nix+uSTT3To0CG9+OKLuuOOO/TCCy8oIyNDHo9Hw4cPV6dOnU7q85w5c6bWrVsnl8ul++67T48//rhCQ0PVrVs3nX322Zo8ebIsFosCAgI0ZswYxcTEaM6cOfrmm29kGIb69OmjK6+88k/+FAEAAADUdQ0+8JqpV69e+v777zVo0CBt2LBB1113nb799lvl5eVp5syZ+te//qWgoCD94x//0LJly2QYhq655hp16tRJa9eu1cyZM9WzZ08dPnxYffv21Z133qmnnnpKP//8s+/KsSQtX75cSUlJCg8P18CBA/XJJ5/o7rvvliQ5HA4988wzysnJ0W233aZu3bpp3rx5eumll9S4cWN99dVXf3gcXq9Xzz//vCZPnqwmTZroP//5j9555x39/e9/15w5c3T33Xfr008/9QXh3NxcjR49WjNnzqy0n0OHDumuu+7yTUdFRemRRx6RVHZF/I477lBWVpYOHDig1157TXa7XX/729903333KTk5WT/88INeeeUVDRs2TAsXLtTkyZNlGIbGjBmjv/zlL0pISDj9HxoAAACAOqPBB17LOX2lP7gaW1P69eunF198Uc2bN1eHDh188zMyMpSTk6MHHnhAklRQUKDMzEx16NBB7777rv773/9KKnu0TrnWrVtLkqKjo1VcXFzpfb744gvt3r1bY8eOVWlpqbZs2aJbb71VktS+fXsZhqGIiAgFBwcrLy9P48aN0+uvv64DBw6oW7dux9Tt9XorTefm5srpdKpJkyaSpI4dO2r69OmV1klLS9OaNWu0fv16SWUjZefm5iosLMy3zom6NMfHx/teN23aVHa7XZK0f/9+JScn+9739ddfV3p6uvbs2aN7771XUlmQzsjIIPACAAAADUyDD7xmiouLU2FhoWbPnq3hw4dr9+7dksoCXXR0tCZNmiSbzaavvvpKycnJmjFjhi655BJ169ZNc+fOrXT19XgjDufm5mr9+vWaNWuW71m/kyZN0tdff63g4GBt3LhRknTgwAG5XC45nU4tXrxY48aNk9fr1bBhw9SnTx8FBARo//79SkhI0ObNmxUVFeV7j7CwMBUUFGj//v1q3LixVq1apebNm1eqIyEhQU2aNNFf//pXFRUV6d1331VISMhJf1YWy+/jq1U81saNG2vr1q1q1aqV733j4+OVmJioiRMnyjAMffTRR0pKSjrp9wIAAABQPxB4TdanTx8tWLBA8fHxvsAbHh6uq6++WnfddZfcbrdiY2PVu3dvnX/++ZoyZYpmzZqlJk2aKDc39w/3//XXX6tXr16+sCtJl1xyiZ555hldd911Kioq0j333COXy6V77rlHAQEBCgkJ0S233KKQkBCdffbZiomJ0ZAhQ/TSSy+pSZMmlcKuJF+34UcffVSGYSgkJMR3dbpFixZ6+umndd9992nSpEkaPXq0CgoKdNlll1UKsdKxXZol+a5EH8+YMWM0efJkeb1eWa1W3XfffYqLi9NZZ52lO+64QyUlJWrTps0xNQMAAACo/wzv0f1T66HMzMxK0wUFBXI6nSZVcyybzVapezJqh7+1A38RFRWl7Oxss8sAJNEe4V9oj/AntEf4E7PbY1xc3HGX8RxeAAAAAEC9ROAFAAAAANRLBF4AAAAAQL1E4AUAAAAA1EsEXpPtPVyi/YeL/3hFAAAAAMCfUqOPJdq8ebNmzZqlxx57TP/85z+Vk5MjSdq3b59at26tu+66S2+++aY2btwoh8MhSRo7dqxsNpsmT56svLw8ORwOjRo1SqGhodq0aZNmzpwpq9Wq1NRUXX311TVZfq0o9XhV6CpRWGCg2aUAAAAAQL1SY4H3008/1XfffaegoCBJ8j1fNT8/X48//rhuuukmSVJ6eroefvhhhYaG+rb94osvlJCQoGuuuUZLlizR7NmzNWzYML3++uu69957FRMTowkTJigtLU1JSUk1dQg1ZuXKlbr77rs1btw4de5xnvYXl6jU49Xfbr1FrVu39j3Dtrp4PB69+uqrSk9Pl2EYstvtuv322084fHdNuu222/Too48qNjbWN2/ChAnavHmzQkJCfPP69++vSy65xIwSAQAAANQDNRZ4Y2JiNGbMGL388suV5n/44YcaOHCgIiIi5PF4lJWVpWnTpik3N1d9+vRR3759tWHDBl166aWSpM6dO2v27NkqKChQaWmpLyR17NhRa9eurZOBV5ISEhL07bff6pzzekuS1m/aosLCwhp5r59//ln79+/XpEmTJEk//PCDpk6dqqeffrpG3u9U/e1vf1PXrl3NLgMAAABAPVFjgbd79+7au3dvpXm5ublau3atbr75ZklSUVGRLrroIg0aNEgej0ePP/64WrVqJZfLJafTKUkKCgpSQUGBXC6Xr9tz+fyj9388UVFRlab37Nkjm63s0L/ZelDzNx881cOs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\n",
      "text/plain": [
       "<Figure size 1152x648 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAA7wAAAImCAYAAABwyYamAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjUuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/YYfK9AAAACXBIWXMAAAsTAAALEwEAmpwYAAB+x0lEQVR4nOzdd3hUVeLG8e+Zkt4TSAhJgNBCr9IElCLFuqK4umvDtioqrn3XRbCw4q5dseFPUbEv9gaCCIgCIh2khh4ChJLeJvf+/ghGsGCAJHeSvJ/nyWPmzJ2Zd+A8z+7LufdcY9u2jYiIiIiIiEgd43I6gIiIiIiIiEh1UOEVERERERGROkmFV0REREREROokFV4RERERERGpk1R4RUREREREpE5S4RUREREREZE6SYVXRESkigwcOJDWrVszffr0Xz138OBB2rVrR8+ePY/rvWfPnk3r1q0rdeyOHTto3bo169ev/91jWrduzezZs48ri4iISG2hwisiIlKFvF4vM2fO/NX4V199RVlZmQOJRERE6i8VXhERkSrUo0cP5syZg8/nO2J8xowZdO7c2ZlQIiIi9ZQKr4iISBU6+eSTKS0t5fvvv68Yy8/PZ+HChQwePLhibM+ePdx+++307t2brl27MmbMGPbs2VPx/NatW7n88svp1KkT55xzDhs3bjzic/bu3cuYMWPo0qULffv25e677yY3N/e4Mtu2zauvvsrQoUPp0KED55xzDnPmzKl4fsOGDVxyySV06dKFXr16cffdd1NQUADA7t27ufbaa+nWrRvdu3fnpptuYt++fceVQ0REpKqp8IqIiFShwMBA+vXrx6xZsyrG5syZQ/v27YmJiQGgtLSUyy+/nF27dvHCCy/wyiuvsHv3bkaPHo1t25SWlnLNNdcQHBzMtGnTGD16NJMnTz7ic2688UZs2+btt9/m2WefZdu2bfz9738/rszPPfccTz31FDfddBMfffQRgwcP5rrrrmPt2rUA3HrrrTRr1oyPPvqI5557ju+++44XXngBgPHjx+Nyufjf//7H1KlT2blzJxMnTjyuHCIiIlXN43QAERGRumbIkCE8+uij/Otf/wLKT2ceOnRoxfPffPMN27Zt4+WXXyY+Ph6Axx9/nEGDBvHtt9/i8/nIyMjg7bffJioqihYtWrBlyxYeeeQRABYsWMC6det49dVXCQgIAODhhx+mf//+rF+/npCQkEpntW2bV155hWuvvZYzzjgDKC/Ty5cvZ/LkyTzyyCPs3LmTgQMHkpiYSHJyMs8++yxerxeAnTt30rZtWxo3bkxAQACPPvoo+fn5J/6HKCIiUgW0wisiIlLFTj31VPbs2cOPP/5IcXEx8+bN47TTTqt4fsOGDSQmJlaUXYCEhAQaN27Mhg0bKp6PioqqeL5Dhw4Vv2/cuJHCwkJ69uxJly5d6NKlC8OGDQNg8+bNR2RZvHhxxTFdunThnnvuOeL5ffv2ceDAgV9dX9ytW7eK06hvu+02nn/+efr06cOtt97K9u3bSU1NBWDMmDF89tln9OrVi9GjR7N06VJatWp1/H94IiIiVUgrvCIiIlUsLCyM3r17M3PmTNq1a0fLli2PKLdBQUG/+TrbtrFt+zef+2lFFcDn85GYmMjLL7/8q+NiY2M5ePBgxeP27dvzwQcfHJHtcEfLYlkWABdddBEDBgzgyy+/ZN68edx888386U9/4oEHHmDQoEF8/fXXzJo1i3nz5jF+/Hg++ugjXnrppd98XxERkZqkFV4REZFqMGTIEGbOnMmXX37JkCFDjniuefPmZGRkHLFJ1e7du8nIyCA1NZXWrVuzc+dOsrKyKp5fs2bNEa/fs2cPoaGhNGnShCZNmuD1epk4cSL79+8/4rOCgoIqjmnSpAmxsbFHPB8WFkbDhg1ZtmzZEeNLly4lNTWVvLw87rvvPowxXHLJJbzwwguMHTuWjz/+GNu2eeihh9izZw8jR47kySef5Omnn2b+/PnauEpERPyCCq+IiEg1GDRoEBs2bGD69OlHnM4M0KdPH1q3bs0tt9zCqlWrWLVqFbfeeitNmzald+/e9O7dm9TUVO68807WrVvHnDlzKjaJgvKdoFu2bMnf//53Vq1axdq1a7n99tvZvn07jRs3Puas11xzDc899xyffvopW7Zs4ZlnnuGbb77hkksuISwsjEWLFnH//fezceNGNm7cyMyZM+nQoQPGGDZt2sR9993HmjVr2Lp1K59++imNGzcmOjr6hP8MRURETpQKr4iISDWIiYmha9euNG3alOTk5COeM8bwzDPPEBMTwyWXXMLll19Ow4YNmTJlCgEBAXg8HiZPnozH4+HPf/4z//73vxk1alTF610uF88++yxRUVFceumlXHzxxURFRfHCCy/gdruPOevFF1/MVVddxX//+1/OOussZs2axXPPPUf37t0BeOqppygsLOTCCy9k5MiRBAYGVmygNWHCBOLi4hg1ahRnn312xc7TLpf+L4aIiDjP2L93sZCIiIiIiIhILaZ/fhUREREREZE6SYVXRERERERE6iQVXhEREREREamTVHhFRERERESkTlLhFRERERERkTpJhVdERERERETqJI/TAWpCRkaG0xGOKi4ujqysLKdjiACaj+JfNB/Fn2g+ij/RfBR/4vR8TExM/N3ntMIrIiIiIiIidZIKr4iIiIiIiNRJKrwiIiIiIiJSJ9WLa3h/ybZtioqKsCwLY4zTcdi9ezfFxcVOx6gxtm3jcrkICgryiz9/ERERERGpm+pl4S0qKsLr9eLx+MfX93g8uN1up2PUKJ/PR1FREcHBwU5HERERERGROqpentJsWZbflN36yuPxYFmW0zFERERERKQOq5eFV6fR+gf9PYiIiIiISHXSMqcDnnnmGdavX8/+/fspLi4mMTGRyMhIxo8f/4evfeONN+jSpQtt2rT5zeeffvppRo4cSXx8/HFl++KLL3j55Zdp1KhRxVhqaio33XTTcb2fiIiIiIiIU1R4HXD99dcD5eVy27ZtXH/99fh8vkq99i9/+ctRn7/hhhtOON+gQYO45pprTvh9REREREREnFTvC6/17VfY82dW6Xuakwfj6jPwmF83ceJEcnJyyMnJYcKECbzwwgvs2bOHnJwcevbsyRVXXMHEiRMZOHAg+/fvZ8GCBRQXF5ORkcFFF13EsGHDuPnmm7nlllv46quv2LVrFwcPHmT37t1cf/319OjRg++++46XX36Z0NBQwsPDSU1N5fLLL69UvlGjRpGcnIzX6yU5OZnVq1dTWFjI7bffzsKFC/nqq69wu9107NiRv/3tb0yZMuWIY5o0aXLMfyYiIiIiIiLHq94XXn/TpUsXRo4cSWZmJm3btuX222+npKSEkSNHcsUVVxxxbH5+Pv/973/ZsWMH//znPxk2bNgRz3u9Xh566CEWL17MO++8Q7du3Xjqqad4+umniYmJ4YEHHvjNDLNmzWLNmjUVj08//XSGDBlCYWEhl1xyCS1btmTKlCmkpKRw4403kp6ezuzZs3n66adxu92MGzeO7777DqDiGBERERERkZpW7wuvq89AOI7V2OqSnJwMQHh4OGvXrmXp0qWE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REREpHZT4fUDiZFBNIkKZOGO3Iox06odoOt4RUREREREjpcKr5/omRTGj3sLySkqP4XZREZDQmPdj1dEREREROQ4qfD6iZ5J4Vg2fL/zsM2rWraDjWuwrTIHk4mIiIiIiNROKrx+onlMILEhHhYefh1vq/ZQWADbtziWS0REREREpLZS4fUTxhh6JoWxdFc+xT6rfOyn63g36DpeERERERGRY6XC60d6JoVTUmazLDMfABPTAOLisdfpOl4REREREZFjpcLrR9o1DCHE6zri9kQmrSOsW4FdWupgMhERERERkdpHhdePeN2GbomhfL8jjzLLBsB07V1+He+apQ6nExERERERqV1UeP1Mz6RwsovLWJ9VWD7QphOEhGIvnu9sMBERERERkVpGhdfPdGsciscFCw6d1mw8XkyXXtjLF+q0ZhERERERkWOgwutnQrxu2seHsmhHLrZ96LTmbn11WrOIiIiIiMgxUuH1Q72SwsjILWVHTkn5QJuOEBKGvfgbZ4OJiIiIiIjUIiq8fqhHUhgAC484rbkn9vJF2KUlTkYTERERERGpNVR4/VBsiJcWMUEs3J5bMWa6HzqtebVOaxYREREREakMFV4/1TM5jPX7ithf6CsfSOtUflrzD9qtWUREREREpDJUeP1Uz6RwAL6vOK3ZU75b87KFOq1ZRERERESkElR4/VRKZAAJYV4W7jj8tOaToahQpzWLiIiIiIhUggqvnzLG0DMpjOWZBRSUlpUP6rRmERERERGRSlPh9WM9k8LxWTZLd+UDOq1ZRERERETkWKjw+rG0BsGEB7pZtD2vYsx076vTmkVERERERCpBhdePuV2GHo3DWLQzj5Iyq3wwrSOEhmMv/sbZcCIiIiIiIn5OhdfP9W8aQUGpxQ87f3Fa8/JFOq1ZRERERETkKFR4/VyH+BCigtzM2ZJdMfbzac1LHEwmIiIiIiLi31R4/ZzbZejXJILvd+aTV3Jot+bWHQ6d1qzdmkVERERERH6PCm8tcEqzCHyWzXfbyu/JazweTNfeOq1ZRERERETkKFR4a4EWMUE0Cvcyd0tOxZjpdrJOaxYRERERETkKFd5awBjDKU0jWLm7gH0FpeWDrTtAWDj29zqtWURERERE5Leo8NYSpzSNxAbmbS1f5S3frfnQac0lxc6GExERERER8UMqvLVEYkQALWODmLP5sNOa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IJVTUFrGDZ9sJszr5pHhTfG6DfaqJVhPjMecdRGusy9yOmKV0HwUf6L5KP5E81H8ieaj+BOn5+PRTmmu1C7NWVlZ/OMf/6BLly4MGzaM+++/3+9LpEhVC/G6ue6kBLZmFzNtzT4ATPuumB79sT9/F3vXDocTioiIiIjI4SpVeKOiogBISEhg+/btxMTE4PP5qjOXiF86KSmMfk3CeXdVFtuyiwEwf74SAgKxpj5DJU6YEBERERGRGlKpwhsREcFHH31EixYtmD17NosXL6akpKS6s4n4pau6xxPscfH0gkzKLBsTEY05fxSsX4X9zZdOxxMRERERkUMqVXivueYaPB4PaWlppKam8s477/DXv/61urOJ+KWoIA9XdY9nXVYhn284AIA5eTC07oD97svYB/Y5nFBERERERKCShTcyMpLBgwezbds2/vKXv3D//ffTo0eP6s4m4rdOaRpBt8RQXlu2lz15pRiXC9elN0CZD+u1STq1WURERETED1Sq8K5fv54bb7yRBx98kAMHDnDdddexbt266s4m4reMMVx7UgIAzy7KxLZtTMNGmHMvgZWLsRd87WxAERERERGpXOGdOnUqY8eOJTw8nNjYWG644QamTJlSzdFE/FvDMC+XdG7Akl35zNmSA4AZeAY0T8N+azJ29gGHE4qIiIiI1G+VKrzFxcUkJSVVPO7atStlZWXVFkqkthjeMprWccG8uHg3B4t8GJcb1+U3QWkJ1uvP6tRmEREREREHVarwejwe8vLyMMYA6B68Ioe4XYYbeiVQ6LN5cfFuAExCEuacv8DSBdiLv3E4oYiIiIhI/VWpwnvuuecyfvx4srKyePzxx/nXv/7FiBEjqjubSK2QEhnIyPaxzNuay6IduQCY086BZq2w33geOzfb4YQiIiIiIvXTUQtvXl4eeXl5pKWlcd111zF8+HBSUlK44447aN++fU1lFPF757WNpWlUIM8szCSnuKz81ObLboKiAuw3X3A6noiIiIhIveQ52pNXXnnlUV/89ttvV2kYkdrK6zaM6d2I277YwuTvd3Nr30RM4xTMmRdifzAVu3tfTNfeTscUEREREalXjlp4+/fvz/r16+nevTsDBgw4YuMqETlSakwQf+4QxxsrsuidEkaflAjM0BHYS77Fev1ZXK3aYcIinI4pIiIiIlJvHPWU5tGjR/Of//yHJk2a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\n",
      "text/plain": [
       "<Figure size 1152x648 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=(16,9))\n",
    "plt.plot(model.history.history['mae'])\n",
    "plt.plot(model.history.history['mse'])\n",
    "plt.title(\"Model's Mean Absolute and Squared Errors\")\n",
    "plt.xlabel('Epoch')\n",
    "plt.ylabel('Error')\n",
    "plt.legend(['Mean Absulote Erroe', 'Mean Squared Error'],loc = 'upper left')\n",
    "plt.show()\n",
    "#summarize history for loss\n",
    "plt.figure(figsize=(16,9))\n",
    "plt.plot(model.history.history['loss'])\n",
    "plt.plot(model.history.history['val_loss'])\n",
    "plt.title('Model-loss')\n",
    "plt.xlabel('Epoch')\n",
    "plt.ylabel('Mean-Absolute-Error')\n",
    "plt.legend(['Training Error', 'Testing Error'],loc='upper left')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 7.2 Predicting values from Model using same dataset<a class=\"anchor\" id=\"7.2\"></a>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 164,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "94/94 [==============================] - 0s 2ms/step\n",
      "(3008, 9)\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>CDP</th>\n",
       "      <th>GTEP</th>\n",
       "      <th>TIT</th>\n",
       "      <th>TAT</th>\n",
       "      <th>AFDP</th>\n",
       "      <th>CO</th>\n",
       "      <th>AT</th>\n",
       "      <th>Actual</th>\n",
       "      <th>Predicted</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>13312</th>\n",
       "      <td>12.219</td>\n",
       "      <td>25.762</td>\n",
       "      <td>1092.5</td>\n",
       "      <td>550.25</td>\n",
       "      <td>4.0023</td>\n",
       "      <td>1.26430</td>\n",
       "      <td>24.0930</td>\n",
       "      <td>134.46</td>\n",
       "      <td>134.671204</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>12627</th>\n",
       "      <td>10.791</td>\n",
       "      <td>20.085</td>\n",
       "      <td>1059.6</td>\n",
       "      <td>549.94</td>\n",
       "      <td>3.2106</td>\n",
       "      <td>2.69370</td>\n",
       "      <td>20.4500</td>\n",
       "      <td>111.88</td>\n",
       "      <td>112.273460</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6393</th>\n",
       "      <td>12.126</td>\n",
       "      <td>25.221</td>\n",
       "      <td>1089.9</td>\n",
       "      <td>549.62</td>\n",
       "      <td>4.5325</td>\n",
       "      <td>1.96250</td>\n",
       "      <td>20.2620</td>\n",
       "      <td>133.72</td>\n",
       "      <td>134.256805</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4990</th>\n",
       "      <td>12.218</td>\n",
       "      <td>25.965</td>\n",
       "      <td>1092.9</td>\n",
       "      <td>549.96</td>\n",
       "      <td>4.4266</td>\n",
       "      <td>1.57120</td>\n",
       "      <td>26.8620</td>\n",
       "      <td>133.79</td>\n",
       "      <td>134.045410</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>12462</th>\n",
       "      <td>10.466</td>\n",
       "      <td>19.688</td>\n",
       "      <td>1056.9</td>\n",
       "      <td>550.01</td>\n",
       "      <td>3.1241</td>\n",
       "      <td>2.29960</td>\n",
       "      <td>19.4090</td>\n",
       "      <td>110.77</td>\n",
       "      <td>111.272774</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7405</th>\n",
       "      <td>10.624</td>\n",
       "      <td>19.387</td>\n",
       "      <td>1058.9</td>\n",
       "      <td>550.17</td>\n",
       "      <td>3.3709</td>\n",
       "      <td>4.27640</td>\n",
       "      <td>2.2158</td>\n",
       "      <td>113.32</td>\n",
       "      <td>113.671730</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>10993</th>\n",
       "      <td>12.088</td>\n",
       "      <td>25.392</td>\n",
       "      <td>1089.7</td>\n",
       "      <td>550.11</td>\n",
       "      <td>3.7871</td>\n",
       "      <td>0.83578</td>\n",
       "      <td>23.8520</td>\n",
       "      <td>133.77</td>\n",
       "      <td>133.790131</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9488</th>\n",
       "      <td>11.498</td>\n",
       "      <td>23.225</td>\n",
       "      <td>1079.4</td>\n",
       "      <td>549.60</td>\n",
       "      <td>4.2837</td>\n",
       "      <td>2.01980</td>\n",
       "      <td>12.3950</td>\n",
       "      <td>128.98</td>\n",
       "      <td>129.272095</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>14201</th>\n",
       "      <td>13.971</td>\n",
       "      <td>32.518</td>\n",
       "      <td>1100.1</td>\n",
       "      <td>528.98</td>\n",
       "      <td>5.1559</td>\n",
       "      <td>0.87760</td>\n",
       "      <td>12.3590</td>\n",
       "      <td>159.42</td>\n",
       "      <td>160.486389</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9757</th>\n",
       "      <td>13.862</td>\n",
       "      <td>32.105</td>\n",
       "      <td>1100.0</td>\n",
       "      <td>530.69</td>\n",
       "      <td>5.9309</td>\n",
       "      <td>10.75000</td>\n",
       "      <td>8.6376</td>\n",
       "      <td>161.86</td>\n",
       "      <td>161.446960</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "          CDP    GTEP     TIT     TAT    AFDP        CO       AT  Actual  \\\n",
       "13312  12.219  25.762  1092.5  550.25  4.0023   1.26430  24.0930  134.46   \n",
       "12627  10.791  20.085  1059.6  549.94  3.2106   2.69370  20.4500  111.88   \n",
       "6393   12.126  25.221  1089.9  549.62  4.5325   1.96250  20.2620  133.72   \n",
       "4990   12.218  25.965  1092.9  549.96  4.4266   1.57120  26.8620  133.79   \n",
       "12462  10.466  19.688  1056.9  550.01  3.1241   2.29960  19.4090  110.77   \n",
       "7405   10.624  19.387  1058.9  550.17  3.3709   4.27640   2.2158  113.32   \n",
       "10993  12.088  25.392  1089.7  550.11  3.7871   0.83578  23.8520  133.77   \n",
       "9488   11.498  23.225  1079.4  549.60  4.2837   2.01980  12.3950  128.98   \n",
       "14201  13.971  32.518  1100.1  528.98  5.1559   0.87760  12.3590  159.42   \n",
       "9757   13.862  32.105  1100.0  530.69  5.9309  10.75000   8.6376  161.86   \n",
       "\n",
       "        Predicted  \n",
       "13312  134.671204  \n",
       "12627  112.273460  \n",
       "6393   134.256805  \n",
       "4990   134.045410  \n",
       "12462  111.272774  \n",
       "7405   113.671730  \n",
       "10993  133.790131  \n",
       "9488   129.272095  \n",
       "14201  160.486389  \n",
       "9757   161.446960  "
      ]
     },
     "execution_count": 164,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# generating predictions for test data\n",
    "y_predict_test = model.predict(x_test_scaled) \n",
    "\n",
    "# creating table with test price & predicted price for test\n",
    "predictions_df = pd.DataFrame(x_test)\n",
    "predictions_df['Actual'] = y_test\n",
    "predictions_df['Predicted'] = y_predict_test\n",
    "print(predictions_df.shape)\n",
    "predictions_df.head(10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 192,
   "metadata": {},
   "outputs": [],
   "source": [
    "predictions_df.drop(['CDP','GTEP','TIT','TAT','AFDP','CO','AT'], axis =1 , inplace = True)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 7.3 Calculating Absolute Percent Error and Error<a class=\"anchor\" id=\"7.3\"></a>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 193,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "The Accuracy for Test Data -- ANN model =  99.64609758691824\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Actual</th>\n",
       "      <th>Predicted</th>\n",
       "      <th>Error</th>\n",
       "      <th>APE %</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>13312</th>\n",
       "      <td>134.46</td>\n",
       "      <td>134.671204</td>\n",
       "      <td>-0.001571</td>\n",
       "      <td>0.157075</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>12627</th>\n",
       "      <td>111.88</td>\n",
       "      <td>112.273460</td>\n",
       "      <td>-0.003517</td>\n",
       "      <td>0.351681</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6393</th>\n",
       "      <td>133.72</td>\n",
       "      <td>134.256805</td>\n",
       "      <td>-0.004014</td>\n",
       "      <td>0.401440</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4990</th>\n",
       "      <td>133.79</td>\n",
       "      <td>134.045410</td>\n",
       "      <td>-0.001909</td>\n",
       "      <td>0.190904</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>12462</th>\n",
       "      <td>110.77</td>\n",
       "      <td>111.272774</td>\n",
       "      <td>-0.004539</td>\n",
       "      <td>0.453890</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "       Actual   Predicted     Error     APE %\n",
       "13312  134.46  134.671204 -0.001571  0.157075\n",
       "12627  111.88  112.273460 -0.003517  0.351681\n",
       "6393   133.72  134.256805 -0.004014  0.401440\n",
       "4990   133.79  134.045410 -0.001909  0.190904\n",
       "12462  110.77  111.272774 -0.004539  0.453890"
      ]
     },
     "execution_count": 193,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Computing the absolute percent error\n",
    "APE=100*(abs(predictions_df['Actual']-predictions_df['Predicted'])/predictions_df['Actual'])\n",
    "print('The Accuracy for Test Data -- ANN model = ', 100-np.mean(APE))\n",
    "\n",
    "# adding absolute percent error to table\n",
    "predictions_df['APE %']=APE\n",
    "predictions_df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 194,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Actual</th>\n",
       "      <th>Predicted</th>\n",
       "      <th>Error</th>\n",
       "      <th>APE %</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>134.46</td>\n",
       "      <td>134.671204</td>\n",
       "      <td>-0.001571</td>\n",
       "      <td>0.157075</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>111.88</td>\n",
       "      <td>112.273460</td>\n",
       "      <td>-0.003517</td>\n",
       "      <td>0.351681</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>133.72</td>\n",
       "      <td>134.256805</td>\n",
       "      <td>-0.004014</td>\n",
       "      <td>0.401440</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>133.79</td>\n",
       "      <td>134.045410</td>\n",
       "      <td>-0.001909</td>\n",
       "      <td>0.190904</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>110.77</td>\n",
       "      <td>111.272774</td>\n",
       "      <td>-0.004539</td>\n",
       "      <td>0.453890</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3003</th>\n",
       "      <td>119.25</td>\n",
       "      <td>119.828529</td>\n",
       "      <td>-0.004851</td>\n",
       "      <td>0.485140</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3004</th>\n",
       "      <td>133.74</td>\n",
       "      <td>134.065872</td>\n",
       "      <td>-0.002437</td>\n",
       "      <td>0.243661</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3005</th>\n",
       "      <td>146.31</td>\n",
       "      <td>147.964294</td>\n",
       "      <td>-0.011307</td>\n",
       "      <td>1.130678</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3006</th>\n",
       "      <td>150.07</td>\n",
       "      <td>150.383911</td>\n",
       "      <td>-0.002092</td>\n",
       "      <td>0.209176</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3007</th>\n",
       "      <td>111.77</td>\n",
       "      <td>111.412468</td>\n",
       "      <td>0.003199</td>\n",
       "      <td>0.319882</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>3008 rows × 4 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "      Actual   Predicted     Error     APE %\n",
       "0     134.46  134.671204 -0.001571  0.157075\n",
       "1     111.88  112.273460 -0.003517  0.351681\n",
       "2     133.72  134.256805 -0.004014  0.401440\n",
       "3     133.79  134.045410 -0.001909  0.190904\n",
       "4     110.77  111.272774 -0.004539  0.453890\n",
       "...      ...         ...       ...       ...\n",
       "3003  119.25  119.828529 -0.004851  0.485140\n",
       "3004  133.74  134.065872 -0.002437  0.243661\n",
       "3005  146.31  147.964294 -0.011307  1.130678\n",
       "3006  150.07  150.383911 -0.002092  0.209176\n",
       "3007  111.77  111.412468  0.003199  0.319882\n",
       "\n",
       "[3008 rows x 4 columns]"
      ]
     },
     "execution_count": 194,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "predictions_df['Error'] = (predictions_df['Actual'] - predictions_df['Predicted'])/(predictions_df['Actual'])\n",
    "predictions_df.reset_index(drop = True)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "[Table of Contents](#0.1)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 7.4 Visualizing the Relationship between the Actual and Predicted ValuesModel Validation<a class=\"anchor\" id=\"7.4\"></a>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 177,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.collections.PathCollection at 0x14c66d86fa0>"
      ]
     },
     "execution_count": 177,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<Figure size 864x576 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=(12,8))\n",
    "plt.xlabel(\"Actual Values\")\n",
    "plt.ylabel(\"Predicted values\")\n",
    "plt.title(\"The Scatterplot of Relationship between Actual Values and Predictions\")\n",
    "plt.scatter(predictions_df['Actual'], predictions_df['Predicted'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 178,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "MAE: 0.4806557943465863\n",
      "MSE: 0.46965791965828263\n",
      "RMSE: 0.6853159268966997\n"
     ]
    }
   ],
   "source": [
    "# We will evaluate our model performance by calculating the residual sum of squares and the explained variance score\n",
    "from sklearn import metrics\n",
    "print(\"MAE:\",metrics.mean_absolute_error(y_test,y_predict_test))\n",
    "print (\"MSE:\",metrics.mean_squared_error(y_test,y_predict_test))\n",
    "print(\"RMSE:\",np.sqrt(metrics.mean_squared_error(y_test,y_predict_test)))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 198,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "94/94 [==============================] - 0s 2ms/step\n",
      "376/376 [==============================] - 1s 2ms/step\n",
      "R2_score (train):  0.9973425736949953\n",
      "R2_score (test):  0.9970628081081423\n"
     ]
    }
   ],
   "source": [
    "from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score\n",
    "y_predict_test = model.predict(x_test_scaled)\n",
    "y_predict_train = model.predict(x_train_scaled) \n",
    "print('R2_score (train): ',r2_score(y_train, y_predict_train))\n",
    "print('R2_score (test): ',r2_score(y_test, y_predict_test))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 203,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "This shows our model predict % 99.74 of the target correctly\n"
     ]
    }
   ],
   "source": [
    "#Evaluation of  the explained variance score (R^2)\n",
    "print('This shows our model predict % {} of the target correctly'.format(np.round(metrics.explained_variance_score(y_test,y_predict_test)*100,2))) "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "[Table of Contents](#0.1)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 7.5 Residual Analysis<a class=\"anchor\" id=\"7.5\"></a>\n",
    "+ Test for Normality of Residuals (Q-Q Plot)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 180,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:ylabel='Density'>"
      ]
     },
     "execution_count": 180,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<Figure size 864x720 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "#Now we will visualize the differences between our predictions and actual y test data\n",
    "plt.figure(figsize=(12,10))\n",
    "sns.distplot(y_test-y_predict_test,bins=50) #this figure also proves that our model fits very good\n",
    "#There is no huge differences between our predictions and actual y data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 186,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "#Residuals values  = y - yhat\n",
    "import statsmodels.api as smf\n",
    "smf.qqplot(predictions_df['Error'], line = 'q')\n",
    "plt.title('Normal Q-Q plot of residuals')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### ^Observation: Error should have Normal / Gaussian distribution~N(0,1) and independently and identically distributed."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<div style=\"display:fill;\n",
    "            border-radius: false;\n",
    "            border-style: solid;\n",
    "            border-color:#000000;\n",
    "            border-style: false;\n",
    "            border-width: 2px;\n",
    "            color:#CF673A;\n",
    "            font-size:15px;\n",
    "            font-family: Georgia;\n",
    "            background-color:#E8DCCC;\n",
    "            text-align:center;\n",
    "            letter-spacing:0.1px;\n",
    "            padding: 0.1em;\">\n",
    "\n",
    "**<h2>♡ Thank you for taking the time ♡**"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}