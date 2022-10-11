{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Simple Linear Regression-1"
   ]
  },
  {
   "cell_type": "raw",
   "metadata": {},
   "source": [
    "Q1) Delivery_time -> Predict delivery time using sorting time \n",
    "Build a simple linear regression model by performing EDA and do necessary transformations and select the best model using R or Python."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "# import libraries\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import seaborn as sns\n",
    "import statsmodels.formula.api as smf"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
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
       "      <th>Delivery Time</th>\n",
       "      <th>Sorting Time</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>21.00</td>\n",
       "      <td>10</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>13.50</td>\n",
       "      <td>4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>19.75</td>\n",
       "      <td>6</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>24.00</td>\n",
       "      <td>9</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>29.00</td>\n",
       "      <td>10</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>15.35</td>\n",
       "      <td>6</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>19.00</td>\n",
       "      <td>7</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>9.50</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>17.90</td>\n",
       "      <td>10</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>18.75</td>\n",
       "      <td>9</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>10</th>\n",
       "      <td>19.83</td>\n",
       "      <td>8</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>11</th>\n",
       "      <td>10.75</td>\n",
       "      <td>4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>12</th>\n",
       "      <td>16.68</td>\n",
       "      <td>7</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>13</th>\n",
       "      <td>11.50</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>14</th>\n",
       "      <td>12.03</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>15</th>\n",
       "      <td>14.88</td>\n",
       "      <td>4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>16</th>\n",
       "      <td>13.75</td>\n",
       "      <td>6</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>17</th>\n",
       "      <td>18.11</td>\n",
       "      <td>7</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>18</th>\n",
       "      <td>8.00</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>19</th>\n",
       "      <td>17.83</td>\n",
       "      <td>7</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>20</th>\n",
       "      <td>21.50</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "    Delivery Time  Sorting Time\n",
       "0           21.00            10\n",
       "1           13.50             4\n",
       "2           19.75             6\n",
       "3           24.00             9\n",
       "4           29.00            10\n",
       "5           15.35             6\n",
       "6           19.00             7\n",
       "7            9.50             3\n",
       "8           17.90            10\n",
       "9           18.75             9\n",
       "10          19.83             8\n",
       "11          10.75             4\n",
       "12          16.68             7\n",
       "13          11.50             3\n",
       "14          12.03             3\n",
       "15          14.88             4\n",
       "16          13.75             6\n",
       "17          18.11             7\n",
       "18           8.00             2\n",
       "19          17.83             7\n",
       "20          21.50             5"
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# import dataset\n",
    "dataset=pd.read_csv('Database/delivery_time.csv')\n",
    "dataset"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## EDA and Data Visualization"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 21 entries, 0 to 20\n",
      "Data columns (total 2 columns):\n",
      " #   Column         Non-Null Count  Dtype  \n",
      "---  ------         --------------  -----  \n",
      " 0   Delivery Time  21 non-null     float64\n",
      " 1   Sorting Time   21 non-null     int64  \n",
      "dtypes: float64(1), int64(1)\n",
      "memory usage: 464.0 bytes\n"
     ]
    }
   ],
   "source": [
    "dataset.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\Hitesh Koli\\anaconda3\\lib\\site-packages\\seaborn\\distributions.py:2557: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).\n",
      "  warnings.warn(msg, FutureWarning)\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x213a9ad7a00>"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.distplot(dataset['Delivery Time'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\Hitesh Koli\\anaconda3\\lib\\site-packages\\seaborn\\distributions.py:2557: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).\n",
      "  warnings.warn(msg, FutureWarning)\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x213b7193e80>"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.distplot(dataset['Sorting Time'])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Feature Engineering"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
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
       "      <th>delivery_time</th>\n",
       "      <th>sorting_time</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>21.00</td>\n",
       "      <td>10</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>13.50</td>\n",
       "      <td>4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>19.75</td>\n",
       "      <td>6</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>24.00</td>\n",
       "      <td>9</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>29.00</td>\n",
       "      <td>10</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>15.35</td>\n",
       "      <td>6</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>19.00</td>\n",
       "      <td>7</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>9.50</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>17.90</td>\n",
       "      <td>10</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>18.75</td>\n",
       "      <td>9</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>10</th>\n",
       "      <td>19.83</td>\n",
       "      <td>8</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>11</th>\n",
       "      <td>10.75</td>\n",
       "      <td>4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>12</th>\n",
       "      <td>16.68</td>\n",
       "      <td>7</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>13</th>\n",
       "      <td>11.50</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>14</th>\n",
       "      <td>12.03</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>15</th>\n",
       "      <td>14.88</td>\n",
       "      <td>4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>16</th>\n",
       "      <td>13.75</td>\n",
       "      <td>6</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>17</th>\n",
       "      <td>18.11</td>\n",
       "      <td>7</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>18</th>\n",
       "      <td>8.00</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>19</th>\n",
       "      <td>17.83</td>\n",
       "      <td>7</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>20</th>\n",
       "      <td>21.50</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "    delivery_time  sorting_time\n",
       "0           21.00            10\n",
       "1           13.50             4\n",
       "2           19.75             6\n",
       "3           24.00             9\n",
       "4           29.00            10\n",
       "5           15.35             6\n",
       "6           19.00             7\n",
       "7            9.50             3\n",
       "8           17.90            10\n",
       "9           18.75             9\n",
       "10          19.83             8\n",
       "11          10.75             4\n",
       "12          16.68             7\n",
       "13          11.50             3\n",
       "14          12.03             3\n",
       "15          14.88             4\n",
       "16          13.75             6\n",
       "17          18.11             7\n",
       "18           8.00             2\n",
       "19          17.83             7\n",
       "20          21.50             5"
      ]
     },
     "execution_count": 38,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Renaming Columns\n",
    "dataset=dataset.rename({'Delivery Time':'delivery_time', 'Sorting Time':'sorting_time'},axis=1)\n",
    "dataset"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Correlation Analysis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
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
       "      <th>delivery_time</th>\n",
       "      <th>sorting_time</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>delivery_time</th>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.825997</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>sorting_time</th>\n",
       "      <td>0.825997</td>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "               delivery_time  sorting_time\n",
       "delivery_time       1.000000      0.825997\n",
       "sorting_time        0.825997      1.000000"
      ]
     },
     "execution_count": 39,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dataset.corr()"
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
       "<matplotlib.axes._subplots.AxesSubplot at 0x213b79d4970>"
      ]
     },
     "execution_count": 40,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.regplot(x=dataset['sorting_time'],y=dataset['delivery_time'])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Model Building"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "metadata": {},
   "outputs": [],
   "source": [
    "model=smf.ols(\"delivery_time~sorting_time\",data=dataset).fit()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Model Testing"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Intercept       6.582734\n",
       "sorting_time    1.649020\n",
       "dtype: float64"
      ]
     },
     "execution_count": 42,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Finding Coefficient parameters\n",
    "model.params"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(Intercept       3.823349\n",
       " sorting_time    6.387447\n",
       " dtype: float64,\n",
       " Intercept       0.001147\n",
       " sorting_time    0.000004\n",
       " dtype: float64)"
      ]
     },
     "execution_count": 43,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Finding tvalues and pvalues\n",
    "model.tvalues , model.pvalues"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(0.6822714748417231, 0.6655489208860244)"
      ]
     },
     "execution_count": 44,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Finding Rsquared Values\n",
    "model.rsquared , model.rsquared_adj"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Model Predictions"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "14.827834"
      ]
     },
     "execution_count": 45,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Manual prediction for say sorting time 5\n",
    "delivery_time = (6.582734) + (1.649020)*(5)\n",
    "delivery_time"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0    5\n",
       "1    8\n",
       "dtype: int64"
      ]
     },
     "execution_count": 46,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Automatic Prediction for say sorting time 5, 8\n",
    "new_data=pd.Series([5,8])\n",
    "new_data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
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
       "      <th>sorting_time</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>8</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   sorting_time\n",
       "0             5\n",
       "1             8"
      ]
     },
     "execution_count": 47,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data_pred=pd.DataFrame(new_data,columns=['sorting_time'])\n",
    "data_pred"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0    14.827833\n",
       "1    19.774893\n",
       "dtype: float64"
      ]
     },
     "execution_count": 48,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model.predict(data_pred)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
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
 "nbformat_minor": 4
}
