{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "UsageError: unrecognized arguments: encoding\n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "%matplotlib inline#for encoding\n",
    "from sklearn.preprocessing import LabelEncoder#for train test splitting\n",
    "from sklearn.model_selection import train_test_split#for decision tree object\n",
    "from sklearn.tree import DecisionTreeClassifier#for checking testing results\n",
    "from sklearn.metrics import classification_report, confusion_matrix#for visualizing tree \n",
    "from sklearn.tree import plot_tree"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
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
       "      <th>Sales</th>\n",
       "      <th>CompPrice</th>\n",
       "      <th>Income</th>\n",
       "      <th>Advertising</th>\n",
       "      <th>Population</th>\n",
       "      <th>Price</th>\n",
       "      <th>ShelveLoc</th>\n",
       "      <th>Age</th>\n",
       "      <th>Education</th>\n",
       "      <th>Urban</th>\n",
       "      <th>US</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>9.50</td>\n",
       "      <td>138</td>\n",
       "      <td>73</td>\n",
       "      <td>11</td>\n",
       "      <td>276</td>\n",
       "      <td>120</td>\n",
       "      <td>Bad</td>\n",
       "      <td>42</td>\n",
       "      <td>17</td>\n",
       "      <td>Yes</td>\n",
       "      <td>Yes</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>11.22</td>\n",
       "      <td>111</td>\n",
       "      <td>48</td>\n",
       "      <td>16</td>\n",
       "      <td>260</td>\n",
       "      <td>83</td>\n",
       "      <td>Good</td>\n",
       "      <td>65</td>\n",
       "      <td>10</td>\n",
       "      <td>Yes</td>\n",
       "      <td>Yes</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>10.06</td>\n",
       "      <td>113</td>\n",
       "      <td>35</td>\n",
       "      <td>10</td>\n",
       "      <td>269</td>\n",
       "      <td>80</td>\n",
       "      <td>Medium</td>\n",
       "      <td>59</td>\n",
       "      <td>12</td>\n",
       "      <td>Yes</td>\n",
       "      <td>Yes</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>7.40</td>\n",
       "      <td>117</td>\n",
       "      <td>100</td>\n",
       "      <td>4</td>\n",
       "      <td>466</td>\n",
       "      <td>97</td>\n",
       "      <td>Medium</td>\n",
       "      <td>55</td>\n",
       "      <td>14</td>\n",
       "      <td>Yes</td>\n",
       "      <td>Yes</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>4.15</td>\n",
       "      <td>141</td>\n",
       "      <td>64</td>\n",
       "      <td>3</td>\n",
       "      <td>340</td>\n",
       "      <td>128</td>\n",
       "      <td>Bad</td>\n",
       "      <td>38</td>\n",
       "      <td>13</td>\n",
       "      <td>Yes</td>\n",
       "      <td>No</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Sales  CompPrice  Income  Advertising  Population  Price ShelveLoc  Age  \\\n",
       "0   9.50        138      73           11         276    120       Bad   42   \n",
       "1  11.22        111      48           16         260     83      Good   65   \n",
       "2  10.06        113      35           10         269     80    Medium   59   \n",
       "3   7.40        117     100            4         466     97    Medium   55   \n",
       "4   4.15        141      64            3         340    128       Bad   38   \n",
       "\n",
       "   Education Urban   US  \n",
       "0         17   Yes  Yes  \n",
       "1         10   Yes  Yes  \n",
       "2         12   Yes  Yes  \n",
       "3         14   Yes  Yes  \n",
       "4         13   Yes   No  "
      ]
     },
     "execution_count": 1,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Pandas is used for data manipulation\n",
    "import pandas as pd\n",
    "# Read in data and display first 5 rows\n",
    "features = pd.read_csv('Company_Data.csv')\n",
    "features.head(5)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 400 entries, 0 to 399\n",
      "Data columns (total 11 columns):\n",
      " #   Column       Non-Null Count  Dtype  \n",
      "---  ------       --------------  -----  \n",
      " 0   Sales        400 non-null    float64\n",
      " 1   CompPrice    400 non-null    int64  \n",
      " 2   Income       400 non-null    int64  \n",
      " 3   Advertising  400 non-null    int64  \n",
      " 4   Population   400 non-null    int64  \n",
      " 5   Price        400 non-null    int64  \n",
      " 6   ShelveLoc    400 non-null    object \n",
      " 7   Age          400 non-null    int64  \n",
      " 8   Education    400 non-null    int64  \n",
      " 9   Urban        400 non-null    object \n",
      " 10  US           400 non-null    object \n",
      "dtypes: float64(1), int64(7), object(3)\n",
      "memory usage: 34.5+ KB\n"
     ]
    }
   ],
   "source": [
    "#getting information of dataset\n",
    "features.info()"
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
      "The shape of our features is: (400, 11)\n"
     ]
    }
   ],
   "source": [
    "print('The shape of our features is:', features.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Sales          False\n",
       "CompPrice      False\n",
       "Income         False\n",
       "Advertising    False\n",
       "Population     False\n",
       "Price          False\n",
       "ShelveLoc      False\n",
       "Age            False\n",
       "Education      False\n",
       "Urban          False\n",
       "US             False\n",
       "dtype: bool"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "features.isnull().any()"
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
       "<seaborn.axisgrid.PairGrid at 0x1a27c448700>"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<Figure size 1516x1440 with 72 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# let's plot pair plot to visualise the attributes all at once\n",
    "sns.pairplot(data=features, hue = 'ShelveLoc')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Creating dummy vairables dropping first dummy variable\n",
    "df=pd.get_dummies(features,columns=['Urban','US'], drop_first=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "   Sales  CompPrice  Income  Advertising  Population  Price ShelveLoc  Age  \\\n",
      "0   9.50        138      73           11         276    120       Bad   42   \n",
      "1  11.22        111      48           16         260     83      Good   65   \n",
      "2  10.06        113      35           10         269     80    Medium   59   \n",
      "3   7.40        117     100            4         466     97    Medium   55   \n",
      "4   4.15        141      64            3         340    128       Bad   38   \n",
      "\n",
      "   Education  Urban_Yes  US_Yes  \n",
      "0         17          1       1  \n",
      "1         10          1       1  \n",
      "2         12          1       1  \n",
      "3         14          1       1  \n",
      "4         13          1       0  \n"
     ]
    }
   ],
   "source": [
    "print(df.head())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.metrics import f1_score\n",
    "from sklearn.model_selection import train_test_split"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [],
   "source": [
    "df['ShelveLoc']=df['ShelveLoc'].map({'Good':1,'Medium':2,'Bad':3})"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "   Sales  CompPrice  Income  Advertising  Population  Price  ShelveLoc  Age  \\\n",
      "0   9.50        138      73           11         276    120          3   42   \n",
      "1  11.22        111      48           16         260     83          1   65   \n",
      "2  10.06        113      35           10         269     80          2   59   \n",
      "3   7.40        117     100            4         466     97          2   55   \n",
      "4   4.15        141      64            3         340    128          3   38   \n",
      "\n",
      "   Education  Urban_Yes  US_Yes  \n",
      "0         17          1       1  \n",
      "1         10          1       1  \n",
      "2         12          1       1  \n",
      "3         14          1       1  \n",
      "4         13          1       0  \n"
     ]
    }
   ],
   "source": [
    "print(df.head())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
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
       "      <th>Sales</th>\n",
       "      <th>CompPrice</th>\n",
       "      <th>Income</th>\n",
       "      <th>Advertising</th>\n",
       "      <th>Population</th>\n",
       "      <th>Price</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>9.50</td>\n",
       "      <td>138</td>\n",
       "      <td>73</td>\n",
       "      <td>11</td>\n",
       "      <td>276</td>\n",
       "      <td>120</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>11.22</td>\n",
       "      <td>111</td>\n",
       "      <td>48</td>\n",
       "      <td>16</td>\n",
       "      <td>260</td>\n",
       "      <td>83</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>10.06</td>\n",
       "      <td>113</td>\n",
       "      <td>35</td>\n",
       "      <td>10</td>\n",
       "      <td>269</td>\n",
       "      <td>80</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>7.40</td>\n",
       "      <td>117</td>\n",
       "      <td>100</td>\n",
       "      <td>4</td>\n",
       "      <td>466</td>\n",
       "      <td>97</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>4.15</td>\n",
       "      <td>141</td>\n",
       "      <td>64</td>\n",
       "      <td>3</td>\n",
       "      <td>340</td>\n",
       "      <td>128</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>395</th>\n",
       "      <td>12.57</td>\n",
       "      <td>138</td>\n",
       "      <td>108</td>\n",
       "      <td>17</td>\n",
       "      <td>203</td>\n",
       "      <td>128</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>396</th>\n",
       "      <td>6.14</td>\n",
       "      <td>139</td>\n",
       "      <td>23</td>\n",
       "      <td>3</td>\n",
       "      <td>37</td>\n",
       "      <td>120</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>397</th>\n",
       "      <td>7.41</td>\n",
       "      <td>162</td>\n",
       "      <td>26</td>\n",
       "      <td>12</td>\n",
       "      <td>368</td>\n",
       "      <td>159</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>398</th>\n",
       "      <td>5.94</td>\n",
       "      <td>100</td>\n",
       "      <td>79</td>\n",
       "      <td>7</td>\n",
       "      <td>284</td>\n",
       "      <td>95</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>399</th>\n",
       "      <td>9.71</td>\n",
       "      <td>134</td>\n",
       "      <td>37</td>\n",
       "      <td>0</td>\n",
       "      <td>27</td>\n",
       "      <td>120</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>400 rows × 6 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "     Sales  CompPrice  Income  Advertising  Population  Price\n",
       "0     9.50        138      73           11         276    120\n",
       "1    11.22        111      48           16         260     83\n",
       "2    10.06        113      35           10         269     80\n",
       "3     7.40        117     100            4         466     97\n",
       "4     4.15        141      64            3         340    128\n",
       "..     ...        ...     ...          ...         ...    ...\n",
       "395  12.57        138     108           17         203    128\n",
       "396   6.14        139      23            3          37    120\n",
       "397   7.41        162      26           12         368    159\n",
       "398   5.94        100      79            7         284     95\n",
       "399   9.71        134      37            0          27    120\n",
       "\n",
       "[400 rows x 6 columns]"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x=df.iloc[:,0:6]\n",
    "y=df['ShelveLoc']\n",
    "x"
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
       "0      3\n",
       "1      1\n",
       "2      2\n",
       "3      2\n",
       "4      3\n",
       "      ..\n",
       "395    1\n",
       "396    2\n",
       "397    2\n",
       "398    3\n",
       "399    1\n",
       "Name: ShelveLoc, Length: 400, dtype: int64"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([3, 1, 2], dtype=int64)"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['ShelveLoc'].unique()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "2    219\n",
       "3     96\n",
       "1     85\n",
       "Name: ShelveLoc, dtype: int64"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.ShelveLoc.value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['Sales',\n",
       " 'CompPrice',\n",
       " 'Income',\n",
       " 'Advertising',\n",
       " 'Population',\n",
       " 'Price',\n",
       " 'ShelveLoc',\n",
       " 'Age',\n",
       " 'Education',\n",
       " 'Urban_Yes',\n",
       " 'US_Yes']"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "colnames = list(df.columns)\n",
    "colnames"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
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
       "      <th>Sales</th>\n",
       "      <th>CompPrice</th>\n",
       "      <th>Income</th>\n",
       "      <th>Advertising</th>\n",
       "      <th>Population</th>\n",
       "      <th>Price</th>\n",
       "      <th>ShelveLoc</th>\n",
       "      <th>Age</th>\n",
       "      <th>Education</th>\n",
       "      <th>Urban_Yes</th>\n",
       "      <th>US_Yes</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>400.000000</td>\n",
       "      <td>400.000000</td>\n",
       "      <td>400.000000</td>\n",
       "      <td>400.000000</td>\n",
       "      <td>400.000000</td>\n",
       "      <td>400.000000</td>\n",
       "      <td>400.000000</td>\n",
       "      <td>400.000000</td>\n",
       "      <td>400.000000</td>\n",
       "      <td>400.000000</td>\n",
       "      <td>400.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>7.496325</td>\n",
       "      <td>124.975000</td>\n",
       "      <td>68.657500</td>\n",
       "      <td>6.635000</td>\n",
       "      <td>264.840000</td>\n",
       "      <td>115.795000</td>\n",
       "      <td>2.027500</td>\n",
       "      <td>53.322500</td>\n",
       "      <td>13.900000</td>\n",
       "      <td>0.705000</td>\n",
       "      <td>0.645000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>2.824115</td>\n",
       "      <td>15.334512</td>\n",
       "      <td>27.986037</td>\n",
       "      <td>6.650364</td>\n",
       "      <td>147.376436</td>\n",
       "      <td>23.676664</td>\n",
       "      <td>0.672961</td>\n",
       "      <td>16.200297</td>\n",
       "      <td>2.620528</td>\n",
       "      <td>0.456614</td>\n",
       "      <td>0.479113</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>0.000000</td>\n",
       "      <td>77.000000</td>\n",
       "      <td>21.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>10.000000</td>\n",
       "      <td>24.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>25.000000</td>\n",
       "      <td>10.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>5.390000</td>\n",
       "      <td>115.000000</td>\n",
       "      <td>42.750000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>139.000000</td>\n",
       "      <td>100.000000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>39.750000</td>\n",
       "      <td>12.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>7.490000</td>\n",
       "      <td>125.000000</td>\n",
       "      <td>69.000000</td>\n",
       "      <td>5.000000</td>\n",
       "      <td>272.000000</td>\n",
       "      <td>117.000000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>54.500000</td>\n",
       "      <td>14.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>9.320000</td>\n",
       "      <td>135.000000</td>\n",
       "      <td>91.000000</td>\n",
       "      <td>12.000000</td>\n",
       "      <td>398.500000</td>\n",
       "      <td>131.000000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>66.000000</td>\n",
       "      <td>16.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>16.270000</td>\n",
       "      <td>175.000000</td>\n",
       "      <td>120.000000</td>\n",
       "      <td>29.000000</td>\n",
       "      <td>509.000000</td>\n",
       "      <td>191.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>80.000000</td>\n",
       "      <td>18.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "            Sales   CompPrice      Income  Advertising  Population  \\\n",
       "count  400.000000  400.000000  400.000000   400.000000  400.000000   \n",
       "mean     7.496325  124.975000   68.657500     6.635000  264.840000   \n",
       "std      2.824115   15.334512   27.986037     6.650364  147.376436   \n",
       "min      0.000000   77.000000   21.000000     0.000000   10.000000   \n",
       "25%      5.390000  115.000000   42.750000     0.000000  139.000000   \n",
       "50%      7.490000  125.000000   69.000000     5.000000  272.000000   \n",
       "75%      9.320000  135.000000   91.000000    12.000000  398.500000   \n",
       "max     16.270000  175.000000  120.000000    29.000000  509.000000   \n",
       "\n",
       "            Price   ShelveLoc         Age   Education   Urban_Yes      US_Yes  \n",
       "count  400.000000  400.000000  400.000000  400.000000  400.000000  400.000000  \n",
       "mean   115.795000    2.027500   53.322500   13.900000    0.705000    0.645000  \n",
       "std     23.676664    0.672961   16.200297    2.620528    0.456614    0.479113  \n",
       "min     24.000000    1.000000   25.000000   10.000000    0.000000    0.000000  \n",
       "25%    100.000000    2.000000   39.750000   12.000000    0.000000    0.000000  \n",
       "50%    117.000000    2.000000   54.500000   14.000000    1.000000    1.000000  \n",
       "75%    131.000000    2.000000   66.000000   16.000000    1.000000    1.000000  \n",
       "max    191.000000    3.000000   80.000000   18.000000    1.000000    1.000000  "
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Descriptive statistics for each column\n",
    "df.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
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
       "      <th>Sales</th>\n",
       "      <th>CompPrice</th>\n",
       "      <th>Income</th>\n",
       "      <th>Advertising</th>\n",
       "      <th>Population</th>\n",
       "      <th>Price</th>\n",
       "      <th>ShelveLoc</th>\n",
       "      <th>Age</th>\n",
       "      <th>Education</th>\n",
       "      <th>Urban_Yes</th>\n",
       "      <th>US_Yes</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>9.50</td>\n",
       "      <td>138</td>\n",
       "      <td>73</td>\n",
       "      <td>11</td>\n",
       "      <td>276</td>\n",
       "      <td>120</td>\n",
       "      <td>3</td>\n",
       "      <td>42</td>\n",
       "      <td>17</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>11.22</td>\n",
       "      <td>111</td>\n",
       "      <td>48</td>\n",
       "      <td>16</td>\n",
       "      <td>260</td>\n",
       "      <td>83</td>\n",
       "      <td>1</td>\n",
       "      <td>65</td>\n",
       "      <td>10</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>10.06</td>\n",
       "      <td>113</td>\n",
       "      <td>35</td>\n",
       "      <td>10</td>\n",
       "      <td>269</td>\n",
       "      <td>80</td>\n",
       "      <td>2</td>\n",
       "      <td>59</td>\n",
       "      <td>12</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>7.40</td>\n",
       "      <td>117</td>\n",
       "      <td>100</td>\n",
       "      <td>4</td>\n",
       "      <td>466</td>\n",
       "      <td>97</td>\n",
       "      <td>2</td>\n",
       "      <td>55</td>\n",
       "      <td>14</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>4.15</td>\n",
       "      <td>141</td>\n",
       "      <td>64</td>\n",
       "      <td>3</td>\n",
       "      <td>340</td>\n",
       "      <td>128</td>\n",
       "      <td>3</td>\n",
       "      <td>38</td>\n",
       "      <td>13</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Sales  CompPrice  Income  Advertising  Population  Price  ShelveLoc  Age  \\\n",
       "0   9.50        138      73           11         276    120          3   42   \n",
       "1  11.22        111      48           16         260     83          1   65   \n",
       "2  10.06        113      35           10         269     80          2   59   \n",
       "3   7.40        117     100            4         466     97          2   55   \n",
       "4   4.15        141      64            3         340    128          3   38   \n",
       "\n",
       "   Education  Urban_Yes  US_Yes  \n",
       "0         17          1       1  \n",
       "1         10          1       1  \n",
       "2         12          1       1  \n",
       "3         14          1       1  \n",
       "4         13          1       0  "
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Labels are the values we want to predict\n",
    "labels = np.array(df['Income'])\n",
    "# Remove the labels from the features\n",
    "# axis 1 refers to the columns\n",
    "features= df.drop('Income', axis = 1)\n",
    "# Saving feature names for later use\n",
    "feature_list = list(df.columns)\n",
    "# Convert to numpy array\n",
    "features = np.array(df)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Using Skicit-learn to split data into training and testing sets\n",
    "from sklearn.model_selection import train_test_split\n",
    "# Split the data into training and testing sets\n",
    "train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Training Features Shape: (300, 11)\n",
      "Training Labels Shape: (300,)\n",
      "Testing Features Shape: (100, 11)\n",
      "Testing Labels Shape: (100,)\n"
     ]
    }
   ],
   "source": [
    "print('Training Features Shape:', train_features.shape)\n",
    "print('Training Labels Shape:', train_labels.shape)\n",
    "print('Testing Features Shape:', test_features.shape)\n",
    "print('Testing Labels Shape:', test_labels.shape)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Establish Baseline"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Average baseline error:  65.26\n"
     ]
    }
   ],
   "source": [
    "# The baseline predictions are the historical averages\n",
    "baseline_preds = test_features[:, feature_list.index('Sales')]\n",
    "# Baseline errors, and display average baseline error\n",
    "baseline_errors = abs(baseline_preds - test_labels)\n",
    "print('Average baseline error: ', round(np.mean(baseline_errors), 2))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Import the model we are using\n",
    "from sklearn.ensemble import RandomForestRegressor\n",
    "# Instantiate model with 1000 decision trees\n",
    "rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)\n",
    "# Train the model on training data\n",
    "rf.fit(train_features, train_labels);"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Mean Absolute Error: 0.27 degrees.\n"
     ]
    }
   ],
   "source": [
    "# Use the forest's predict method on the test data\n",
    "predictions = rf.predict(test_features)\n",
    "# Calculate the absolute errors\n",
    "errors = abs(predictions - test_labels)\n",
    "# Print out the mean absolute error (mae)\n",
    "print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Determine Performance Metrics"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy: 99.58 %.\n"
     ]
    }
   ],
   "source": [
    "# Calculate mean absolute percentage error (MAPE)\n",
    "mape = 100 * (errors / test_labels)\n",
    "# Calculate and display accuracy\n",
    "accuracy = 100 - np.mean(mape)\n",
    "print('Accuracy:', round(accuracy, 2), '%.')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Import tools needed for visualization\n",
    "from sklearn.tree import export_graphviz\n",
    "import pydot\n",
    "# Pull out one tree from the forest\n",
    "tree = rf.estimators_[5]\n",
    "# Import tools needed for visualization\n",
    "from sklearn.tree import export_graphviz\n",
    "import pydot\n",
    "# Pull out one tree from the forest\n",
    "tree = rf.estimators_[5]\n",
    "# Export the image to a dot file\n",
    "export_graphviz(tree, out_file = 'tree.dot', feature_names = feature_list, rounded = True, precision = 1)\n",
    "# Use dot file to create a graph\n",
    "(graph, ) = pydot.graph_from_dot_file('tree.dot')\n",
    "# Write graph to a png file\n",
    "graph.write_png('tree.png')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Limit depth of tree to 3 levels\n",
    "rf_small = RandomForestRegressor(n_estimators=10, max_depth = 3)\n",
    "rf_small.fit(train_features, train_labels)\n",
    "# Extract the small tree\n",
    "tree_small = rf_small.estimators_[5]\n",
    "# Save the tree as a png image\n",
    "export_graphviz(tree_small, out_file = 'small_tree.dot', feature_names = feature_list, rounded = True, precision = 1)\n",
    "(graph, ) = pydot.graph_from_dot_file('small_tree.dot')\n",
    "graph.write_png('small_tree.png');"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Variable: Income               Importance: 1.0\n",
      "Variable: Sales                Importance: 0.0\n",
      "Variable: CompPrice            Importance: 0.0\n",
      "Variable: Advertising          Importance: 0.0\n",
      "Variable: Population           Importance: 0.0\n",
      "Variable: Price                Importance: 0.0\n",
      "Variable: ShelveLoc            Importance: 0.0\n",
      "Variable: Age                  Importance: 0.0\n",
      "Variable: Education            Importance: 0.0\n",
      "Variable: Urban_Yes            Importance: 0.0\n",
      "Variable: US_Yes               Importance: 0.0\n"
     ]
    }
   ],
   "source": [
    "# Get numerical feature importances\n",
    "importances = list(rf.feature_importances_)\n",
    "# List of tuples with variable and importance\n",
    "feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]\n",
    "# Sort the feature importances by most important first\n",
    "feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)\n",
    "# Print out the feature and importances \n",
    "[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Mean Absolute Error: 0.16 degrees.\n",
      "Accuracy: 99.75 %.\n"
     ]
    }
   ],
   "source": [
    "# New random forest with only the two most important variables\n",
    "rf_most_important = RandomForestRegressor(n_estimators= 1000, random_state=42)\n",
    "# Extract the two most important features\n",
    "important_indices = [feature_list.index('Sales'), feature_list.index('Income')]\n",
    "train_important = train_features[:, important_indices]\n",
    "test_important = test_features[:, important_indices]\n",
    "# Train the random forest\n",
    "rf_most_important.fit(train_important, train_labels)\n",
    "# Make predictions and determine the error\n",
    "predictions = rf_most_important.predict(test_important)\n",
    "errors = abs(predictions - test_labels)\n",
    "# Display the performance metrics\n",
    "print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')\n",
    "mape = np.mean(100 * (errors / test_labels))\n",
    "accuracy = 100 - mape\n",
    "print('Accuracy:', round(accuracy, 2), '%.')\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
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
    }
   ],
   "source": [
    "# Import matplotlib for plotting and use magic command for Jupyter Notebooks\n",
    "import matplotlib.pyplot as plt\n",
    "%matplotlib inline\n",
    "# Set the style\n",
    "plt.style.use('fivethirtyeight')\n",
    "# list of x locations for plotting\n",
    "x_values = list(range(len(importances)))\n",
    "# Make a bar chart\n",
    "plt.bar(x_values, importances, orientation = 'vertical')\n",
    "# Tick labels for x axis\n",
    "plt.xticks(x_values, feature_list, rotation='vertical')\n",
    "# Axis labels and title\n",
    "plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances');"
   ]
  }
 ],
 "metadata": {
  "_draft": {
   "nbviewer_url": "https://gist.github.com/2cf79d3dacb6c4da725db40bdd53b0e6"
  },
  "gist": {
   "data": {
    "description": "Assignment 15- Random Forest.ipynb",
    "public": true
   },
   "id": "2cf79d3dacb6c4da725db40bdd53b0e6"
  },
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