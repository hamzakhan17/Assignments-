{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Assignment - Naive Bayes"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "1. Prepare a classification model using Naive Bayes for salary data \n",
    "\n",
    "\n",
    "Data Description:\n",
    "\n",
    "age -- age of a person\n",
    "\n",
    "workclass-- A work class is a grouping of work \n",
    "\n",
    "education-- Education of an individuals\n",
    "\n",
    "maritalstatus -- Marital status of an individulas\n",
    "\n",
    "occupation-- occupation of an individuals\n",
    "\n",
    "relationship -- \n",
    "\n",
    "race --  Race of an Individual\n",
    "\n",
    "sex --  Gender of an Individual\n",
    "\n",
    "capitalgain --  profit received from the sale of an investment\n",
    "\n",
    "capitalloss\t-- A decrease in the value of a capital asset\n",
    "\n",
    "hoursperweek -- number of hours work per week\n",
    "\n",
    "native -- Native of an individual\n",
    "\n",
    "Salary -- salary of an individual\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Import libraries"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "%matplotlib inline\n",
    "import os\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')\n",
    "\n",
    "from pandas.plotting import scatter_matrix\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.model_selection import train_test_split \n",
    "from sklearn.model_selection import KFold\n",
    "from sklearn.model_selection import cross_val_score\n",
    "from sklearn import metrics\n",
    "import statsmodels.api as sm\n",
    "\n",
    "from sklearn.datasets import fetch_20newsgroups\n",
    "from sklearn.feature_extraction.text import CountVectorizer\n",
    "from sklearn.naive_bayes import GaussianNB\n",
    "from sklearn.metrics import confusion_matrix, plot_confusion_matrix"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Import dataset"
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
       "      <th>age</th>\n",
       "      <th>workclass</th>\n",
       "      <th>education</th>\n",
       "      <th>educationno</th>\n",
       "      <th>maritalstatus</th>\n",
       "      <th>occupation</th>\n",
       "      <th>relationship</th>\n",
       "      <th>race</th>\n",
       "      <th>sex</th>\n",
       "      <th>capitalgain</th>\n",
       "      <th>capitalloss</th>\n",
       "      <th>hoursperweek</th>\n",
       "      <th>native</th>\n",
       "      <th>Salary</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>39</td>\n",
       "      <td>State-gov</td>\n",
       "      <td>Bachelors</td>\n",
       "      <td>13</td>\n",
       "      <td>Never-married</td>\n",
       "      <td>Adm-clerical</td>\n",
       "      <td>Not-in-family</td>\n",
       "      <td>White</td>\n",
       "      <td>Male</td>\n",
       "      <td>2174</td>\n",
       "      <td>0</td>\n",
       "      <td>40</td>\n",
       "      <td>United-States</td>\n",
       "      <td>&lt;=50K</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>50</td>\n",
       "      <td>Self-emp-not-inc</td>\n",
       "      <td>Bachelors</td>\n",
       "      <td>13</td>\n",
       "      <td>Married-civ-spouse</td>\n",
       "      <td>Exec-managerial</td>\n",
       "      <td>Husband</td>\n",
       "      <td>White</td>\n",
       "      <td>Male</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>13</td>\n",
       "      <td>United-States</td>\n",
       "      <td>&lt;=50K</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>38</td>\n",
       "      <td>Private</td>\n",
       "      <td>HS-grad</td>\n",
       "      <td>9</td>\n",
       "      <td>Divorced</td>\n",
       "      <td>Handlers-cleaners</td>\n",
       "      <td>Not-in-family</td>\n",
       "      <td>White</td>\n",
       "      <td>Male</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>40</td>\n",
       "      <td>United-States</td>\n",
       "      <td>&lt;=50K</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>53</td>\n",
       "      <td>Private</td>\n",
       "      <td>11th</td>\n",
       "      <td>7</td>\n",
       "      <td>Married-civ-spouse</td>\n",
       "      <td>Handlers-cleaners</td>\n",
       "      <td>Husband</td>\n",
       "      <td>Black</td>\n",
       "      <td>Male</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>40</td>\n",
       "      <td>United-States</td>\n",
       "      <td>&lt;=50K</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>28</td>\n",
       "      <td>Private</td>\n",
       "      <td>Bachelors</td>\n",
       "      <td>13</td>\n",
       "      <td>Married-civ-spouse</td>\n",
       "      <td>Prof-specialty</td>\n",
       "      <td>Wife</td>\n",
       "      <td>Black</td>\n",
       "      <td>Female</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>40</td>\n",
       "      <td>Cuba</td>\n",
       "      <td>&lt;=50K</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   age          workclass   education  educationno        maritalstatus  \\\n",
       "0   39          State-gov   Bachelors           13        Never-married   \n",
       "1   50   Self-emp-not-inc   Bachelors           13   Married-civ-spouse   \n",
       "2   38            Private     HS-grad            9             Divorced   \n",
       "3   53            Private        11th            7   Married-civ-spouse   \n",
       "4   28            Private   Bachelors           13   Married-civ-spouse   \n",
       "\n",
       "           occupation    relationship    race      sex  capitalgain  \\\n",
       "0        Adm-clerical   Not-in-family   White     Male         2174   \n",
       "1     Exec-managerial         Husband   White     Male            0   \n",
       "2   Handlers-cleaners   Not-in-family   White     Male            0   \n",
       "3   Handlers-cleaners         Husband   Black     Male            0   \n",
       "4      Prof-specialty            Wife   Black   Female            0   \n",
       "\n",
       "   capitalloss  hoursperweek          native  Salary  \n",
       "0            0            40   United-States   <=50K  \n",
       "1            0            13   United-States   <=50K  \n",
       "2            0            40   United-States   <=50K  \n",
       "3            0            40   United-States   <=50K  \n",
       "4            0            40            Cuba   <=50K  "
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "salarydata_train = pd.read_csv('SalaryData_Train.csv')\n",
    "salarydata_train.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
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
       "      <th>age</th>\n",
       "      <th>workclass</th>\n",
       "      <th>education</th>\n",
       "      <th>educationno</th>\n",
       "      <th>maritalstatus</th>\n",
       "      <th>occupation</th>\n",
       "      <th>relationship</th>\n",
       "      <th>race</th>\n",
       "      <th>sex</th>\n",
       "      <th>capitalgain</th>\n",
       "      <th>capitalloss</th>\n",
       "      <th>hoursperweek</th>\n",
       "      <th>native</th>\n",
       "      <th>Salary</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>25</td>\n",
       "      <td>Private</td>\n",
       "      <td>11th</td>\n",
       "      <td>7</td>\n",
       "      <td>Never-married</td>\n",
       "      <td>Machine-op-inspct</td>\n",
       "      <td>Own-child</td>\n",
       "      <td>Black</td>\n",
       "      <td>Male</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>40</td>\n",
       "      <td>United-States</td>\n",
       "      <td>&lt;=50K</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>38</td>\n",
       "      <td>Private</td>\n",
       "      <td>HS-grad</td>\n",
       "      <td>9</td>\n",
       "      <td>Married-civ-spouse</td>\n",
       "      <td>Farming-fishing</td>\n",
       "      <td>Husband</td>\n",
       "      <td>White</td>\n",
       "      <td>Male</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>50</td>\n",
       "      <td>United-States</td>\n",
       "      <td>&lt;=50K</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>28</td>\n",
       "      <td>Local-gov</td>\n",
       "      <td>Assoc-acdm</td>\n",
       "      <td>12</td>\n",
       "      <td>Married-civ-spouse</td>\n",
       "      <td>Protective-serv</td>\n",
       "      <td>Husband</td>\n",
       "      <td>White</td>\n",
       "      <td>Male</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>40</td>\n",
       "      <td>United-States</td>\n",
       "      <td>&gt;50K</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>44</td>\n",
       "      <td>Private</td>\n",
       "      <td>Some-college</td>\n",
       "      <td>10</td>\n",
       "      <td>Married-civ-spouse</td>\n",
       "      <td>Machine-op-inspct</td>\n",
       "      <td>Husband</td>\n",
       "      <td>Black</td>\n",
       "      <td>Male</td>\n",
       "      <td>7688</td>\n",
       "      <td>0</td>\n",
       "      <td>40</td>\n",
       "      <td>United-States</td>\n",
       "      <td>&gt;50K</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>34</td>\n",
       "      <td>Private</td>\n",
       "      <td>10th</td>\n",
       "      <td>6</td>\n",
       "      <td>Never-married</td>\n",
       "      <td>Other-service</td>\n",
       "      <td>Not-in-family</td>\n",
       "      <td>White</td>\n",
       "      <td>Male</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>30</td>\n",
       "      <td>United-States</td>\n",
       "      <td>&lt;=50K</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   age   workclass      education  educationno        maritalstatus  \\\n",
       "0   25     Private           11th            7        Never-married   \n",
       "1   38     Private        HS-grad            9   Married-civ-spouse   \n",
       "2   28   Local-gov     Assoc-acdm           12   Married-civ-spouse   \n",
       "3   44     Private   Some-college           10   Married-civ-spouse   \n",
       "4   34     Private           10th            6        Never-married   \n",
       "\n",
       "           occupation    relationship    race    sex  capitalgain  \\\n",
       "0   Machine-op-inspct       Own-child   Black   Male            0   \n",
       "1     Farming-fishing         Husband   White   Male            0   \n",
       "2     Protective-serv         Husband   White   Male            0   \n",
       "3   Machine-op-inspct         Husband   Black   Male         7688   \n",
       "4       Other-service   Not-in-family   White   Male            0   \n",
       "\n",
       "   capitalloss  hoursperweek          native  Salary  \n",
       "0            0            40   United-States   <=50K  \n",
       "1            0            50   United-States   <=50K  \n",
       "2            0            40   United-States    >50K  \n",
       "3            0            40   United-States    >50K  \n",
       "4            0            30   United-States   <=50K  "
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "salarydata_test = pd.read_csv('SalaryData_Test.csv')\n",
    "salarydata_test.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Exploratory data analysis"
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
       "(30161, 14)"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "salarydata_train.shape"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We can see that there are 30161 instances and 14 attributes in the training data set."
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
       "(15060, 14)"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "salarydata_test.shape"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We can see that there are 15060 instances and 14 attributes in the test data set."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# View top 5 rows of dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
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
       "      <th>age</th>\n",
       "      <th>workclass</th>\n",
       "      <th>education</th>\n",
       "      <th>educationno</th>\n",
       "      <th>maritalstatus</th>\n",
       "      <th>occupation</th>\n",
       "      <th>relationship</th>\n",
       "      <th>race</th>\n",
       "      <th>sex</th>\n",
       "      <th>capitalgain</th>\n",
       "      <th>capitalloss</th>\n",
       "      <th>hoursperweek</th>\n",
       "      <th>native</th>\n",
       "      <th>Salary</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>39</td>\n",
       "      <td>State-gov</td>\n",
       "      <td>Bachelors</td>\n",
       "      <td>13</td>\n",
       "      <td>Never-married</td>\n",
       "      <td>Adm-clerical</td>\n",
       "      <td>Not-in-family</td>\n",
       "      <td>White</td>\n",
       "      <td>Male</td>\n",
       "      <td>2174</td>\n",
       "      <td>0</td>\n",
       "      <td>40</td>\n",
       "      <td>United-States</td>\n",
       "      <td>&lt;=50K</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>50</td>\n",
       "      <td>Self-emp-not-inc</td>\n",
       "      <td>Bachelors</td>\n",
       "      <td>13</td>\n",
       "      <td>Married-civ-spouse</td>\n",
       "      <td>Exec-managerial</td>\n",
       "      <td>Husband</td>\n",
       "      <td>White</td>\n",
       "      <td>Male</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>13</td>\n",
       "      <td>United-States</td>\n",
       "      <td>&lt;=50K</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>38</td>\n",
       "      <td>Private</td>\n",
       "      <td>HS-grad</td>\n",
       "      <td>9</td>\n",
       "      <td>Divorced</td>\n",
       "      <td>Handlers-cleaners</td>\n",
       "      <td>Not-in-family</td>\n",
       "      <td>White</td>\n",
       "      <td>Male</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>40</td>\n",
       "      <td>United-States</td>\n",
       "      <td>&lt;=50K</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>53</td>\n",
       "      <td>Private</td>\n",
       "      <td>11th</td>\n",
       "      <td>7</td>\n",
       "      <td>Married-civ-spouse</td>\n",
       "      <td>Handlers-cleaners</td>\n",
       "      <td>Husband</td>\n",
       "      <td>Black</td>\n",
       "      <td>Male</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>40</td>\n",
       "      <td>United-States</td>\n",
       "      <td>&lt;=50K</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>28</td>\n",
       "      <td>Private</td>\n",
       "      <td>Bachelors</td>\n",
       "      <td>13</td>\n",
       "      <td>Married-civ-spouse</td>\n",
       "      <td>Prof-specialty</td>\n",
       "      <td>Wife</td>\n",
       "      <td>Black</td>\n",
       "      <td>Female</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>40</td>\n",
       "      <td>Cuba</td>\n",
       "      <td>&lt;=50K</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   age          workclass   education  educationno        maritalstatus  \\\n",
       "0   39          State-gov   Bachelors           13        Never-married   \n",
       "1   50   Self-emp-not-inc   Bachelors           13   Married-civ-spouse   \n",
       "2   38            Private     HS-grad            9             Divorced   \n",
       "3   53            Private        11th            7   Married-civ-spouse   \n",
       "4   28            Private   Bachelors           13   Married-civ-spouse   \n",
       "\n",
       "           occupation    relationship    race      sex  capitalgain  \\\n",
       "0        Adm-clerical   Not-in-family   White     Male         2174   \n",
       "1     Exec-managerial         Husband   White     Male            0   \n",
       "2   Handlers-cleaners   Not-in-family   White     Male            0   \n",
       "3   Handlers-cleaners         Husband   Black     Male            0   \n",
       "4      Prof-specialty            Wife   Black   Female            0   \n",
       "\n",
       "   capitalloss  hoursperweek          native  Salary  \n",
       "0            0            40   United-States   <=50K  \n",
       "1            0            13   United-States   <=50K  \n",
       "2            0            40   United-States   <=50K  \n",
       "3            0            40   United-States   <=50K  \n",
       "4            0            40            Cuba   <=50K  "
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# preview the Training dataset\n",
    "\n",
    "salarydata_train.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
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
       "      <th>age</th>\n",
       "      <th>workclass</th>\n",
       "      <th>education</th>\n",
       "      <th>educationno</th>\n",
       "      <th>maritalstatus</th>\n",
       "      <th>occupation</th>\n",
       "      <th>relationship</th>\n",
       "      <th>race</th>\n",
       "      <th>sex</th>\n",
       "      <th>capitalgain</th>\n",
       "      <th>capitalloss</th>\n",
       "      <th>hoursperweek</th>\n",
       "      <th>native</th>\n",
       "      <th>Salary</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>25</td>\n",
       "      <td>Private</td>\n",
       "      <td>11th</td>\n",
       "      <td>7</td>\n",
       "      <td>Never-married</td>\n",
       "      <td>Machine-op-inspct</td>\n",
       "      <td>Own-child</td>\n",
       "      <td>Black</td>\n",
       "      <td>Male</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>40</td>\n",
       "      <td>United-States</td>\n",
       "      <td>&lt;=50K</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>38</td>\n",
       "      <td>Private</td>\n",
       "      <td>HS-grad</td>\n",
       "      <td>9</td>\n",
       "      <td>Married-civ-spouse</td>\n",
       "      <td>Farming-fishing</td>\n",
       "      <td>Husband</td>\n",
       "      <td>White</td>\n",
       "      <td>Male</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>50</td>\n",
       "      <td>United-States</td>\n",
       "      <td>&lt;=50K</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>28</td>\n",
       "      <td>Local-gov</td>\n",
       "      <td>Assoc-acdm</td>\n",
       "      <td>12</td>\n",
       "      <td>Married-civ-spouse</td>\n",
       "      <td>Protective-serv</td>\n",
       "      <td>Husband</td>\n",
       "      <td>White</td>\n",
       "      <td>Male</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>40</td>\n",
       "      <td>United-States</td>\n",
       "      <td>&gt;50K</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>44</td>\n",
       "      <td>Private</td>\n",
       "      <td>Some-college</td>\n",
       "      <td>10</td>\n",
       "      <td>Married-civ-spouse</td>\n",
       "      <td>Machine-op-inspct</td>\n",
       "      <td>Husband</td>\n",
       "      <td>Black</td>\n",
       "      <td>Male</td>\n",
       "      <td>7688</td>\n",
       "      <td>0</td>\n",
       "      <td>40</td>\n",
       "      <td>United-States</td>\n",
       "      <td>&gt;50K</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>34</td>\n",
       "      <td>Private</td>\n",
       "      <td>10th</td>\n",
       "      <td>6</td>\n",
       "      <td>Never-married</td>\n",
       "      <td>Other-service</td>\n",
       "      <td>Not-in-family</td>\n",
       "      <td>White</td>\n",
       "      <td>Male</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>30</td>\n",
       "      <td>United-States</td>\n",
       "      <td>&lt;=50K</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   age   workclass      education  educationno        maritalstatus  \\\n",
       "0   25     Private           11th            7        Never-married   \n",
       "1   38     Private        HS-grad            9   Married-civ-spouse   \n",
       "2   28   Local-gov     Assoc-acdm           12   Married-civ-spouse   \n",
       "3   44     Private   Some-college           10   Married-civ-spouse   \n",
       "4   34     Private           10th            6        Never-married   \n",
       "\n",
       "           occupation    relationship    race    sex  capitalgain  \\\n",
       "0   Machine-op-inspct       Own-child   Black   Male            0   \n",
       "1     Farming-fishing         Husband   White   Male            0   \n",
       "2     Protective-serv         Husband   White   Male            0   \n",
       "3   Machine-op-inspct         Husband   Black   Male         7688   \n",
       "4       Other-service   Not-in-family   White   Male            0   \n",
       "\n",
       "   capitalloss  hoursperweek          native  Salary  \n",
       "0            0            40   United-States   <=50K  \n",
       "1            0            50   United-States   <=50K  \n",
       "2            0            40   United-States    >50K  \n",
       "3            0            40   United-States    >50K  \n",
       "4            0            30   United-States   <=50K  "
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# preview the Test dataset\n",
    "\n",
    "salarydata_test.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# View summary of Training dataset"
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
      "RangeIndex: 30161 entries, 0 to 30160\n",
      "Data columns (total 14 columns):\n",
      " #   Column         Non-Null Count  Dtype \n",
      "---  ------         --------------  ----- \n",
      " 0   age            30161 non-null  int64 \n",
      " 1   workclass      30161 non-null  object\n",
      " 2   education      30161 non-null  object\n",
      " 3   educationno    30161 non-null  int64 \n",
      " 4   maritalstatus  30161 non-null  object\n",
      " 5   occupation     30161 non-null  object\n",
      " 6   relationship   30161 non-null  object\n",
      " 7   race           30161 non-null  object\n",
      " 8   sex            30161 non-null  object\n",
      " 9   capitalgain    30161 non-null  int64 \n",
      " 10  capitalloss    30161 non-null  int64 \n",
      " 11  hoursperweek   30161 non-null  int64 \n",
      " 12  native         30161 non-null  object\n",
      " 13  Salary         30161 non-null  object\n",
      "dtypes: int64(5), object(9)\n",
      "memory usage: 3.2+ MB\n"
     ]
    }
   ],
   "source": [
    "salarydata_train.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
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
       "      <th>age</th>\n",
       "      <th>educationno</th>\n",
       "      <th>capitalgain</th>\n",
       "      <th>capitalloss</th>\n",
       "      <th>hoursperweek</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>30161.000000</td>\n",
       "      <td>30161.000000</td>\n",
       "      <td>30161.000000</td>\n",
       "      <td>30161.000000</td>\n",
       "      <td>30161.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>38.438115</td>\n",
       "      <td>10.121316</td>\n",
       "      <td>1092.044064</td>\n",
       "      <td>88.302311</td>\n",
       "      <td>40.931269</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>13.134830</td>\n",
       "      <td>2.550037</td>\n",
       "      <td>7406.466611</td>\n",
       "      <td>404.121321</td>\n",
       "      <td>11.980182</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>17.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>28.000000</td>\n",
       "      <td>9.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>40.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>37.000000</td>\n",
       "      <td>10.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>40.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>47.000000</td>\n",
       "      <td>13.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>45.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>90.000000</td>\n",
       "      <td>16.000000</td>\n",
       "      <td>99999.000000</td>\n",
       "      <td>4356.000000</td>\n",
       "      <td>99.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                age   educationno   capitalgain   capitalloss  hoursperweek\n",
       "count  30161.000000  30161.000000  30161.000000  30161.000000  30161.000000\n",
       "mean      38.438115     10.121316   1092.044064     88.302311     40.931269\n",
       "std       13.134830      2.550037   7406.466611    404.121321     11.980182\n",
       "min       17.000000      1.000000      0.000000      0.000000      1.000000\n",
       "25%       28.000000      9.000000      0.000000      0.000000     40.000000\n",
       "50%       37.000000     10.000000      0.000000      0.000000     40.000000\n",
       "75%       47.000000     13.000000      0.000000      0.000000     45.000000\n",
       "max       90.000000     16.000000  99999.000000   4356.000000     99.000000"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "salarydata_train.describe()"
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
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 15060 entries, 0 to 15059\n",
      "Data columns (total 14 columns):\n",
      " #   Column         Non-Null Count  Dtype \n",
      "---  ------         --------------  ----- \n",
      " 0   age            15060 non-null  int64 \n",
      " 1   workclass      15060 non-null  object\n",
      " 2   education      15060 non-null  object\n",
      " 3   educationno    15060 non-null  int64 \n",
      " 4   maritalstatus  15060 non-null  object\n",
      " 5   occupation     15060 non-null  object\n",
      " 6   relationship   15060 non-null  object\n",
      " 7   race           15060 non-null  object\n",
      " 8   sex            15060 non-null  object\n",
      " 9   capitalgain    15060 non-null  int64 \n",
      " 10  capitalloss    15060 non-null  int64 \n",
      " 11  hoursperweek   15060 non-null  int64 \n",
      " 12  native         15060 non-null  object\n",
      " 13  Salary         15060 non-null  object\n",
      "dtypes: int64(5), object(9)\n",
      "memory usage: 1.6+ MB\n"
     ]
    }
   ],
   "source": [
    "salarydata_test.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
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
       "      <th>age</th>\n",
       "      <th>educationno</th>\n",
       "      <th>capitalgain</th>\n",
       "      <th>capitalloss</th>\n",
       "      <th>hoursperweek</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>15060.000000</td>\n",
       "      <td>15060.000000</td>\n",
       "      <td>15060.000000</td>\n",
       "      <td>15060.000000</td>\n",
       "      <td>15060.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>38.768327</td>\n",
       "      <td>10.112749</td>\n",
       "      <td>1120.301594</td>\n",
       "      <td>89.041899</td>\n",
       "      <td>40.951594</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>13.380676</td>\n",
       "      <td>2.558727</td>\n",
       "      <td>7703.181842</td>\n",
       "      <td>406.283245</td>\n",
       "      <td>12.062831</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>17.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>28.000000</td>\n",
       "      <td>9.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>40.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>37.000000</td>\n",
       "      <td>10.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>40.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>48.000000</td>\n",
       "      <td>13.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>45.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>90.000000</td>\n",
       "      <td>16.000000</td>\n",
       "      <td>99999.000000</td>\n",
       "      <td>3770.000000</td>\n",
       "      <td>99.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                age   educationno   capitalgain   capitalloss  hoursperweek\n",
       "count  15060.000000  15060.000000  15060.000000  15060.000000  15060.000000\n",
       "mean      38.768327     10.112749   1120.301594     89.041899     40.951594\n",
       "std       13.380676      2.558727   7703.181842    406.283245     12.062831\n",
       "min       17.000000      1.000000      0.000000      0.000000      1.000000\n",
       "25%       28.000000      9.000000      0.000000      0.000000     40.000000\n",
       "50%       37.000000     10.000000      0.000000      0.000000     40.000000\n",
       "75%       48.000000     13.000000      0.000000      0.000000     45.000000\n",
       "max       90.000000     16.000000  99999.000000   3770.000000     99.000000"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "salarydata_test.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "age              0\n",
       "workclass        0\n",
       "education        0\n",
       "educationno      0\n",
       "maritalstatus    0\n",
       "occupation       0\n",
       "relationship     0\n",
       "race             0\n",
       "sex              0\n",
       "capitalgain      0\n",
       "capitalloss      0\n",
       "hoursperweek     0\n",
       "native           0\n",
       "Salary           0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Finding the special characters in the data frame \n",
    "salarydata_train.isin(['?']).sum(axis=0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "age              0\n",
       "workclass        0\n",
       "education        0\n",
       "educationno      0\n",
       "maritalstatus    0\n",
       "occupation       0\n",
       "relationship     0\n",
       "race             0\n",
       "sex              0\n",
       "capitalgain      0\n",
       "capitalloss      0\n",
       "hoursperweek     0\n",
       "native           0\n",
       "Salary           0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Finding the special characters in the data frame \n",
    "salarydata_test.isin(['?']).sum(axis=0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "   age          workclass   education  educationno        maritalstatus  \\\n",
      "0   39          State-gov   Bachelors           13        Never-married   \n",
      "1   50   Self-emp-not-inc   Bachelors           13   Married-civ-spouse   \n",
      "2   38            Private     HS-grad            9             Divorced   \n",
      "3   53            Private        11th            7   Married-civ-spouse   \n",
      "4   28            Private   Bachelors           13   Married-civ-spouse   \n",
      "\n",
      "           occupation    relationship    race      sex  capitalgain  \\\n",
      "0        Adm-clerical   Not-in-family   White     Male         2174   \n",
      "1     Exec-managerial         Husband   White     Male            0   \n",
      "2   Handlers-cleaners   Not-in-family   White     Male            0   \n",
      "3   Handlers-cleaners         Husband   Black     Male            0   \n",
      "4      Prof-specialty            Wife   Black   Female            0   \n",
      "\n",
      "   capitalloss  hoursperweek          native  Salary  \n",
      "0            0            40   United-States   <=50K  \n",
      "1            0            13   United-States   <=50K  \n",
      "2            0            40   United-States   <=50K  \n",
      "3            0            40   United-States   <=50K  \n",
      "4            0            40            Cuba   <=50K  \n"
     ]
    }
   ],
   "source": [
    "print(salarydata_train[0:5])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Explore categorical variables"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "There are 9 categorical variables\n",
      "\n",
      "The categorical variables are :\n",
      "\n",
      " ['workclass', 'education', 'maritalstatus', 'occupation', 'relationship', 'race', 'sex', 'native', 'Salary']\n"
     ]
    }
   ],
   "source": [
    "# find categorical variables\n",
    "\n",
    "categorical = [var for var in salarydata_train.columns if salarydata_train[var].dtype=='O']\n",
    "\n",
    "print('There are {} categorical variables\\n'.format(len(categorical)))\n",
    "\n",
    "print('The categorical variables are :\\n\\n', categorical)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
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
       "      <th>workclass</th>\n",
       "      <th>education</th>\n",
       "      <th>maritalstatus</th>\n",
       "      <th>occupation</th>\n",
       "      <th>relationship</th>\n",
       "      <th>race</th>\n",
       "      <th>sex</th>\n",
       "      <th>native</th>\n",
       "      <th>Salary</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>State-gov</td>\n",
       "      <td>Bachelors</td>\n",
       "      <td>Never-married</td>\n",
       "      <td>Adm-clerical</td>\n",
       "      <td>Not-in-family</td>\n",
       "      <td>White</td>\n",
       "      <td>Male</td>\n",
       "      <td>United-States</td>\n",
       "      <td>&lt;=50K</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Self-emp-not-inc</td>\n",
       "      <td>Bachelors</td>\n",
       "      <td>Married-civ-spouse</td>\n",
       "      <td>Exec-managerial</td>\n",
       "      <td>Husband</td>\n",
       "      <td>White</td>\n",
       "      <td>Male</td>\n",
       "      <td>United-States</td>\n",
       "      <td>&lt;=50K</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Private</td>\n",
       "      <td>HS-grad</td>\n",
       "      <td>Divorced</td>\n",
       "      <td>Handlers-cleaners</td>\n",
       "      <td>Not-in-family</td>\n",
       "      <td>White</td>\n",
       "      <td>Male</td>\n",
       "      <td>United-States</td>\n",
       "      <td>&lt;=50K</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Private</td>\n",
       "      <td>11th</td>\n",
       "      <td>Married-civ-spouse</td>\n",
       "      <td>Handlers-cleaners</td>\n",
       "      <td>Husband</td>\n",
       "      <td>Black</td>\n",
       "      <td>Male</td>\n",
       "      <td>United-States</td>\n",
       "      <td>&lt;=50K</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Private</td>\n",
       "      <td>Bachelors</td>\n",
       "      <td>Married-civ-spouse</td>\n",
       "      <td>Prof-specialty</td>\n",
       "      <td>Wife</td>\n",
       "      <td>Black</td>\n",
       "      <td>Female</td>\n",
       "      <td>Cuba</td>\n",
       "      <td>&lt;=50K</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "           workclass   education        maritalstatus          occupation  \\\n",
       "0          State-gov   Bachelors        Never-married        Adm-clerical   \n",
       "1   Self-emp-not-inc   Bachelors   Married-civ-spouse     Exec-managerial   \n",
       "2            Private     HS-grad             Divorced   Handlers-cleaners   \n",
       "3            Private        11th   Married-civ-spouse   Handlers-cleaners   \n",
       "4            Private   Bachelors   Married-civ-spouse      Prof-specialty   \n",
       "\n",
       "     relationship    race      sex          native  Salary  \n",
       "0   Not-in-family   White     Male   United-States   <=50K  \n",
       "1         Husband   White     Male   United-States   <=50K  \n",
       "2   Not-in-family   White     Male   United-States   <=50K  \n",
       "3         Husband   Black     Male   United-States   <=50K  \n",
       "4            Wife   Black   Female            Cuba   <=50K  "
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# view the categorical variables\n",
    "\n",
    "salarydata_train[categorical].head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Summary of categorical variables\n",
    "\n",
    "There are 9 categorical variables.\n",
    "\n",
    "The categorical variables are given by workclass, education, maritalstatus, occupation, relationship, race, sex, native and Salary.\n",
    "\n",
    "Salary is the target variable."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Explore problems within categorical variables"
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
       "workclass        0\n",
       "education        0\n",
       "maritalstatus    0\n",
       "occupation       0\n",
       "relationship     0\n",
       "race             0\n",
       "sex              0\n",
       "native           0\n",
       "Salary           0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# check missing values in categorical variables\n",
    "salarydata_train[categorical].isnull().sum()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We can see that there are no missing values in the categorical variables. I will confirm this further."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " Private             22285\n",
      " Self-emp-not-inc     2499\n",
      " Local-gov            2067\n",
      " State-gov            1279\n",
      " Self-emp-inc         1074\n",
      " Federal-gov           943\n",
      " Without-pay            14\n",
      "Name: workclass, dtype: int64\n",
      " HS-grad         9840\n",
      " Some-college    6677\n",
      " Bachelors       5044\n",
      " Masters         1627\n",
      " Assoc-voc       1307\n",
      " 11th            1048\n",
      " Assoc-acdm      1008\n",
      " 10th             820\n",
      " 7th-8th          557\n",
      " Prof-school      542\n",
      " 9th              455\n",
      " 12th             377\n",
      " Doctorate        375\n",
      " 5th-6th          288\n",
      " 1st-4th          151\n",
      " Preschool         45\n",
      "Name: education, dtype: int64\n",
      " Married-civ-spouse       14065\n",
      " Never-married             9725\n",
      " Divorced                  4214\n",
      " Separated                  939\n",
      " Widowed                    827\n",
      " Married-spouse-absent      370\n",
      " Married-AF-spouse           21\n",
      "Name: maritalstatus, dtype: int64\n",
      " Prof-specialty       4038\n",
      " Craft-repair         4030\n",
      " Exec-managerial      3992\n",
      " Adm-clerical         3721\n",
      " Sales                3584\n",
      " Other-service        3212\n",
      " Machine-op-inspct    1965\n",
      " Transport-moving     1572\n",
      " Handlers-cleaners    1350\n",
      " Farming-fishing       989\n",
      " Tech-support          912\n",
      " Protective-serv       644\n",
      " Priv-house-serv       143\n",
      " Armed-Forces            9\n",
      "Name: occupation, dtype: int64\n",
      " Husband           12463\n",
      " Not-in-family      7726\n",
      " Own-child          4466\n",
      " Unmarried          3212\n",
      " Wife               1406\n",
      " Other-relative      888\n",
      "Name: relationship, dtype: int64\n",
      " White                 25932\n",
      " Black                  2817\n",
      " Asian-Pac-Islander      895\n",
      " Amer-Indian-Eskimo      286\n",
      " Other                   231\n",
      "Name: race, dtype: int64\n",
      " Male      20380\n",
      " Female     9781\n",
      "Name: sex, dtype: int64\n",
      " United-States                 27504\n",
      " Mexico                          610\n",
      " Philippines                     188\n",
      " Germany                         128\n",
      " Puerto-Rico                     109\n",
      " Canada                          107\n",
      " El-Salvador                     100\n",
      " India                           100\n",
      " Cuba                             92\n",
      " England                          86\n",
      " Jamaica                          80\n",
      " South                            71\n",
      " Italy                            68\n",
      " China                            68\n",
      " Dominican-Republic               67\n",
      " Vietnam                          64\n",
      " Guatemala                        63\n",
      " Japan                            59\n",
      " Columbia                         56\n",
      " Poland                           56\n",
      " Haiti                            42\n",
      " Iran                             42\n",
      " Taiwan                           42\n",
      " Portugal                         34\n",
      " Nicaragua                        33\n",
      " Peru                             30\n",
      " Greece                           29\n",
      " Ecuador                          27\n",
      " France                           27\n",
      " Ireland                          24\n",
      " Hong                             19\n",
      " Cambodia                         18\n",
      " Trinadad&Tobago                  18\n",
      " Laos                             17\n",
      " Thailand                         17\n",
      " Yugoslavia                       16\n",
      " Outlying-US(Guam-USVI-etc)       14\n",
      " Hungary                          13\n",
      " Honduras                         12\n",
      " Scotland                         11\n",
      "Name: native, dtype: int64\n",
      " <=50K    22653\n",
      " >50K      7508\n",
      "Name: Salary, dtype: int64\n"
     ]
    }
   ],
   "source": [
    "# view frequency counts of values in categorical variables\n",
    "\n",
    "for var in categorical: \n",
    "    \n",
    "    print(salarydata_train[var].value_counts())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " Private             0.738868\n",
      " Self-emp-not-inc    0.082855\n",
      " Local-gov           0.068532\n",
      " State-gov           0.042406\n",
      " Self-emp-inc        0.035609\n",
      " Federal-gov         0.031266\n",
      " Without-pay         0.000464\n",
      "Name: workclass, dtype: float64\n",
      " HS-grad         0.326249\n",
      " Some-college    0.221379\n",
      " Bachelors       0.167236\n",
      " Masters         0.053944\n",
      " Assoc-voc       0.043334\n",
      " 11th            0.034747\n",
      " Assoc-acdm      0.033421\n",
      " 10th            0.027187\n",
      " 7th-8th         0.018468\n",
      " Prof-school     0.017970\n",
      " 9th             0.015086\n",
      " 12th            0.012500\n",
      " Doctorate       0.012433\n",
      " 5th-6th         0.009549\n",
      " 1st-4th         0.005006\n",
      " Preschool       0.001492\n",
      "Name: education, dtype: float64\n",
      " Married-civ-spouse       0.466331\n",
      " Never-married            0.322436\n",
      " Divorced                 0.139717\n",
      " Separated                0.031133\n",
      " Widowed                  0.027420\n",
      " Married-spouse-absent    0.012267\n",
      " Married-AF-spouse        0.000696\n",
      "Name: maritalstatus, dtype: float64\n",
      " Prof-specialty       0.133882\n",
      " Craft-repair         0.133616\n",
      " Exec-managerial      0.132356\n",
      " Adm-clerical         0.123371\n",
      " Sales                0.118829\n",
      " Other-service        0.106495\n",
      " Machine-op-inspct    0.065150\n",
      " Transport-moving     0.052120\n",
      " Handlers-cleaners    0.044760\n",
      " Farming-fishing      0.032791\n",
      " Tech-support         0.030238\n",
      " Protective-serv      0.021352\n",
      " Priv-house-serv      0.004741\n",
      " Armed-Forces         0.000298\n",
      "Name: occupation, dtype: float64\n",
      " Husband           0.413216\n",
      " Not-in-family     0.256159\n",
      " Own-child         0.148072\n",
      " Unmarried         0.106495\n",
      " Wife              0.046616\n",
      " Other-relative    0.029442\n",
      "Name: relationship, dtype: float64\n",
      " White                 0.859786\n",
      " Black                 0.093399\n",
      " Asian-Pac-Islander    0.029674\n",
      " Amer-Indian-Eskimo    0.009482\n",
      " Other                 0.007659\n",
      "Name: race, dtype: float64\n",
      " Male      0.675707\n",
      " Female    0.324293\n",
      "Name: sex, dtype: float64\n",
      " United-States                 0.911906\n",
      " Mexico                        0.020225\n",
      " Philippines                   0.006233\n",
      " Germany                       0.004244\n",
      " Puerto-Rico                   0.003614\n",
      " Canada                        0.003548\n",
      " El-Salvador                   0.003316\n",
      " India                         0.003316\n",
      " Cuba                          0.003050\n",
      " England                       0.002851\n",
      " Jamaica                       0.002652\n",
      " South                         0.002354\n",
      " Italy                         0.002255\n",
      " China                         0.002255\n",
      " Dominican-Republic            0.002221\n",
      " Vietnam                       0.002122\n",
      " Guatemala                     0.002089\n",
      " Japan                         0.001956\n",
      " Columbia                      0.001857\n",
      " Poland                        0.001857\n",
      " Haiti                         0.001393\n",
      " Iran                          0.001393\n",
      " Taiwan                        0.001393\n",
      " Portugal                      0.001127\n",
      " Nicaragua                     0.001094\n",
      " Peru                          0.000995\n",
      " Greece                        0.000962\n",
      " Ecuador                       0.000895\n",
      " France                        0.000895\n",
      " Ireland                       0.000796\n",
      " Hong                          0.000630\n",
      " Cambodia                      0.000597\n",
      " Trinadad&Tobago               0.000597\n",
      " Laos                          0.000564\n",
      " Thailand                      0.000564\n",
      " Yugoslavia                    0.000530\n",
      " Outlying-US(Guam-USVI-etc)    0.000464\n",
      " Hungary                       0.000431\n",
      " Honduras                      0.000398\n",
      " Scotland                      0.000365\n",
      "Name: native, dtype: float64\n",
      " <=50K    0.751069\n",
      " >50K     0.248931\n",
      "Name: Salary, dtype: float64\n"
     ]
    }
   ],
   "source": [
    "# view frequency distribution of categorical variables\n",
    "\n",
    "for var in categorical: \n",
    "    \n",
    "    print(salarydata_train[var].value_counts()/np.float(len(salarydata_train)))"
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
       "array([' State-gov', ' Self-emp-not-inc', ' Private', ' Federal-gov',\n",
       "       ' Local-gov', ' Self-emp-inc', ' Without-pay'], dtype=object)"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# check labels in workclass variable\n",
    "\n",
    "salarydata_train.workclass.unique()"
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
       " Private             22285\n",
       " Self-emp-not-inc     2499\n",
       " Local-gov            2067\n",
       " State-gov            1279\n",
       " Self-emp-inc         1074\n",
       " Federal-gov           943\n",
       " Without-pay            14\n",
       "Name: workclass, dtype: int64"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# check frequency distribution of values in workclass variable\n",
    "\n",
    "salarydata_train.workclass.value_counts()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Explore occupation variable"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([' Adm-clerical', ' Exec-managerial', ' Handlers-cleaners',\n",
       "       ' Prof-specialty', ' Other-service', ' Sales', ' Transport-moving',\n",
       "       ' Farming-fishing', ' Machine-op-inspct', ' Tech-support',\n",
       "       ' Craft-repair', ' Protective-serv', ' Armed-Forces',\n",
       "       ' Priv-house-serv'], dtype=object)"
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# check labels in occupation variable\n",
    "\n",
    "salarydata_train.occupation.unique()"
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
       " Prof-specialty       4038\n",
       " Craft-repair         4030\n",
       " Exec-managerial      3992\n",
       " Adm-clerical         3721\n",
       " Sales                3584\n",
       " Other-service        3212\n",
       " Machine-op-inspct    1965\n",
       " Transport-moving     1572\n",
       " Handlers-cleaners    1350\n",
       " Farming-fishing       989\n",
       " Tech-support          912\n",
       " Protective-serv       644\n",
       " Priv-house-serv       143\n",
       " Armed-Forces            9\n",
       "Name: occupation, dtype: int64"
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# check frequency distribution of values in occupation variable\n",
    "\n",
    "salarydata_train.occupation.value_counts()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Explore native_country variable"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([' United-States', ' Cuba', ' Jamaica', ' India', ' Mexico',\n",
       "       ' Puerto-Rico', ' Honduras', ' England', ' Canada', ' Germany',\n",
       "       ' Iran', ' Philippines', ' Poland', ' Columbia', ' Cambodia',\n",
       "       ' Thailand', ' Ecuador', ' Laos', ' Taiwan', ' Haiti', ' Portugal',\n",
       "       ' Dominican-Republic', ' El-Salvador', ' France', ' Guatemala',\n",
       "       ' Italy', ' China', ' South', ' Japan', ' Yugoslavia', ' Peru',\n",
       "       ' Outlying-US(Guam-USVI-etc)', ' Scotland', ' Trinadad&Tobago',\n",
       "       ' Greece', ' Nicaragua', ' Vietnam', ' Hong', ' Ireland',\n",
       "       ' Hungary'], dtype=object)"
      ]
     },
     "execution_count": 24,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# check labels in native_country variable\n",
    "\n",
    "salarydata_train.native.unique()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       " United-States                 27504\n",
       " Mexico                          610\n",
       " Philippines                     188\n",
       " Germany                         128\n",
       " Puerto-Rico                     109\n",
       " Canada                          107\n",
       " El-Salvador                     100\n",
       " India                           100\n",
       " Cuba                             92\n",
       " England                          86\n",
       " Jamaica                          80\n",
       " South                            71\n",
       " Italy                            68\n",
       " China                            68\n",
       " Dominican-Republic               67\n",
       " Vietnam                          64\n",
       " Guatemala                        63\n",
       " Japan                            59\n",
       " Columbia                         56\n",
       " Poland                           56\n",
       " Haiti                            42\n",
       " Iran                             42\n",
       " Taiwan                           42\n",
       " Portugal                         34\n",
       " Nicaragua                        33\n",
       " Peru                             30\n",
       " Greece                           29\n",
       " Ecuador                          27\n",
       " France                           27\n",
       " Ireland                          24\n",
       " Hong                             19\n",
       " Cambodia                         18\n",
       " Trinadad&Tobago                  18\n",
       " Laos                             17\n",
       " Thailand                         17\n",
       " Yugoslavia                       16\n",
       " Outlying-US(Guam-USVI-etc)       14\n",
       " Hungary                          13\n",
       " Honduras                         12\n",
       " Scotland                         11\n",
       "Name: native, dtype: int64"
      ]
     },
     "execution_count": 25,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# check frequency distribution of values in native_country variable\n",
    "\n",
    "salarydata_train.native.value_counts()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Number of labels: cardinality"
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
      "workclass  contains  7  labels\n",
      "education  contains  16  labels\n",
      "maritalstatus  contains  7  labels\n",
      "occupation  contains  14  labels\n",
      "relationship  contains  6  labels\n",
      "race  contains  5  labels\n",
      "sex  contains  2  labels\n",
      "native  contains  40  labels\n",
      "Salary  contains  2  labels\n"
     ]
    }
   ],
   "source": [
    "# check for cardinality in categorical variables\n",
    "\n",
    "for var in categorical:\n",
    "    \n",
    "    print(var, ' contains ', len(salarydata_train[var].unique()), ' labels')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Explore Numerical Variables"
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
      "There are 5 numerical variables\n",
      "\n",
      "The numerical variables are : ['age', 'educationno', 'capitalgain', 'capitalloss', 'hoursperweek']\n"
     ]
    }
   ],
   "source": [
    "# find numerical variables\n",
    "\n",
    "numerical = [var for var in salarydata_train.columns if salarydata_train[var].dtype!='O']\n",
    "\n",
    "print('There are {} numerical variables\\n'.format(len(numerical)))\n",
    "\n",
    "print('The numerical variables are :', numerical)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
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
       "      <th>age</th>\n",
       "      <th>educationno</th>\n",
       "      <th>capitalgain</th>\n",
       "      <th>capitalloss</th>\n",
       "      <th>hoursperweek</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>39</td>\n",
       "      <td>13</td>\n",
       "      <td>2174</td>\n",
       "      <td>0</td>\n",
       "      <td>40</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>50</td>\n",
       "      <td>13</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>13</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>38</td>\n",
       "      <td>9</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>40</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>53</td>\n",
       "      <td>7</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>40</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>28</td>\n",
       "      <td>13</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>40</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   age  educationno  capitalgain  capitalloss  hoursperweek\n",
       "0   39           13         2174            0            40\n",
       "1   50           13            0            0            13\n",
       "2   38            9            0            0            40\n",
       "3   53            7            0            0            40\n",
       "4   28           13            0            0            40"
      ]
     },
     "execution_count": 28,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# view the numerical variables\n",
    "\n",
    "salarydata_train[numerical].head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Summary of numerical variables\n",
    "\n",
    "There are 5 numerical variables.\n",
    "\n",
    "These are given by age, educationno, capitalgain, capitalloss and hoursperweek.\n",
    "All of the numerical variables are of discrete data type."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Explore problems within numerical variables"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "age             0\n",
       "educationno     0\n",
       "capitalgain     0\n",
       "capitalloss     0\n",
       "hoursperweek    0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# check missing values in numerical variables\n",
    "\n",
    "salarydata_train[numerical].isnull().sum()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Declare feature vector and target variable"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [],
   "source": [
    "X = salarydata_train.drop(['Salary'], axis=1)\n",
    "\n",
    "y = salarydata_train['Salary']"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Split data into separate training and test set"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [],
   "source": [
    "# split X and y into training and testing sets\n",
    "\n",
    "from sklearn.model_selection import train_test_split\n",
    "\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "((21112, 13), (9049, 13))"
      ]
     },
     "execution_count": 32,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# check the shape of X_train and X_test\n",
    "\n",
    "X_train.shape, X_test.shape"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Feature Engineering "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "age               int64\n",
       "workclass        object\n",
       "education        object\n",
       "educationno       int64\n",
       "maritalstatus    object\n",
       "occupation       object\n",
       "relationship     object\n",
       "race             object\n",
       "sex              object\n",
       "capitalgain       int64\n",
       "capitalloss       int64\n",
       "hoursperweek      int64\n",
       "native           object\n",
       "dtype: object"
      ]
     },
     "execution_count": 33,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_train.dtypes"
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
       "age               int64\n",
       "workclass        object\n",
       "education        object\n",
       "educationno       int64\n",
       "maritalstatus    object\n",
       "occupation       object\n",
       "relationship     object\n",
       "race             object\n",
       "sex              object\n",
       "capitalgain       int64\n",
       "capitalloss       int64\n",
       "hoursperweek      int64\n",
       "native           object\n",
       "dtype: object"
      ]
     },
     "execution_count": 34,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_test.dtypes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['workclass',\n",
       " 'education',\n",
       " 'maritalstatus',\n",
       " 'occupation',\n",
       " 'relationship',\n",
       " 'race',\n",
       " 'sex',\n",
       " 'native']"
      ]
     },
     "execution_count": 35,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# display categorical variables\n",
    "\n",
    "categorical = [col for col in X_train.columns if X_train[col].dtypes == 'O']\n",
    "\n",
    "categorical"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['age', 'educationno', 'capitalgain', 'capitalloss', 'hoursperweek']"
      ]
     },
     "execution_count": 36,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# display numerical variables\n",
    "\n",
    "numerical = [col for col in X_train.columns if X_train[col].dtypes != 'O']\n",
    "\n",
    "numerical"
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
       "workclass        0.0\n",
       "education        0.0\n",
       "maritalstatus    0.0\n",
       "occupation       0.0\n",
       "relationship     0.0\n",
       "race             0.0\n",
       "sex              0.0\n",
       "native           0.0\n",
       "dtype: float64"
      ]
     },
     "execution_count": 37,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# print percentage of missing values in the categorical variables in training set\n",
    "\n",
    "X_train[categorical].isnull().mean()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {},
   "outputs": [],
   "source": [
    "# print categorical variables with missing data\n",
    "\n",
    "for col in categorical:\n",
    "    if X_train[col].isnull().mean()>0:\n",
    "        print(col, (X_train[col].isnull().mean()))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {},
   "outputs": [],
   "source": [
    "# impute missing categorical variables with most frequent value\n",
    "\n",
    "for df2 in [X_train, X_test]:\n",
    "    df2['workclass'].fillna(X_train['workclass'].mode()[0], inplace=True)\n",
    "    df2['occupation'].fillna(X_train['occupation'].mode()[0], inplace=True)\n",
    "    df2['native'].fillna(X_train['native'].mode()[0], inplace=True)  "
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
       "workclass        0\n",
       "education        0\n",
       "maritalstatus    0\n",
       "occupation       0\n",
       "relationship     0\n",
       "race             0\n",
       "sex              0\n",
       "native           0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 40,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# check missing values in categorical variables in X_train\n",
    "\n",
    "X_train[categorical].isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "workclass        0\n",
       "education        0\n",
       "maritalstatus    0\n",
       "occupation       0\n",
       "relationship     0\n",
       "race             0\n",
       "sex              0\n",
       "native           0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 41,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# check missing values in categorical variables in X_test\n",
    "\n",
    "X_test[categorical].isnull().sum()"
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
       "age              0\n",
       "workclass        0\n",
       "education        0\n",
       "educationno      0\n",
       "maritalstatus    0\n",
       "occupation       0\n",
       "relationship     0\n",
       "race             0\n",
       "sex              0\n",
       "capitalgain      0\n",
       "capitalloss      0\n",
       "hoursperweek     0\n",
       "native           0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 42,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# check missing values in X_train\n",
    "\n",
    "X_train.isnull().sum()"
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
       "age              0\n",
       "workclass        0\n",
       "education        0\n",
       "educationno      0\n",
       "maritalstatus    0\n",
       "occupation       0\n",
       "relationship     0\n",
       "race             0\n",
       "sex              0\n",
       "capitalgain      0\n",
       "capitalloss      0\n",
       "hoursperweek     0\n",
       "native           0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 43,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# check missing values in X_test\n",
    "\n",
    "X_test.isnull().sum()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Encode categorical variables"
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
       "['workclass',\n",
       " 'education',\n",
       " 'maritalstatus',\n",
       " 'occupation',\n",
       " 'relationship',\n",
       " 'race',\n",
       " 'sex',\n",
       " 'native']"
      ]
     },
     "execution_count": 44,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# print categorical variables\n",
    "\n",
    "categorical"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
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
       "      <th>workclass</th>\n",
       "      <th>education</th>\n",
       "      <th>maritalstatus</th>\n",
       "      <th>occupation</th>\n",
       "      <th>relationship</th>\n",
       "      <th>race</th>\n",
       "      <th>sex</th>\n",
       "      <th>native</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>8166</th>\n",
       "      <td>Local-gov</td>\n",
       "      <td>Some-college</td>\n",
       "      <td>Married-civ-spouse</td>\n",
       "      <td>Protective-serv</td>\n",
       "      <td>Husband</td>\n",
       "      <td>White</td>\n",
       "      <td>Male</td>\n",
       "      <td>United-States</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7138</th>\n",
       "      <td>Private</td>\n",
       "      <td>Some-college</td>\n",
       "      <td>Never-married</td>\n",
       "      <td>Other-service</td>\n",
       "      <td>Own-child</td>\n",
       "      <td>White</td>\n",
       "      <td>Male</td>\n",
       "      <td>United-States</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>437</th>\n",
       "      <td>Private</td>\n",
       "      <td>HS-grad</td>\n",
       "      <td>Never-married</td>\n",
       "      <td>Transport-moving</td>\n",
       "      <td>Not-in-family</td>\n",
       "      <td>White</td>\n",
       "      <td>Male</td>\n",
       "      <td>United-States</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5436</th>\n",
       "      <td>Private</td>\n",
       "      <td>HS-grad</td>\n",
       "      <td>Married-civ-spouse</td>\n",
       "      <td>Craft-repair</td>\n",
       "      <td>Husband</td>\n",
       "      <td>White</td>\n",
       "      <td>Male</td>\n",
       "      <td>United-States</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6541</th>\n",
       "      <td>Self-emp-not-inc</td>\n",
       "      <td>HS-grad</td>\n",
       "      <td>Married-civ-spouse</td>\n",
       "      <td>Tech-support</td>\n",
       "      <td>Husband</td>\n",
       "      <td>White</td>\n",
       "      <td>Male</td>\n",
       "      <td>United-States</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "              workclass      education        maritalstatus  \\\n",
       "8166          Local-gov   Some-college   Married-civ-spouse   \n",
       "7138            Private   Some-college        Never-married   \n",
       "437             Private        HS-grad        Never-married   \n",
       "5436            Private        HS-grad   Married-civ-spouse   \n",
       "6541   Self-emp-not-inc        HS-grad   Married-civ-spouse   \n",
       "\n",
       "             occupation    relationship    race    sex          native  \n",
       "8166    Protective-serv         Husband   White   Male   United-States  \n",
       "7138      Other-service       Own-child   White   Male   United-States  \n",
       "437    Transport-moving   Not-in-family   White   Male   United-States  \n",
       "5436       Craft-repair         Husband   White   Male   United-States  \n",
       "6541       Tech-support         Husband   White   Male   United-States  "
      ]
     },
     "execution_count": 45,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_train[categorical].head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: category_encoders in c:\\programdata\\anaconda3\\lib\\site-packages (2.2.2)\n",
      "Requirement already satisfied: numpy>=1.14.0 in c:\\programdata\\anaconda3\\lib\\site-packages (from category_encoders) (1.18.5)\n",
      "Requirement already satisfied: statsmodels>=0.9.0 in c:\\programdata\\anaconda3\\lib\\site-packages (from category_encoders) (0.11.1)\n",
      "Requirement already satisfied: scipy>=1.0.0 in c:\\programdata\\anaconda3\\lib\\site-packages (from category_encoders) (1.5.0)\n",
      "Requirement already satisfied: scikit-learn>=0.20.0 in c:\\programdata\\anaconda3\\lib\\site-packages (from category_encoders) (0.23.1)\n",
      "Requirement already satisfied: patsy>=0.5.1 in c:\\programdata\\anaconda3\\lib\\site-packages (from category_encoders) (0.5.1)\n",
      "Requirement already satisfied: pandas>=0.21.1 in c:\\programdata\\anaconda3\\lib\\site-packages (from category_encoders) (1.0.5)\n",
      "Requirement already satisfied: joblib>=0.11 in c:\\programdata\\anaconda3\\lib\\site-packages (from scikit-learn>=0.20.0->category_encoders) (0.16.0)\n",
      "Requirement already satisfied: threadpoolctl>=2.0.0 in c:\\programdata\\anaconda3\\lib\\site-packages (from scikit-learn>=0.20.0->category_encoders) (2.1.0)\n",
      "Requirement already satisfied: six in c:\\programdata\\anaconda3\\lib\\site-packages (from patsy>=0.5.1->category_encoders) (1.15.0)\n",
      "Requirement already satisfied: pytz>=2017.2 in c:\\programdata\\anaconda3\\lib\\site-packages (from pandas>=0.21.1->category_encoders) (2020.1)\n",
      "Requirement already satisfied: python-dateutil>=2.6.1 in c:\\programdata\\anaconda3\\lib\\site-packages (from pandas>=0.21.1->category_encoders) (2.8.1)\n"
     ]
    }
   ],
   "source": [
    "!pip install category_encoders"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "metadata": {},
   "outputs": [],
   "source": [
    "# import category encoders\n",
    "\n",
    "import category_encoders as ce"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "metadata": {},
   "outputs": [],
   "source": [
    "# encode remaining variables with one-hot encoding\n",
    "\n",
    "encoder = ce.OneHotEncoder(cols=['workclass', 'education', 'maritalstatus', 'occupation', 'relationship', \n",
    "                                 'race', 'sex', 'native'])\n",
    "\n",
    "X_train = encoder.fit_transform(X_train)\n",
    "\n",
    "X_test = encoder.transform(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
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
       "      <th>age</th>\n",
       "      <th>workclass_1</th>\n",
       "      <th>workclass_2</th>\n",
       "      <th>workclass_3</th>\n",
       "      <th>workclass_4</th>\n",
       "      <th>workclass_5</th>\n",
       "      <th>workclass_6</th>\n",
       "      <th>workclass_7</th>\n",
       "      <th>education_1</th>\n",
       "      <th>education_2</th>\n",
       "      <th>...</th>\n",
       "      <th>native_31</th>\n",
       "      <th>native_32</th>\n",
       "      <th>native_33</th>\n",
       "      <th>native_34</th>\n",
       "      <th>native_35</th>\n",
       "      <th>native_36</th>\n",
       "      <th>native_37</th>\n",
       "      <th>native_38</th>\n",
       "      <th>native_39</th>\n",
       "      <th>native_40</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>8166</th>\n",
       "      <td>54</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7138</th>\n",
       "      <td>21</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>437</th>\n",
       "      <td>30</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5436</th>\n",
       "      <td>42</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6541</th>\n",
       "      <td>37</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 102 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "      age  workclass_1  workclass_2  workclass_3  workclass_4  workclass_5  \\\n",
       "8166   54            1            0            0            0            0   \n",
       "7138   21            0            1            0            0            0   \n",
       "437    30            0            1            0            0            0   \n",
       "5436   42            0            1            0            0            0   \n",
       "6541   37            0            0            1            0            0   \n",
       "\n",
       "      workclass_6  workclass_7  education_1  education_2  ...  native_31  \\\n",
       "8166            0            0            1            0  ...          0   \n",
       "7138            0            0            1            0  ...          0   \n",
       "437             0            0            0            1  ...          0   \n",
       "5436            0            0            0            1  ...          0   \n",
       "6541            0            0            0            1  ...          0   \n",
       "\n",
       "      native_32  native_33  native_34  native_35  native_36  native_37  \\\n",
       "8166          0          0          0          0          0          0   \n",
       "7138          0          0          0          0          0          0   \n",
       "437           0          0          0          0          0          0   \n",
       "5436          0          0          0          0          0          0   \n",
       "6541          0          0          0          0          0          0   \n",
       "\n",
       "      native_38  native_39  native_40  \n",
       "8166          0          0          0  \n",
       "7138          0          0          0  \n",
       "437           0          0          0  \n",
       "5436          0          0          0  \n",
       "6541          0          0          0  \n",
       "\n",
       "[5 rows x 102 columns]"
      ]
     },
     "execution_count": 49,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_train.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(21112, 102)"
      ]
     },
     "execution_count": 50,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_train.shape"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We can see that from the initial 14 columns, we now have 102 columns."
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
       "      <th>age</th>\n",
       "      <th>workclass_1</th>\n",
       "      <th>workclass_2</th>\n",
       "      <th>workclass_3</th>\n",
       "      <th>workclass_4</th>\n",
       "      <th>workclass_5</th>\n",
       "      <th>workclass_6</th>\n",
       "      <th>workclass_7</th>\n",
       "      <th>education_1</th>\n",
       "      <th>education_2</th>\n",
       "      <th>...</th>\n",
       "      <th>native_31</th>\n",
       "      <th>native_32</th>\n",
       "      <th>native_33</th>\n",
       "      <th>native_34</th>\n",
       "      <th>native_35</th>\n",
       "      <th>native_36</th>\n",
       "      <th>native_37</th>\n",
       "      <th>native_38</th>\n",
       "      <th>native_39</th>\n",
       "      <th>native_40</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>25338</th>\n",
       "      <td>21</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>18840</th>\n",
       "      <td>21</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8391</th>\n",
       "      <td>56</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>18258</th>\n",
       "      <td>43</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>16669</th>\n",
       "      <td>53</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 102 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "       age  workclass_1  workclass_2  workclass_3  workclass_4  workclass_5  \\\n",
       "25338   21            0            1            0            0            0   \n",
       "18840   21            0            1            0            0            0   \n",
       "8391    56            0            1            0            0            0   \n",
       "18258   43            1            0            0            0            0   \n",
       "16669   53            0            0            0            1            0   \n",
       "\n",
       "       workclass_6  workclass_7  education_1  education_2  ...  native_31  \\\n",
       "25338            0            0            0            1  ...          0   \n",
       "18840            0            0            0            0  ...          0   \n",
       "8391             0            0            0            0  ...          0   \n",
       "18258            0            0            1            0  ...          0   \n",
       "16669            0            0            0            0  ...          0   \n",
       "\n",
       "       native_32  native_33  native_34  native_35  native_36  native_37  \\\n",
       "25338          0          0          0          0          0          0   \n",
       "18840          0          0          0          0          0          0   \n",
       "8391           0          0          0          0          0          0   \n",
       "18258          0          0          0          0          0          0   \n",
       "16669          1          0          0          0          0          0   \n",
       "\n",
       "       native_38  native_39  native_40  \n",
       "25338          0          0          0  \n",
       "18840          0          0          0  \n",
       "8391           0          0          0  \n",
       "18258          0          0          0  \n",
       "16669          0          0          0  \n",
       "\n",
       "[5 rows x 102 columns]"
      ]
     },
     "execution_count": 51,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_test.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(9049, 102)"
      ]
     },
     "execution_count": 52,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_test.shape"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We now have training and testing set ready for model building. \n",
    "Before that, we should map all the feature variables onto the same scale. \n",
    "It is called feature scaling."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Feature Scaling"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "metadata": {},
   "outputs": [],
   "source": [
    "cols = X_train.columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.preprocessing import RobustScaler\n",
    "\n",
    "scaler = RobustScaler()\n",
    "\n",
    "X_train = scaler.fit_transform(X_train)\n",
    "\n",
    "X_test = scaler.transform(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train = pd.DataFrame(X_train, columns=[cols])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_test = pd.DataFrame(X_test, columns=[cols])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
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
       "    .dataframe thead tr th {\n",
       "        text-align: left;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr>\n",
       "      <th></th>\n",
       "      <th>age</th>\n",
       "      <th>workclass_1</th>\n",
       "      <th>workclass_2</th>\n",
       "      <th>workclass_3</th>\n",
       "      <th>workclass_4</th>\n",
       "      <th>workclass_5</th>\n",
       "      <th>workclass_6</th>\n",
       "      <th>workclass_7</th>\n",
       "      <th>education_1</th>\n",
       "      <th>education_2</th>\n",
       "      <th>...</th>\n",
       "      <th>native_31</th>\n",
       "      <th>native_32</th>\n",
       "      <th>native_33</th>\n",
       "      <th>native_34</th>\n",
       "      <th>native_35</th>\n",
       "      <th>native_36</th>\n",
       "      <th>native_37</th>\n",
       "      <th>native_38</th>\n",
       "      <th>native_39</th>\n",
       "      <th>native_40</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0.894737</td>\n",
       "      <td>1.0</td>\n",
       "      <td>-1.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>...</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>-0.842105</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>...</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>-0.368421</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>...</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>0.263158</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>...</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.0</td>\n",
       "      <td>-1.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>...</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 102 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "        age workclass_1 workclass_2 workclass_3 workclass_4 workclass_5  \\\n",
       "0  0.894737         1.0        -1.0         0.0         0.0         0.0   \n",
       "1 -0.842105         0.0         0.0         0.0         0.0         0.0   \n",
       "2 -0.368421         0.0         0.0         0.0         0.0         0.0   \n",
       "3  0.263158         0.0         0.0         0.0         0.0         0.0   \n",
       "4  0.000000         0.0        -1.0         1.0         0.0         0.0   \n",
       "\n",
       "  workclass_6 workclass_7 education_1 education_2  ... native_31 native_32  \\\n",
       "0         0.0         0.0         1.0         0.0  ...       0.0       0.0   \n",
       "1         0.0         0.0         1.0         0.0  ...       0.0       0.0   \n",
       "2         0.0         0.0         0.0         1.0  ...       0.0       0.0   \n",
       "3         0.0         0.0         0.0         1.0  ...       0.0       0.0   \n",
       "4         0.0         0.0         0.0         1.0  ...       0.0       0.0   \n",
       "\n",
       "  native_33 native_34 native_35 native_36 native_37 native_38 native_39  \\\n",
       "0       0.0       0.0       0.0       0.0       0.0       0.0       0.0   \n",
       "1       0.0       0.0       0.0       0.0       0.0       0.0       0.0   \n",
       "2       0.0       0.0       0.0       0.0       0.0       0.0       0.0   \n",
       "3       0.0       0.0       0.0       0.0       0.0       0.0       0.0   \n",
       "4       0.0       0.0       0.0       0.0       0.0       0.0       0.0   \n",
       "\n",
       "  native_40  \n",
       "0       0.0  \n",
       "1       0.0  \n",
       "2       0.0  \n",
       "3       0.0  \n",
       "4       0.0  \n",
       "\n",
       "[5 rows x 102 columns]"
      ]
     },
     "execution_count": 57,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_train.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We now have X_train dataset ready to be fed into the Gaussian Naive Bayes classifier."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Model training"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 58,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "GaussianNB()"
      ]
     },
     "execution_count": 58,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# train a Gaussian Naive Bayes classifier on the training set\n",
    "from sklearn.naive_bayes import GaussianNB\n",
    "\n",
    "\n",
    "# instantiate the model\n",
    "gnb = GaussianNB()\n",
    "\n",
    "\n",
    "# fit the model\n",
    "gnb.fit(X_train, y_train)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Predict the results "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 59,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([' <=50K', ' <=50K', ' <=50K', ..., ' <=50K', ' <=50K', ' >50K'],\n",
       "      dtype='<U6')"
      ]
     },
     "execution_count": 59,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_pred = gnb.predict(X_test)\n",
    "\n",
    "y_pred"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Check accuracy score"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Model accuracy score: 0.7995\n"
     ]
    }
   ],
   "source": [
    "from sklearn.metrics import accuracy_score\n",
    "\n",
    "print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Here, y_test are the true class labels and y_pred are the predicted class labels in the test-set."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Compare the train-set and test-set accuracy"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([' >50K', ' <=50K', ' <=50K', ..., ' <=50K', ' >50K', ' <=50K'],\n",
       "      dtype='<U6')"
      ]
     },
     "execution_count": 61,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_pred_train = gnb.predict(X_train)\n",
    "\n",
    "y_pred_train"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 62,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Training-set accuracy score: 0.8023\n"
     ]
    }
   ],
   "source": [
    "print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train)))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Check for overfitting and underfitting"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 63,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Training set score: 0.8023\n",
      "Test set score: 0.7995\n"
     ]
    }
   ],
   "source": [
    "# print the scores on training and test set\n",
    "\n",
    "print('Training set score: {:.4f}'.format(gnb.score(X_train, y_train)))\n",
    "\n",
    "print('Test set score: {:.4f}'.format(gnb.score(X_test, y_test)))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The training-set accuracy score is 0.8023 while the test-set accuracy to be 0.7995. \n",
    "These two values are quite comparable. So, there is no sign of overfitting"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Compare model accuracy with null accuracy"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 64,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       " <=50K    6798\n",
       " >50K     2251\n",
       "Name: Salary, dtype: int64"
      ]
     },
     "execution_count": 64,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# check class distribution in test set\n",
    "\n",
    "y_test.value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 65,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Null accuracy score: 0.7582\n"
     ]
    }
   ],
   "source": [
    "# check null accuracy score\n",
    "\n",
    "null_accuracy = (7407/(7407+2362))\n",
    "\n",
    "print('Null accuracy score: {0:0.4f}'. format(null_accuracy))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We can see that our model accuracy score is 0.8023 but null accuracy score is 0.7582. \n",
    "So, we can conclude that our Gaussian Naive Bayes Classification model is doing a very good job in predicting the class labels."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Confusion matrix "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 66,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Confusion matrix\n",
      "\n",
      " [[5422 1376]\n",
      " [ 438 1813]]\n",
      "\n",
      "True Positives(TP) =  5422\n",
      "\n",
      "True Negatives(TN) =  1813\n",
      "\n",
      "False Positives(FP) =  1376\n",
      "\n",
      "False Negatives(FN) =  438\n"
     ]
    }
   ],
   "source": [
    "# Print the Confusion Matrix and slice it into four pieces\n",
    "\n",
    "from sklearn.metrics import confusion_matrix\n",
    "\n",
    "cm = confusion_matrix(y_test, y_pred)\n",
    "\n",
    "print('Confusion matrix\\n\\n', cm)\n",
    "\n",
    "print('\\nTrue Positives(TP) = ', cm[0,0])\n",
    "\n",
    "print('\\nTrue Negatives(TN) = ', cm[1,1])\n",
    "\n",
    "print('\\nFalse Positives(FP) = ', cm[0,1])\n",
    "\n",
    "print('\\nFalse Negatives(FN) = ', cm[1,0])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 67,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x22ef3f1d3a0>"
      ]
     },
     "execution_count": 67,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# visualize confusion matrix with seaborn heatmap\n",
    "\n",
    "cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'], \n",
    "                                 index=['Predict Positive:1', 'Predict Negative:0'])\n",
    "\n",
    "sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Classification metrices "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 68,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "       <=50K       0.93      0.80      0.86      6798\n",
      "        >50K       0.57      0.81      0.67      2251\n",
      "\n",
      "    accuracy                           0.80      9049\n",
      "   macro avg       0.75      0.80      0.76      9049\n",
      "weighted avg       0.84      0.80      0.81      9049\n",
      "\n"
     ]
    }
   ],
   "source": [
    "from sklearn.metrics import classification_report\n",
    "\n",
    "print(classification_report(y_test, y_pred))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Classification accuracy"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 69,
   "metadata": {},
   "outputs": [],
   "source": [
    "TP = cm[0,0]\n",
    "TN = cm[1,1]\n",
    "FP = cm[0,1]\n",
    "FN = cm[1,0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 70,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Classification accuracy : 0.7995\n"
     ]
    }
   ],
   "source": [
    "# print classification accuracy\n",
    "\n",
    "classification_accuracy = (TP + TN) / float(TP + TN + FP + FN)\n",
    "\n",
    "print('Classification accuracy : {0:0.4f}'.format(classification_accuracy))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Classification error"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 71,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Classification error : 0.2005\n"
     ]
    }
   ],
   "source": [
    "# print classification error\n",
    "\n",
    "classification_error = (FP + FN) / float(TP + TN + FP + FN)\n",
    "\n",
    "print('Classification error : {0:0.4f}'.format(classification_error))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Precision"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 72,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Precision : 0.7976\n"
     ]
    }
   ],
   "source": [
    "# print precision score\n",
    "\n",
    "precision = TP / float(TP + FP)\n",
    "\n",
    "\n",
    "print('Precision : {0:0.4f}'.format(precision))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Recall"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 73,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Recall or Sensitivity : 0.9253\n"
     ]
    }
   ],
   "source": [
    "recall = TP / float(TP + FN)\n",
    "\n",
    "print('Recall or Sensitivity : {0:0.4f}'.format(recall))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "True Positive Rate\n",
    "\n",
    "True Positive Rate is synonymous with Recall."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 74,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "True Positive Rate : 0.9253\n"
     ]
    }
   ],
   "source": [
    "true_positive_rate = TP / float(TP + FN)\n",
    "\n",
    "\n",
    "print('True Positive Rate : {0:0.4f}'.format(true_positive_rate))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# False Positive Rate"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 75,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "False Positive Rate : 0.4315\n"
     ]
    }
   ],
   "source": [
    "false_positive_rate = FP / float(FP + TN)\n",
    "\n",
    "\n",
    "print('False Positive Rate : {0:0.4f}'.format(false_positive_rate))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Specificity"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 76,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Specificity : 0.5685\n"
     ]
    }
   ],
   "source": [
    "specificity = TN / (TN + FP)\n",
    "\n",
    "print('Specificity : {0:0.4f}'.format(specificity))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Calculate class probabilities "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 77,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[9.99955511e-01, 4.44887598e-05],\n",
       "       [9.95935549e-01, 4.06445120e-03],\n",
       "       [8.63901480e-01, 1.36098520e-01],\n",
       "       [9.99999906e-01, 9.37239455e-08],\n",
       "       [8.80888343e-02, 9.11911166e-01],\n",
       "       [9.99562896e-01, 4.37103927e-04],\n",
       "       [5.34482750e-06, 9.99994655e-01],\n",
       "       [6.28497161e-01, 3.71502839e-01],\n",
       "       [5.46536963e-04, 9.99453463e-01],\n",
       "       [9.99999570e-01, 4.30495598e-07]])"
      ]
     },
     "execution_count": 77,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# print the first 10 predicted probabilities of two classes- 0 and 1\n",
    "\n",
    "y_pred_prob = gnb.predict_proba(X_test)[0:10]\n",
    "\n",
    "y_pred_prob"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Observations\n",
    "\n",
    "* In each row, the numbers sum to 1.\n",
    "\n",
    "* There are 2 columns which correspond to 2 classes - <=50K and >50K.\n",
    "\n",
    "        * Class 0 => <=50K - Class that a person makes less than equal to 50K.\n",
    "\n",
    "        * Class 1 => >50K - Class that a person makes more than 50K.\n",
    "\n",
    "* Importance of predicted probabilities\n",
    "\n",
    "    * We can rank the observations by probability of whether a person makes less than or equal to 50K or more than 50K.\n",
    "\n",
    "* predict_proba process\n",
    "\n",
    "    * Predicts the probabilities\n",
    "\n",
    "    * Choose the class with the highest probability\n",
    "\n",
    "* Classification threshold level\n",
    "\n",
    "    * There is a classification threshold level of 0.5.\n",
    "\n",
    "    * Class 0 => <=50K - probability of salary less than or equal to 50K is predicted if probability < 0.5.\n",
    "\n",
    "    * Class 1 => >50K - probability of salary more than 50K is predicted if probability > 0.5."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 78,
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
       "      <th>Prob of - &lt;=50K</th>\n",
       "      <th>Prob of - &gt;50K</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0.999956</td>\n",
       "      <td>4.448876e-05</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0.995936</td>\n",
       "      <td>4.064451e-03</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>0.863901</td>\n",
       "      <td>1.360985e-01</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1.000000</td>\n",
       "      <td>9.372395e-08</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0.088089</td>\n",
       "      <td>9.119112e-01</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>0.999563</td>\n",
       "      <td>4.371039e-04</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>0.000005</td>\n",
       "      <td>9.999947e-01</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>0.628497</td>\n",
       "      <td>3.715028e-01</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>0.000547</td>\n",
       "      <td>9.994535e-01</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>1.000000</td>\n",
       "      <td>4.304956e-07</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Prob of - <=50K  Prob of - >50K\n",
       "0         0.999956    4.448876e-05\n",
       "1         0.995936    4.064451e-03\n",
       "2         0.863901    1.360985e-01\n",
       "3         1.000000    9.372395e-08\n",
       "4         0.088089    9.119112e-01\n",
       "5         0.999563    4.371039e-04\n",
       "6         0.000005    9.999947e-01\n",
       "7         0.628497    3.715028e-01\n",
       "8         0.000547    9.994535e-01\n",
       "9         1.000000    4.304956e-07"
      ]
     },
     "execution_count": 78,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# store the probabilities in dataframe\n",
    "\n",
    "y_pred_prob_df = pd.DataFrame(data=y_pred_prob, columns=['Prob of - <=50K', 'Prob of - >50K'])\n",
    "\n",
    "y_pred_prob_df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 79,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([4.44887598e-05, 4.06445120e-03, 1.36098520e-01, 9.37239455e-08,\n",
       "       9.11911166e-01, 4.37103927e-04, 9.99994655e-01, 3.71502839e-01,\n",
       "       9.99453463e-01, 4.30495598e-07])"
      ]
     },
     "execution_count": 79,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# print the first 10 predicted probabilities for class 1 - Probability of >50K\n",
    "\n",
    "gnb.predict_proba(X_test)[0:10, 1]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 80,
   "metadata": {},
   "outputs": [],
   "source": [
    "# store the predicted probabilities for class 1 - Probability of >50K\n",
    "\n",
    "y_pred1 = gnb.predict_proba(X_test)[:, 1]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 81,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Text(0, 0.5, 'Frequency')"
      ]
     },
     "execution_count": 81,
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
    "# plot histogram of predicted probabilities\n",
    "\n",
    "\n",
    "# adjust the font size \n",
    "plt.rcParams['font.size'] = 12\n",
    "\n",
    "\n",
    "# plot histogram with 10 bins\n",
    "plt.hist(y_pred1, bins = 10)\n",
    "\n",
    "\n",
    "# set the title of predicted probabilities\n",
    "plt.title('Histogram of predicted probabilities of salaries >50K')\n",
    "\n",
    "\n",
    "# set the x-axis limit\n",
    "plt.xlim(0,1)\n",
    "\n",
    "\n",
    "# set the title\n",
    "plt.xlabel('Predicted probabilities of salaries >50K')\n",
    "plt.ylabel('Frequency')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# ROC - AUC"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 82,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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zz3jDss557JWku9nNzOzewEkDQKuKOV06gEL0/otBAq98Ry9kCiX8IjieqJHFYcC5ObmWiljqxJ++OEHtthiC+rUqcO9995L/fr12W233eIOyznnsl6SSwL+iN3MVlP6hCYPaJDWrwGhsmFGko4BBgJHmNm8Us63ylu7di2DBg1ap8Gfdu3aeQLgnHMVJMklAXUkvZvSXT+tGzPrWIzpTAc2krSLmc2I+rWh4GL+w4EHgC5m9mUp4nbAtGnT6NmzJ++//z6HHXYY5557btwhOedclZPkJKBnWvdDpZmImS2R9DxwnaSzgLbA0cB6tdEkHUyoDHismY0vzfwcjBw5kjPPPJM6deowfPhwunfv7u/7d865GCQ2CTCz4WU4uT7AMOBXwtsGzzWzyZKaAFOAHDP7gdBUcUPglZQfrffM7IgyjCVrmRmS2GuvvTjuuOO44447aNy4cdxhOedclaX8lthc+cnNzbUJEybEHUZsli9fzrXXXsvMmTMZOXJk3OE45xJC0idmlht3HNksyRUDXQKMHTuWNm3aMHDgQGrVquUN/jjnXCXiSYArF3l5efTt25eOHTuycuVK3njjDYYNG+YN/jjnXCXiSYArF0uXLuWpp56ib9++fPnll/zlL3+JOyTnnHNpEp8ESNpY0g2SvpW0MOp3qKTz446tqpk/fz4DBgxgzZo1bLnllkyfPp277rqLevXqxR2ac865DBKfBAB3AK0IDf/k13KcDPiD5xXo2WefpUWLFlx77bWMHx+entxkk01ijso551xhsiEJOBboZmYfAGsBosZ9to01qiril19+4bjjjuOEE05gu+224+OPP2bfffeNOyznnHPFkNj3BKRYSdpySGpEeN7flSMz45hjjmHixIkMHDiQSy65hI02yoZdyjnnqoZsOGM/AwyXdBGApK2BO4EnY40qi33//fc0atSIunXrMmjQIBo0aEDz5s3jDss551wJZcPtgMuB74EvgU2AGcDPwLUxxpSV1qxZw913371Ogz+5ubmeADjnXEIlviTAzFYCFwIXRrcB5pm/BrHMTZ06lZ49e/LBBx9wxBFHcN5558UdknPOuQ2U+CRAUrO0XvXz3+tvZt9WfETZZ8SIEZx55pnUr1+fxx9/nG7dunmDP845lwUSnwQAXxMeDUz9VcovCahe8eFkj/wGf3JzcznhhBO4/fbb2XLLLeMOyznnXBlJfJ0AM6tmZtWj/9WAbYChQPeYQ0usZcuWcdlll9GtWzcAdt11Vx5//HFPAJxzLsskPglIZ2azCXUEboo7liR69913adOmDTfffDN169b1Bn+ccy6LZV0SENkVqBN3EEmyePFi+vTpw4EHHsjq1at56623ePDBB73BH+ecy2KJrxMg6T3+rAMA4ce/JXBdPBEl0/Lly3nuuee48MILGTBgAHXr1o07JOecc+Us8UkA8GBa9xJgopnNiCOYJJk3bx6DBg3iyiuvpFGjRkyfPp2GDRvGHZZzzrkKkugkQFJ14GCgl5mtiDuepDAznn76afr27cuCBQs47LDDaN++vScAzjlXxSS6ToCZrQEOJWo4yBXt559/5phjjuGkk05ihx124NNPP6V9+/Zxh+Wccy4GiU4CIncA10ryGmxFMDOOPfZY3njjDW699VY++OADWrduHXdYzjnnYpLY2wGSTjazJ4C+wFbAxZLmklJJ0MyaxBVfZfLtt9/SuHFj6taty+DBg2nYsCE777xz3GE555yLWZJLAu6P/p8KdAYOiz53T/mr0tasWcMdd9xBq1at/mjwZ6+99vIEwDnnHJDgkgCi1wSb2f/iDqQymjx5Mj179uSjjz6iS5cu9O3bN+6QnHPOVTJJTgKqSzqIddsMWIeZja7AeCqNxx9/nDPPPJOGDRsycuRITjrpJG/wxznn3HqSnARsDDxEwUmAAektDGa1tWvXUq1aNdq1a8dJJ53EbbfdRqNGjeIOyznnXCWV5CRgiZlVqR/5gixdupSrr76aH374gaeffppdd92VRx99NO6wnHPOVXJJrhjogHfeeYfWrVtz2223sdlmm3mDP84554otyUlAlb7JvWjRIs455xwOPvhgJPHOO+8wZMgQb/DHOedcsSU2CTCz+nHHEKcVK1YwatQoLr30Ur744gs6deoUd0jOOecSJsl1AqqcuXPncu+993L11VfTqFEjZsyYQf36VToXcs45twESWxJQliRtJukFSUskzZTUrZBhL5I0W9JCScMkbVze8ZkZTzzxBDk5Odx00018/PHHAJ4AOOec2yCeBASDgJVAY+AU4D5JLdMHknQYcBlwCNCU8AjiteUZ2I8//shRRx1Ft27d2Gmnnfjss8+8wR/nnHNlosonAZLqAscDV5lZnpmNBV4k82uHTwceMrPJZrYA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\n",
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
    "# plot ROC Curve\n",
    "\n",
    "from sklearn.metrics import roc_curve\n",
    "\n",
    "fpr, tpr, thresholds = roc_curve(y_test, y_pred1, pos_label = '>50K')\n",
    "\n",
    "plt.figure(figsize=(6,4))\n",
    "\n",
    "plt.plot(fpr, tpr, linewidth=2)\n",
    "\n",
    "plt.plot([0,1], [0,1], 'k--' )\n",
    "\n",
    "plt.rcParams['font.size'] = 12\n",
    "\n",
    "plt.title('ROC curve for Gaussian Naive Bayes Classifier for Predicting Salaries')\n",
    "\n",
    "plt.xlabel('False Positive Rate (1 - Specificity)')\n",
    "\n",
    "plt.ylabel('True Positive Rate (Sensitivity)')\n",
    "\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 83,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "ROC AUC : 0.8902\n"
     ]
    }
   ],
   "source": [
    "# compute ROC AUC\n",
    "\n",
    "from sklearn.metrics import roc_auc_score\n",
    "\n",
    "ROC_AUC = roc_auc_score(y_test, y_pred1)\n",
    "\n",
    "print('ROC AUC : {:.4f}'.format(ROC_AUC))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Interpretation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 84,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Cross validated ROC AUC : 0.8923\n"
     ]
    }
   ],
   "source": [
    "# calculate cross-validated ROC AUC \n",
    "\n",
    "from sklearn.model_selection import cross_val_score\n",
    "\n",
    "Cross_validated_ROC_AUC = cross_val_score(gnb, X_train, y_train, cv=5, scoring='roc_auc').mean()\n",
    "\n",
    "print('Cross validated ROC AUC : {:.4f}'.format(Cross_validated_ROC_AUC))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# k-Fold Cross Validation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 85,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Cross-validation scores:[0.81676136 0.79829545 0.79014685 0.81288489 0.80388441 0.79062056\n",
      " 0.80767409 0.7925154  0.79630507 0.80909522]\n"
     ]
    }
   ],
   "source": [
    "# Applying 10-Fold Cross Validation\n",
    "\n",
    "from sklearn.model_selection import cross_val_score\n",
    "\n",
    "scores = cross_val_score(gnb, X_train, y_train, cv = 10, scoring='accuracy')\n",
    "\n",
    "print('Cross-validation scores:{}'.format(scores))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 86,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Average cross-validation score: 0.8018\n"
     ]
    }
   ],
   "source": [
    "# compute Average cross-validation score\n",
    "\n",
    "print('Average cross-validation score: {:.4f}'.format(scores.mean()))"
   ]
  }
 ],
 "metadata": {
  "gist": {
   "data": {
    "description": "Assignment 12 - Naive Bayes.ipynb",
    "public": true
   },
   "id": ""
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
