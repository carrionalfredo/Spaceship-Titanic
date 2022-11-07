

## Machine Learning Zoomcamp 2022 - Mid Term Project
### Spaceship Titanic - Predict which passengers are transported to an alternate dimension

This machine learning project was prepared for the Mid term Project for the [Machine Learning Zoomcamp 2022](https://github.com/alexeygrigorev/mlbookcamp-code/tree/master/course-zoomcamp) edition online course, prepared by [DataTalks.Club.com](https://datatalks.club/) and [Alexey Grigorev
](https://github.com/alexeygrigorev)



## Problem description
This problem and the data used are part of the Kaggle GettingStarted Prediction Competition.

Welcome to the year 2912, where your data science skills are needed to solve a cosmic mystery. We've received a transmission from four lightyears away and things aren't looking good.

The Spaceship Titanic was an interstellar passenger liner launched a month ago. With almost 13,000 passengers on board, the vessel set out on its maiden voyage transporting emigrants from our solar system to three newly habitable exoplanets orbiting nearby stars.

While rounding Alpha Centauri en route to its first destination—the torrid 55 Cancri E—the unwary Spaceship Titanic collided with a spacetime anomaly hidden within a dust cloud. Sadly, it met a similar fate as its namesake from 1000 years before. Though the ship stayed intact, almost half of the passengers were transported to an alternate dimension!

![](https://storage.googleapis.com/kaggle-media/competitions/Spaceship%20Titanic/joel-filipe-QwoNAhbmLLo-unsplash.jpg)
> Fuente：[Kaggle](https://www.kaggle.com/competitions/spaceship-titanic/overview)

To help rescue crews and retrieve the lost passengers, you are challenged to predict which passengers were transported by the anomaly using records recovered from the spaceship’s damaged computer system.

Help save them and change history!

For this, is available a data set ('train.csv') with information of the passengers. This data set is available in this repo, and also can be downloaded from Kaggle ([Data](https://www.kaggle.com/competitions/spaceship-titanic/data?select=train.csv)).

Since the objective is predict if a passenger is transported or not, the problem is considered as a classification problem.

## Training of the model
### Python libraries
For the training of the model in this project, the pandas, numpy, matplolib and seabron were used.
### EDA
The data set consist of 8693 rows of 14 columns of data.

```python
PassengerId      object
HomePlanet       object
CryoSleep        object
Cabin            object
Destination      object
Age             float64
VIP              object
RoomService     float64
FoodCourt       float64
ShoppingMall    float64
Spa             float64
VRDeck          float64
Name             object
Transported        bool
```
The 'cabin' feature has the number where the passenger is staying, in the format 'deck'/'number'/'side', so will be splitted into the 3 new deck', 'num' and 'side' variables, and 'cabin' will be dropped from the data set.

This is the results after verify null and missed values.

```python
RangeIndex: 8693 entries, 0 to 8692
Data columns (total 16 columns):
 #   Column        Non-Null Count  Dtype  
---  ------        --------------  -----  
 0   passengerid   8693 non-null   object 
 1   homeplanet    8492 non-null   object 
 2   cryosleep     8693 non-null   bool   
 3   destination   8511 non-null   object 
 4   age           8514 non-null   float64
 5   vip           8693 non-null   bool   
 6   roomservice   8512 non-null   float64
 7   foodcourt     8510 non-null   float64
 8   shoppingmall  8485 non-null   float64
 9   spa           8510 non-null   float64
 10  vrdeck        8505 non-null   float64
 11  name          8493 non-null   object 
 12  transported   8693 non-null   bool   
 13  deck          8494 non-null   object 
 14  num           8494 non-null   float64
 15  side          8494 non-null   object 
dtypes: bool(3), float64(7), object(6)
```
Is posible to see that about 2% of the data is null or missing, so the next step was group variables by type and replace null and missing values.

For the 'object' type variables, the replacement was with 'unk' value.

For the numerical variables, except 'age', the replacement was with '0'.

For the 'age' variable, the replacement was with the median value.

After this replacements, the data set remain as follow:

```python
        age	        roomservice	    foodcourt	    shoppingmall	spa	            vrdeck	        num
count	8693.000000	8693.000000	    8693.000000	    8693.000000	    8693.000000	    8693.000000	    8693.000000
mean	29.343150	220.009318	    448.434027	    169.572300	    304.588865	    298.261820	    586.624065
std	    13.728128	660.519050	    1595.790627	    598.007164	    1125.562559	    1134.126417	    513.880084
min	    1.000000	0.000000	    0.000000	    0.000000	    0.000000	    0.000000	    0.000000
25%	    20.000000	0.000000	    0.000000	    0.000000    	0.000000	    0.000000	    152.000000
50%	    27.000000	0.000000	    0.000000	    0.000000    	0.000000	    0.000000	    407.000000
75%	    37.000000	41.000000	    61.000000	    22.000000   	53.000000	    40.000000	    983.000000
max	    79.000000	14327.000000	29813.000000	23492.000000	22408.000000	24133.000000	1894.000000
```
#### Target variable
The target variable for this project will be the  'transported' feature. This variable is boolen type with tehe following description.
```python
count     8693
unique       2
top       True
freq      4378
Name: transported, dtype: object
```
The values are almost equal distributed between True and False.
#### Feature importance analysis
In order to analyze the importance of the variables, the following steps were made.

For categorical variables, the mutual_info_score is calculated, and the result is:
```python
passengerid    0.693121
name           0.675740
cryosleep      0.107255
deck           0.023157
homeplanet     0.018931
destination    0.006161
side           0.005271
vip            0.000303
```
And the quantity of unique values for categorical variables is:
```python
passengerid    8693
homeplanet        4
destination       4
name           8474
deck              9
side              3
```
From the above results, the variables 'passengerid' and ''name' are descarted for the high number of differente values in the data set. So the categorical variables to be used in the training of the model will be:
```python
categorical_columns = ['homeplanet', 'destination', 'deck', 'side', 'cryosleep', 'vip', 'transported']
```

For the numerical variables, the result of correlation analysis is the following:
```python
age            -0.052951
roomservice    -0.241124
foodcourt       0.045583
shoppingmall    0.009391
spa            -0.218545
vrdeck         -0.204874
num            -0.043832
```
None numerical variable will be descarted this time.

## Model Training
For the training of the model, the data set was splitted in train, validation and test sets, with a 60/20/20 split. Then it was applied one hot encoding, and the resultant vector is:
```python
['age',
 'cryosleep',
 'deck=a',
 'deck=b',
 'deck=c',
 'deck=d',
 'deck=e',
 'deck=f',
 'deck=g',
 'deck=t',
 'deck=unk',
 'destination=55_cancri_e',
 'destination=pso_j318.5-22',
 'destination=trappist-1e',
 'destination=unk',
 'foodcourt',
 'homeplanet=earth',
 'homeplanet=europa',
 'homeplanet=mars',
 'homeplanet=unk',
 'num',
 'roomservice',
 'shoppingmall',
 'side=p',
 'side=s',
 'side=unk',
 'spa',
 'vip',
 'vrdeck']
```
### Logistic Regression
 The first model type employed to train the model was simple logistic regression. In this case, the model was setup with max_iter=2000, class_weight='balanced and was tunned for 10 C values between 1  to 10. For this setup, the roc_auc_score between predicted values and validation values is:
 ```Python
C	auc
6	0.883357
4	0.883244
9	0.883141
1	0.883109
3	0.883012
10	0.882922
7	0.882916
8	0.882897
5	0.882795
2	0.882772
```
So the C value that gives the high AUC is C=6.

```python
LogisticRegression(max_iter=2000, C=C, class_weight='balanced')
```
With this setup, the coeficients of the model are:
![](https://github.com/carrionalfredo/Spaceship-Titanic/blob/main/images/Fig_01.png)

The ROC Curve for this model is:
![](https://github.com/carrionalfredo/Spaceship-Titanic/blob/main/images/Fig_02.png)





