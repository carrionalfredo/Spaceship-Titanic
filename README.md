

## Machine Learning Zoomcamp 2022 - Mid Term Project
### Spaceship Titanic - Predict which passengers are transported to an alternate dimension

This machine learning project was prepared for the Mid Term Project for the [Machine Learning Zoomcamp 2022](https://github.com/alexeygrigorev/mlbookcamp-code/tree/master/course-zoomcamp) edition online course, prepared by [DataTalks.Club.com](https://datatalks.club/) and [Alexey Grigorev
](https://github.com/alexeygrigorev)



## Problem description
This problem and the data used are part of the **Kaggle GettingStarted Prediction Competition**.

Welcome to the year 2912, where your data science skills are needed to solve a cosmic mystery. We've received a transmission from four lightyears away and things aren't looking good.

The Spaceship Titanic was an interstellar passenger liner launched a month ago. With almost 13,000 passengers on board, the vessel set out on its maiden voyage transporting emigrants from our solar system to three newly habitable exoplanets orbiting nearby stars.

While rounding Alpha Centauri en route to its first destination—the torrid 55 Cancri E—the unwary Spaceship Titanic collided with a spacetime anomaly hidden within a dust cloud. Sadly, it met a similar fate as its namesake from 1000 years before. Though the ship stayed intact, almost half of the passengers were transported to an alternate dimension!

![](https://storage.googleapis.com/kaggle-media/competitions/Spaceship%20Titanic/joel-filipe-QwoNAhbmLLo-unsplash.jpg)
> Source：[Kaggle](https://www.kaggle.com/competitions/spaceship-titanic/overview)

To help rescue crews and retrieve the lost passengers, you are challenged to predict which passengers were transported by the anomaly using records recovered from the spaceship’s damaged computer system.

Help save them and change history!

A dataset is available for this purpose ([train.csv](https://github.com/carrionalfredo/Spaceship-Titanic/blob/main/train.csv)) with information of the passengers. This dataset is available in this repo, and also can be downloaded from Kaggle ([Data](https://www.kaggle.com/competitions/spaceship-titanic/data?select=train.csv)).

Since the objective is to predict if a passenger is transported or not, the problem is considered as a classification problem.

## Training of the model
### Python libraries
For the training of the model in this project, the **pandas**, **numpy**, **matplolib** and **seaborn** libraries were used.
### EDA
The dataset consist of 8693 rows of 14 columns of data.

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
The `cabin` feature has the number where the passenger is staying, in the format `deck/number/side`, so will be splitted into the 3 new `deck`, `num` and `side` variables, and `cabin` will be dropped from the dataset.

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

For the `object` type variables, the replacement was with `unk` value.

For the numerical variables, except `age`, the replacement was with `0`.

For the `age` variable, the replacement was with the `df['age'].median()` value.

After this replacements, the dataset remain as follow:

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
The target variable for this project will be the `transported` feature. This variable is `boolean` type with the following description.
```python
count     8693
unique       2
top       True
freq      4378
Name: transported, dtype: object
```
The values are almost equal distributed between True and False (50.36% `True`).

#### Feature importance analysis
In order to analyze the importance of the variables, the following steps were made.

For categorical variables, the `mutual_info_score` is calculated, and the results are:

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
The quantity of unique values for categorical variables is:

```python
passengerid    8693
homeplanet        4
destination       4
name           8474
deck              9
side              3
```
From the above results, the variables `passengerid` and `name` are descarded for the high number of different values in the dataset. So the categorical variables to be used in the training of the model will be:

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
None numerical variable will be descarded this time.

## Model Training

For the training of the model, the dataset was split in to train, validation and test sets, with a 60/20/20 split. Then it was applied one hot encoding, and the resultant vector is:

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
The first model type employed to train the model was a simple **logistic regression**. In this case, the model was set up with `max_iter=2000`, `class_weight='balanced'` and was tuned for 10 Inverse of regularization strength `C` values between `1` to `10`. For this setup, the `roc_auc_score` between predicted values and validation values is:

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
So, the `C` value that gives the high `AUC` is `C=6`.

Then, the final Logistic Regression model (`lgr`) is:

```python
lgr = LogisticRegression(max_iter=2000, C=6, class_weight='balanced')
```
With this setup, after fit the model with `x_train`  and `y_train`, the coefficients of the model are:

![](https://github.com/carrionalfredo/Spaceship-Titanic/blob/main/images/Fig_01.png)

The ROC Curve for this model is:

![](https://github.com/carrionalfredo/Spaceship-Titanic/blob/main/images/Fig_02.png)

### Decision Tree

The next model type employed to train the model was a **classifier decision tree**. In this case, first the model was trained with default parameters.

```pyhon
DecisionTreeClassifier()
```
With this setup, the valdiation `roc_auc_score` is:
```python
0.7340163934426229
```
Next, the `max_depth` and `min_samples_leaf` of the setup were tuned. The results are the following:

![](https://github.com/carrionalfredo/Spaceship-Titanic/blob/main/images/Fig_03.png)

From this result, the `max_depth = 7` and `min_samples_leaf = 14` were selected for the final decision tree model (`dt`).

```python
dt = DecisionTreeClassifier(max_depth=7, min_samples_leaf=16)
```
The Decision Tree for this model is the following. The decision tree render for this model is available [here](https://raw.githubusercontent.com/carrionalfredo/Spaceship-Titanic/main/dtree_render).

![](https://github.com/carrionalfredo/Spaceship-Titanic/blob/main/images/Fig_04.png)

The ROC Curve for this model is:

![](https://github.com/carrionalfredo/Spaceship-Titanic/blob/main/images/Fig_05.png)

With this setup, the validation `roc_auc_score` is:
```python
0.8692589792561939
```

### Random Forest

With the base of the decision tree model, the next model type employed to train the model was a random forest classifier. In this case, the model was trained with the `max_depth` parameter value from the decision tree final model, and the parameters `min_samples_leaf` and `n_estimators` were tuned, with the following results.

![](https://github.com/carrionalfredo/Spaceship-Titanic/blob/main/images/Fig_06.png)

From this results, the `min_samples_leaf = 3` and `n_estimators = 30` were selected for the final random forest model (`rf`).

```python
rf = RandomForestClassifier(n_estimators=30, max_depth=6, min_samples_leaf=3, random_state=1)
```

The ROC Curve for this model is:

![](https://github.com/carrionalfredo/Spaceship-Titanic/blob/main/images/Fig_07.png)

With this setup, the validation `roc_auc_score` is:

```python
0.8806727147328772
```
## Model selection
In order to select the best classifier model to predict if a passenger is transported or not, next table shows a comparison of the validation and test `roc_auc_score` for the models trainned.
```python
            Simple Logistic Reg.	Decision Tree Classifier	Random Forest
Validation	0.883357	                0.869259	                0.880673
Test    	0.882395	                0.867580	                0.873317
```
The model with the highest 'roc_auc_score`, both for validation and test, is Logistic Regression with this setup:
```python
LogisticRegression(max_iter=2000, C=6, class_weight='balanced')
```

## Dependency and Environment Managenent

The selected model and its training logic has been exported to the `train.py` script, that generates the `LGRmodel.bin` pickle file.

The file `predict.py` loads the `LGRmodel.bin` and deploy it via web service with **Flask**.

All the dependencies and the virtual environment used in this project are provided in the [`pipfile`](https://raw.githubusercontent.com/carrionalfredo/Spaceship-Titanic/main/Pipfile) uploaded in this repository.

In order to install this dependencies and virtual environment, with `Pipenv` installed and once downloaded the  `pipfile` and `pipfile.lock` files in the working directory, execute the next command:

        pipenv install

This will install the dependencies from the `pipfile.lock` file. To activate the virtual environment for this project, run:

        pipenv shell


Also, its posible run a command inside this virtual environment with:

        pipenv run

Once activated the virtual environment, the model can be deployed via web service running the following command:

        predict.py

This will serve the Flask app `transport_predictor` in the port `9696`.

To verify that the `transport_predictor` is working, use the [`test.py`](https://raw.githubusercontent.com/carrionalfredo/Spaceship-Titanic/main/test.py) script. In another command window, go to the working directory, and run:

        test.py

If all is working OK, in the virtual environment command window, should return a `"POST /classify HTTP/1.1" 200 -` message, and in the another command window, should show the results of the prediction.

For this example, the `test.py` script uses the following passenger data in **JSON()** format:

```python
data = {
        "age": 37,
    "cryosleep": 0,
    "deck_a": 0,
    "deck_b": 0,
    "deck_c": 0,
    "deck_d": 0,
    "deck_e": 0,
    "deck_f": 1,
    "deck_g": 0,
    "deck_t": 0,
    "deck_unk": 0,
    "destination_55_cancri_e": 0,
    "destination_pso_j318_5_22": 0,
    "destination_trappist_1e": 1,
    "destination_unk": 0,
    "foodcourt": 27,
    "homeplanet_earth": 1,
    "homeplanet_europa": 0,
    "homeplanet_mars": 0,
    "homeplanet_unk": 0,
    "num": 309,
    "roomservice": 0,
    "shoppingmall": 11,
    "side_p": 0,
    "side_s": 1,
    "side_unk": 0,
    "spa": 732,
    "vip": 0,
    "vrdeck": 5
    }
```
The result of this test should be:

        Transported?:  False

## Containerization

The model and dependencies were containerizated with **Docker**.

To create a Docker image denominated `mtp` with the virtual environment and dependencies used in the model, start the Docker service, go to the working directory where the necesary `dockerfile` is, and run the following command:

        docker build -t mtp .

The [`dockerfile`](https://raw.githubusercontent.com/carrionalfredo/Spaceship-Titanic/main/Dockerfile) used to create the Docker image in this project, has been uploaded to his repository.

To run the Docker image recently created, run this command:

        docker run -it --rm --entrypoint=bash mtp

 And for run the web service via **Gunicorn** of the `transport_predictor` app in the port `9696`, run the following command:

        docker run -it --rm -p 9696:9696 mtp

The following messages should be show:

        [INFO] Starting gunicorn 20.1.0
        [1] [INFO] Listening at: http://0.0.0.0:9696 (1)
        [1] [INFO] Using worker: sync
        [8] [INFO] Booting worker with pid: 8

After that, to test the deployed model, in another command window, run the `test.py` script.

The result `Transported?:  False` should be show as response.

## Cloud Deployment

Adicionally, the model was deployed in the cloud through **AWS Elastic Beanstalk**. For this, first an application called `mtp_predictor` was created under the Docker platform, with the following command:

        eb init -p docker -r us-east-1 mtp_predictor

After create the application, con be tested locally executing:

        eb local run --port 9696

If the application in online, the following message will appear.

![](https://github.com/carrionalfredo/Spaceship-Titanic/blob/main/images/Fig_08.png)

In another command window, the application can be tested, executing the `test.py` script:

And the result should be:

![](https://github.com/carrionalfredo/Spaceship-Titanic/blob/main/images/Fig_09.png)

For deploy the web service in the cloud, the `mtp-env` environment was created with:

        eb create mtp-env

With the application addres provided by AWS, another test script was created ([`cloud_test.py`](https://raw.githubusercontent.com/carrionalfredo/Spaceship-Titanic/main/cloud_test.py)).

Running the `cloud_test.py` script in a command window, will return the prediction using the `mtp_env` environment and `mtp_predictor` application created in the AWS cloud.

![](https://github.com/carrionalfredo/Spaceship-Titanic/blob/main/images/Fig_10.png)