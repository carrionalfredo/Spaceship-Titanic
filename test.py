# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 01:45:02 2022

@author: USUARIO
"""

import requests

url = 'http://localhost:9696/classify'

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

result = requests.post(url, json=data).json()

print('Transported?: ', result.get('Result'))