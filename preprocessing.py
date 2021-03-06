#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 19:37:58 2022

@author: PyBoys
"""

import pandas as pd




#from sklearn.preprocessing import MinMaxScaler

def preprocess(df, option):
    """
    This function is to cover all the preprocessing steps on the churn dataframe. It involves selecting important features, encoding categorical data, handling missing values,feature scaling and splitting the data
    """
    #Defining the map function


    
    #Drop values based on operational options
    if (option == "Online"):
        # Encode binary categorical features
        #Encoding the other categorical categoric features with more than two categories
        #print(df.head())
        pass
                    
    else:
        print("Incorrect operational options")


    #feature scaling
    #sc = MinMaxScaler()
    #df['tenure'] = sc.fit_transform(df[['tenure']])
    #df['MonthlyCharges'] = sc.fit_transform(df[['MonthlyCharges']])
    #df['TotalCharges'] = sc.fit_transform(df[['TotalCharges']])
    return df
        
make_model_dict = {'Audi-A1': 0,
 'Audi-A3': 1,
 'Audi-A4': 2,
 'Audi-A4 allroad': 3,
 'Audi-A5': 4,
 'Audi-A6': 5,
 'Audi-A6 allroad': 6,
 'Audi-A7': 7,
 'Audi-A8': 8,
 'Audi-Q2': 9,
 'Audi-Q3': 10,
 'Audi-Q4 e-tron': 11,
 'Audi-Q5': 12,
 'Audi-Q7': 13,
 'Audi-Q8': 14,
 'Audi-R8': 15,
 'Audi-S5': 16,
 'Audi-SQ5': 17,
 'Audi-SQ7': 18,
 'Audi-TT': 19,
 'Audi-e-tron': 20,
 'Audi-e-tron GT': 21,
 'BMW-116': 22,
 'BMW-118': 23,
 'BMW-120': 24,
 'BMW-216': 25,
 'BMW-218': 26,
 'BMW-220': 27,
 'BMW-225': 28,
 'BMW-316': 29,
 'BMW-318': 30,
 'BMW-320': 31,
 'BMW-325': 32,
 'BMW-328': 33,
 'BMW-330': 34,
 'BMW-335': 35,
 'BMW-418': 36,
 'BMW-420': 37,
 'BMW-428': 38,
 'BMW-435': 39,
 'BMW-520': 40,
 'BMW-523': 41,
 'BMW-525': 42,
 'BMW-530': 43,
 'BMW-535': 44,
 'BMW-640': 45,
 'BMW-650': 46,
 'BMW-730': 47,
 'BMW-740': 48,
 'BMW-M3': 49,
 'BMW-M4': 50,
 'BMW-Others': 51,
 'BMW-X1': 52,
 'BMW-X2': 53,
 'BMW-X3': 54,
 'BMW-X5': 55,
 'BMW-X6': 56,
 'BMW-Z3': 57,
 'BMW-Z4': 58,
 'BMW-i3': 59,
 'BMW-i4': 60,
 'BMW-iX': 61,
 'BMW-iX3': 62,
 'Chevrolet-Aveo': 63,
 'Chevrolet-Camaro': 64,
 'Chevrolet-Captiva': 65,
 'Chevrolet-Corvette': 66,
 'Chevrolet-Cruze': 67,
 'Chevrolet-Spark': 68,
 'Citroen-Berlingo': 69,
 'Citroen-C1': 70,
 'Citroen-C2': 71,
 'Citroen-C3': 72,
 'Citroen-C3 Aircross': 73,
 'Citroen-C3 Picasso': 74,
 'Citroen-C4': 75,
 'Citroen-C4 Cactus': 76,
 'Citroen-C4 Picasso': 77,
 'Citroen-C5': 78,
 'Citroen-C5 Aircross': 79,
 'Citroen-DS3': 80,
 'Citroen-Grand C4 Picasso': 81,
 'Citroen-Jumper': 82,
 'Citroen-Jumpy': 83,
 'Citroen-Others': 84,
 'Citroen-Xsara Picasso': 85,
 'Dacia-Dokker': 86,
 'Dacia-Duster': 87,
 'Dacia-Lodgy': 88,
 'Dacia-Logan': 89,
 'Dacia-Sandero': 90,
 'Dacia-Spring': 91,
 'Fiat-500': 92,
 'Fiat-500C': 93,
 'Fiat-500L': 94,
 'Fiat-500X': 95,
 'Fiat-500e': 96,
 'Fiat-Doblo': 97,
 'Fiat-Ducato': 98,
 'Fiat-Grande Punto': 99,
 'Fiat-Panda': 100,
 'Fiat-Punto': 101,
 'Fiat-Punto Evo': 102,
 'Fiat-Sedici': 103,
 'Fiat-Stilo': 104,
 'Fiat-Talento': 105,
 'Fiat-Tipo': 106,
 'Ford-B-Max': 107,
 'Ford-C-Max': 108,
 'Ford-EcoSport': 109,
 'Ford-Explorer': 110,
 'Ford-F 150': 111,
 'Ford-Fiesta': 112,
 'Ford-Focus': 113,
 'Ford-Focus C-Max': 114,
 'Ford-Focus CC': 115,
 'Ford-Fusion': 116,
 'Ford-Galaxy': 117,
 'Ford-Ka/Ka+': 118,
 'Ford-Kuga': 119,
 'Ford-Mondeo': 120,
 'Ford-Mustang': 121,
 'Ford-Mustang Mach-E': 122,
 'Ford-Puma': 123,
 'Ford-Ranger': 124,
 'Ford-S-Max': 125,
 'Ford-Transit': 126,
 'Ford-Transit Connect': 127,
 'Ford-Transit Custom': 128,
 'Honda-Accord': 129,
 'Honda-CR-V': 130,
 'Honda-Civic': 131,
 'Honda-HR-V': 132,
 'Honda-Insight': 133,
 'Honda-Jazz': 134,
 'Honda-e': 135,
 'Hyundai-Bayon': 136,
 'Hyundai-Getz': 137,
 'Hyundai-Ioniq': 138,
 'Hyundai-Ioniq 5': 139,
 'Hyundai-Kona': 140,
 'Hyundai-Nexo': 141,
 'Hyundai-Santa Fe': 142,
 'Hyundai-Tucson': 143,
 'Hyundai-i10': 144,
 'Hyundai-i20': 145,
 'Hyundai-i30': 146,
 'Hyundai-i40': 147,
 'Hyundai-iX20': 148,
 'Hyundai-iX35': 149,
 'Kia-Carens': 150,
 "Kia-Ceed / cee'd": 151,
 "Kia-Ceed SW / cee'd SW": 152,
 'Kia-EV6': 153,
 'Kia-Niro': 154,
 'Kia-Optima': 155,
 'Kia-Picanto': 156,
 "Kia-ProCeed / pro_cee'd": 157,
 'Kia-Rio': 158,
 'Kia-Sorento': 159,
 'Kia-Soul': 160,
 'Kia-Sportage': 161,
 'Kia-Stonic': 162,
 'Kia-Venga': 163,
 'Kia-XCeed': 164,
 'Kia-e-Niro': 165,
 'Mazda-2': 166,
 'Mazda-3': 167,
 'Mazda-5': 168,
 'Mazda-6': 169,
 'Mazda-CX-3': 170,
 'Mazda-CX-30': 171,
 'Mazda-CX-5': 172,
 'Mazda-MX-30': 173,
 'Mazda-MX-5': 174,
 'Mercedes-Benz-A 150': 175,
 'Mercedes-Benz-A 160': 176,
 'Mercedes-Benz-A 170': 177,
 'Mercedes-Benz-A 180': 178,
 'Mercedes-Benz-A 200': 179,
 'Mercedes-Benz-A 250': 180,
 'Mercedes-Benz-A 45 AMG': 181,
 'Mercedes-Benz-B 160': 182,
 'Mercedes-Benz-B 170': 183,
 'Mercedes-Benz-B 180': 184,
 'Mercedes-Benz-B 200': 185,
 'Mercedes-Benz-B 250': 186,
 'Mercedes-Benz-C 180': 187,
 'Mercedes-Benz-C 200': 188,
 'Mercedes-Benz-C 220': 189,
 'Mercedes-Benz-C 250': 190,
 'Mercedes-Benz-C 300': 191,
 'Mercedes-Benz-C 350': 192,
 'Mercedes-Benz-CLA 180': 193,
 'Mercedes-Benz-CLA 200': 194,
 'Mercedes-Benz-CLA 250': 195,
 'Mercedes-Benz-CLK 200': 196,
 'Mercedes-Benz-CLS 350': 197,
 'Mercedes-Benz-Citan': 198,
 'Mercedes-Benz-E 200': 199,
 'Mercedes-Benz-E 220': 200,
 'Mercedes-Benz-E 240': 201,
 'Mercedes-Benz-E 250': 202,
 'Mercedes-Benz-E 300': 203,
 'Mercedes-Benz-E 320': 204,
 'Mercedes-Benz-E 350': 205,
 'Mercedes-Benz-EQC 400': 206,
 'Mercedes-Benz-EQS': 207,
 'Mercedes-Benz-GLA 180': 208,
 'Mercedes-Benz-GLA 200': 209,
 'Mercedes-Benz-GLA 250': 210,
 'Mercedes-Benz-GLA 45 AMG': 211,
 'Mercedes-Benz-GLC 220': 212,
 'Mercedes-Benz-GLC 250': 213,
 'Mercedes-Benz-GLC 300': 214,
 'Mercedes-Benz-GLC 350': 215,
 'Mercedes-Benz-GLE 350': 216,
 'Mercedes-Benz-ML 320': 217,
 'Mercedes-Benz-ML 350': 218,
 'Mercedes-Benz-Others': 219,
 'Mercedes-Benz-S 350': 220,
 'Mercedes-Benz-SL 500': 221,
 'Mercedes-Benz-SLK 200': 222,
 'Mercedes-Benz-Sprinter': 223,
 'Mercedes-Benz-Vito': 224,
 'Opel-Adam': 225,
 'Opel-Agila': 226,
 'Opel-Ampera': 227,
 'Opel-Antara': 228,
 'Opel-Astra': 229,
 'Opel-Cascada': 230,
 'Opel-Combo': 231,
 'Opel-Corsa': 232,
 'Opel-Corsa-e': 233,
 'Opel-Crossland X': 234,
 'Opel-Grandland X': 235,
 'Opel-Insignia': 236,
 'Opel-Karl': 237,
 'Opel-Meriva': 238,
 'Opel-Mokka': 239,
 'Opel-Mokka X': 240,
 'Opel-Mokka-E': 241,
 'Opel-Tigra': 242,
 'Opel-Vectra': 243,
 'Opel-Vivaro': 244,
 'Peugeot-108': 245,
 'Peugeot-2008': 246,
 'Peugeot-206': 247,
 'Peugeot-207': 248,
 'Peugeot-208': 249,
 'Peugeot-3008': 250,
 'Peugeot-307': 251,
 'Peugeot-308': 252,
 'Peugeot-407': 253,
 'Peugeot-5008': 254,
 'Peugeot-508': 255,
 'Peugeot-Boxer': 256,
 'Peugeot-Expert': 257,
 'Peugeot-Partner': 258,
 'Peugeot-RCZ': 259,
 'Peugeot-Rifter': 260,
 'Renault-Arkana': 261,
 'Renault-Captur': 262,
 'Renault-Clio': 263,
 'Renault-Espace': 264,
 'Renault-Grand Scenic': 265,
 'Renault-Kadjar': 266,
 'Renault-Kangoo': 267,
 'Renault-Laguna': 268,
 'Renault-Master': 269,
 'Renault-Megane': 270,
 'Renault-Modus': 271,
 'Renault-Scenic': 272,
 'Renault-Talisman': 273,
 'Renault-Trafic': 274,
 'Renault-Twingo': 275,
 'Renault-ZOE': 276,
 'Skoda-Citigo': 277,
 'Skoda-Enyaq': 278,
 'Skoda-Fabia': 279,
 'Skoda-Kamiq': 280,
 'Skoda-Karoq': 281,
 'Skoda-Kodiaq': 282,
 'Skoda-Octavia': 283,
 'Skoda-Rapid/Spaceback': 284,
 'Skoda-Roomster': 285,
 'Skoda-Scala': 286,
 'Skoda-Superb': 287,
 'Skoda-Yeti': 288,
 'Tesla-Model 3': 289,
 'Tesla-Model S': 290,
 'Tesla-Model X': 291,
 'Toyota-Auris': 292,
 'Toyota-Avensis': 293,
 'Toyota-Aygo': 294,
 'Toyota-C-HR': 295,
 'Toyota-Camry': 296,
 'Toyota-Corolla': 297,
 'Toyota-Corolla Verso': 298,
 'Toyota-Hilux': 299,
 'Toyota-Land Cruiser': 300,
 'Toyota-Mirai': 301,
 'Toyota-Prius': 302,
 'Toyota-Prius+': 303,
 'Toyota-Proace': 304,
 'Toyota-RAV 4': 305,
 'Toyota-Verso': 306,
 'Toyota-Verso-S': 307,
 'Toyota-Yaris': 308,
 'Toyota-Yaris Cross': 309,
 'Volkswagen-Amarok': 310,
 'Volkswagen-Arteon': 311,
 'Volkswagen-Beetle': 312,
 'Volkswagen-Caddy': 313,
 'Volkswagen-Crafter': 314,
 'Volkswagen-Eos': 315,
 'Volkswagen-Golf': 316,
 'Volkswagen-Golf Cabriolet': 317,
 'Volkswagen-Golf GTE': 318,
 'Volkswagen-Golf GTI': 319,
 'Volkswagen-Golf Plus': 320,
 'Volkswagen-Golf Sportsvan': 321,
 'Volkswagen-Golf Variant': 322,
 'Volkswagen-ID.3': 323,
 'Volkswagen-ID.4': 324,
 'Volkswagen-Jetta': 325,
 'Volkswagen-Others': 326,
 'Volkswagen-Passat': 327,
 'Volkswagen-Passat CC': 328,
 'Volkswagen-Passat Variant': 329,
 'Volkswagen-Polo': 330,
 'Volkswagen-Scirocco': 331,
 'Volkswagen-Sharan': 332,
 'Volkswagen-T-Cross': 333,
 'Volkswagen-T-Roc': 334,
 'Volkswagen-T5 Transporter': 335,
 'Volkswagen-T6 Transporter': 336,
 'Volkswagen-Tiguan': 337,
 'Volkswagen-Touareg': 338,
 'Volkswagen-Touran': 339,
 'Volkswagen-Transporter': 340,
 'Volkswagen-e-Golf': 341,
 'Volkswagen-e-up!': 342,
 'Volkswagen-up!': 343,
 'Volvo-C30': 344,
 'Volvo-C70': 345,
 'Volvo-S40': 346,
 'Volvo-S60': 347,
 'Volvo-S80': 348,
 'Volvo-S90': 349,
 'Volvo-V40': 350,
 'Volvo-V40 Cross Country': 351,
 'Volvo-V50': 352,
 'Volvo-V60': 353,
 'Volvo-V60 Cross Country': 354,
 'Volvo-V70': 355,
 'Volvo-V90': 356,
 'Volvo-XC40': 357,
 'Volvo-XC60': 358,
 'Volvo-XC70': 359,
 'Volvo-XC90': 360}

n_dict = {'1' : 1,'2': 2,'3': 3,'4': 4,'5': 5,'6': 6,'7': 7,'8': 8,'9': 9, '10' : 10}