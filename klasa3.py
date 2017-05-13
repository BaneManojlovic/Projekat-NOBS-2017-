_author_='Bane'

import sys
import os
import csv
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.cross_validation import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import math
from random import randint


class Klasa3:
#-------------------- Funkcije za opciju 4. ----------------------------------------------------------------------------
    def linearna_regresija(self):
        # 1.Korak - Ucitavam podatke iz fajla
        baza = pd.read_csv("baza.csv")
        baza.columns = ['PAK', 'Objekti', 'Broj spratova', 'Domacinstva', 'Pravna lica stan',  'Pravna lica lokal',
                        'Pravna lica', 'Korisnici', 'Liftovi', 'Sanducici', 'O', 'OS', 'R' ,'RS', 'AR', 'ARS', 'PU',
                        'TR10', 'V', 'Votk', 'Ropt', 'Rino', 'TG', 'Posiljke']
        print('\na. Analiza podataka pomocu Linearne regresije ')
        X = baza[['Korisnici']].values
        Y = baza['Posiljke'].values
        slr = LinearRegression()
        slr.fit(X, Y)    # Fit regression model
        print('\nKoeficijenti:')
        print(' b1 = %.3f ' % slr.coef_[0], ', (slope)')              # Koeficijent b1 (beta1)
        print(' b0 = %.3f' % slr.intercept_, ', (intercept)')         # Koeficijent bo (beta0)
        print('\nRegresiona jednacina:')
        print('\t Y = %.3f'  % slr.intercept_,'+ %.3f'  % slr.coef_[0], '* X' )

       # 2. Korak - Iscrtava se grafik na osnovu uzetih podataka za tacke i odredjenih koeficijenata b0 i b1
        plt.scatter(X, Y, color='red', label = 'Stvarni broj posiljaka')
        plt.xlabel('Korisnici')
        plt.ylabel('Posiljke')
        plt.plot(X, slr.intercept_ + slr.coef_[0] * X, color='blue', linewidth=2, label = 'Regresiona prava')   #<-- Formula Y = b0 + b1*X
        plt.grid(True)
        plt.title('Prikaz stvarnog broja posiljaka i regresione prave')
        plt.legend(loc='upper left')
        plt.show()
        input('\nPress any key to continue . . .\n')

        print('\nb. Vrsimo predikciju kolicine posiljaka ')
        attr_list = list(baza.columns.values)
        attr_list.remove('PAK')
      #  attr_list.remove('Domacinstva')
        attr_list.remove('Pravna lica')
      #  attr_list.remove('Pravna lica stan')
      #  attr_list.remove('Pravna lica lokal')
      #  attr_list.remove('Liftovi')
      #  attr_list.remove('Sanducici')
      #  attr_list.remove('Broj spratova')
      #  attr_list.remove('Objekti')
        attr_list.remove('Korisnici')
        attr_list.remove('O')
        attr_list.remove('OS')
        attr_list.remove('R')
        attr_list.remove('RS')
        attr_list.remove('AR')
        attr_list.remove('ARS')
        attr_list.remove('PU')
        attr_list.remove('TR10')
        attr_list.remove('V')
        attr_list.remove('Votk')
        attr_list.remove('Ropt')
        attr_list.remove('Rino')
        attr_list.remove('TG')
        attr_list.remove('Posiljke')
        X = baza[attr_list].values

        slr.fit(X, Y)

        model_coef = {'Feature': attr_list, 'LR': slr.coef_}   #  Estimated coefficients for the linear regression problem
        model_coef = pd.DataFrame(model_coef, columns=['Feature', 'LR'])
        print('\nKoeficijent modela:\n', model_coef, '\n')  # parameters of the regression line

        y_pred = slr.predict(X)   # Predict using the fitted model
        plt.scatter(Y, y_pred, c='b', label = 'Predvidjen broj posiljaka')
        plt.plot([0,3000],[0,3000], 'r-', label = 'Regresiona prava')
        plt.xlabel('Stvarne vrednosti Posiljaka')
        plt.ylabel('Predvidjene vrednosti Posiljaka')
        plt.title('Predikcija kolicine posiljaka pomocu modela')
        plt.grid(True)
        plt.legend(loc='upper left')
        plt.show()
        input('Press any key to continue . . . \n')

        # 3. Korak - Evaluacija modela Linearne regresije, odredjivanje R^2, SEE, MSE ----------------------------------
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

        print("c. Treniranje modela Linearne regresije \n")
        slr.fit(X_train, y_train)
        y_train_pred = slr.predict(X_train)
        y_test_pred = slr.predict(X_test)

        plt.scatter(y_train_pred, y_train_pred - y_train, c='blue', marker='o', label='Trening podaci')
        plt.scatter(y_test_pred, y_test_pred - y_test, c='lightgreen', marker='s', label='Test podaci')
        plt.title('Podaci za trening i podaci za testiranje modela')
        plt.xlabel('Predicted values')
        plt.ylabel('Residuals')
        plt.legend(loc='upper left')
        plt.hlines(y=0, xmin=-10, xmax=2000, lw=2, color='red')
        plt.xlim([-10, 2000])
        plt.tight_layout()
        plt.show()

        print('Koeficijent modela:\n', model_coef, '\n')
        model_coef['LR train'] = slr.coef_
        print(model_coef, '\n')
        input('Press any key to continue . . .\n ')

        print("d. Ocena pouzdanosti Linearne regresije ")
        print('\nOdredjujem koeficijent R^2 i greske SEE i MSE')
        r2_train = r2_score(y_train, y_train_pred)     # Coefficient of determination
        r2_test = r2_score(y_test, y_test_pred)

        mse_train = mean_squared_error(y_train, y_train_pred)  # Mean squared error regression loss
        mse_test = mean_squared_error(y_test, y_test_pred)

        see_train = math.sqrt(mse_train)  # Standard Error of Estimate
        see_test = math.sqrt(mse_test)

        model_evaluate = {'Score': ['R^2 train', 'R^2 test ', 'SEE train', 'SEE test ', 'MSE train', 'MSE test '],
                             'LR': [r2_train, r2_test, see_train, see_test, mse_train, mse_test]}
        model_evaluate = pd.DataFrame(model_evaluate, columns=['Score', 'LR'])
        print(model_evaluate, '\n')

#--------- Funkcija za interval poverenja -----------------------------------------------------------------------------
    def interval_poverenja(self):
        print('e. Interval poverenja \n')
        baza = pd.read_csv("baza.csv")
        baza.columns = ['PAK', 'Objekti', 'Broj spratova', 'Domacinstva', 'Pravna lica stan',  'Pravna lica lokal',
                        'Pravna lica', 'Korisnici', 'Liftovi', 'Sanducici', 'O', 'OS', 'R' ,'RS', 'AR', 'ARS', 'PU',
                        'TR10', 'V', 'Votk', 'Ropt', 'Rino', 'TG', 'Posiljke']
        X = baza[['Korisnici']].values
        Y = baza['Posiljke'].values
        slr = LinearRegression()
        slr.fit(X, Y)    # Fit regression model
        beta0 = slr.intercept_
        beta1 = slr.coef_

        #----- Testiranje modela Linearne Regresije --------------------------------------------------------------------
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=0)

        slr.fit(X_train, y_train)
        y_train_pred = slr.predict(X_train)
        y_test_pred = slr.predict(X_test)
        r2_train = r2_score(y_train, y_train_pred)     # Coefficient of determination
        r2_test = r2_score(y_test, y_test_pred)
        mse_train = mean_squared_error(y_train, y_train_pred)  # Mean squared error regression loss
        mse_test = mean_squared_error(y_test, y_test_pred)
        see_train = math.sqrt(mse_train)  # Standard Error of Estimate
        see_test = math.sqrt(mse_test)

        #--------- Odredjujem Interval poverenja -----------------------------------------------------------------------
        print('Uzmemo test vrednosti za X koje obelezim sa Xnovo')
        Xnovo = X_test
        #print('\nXnovo = ', Xnovo)
        Ynovo = y_test
        #print('\nYnovo = ', Ynovo)
        print('\nZatim izracunamo Y^ = bo + b1 * Xnovo')
        print('bo = %.3f' % beta0)
        print('b1 = %.3f' % beta1)
        print('\nY^ = %.3f' % beta0,'+ %.3f' % beta1, '* Xnovo')
        Ymodel = beta0 + beta1 * Xnovo
        #print('\nY^ = ', Ymodel)
        print('\nIz Ygornje = Y^ + 2 * SEE sledi gornja granica intervala')
        Ygornje = Ymodel+2*see_train
        #print('\nYgornje = ', Ygornje)
        print('\nIz Ydonje = Y^ - 2 * SEE sledi donja granica intervala')
        Ydonje = Ymodel-2*see_train
        #print('\nYdonje = ', Ydonje)

        #------ Graficki prikaz Intervala poverenja --------------------------------------------------------------------
        print('\nGraficki prikaz Intervala poverenja')
        plt.scatter(Xnovo, Ynovo, color='blue', label ='Posiljke')
        plt.plot(Xnovo, Ymodel, color='red', label ='Regresiona prava')
        plt.plot(Xnovo, Ygornje, color='green', label ='Gornja granica intervala')
        plt.plot(Xnovo, Ydonje, color='lightgreen', label ='Donja granica intervala')
        plt.legend(loc='upper left')
        plt.xlabel('Xnovo - proizvoljan broj korisnika')
        plt.ylabel('Y - broj posiljaka')
        plt.grid(True)
        plt.title('Interval poverenja')
        plt.show()

#--------- Funkcija za opciju 5 poredjenje LR, Ridge, Lasso i ElasticNet -----------------------------------------------
    def ostale_metode(self):
        data = pd.read_csv("baza.csv")
        data.columns = ['PAK', 'Objekti', 'Broj spratova', 'Domacinstva', 'Pravna lica stan',  'Pravna lica lokal',
                        'Pravna lica', 'Korisnici', 'Liftovi', 'Sanducici', 'O', 'OS', 'R' ,'RS', 'AR', 'ARS', 'PU',
                        'TR10', 'V', 'Votk', 'Ropt', 'Rino', 'TG', 'Posiljke']
        X = data[['Korisnici']].values
        y = data['Posiljke'].values
        slr = LinearRegression()
        slr.fit(X, y)
        print('Slope: %.3f' % slr.coef_[0])
        print('Intercept: %.3f' % slr.intercept_)

        attr_list = list(data.columns.values)
        attr_list.remove('PAK')
       # attr_list.remove('Objekti')
       # attr_list.remove('Broj spratova')
       # attr_list.remove('Domacinstva')
       # attr_list.remove('Pravna lica stan')
       # attr_list.remove('Pravna lica lokal')
        attr_list.remove('Pravna lica')
       # attr_list.remove('Liftovi')
       # attr_list.remove('Sanducici')
        attr_list.remove('O')
        attr_list.remove('OS')
        attr_list.remove('R')
        attr_list.remove('RS')
        attr_list.remove('AR')
        attr_list.remove('ARS')
        attr_list.remove('PU')
        attr_list.remove('TR10')
        attr_list.remove('V')
        attr_list.remove('Votk')
        attr_list.remove('Ropt')
        attr_list.remove('Rino')
        attr_list.remove('TG')
        attr_list.remove('Posiljke')

        X = data[attr_list].values
        slr.fit(X, y)

        model_coef = {'Feature': attr_list, 'LR': slr.coef_}   #  Estimated coefficients for the linear regression problem
        model_coef = pd.DataFrame(model_coef, columns=['Feature', 'LR'])
        print('\nKoeficijenti modela:\n', model_coef, '\n')  # parameters of the regression line
        input('Press any key to continue . . . ')

#-----------------------------------------------------------------------------------------------------------------------
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

        print("Treniranje modela Linearne regresije\n")
        slr.fit(X_train, y_train)
        y_train_pred = slr.predict(X_train)
        y_test_pred = slr.predict(X_test)

        print('Koeficijenti modela:\n', model_coef, '\n')
        model_coef['LR train'] = slr.coef_
        print(model_coef, '\n')
        input('Press any key to continue . . .\n ')

        print("Ocena modela Linearne regresije")
        mse_train = mean_squared_error(y_train, y_train_pred)  # Mean squared error regression loss
        mse_test = mean_squared_error(y_test, y_test_pred)

        r2_train = r2_score(y_train, y_train_pred)     # Coefficient of determination
        r2_test = r2_score(y_test, y_test_pred)

        model_evaluate = {'Score': ['MSE train', 'MSE test', 'R^2 train', 'R^2 test'], 'LR': [mse_train, mse_test, r2_train, r2_test]}
        model_evaluate = pd.DataFrame(model_evaluate, columns=['Score', 'LR'])
        print(model_evaluate, '\n')
        input('Press any key to continue . . . \n')

        print("Treniranje modela Ridge regresije\n")
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train, y_train)
        y_train_pred = ridge.predict(X_train)
        y_test_pred = ridge.predict(X_test)

        model_coef['Ridge'] = ridge.coef_
        print(model_coef, '\n')
        input('Press any key to continue . . . \n')

        print("Ocena modela Ridge regresije\n")
        mse_train = mean_squared_error(y_train, y_train_pred)  # Mean squared error regression loss
        mse_test = mean_squared_error(y_test, y_test_pred)

        r2_train = r2_score(y_train, y_train_pred)     # Coefficient of determination
        r2_test = r2_score(y_test, y_test_pred)

        model_evaluate['Ridge'] = [mse_train, mse_test, r2_train, r2_test]
        print(model_evaluate, '\n')
        input('Press any key to continue . . . \n')

        print("Treniranje modela Lasso regresije\n")
        lasso = Lasso(alpha=1.0)   # alpha parameter controls the degree of sparsity of the coefficients estimated
        lasso.fit(X_train, y_train)
        y_train_pred = lasso.predict(X_train)
        y_test_pred = lasso.predict(X_test)

        model_coef['Lasso'] = lasso.coef_
        print(model_coef, '\n')

        print("Ocena modela Lasso regresije\n")
        mse_train = mean_squared_error(y_train, y_train_pred)  # Mean squared error regression loss
        mse_test = mean_squared_error(y_test, y_test_pred)

        r2_train = r2_score(y_train, y_train_pred)     # Coefficient of determination
        r2_test = r2_score(y_test, y_test_pred)

        model_evaluate['Lasso'] = [mse_train, mse_test, r2_train, r2_test]
        print(model_evaluate, '\n')
        input('Press any key to continue . . . \n')

        print("Treniranje modela ElasticNet regresije\n")
        elastic_net = ElasticNet(alpha=1.0, l1_ratio=0.5, tol=0.1,)   # ElasticNet model, alpha parameter controls the degree of sparsity of the coefficients estimated, l1_ratio - mixing parameter
        elastic_net.fit(X_train, y_train)
        y_train_pred = elastic_net.predict(X_train)
        y_test_pred = elastic_net.predict(X_test)

        model_coef['ElasticNet'] = elastic_net.coef_
        print(model_coef, '\n')
        input('Press any key to continue . . . \n')

        print("Ocena modela ElasticNet regresije\n")
        mse_train = mean_squared_error(y_train, y_train_pred)  # Mean squared error regression loss
        mse_test = mean_squared_error(y_test, y_test_pred)

        r2_train = r2_score(y_train, y_train_pred)     # Coefficient of determination
        r2_test = r2_score(y_test, y_test_pred)

        model_evaluate['ElasticNet'] = [mse_train, mse_test, r2_train, r2_test]
        print(model_evaluate, '\n')