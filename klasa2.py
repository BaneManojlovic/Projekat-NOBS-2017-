_author_='Bane'

import sys
import os
import matplotlib.pyplot as plt
import csv
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


class Klasa2:
#----------- Funkcije za opciju 3. ------------------------------------------------------------------------------------
    def kljucne_varijable(self):
        print('\na. Parovi grafikona koji opisuju vezu izmedju kljucnih varijabli')
        baza = pd.read_csv("baza.csv")
        baza.columns = ['PAK', 'Objekti', 'Broj spratova', 'Domacinstva', 'Pravna lica stan',  'Pravna lica lokal',
                        'Pravna lica', 'Korisnici', 'Liftovi', 'Sanducici', 'O', 'OS', 'R' ,'RS', 'AR', 'ARS', 'PU',
                        'TR10', 'V', 'Votk', 'Ropt', 'Rino', 'TG', 'Posiljke']

        sns.pairplot(baza, vars=['Korisnici', 'Posiljke'])    # Plot a subset of variables
        print('working ...')
        plt.show()

        sns.pairplot(baza, vars=['Objekti', 'Posiljke'])    # Plot a subset of variables
        print('working ...')
        plt.show()

        sns.pairplot(baza, vars=['Broj spratova', 'Posiljke'])    # Plot a subset of variables
        print('working ...')
        plt.show()

        sns.pairplot(baza, vars=['Sanducici', 'Posiljke'])    # Plot a subset of variables
        print('working ...')
        plt.show()

        sns.pairplot(baza, vars=['Korisnici', 'Objekti', 'Broj spratova', 'Sanducici', 'Posiljke'], diag_kind='kde')    # Kernel density estimates for univariate plots
        print('working ...')
        plt.show()

        sns.pairplot(baza, vars=['Korisnici', 'Objekti', 'Broj spratova', 'Sanducici', 'Posiljke'], kind='reg')    # Fit linear regression models to the scatter plots
        print('working ...')
        plt.show()

    def univarijantna_raspodela(self):
        print('b. Prikaz univarijantne raspodele medju varijablama ')
        baza = pd.read_csv("baza.csv")
        baza.columns = ['PAK', 'Objekti', 'Broj spratova', 'Domacinstva', 'Pravna lica stan',  'Pravna lica lokal',
                        'Pravna lica', 'Korisnici', 'Liftovi', 'Sanducici', 'O', 'OS', 'R' ,'RS', 'AR', 'ARS', 'PU',
                        'TR10', 'V', 'Votk', 'Ropt', 'Rino', 'TG', 'Posiljke']
        for col in baza.columns.values[:-1]:
                plt.subplot(211)
                plt.title('KDE - kernelova procena gustine raspodele')
                sns.kdeplot(baza[col], shade=True)    # Fit and plot a univariate or bivariate kernel density estimate
                plt.subplot(212)
                plt.title('Histogram i KDE')
                sns.distplot(baza[col])    # Flexibly plot a univariate distribution of observations
                plt.show()
        input('Press any key to continue . . . ')

        print('c. Prikaz odnosa razlicitih varijabli pomocu boxplot-ova ')
        sns.set_style("whitegrid")
        plt.title('Prikaz odnosa razlicitih tipova posiljaka medjusobno')
        sns.boxplot(data=baza[['O', 'OS', 'R' ,'RS', 'AR', 'ARS', 'PU','TR10', 'V', 'Votk', 'Ropt', 'Rino', 'TG']])
        plt.show()
        plt.title('Odnos razlicitih tipova posiljaka prema ukupno dostavljenim posiljkama')
        sns.boxplot(data=baza[['O', 'OS', 'R' ,'RS', 'AR', 'ARS', 'PU','TR10', 'V', 'Votk', 'Ropt', 'Rino', 'TG', 'Posiljke']])
        plt.show()
        plt.title('Prikaz odnosa broja korisnika')
        sns.boxplot(data=baza[['Domacinstva', 'Pravna lica']])
        plt.show()
        plt.title('Odnos broja razlicitih tipova korisnika prema ukupnom broju korisnika')
        sns.boxplot(data=baza[['Domacinstva', 'Pravna lica', 'Korisnici']])
        plt.show()
        input('Press any key to continue . . . ')

        print('d. Prikaz bivarijantne raspodele ')
        g = sns.JointGrid('Korisnici', 'Posiljke', baza)
        g.plot_marginals(sns.distplot, kde=False, color=".5")
        g.plot_joint(plt.scatter, color=".5", edgecolor="white")
        g.annotate(stats.pearsonr, template="{stat} = {val:.3f} (p = {p:.3g})");
        plt.show()

        g = sns.JointGrid('Objekti', 'Posiljke', baza)
        g.plot_marginals(sns.distplot, kde=False, color=".5")
        g.plot_joint(plt.scatter, color=".5", edgecolor="white")
        g.annotate(stats.pearsonr, template="{stat} = {val:.3f} (p = {p:.3g})");
        plt.show()

        g = sns.JointGrid('Broj spratova', 'Posiljke', baza)
        g.plot_marginals(sns.distplot, kde=False, color=".5")
        g.plot_joint(plt.scatter, color=".5", edgecolor="white")
        g.annotate(stats.pearsonr, template="{stat} = {val:.3f} (p = {p:.3g})");
        plt.show()

        g = sns.JointGrid('Sanducici', 'Posiljke', baza)
        g.plot_marginals(sns.distplot, kde=False, color=".5")
        g.plot_joint(plt.scatter, color=".5", edgecolor="white")
        g.annotate(stats.pearsonr, template="{stat} = {val:.3f} (p = {p:.3g})");
        plt.show()

        sns.jointplot("Korisnici", "Posiljke", data=baza[0:1000], kind="kde", stat_func=stats.pearsonr, color="r")
        plt.show()

        sns.jointplot("Objekti", "Posiljke", data=baza[0:1000], kind="kde", stat_func=stats.pearsonr, color="r")
        plt.show()

        sns.jointplot("Broj spratova", "Posiljke", data=baza[0:1000], kind="kde", stat_func=stats.pearsonr, color="r")
        plt.show()

        sns.jointplot("Sanducici", "Posiljke", data=baza[0:1000], kind="kde", stat_func=stats.pearsonr, color="r")
        plt.show()

        sns.jointplot("Korisnici", "Posiljke", data=baza, kind="hex", stat_func=stats.spearmanr)
        plt.show()

        sns.jointplot("Objekti", "Posiljke", data=baza, kind="hex", stat_func=stats.spearmanr)
        plt.show()

        sns.jointplot("Broj spratova", "Posiljke", data=baza, kind="hex", stat_func=stats.spearmanr)
        plt.show()

        sns.jointplot("Sanducici", "Posiljke", data=baza, kind="hex", stat_func=stats.spearmanr)
        plt.show()