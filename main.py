_author_='Bane'

import sys
import os
import random
from klasa1 import *
from klasa2 import *
from klasa3 import *

print("*************************************************************************************************************")
print(" Dobrodosli u Python projekat: Analiza dostave pismonosnih posiljaka")
print("*************************************************************************************************************")
print(" Autor: student Branislav Manojlovic, broj indeksa: E1 88/2015")
print("*************************************************************************************************************")
print("\n")

#-------- Razne funkcije koje sam upotrebio -------------------------------
def funkcija_meni():
    print("*********** MENI ***********")
    print("Izaberite zeljenu opciju:")
    print("1. Prikaz analiziranog podrucja")
    print("2. Prikaz podataka iz baze")
    print("3. Analiza zavisnosti izmedju varijabli")
    print("4. Analiza pomocu Linearne Regresije")
    print("5. Poredjenje LR, Ridge, Lasso, ElasticNet regresije")
    print("6. Izlaz iz programa")
    print("\n")
    return

def izbor_opcije():
    print("Odaberite zeljenu opciju:")
    opcija = sys.stdin.readline()
    return int(opcija)

#-------- MENI sekcija ----------------------------------------------
odgovor = 1
while (odgovor != 0):
    funkcija_meni()
    opcija = izbor_opcije()
    if (opcija == 1):
        print("1. Prikaz analiziranog podrucja")
        Klasa1().prikaz_slike()
        input('Press any key to continue . . . ')
        Klasa1().korisnici()
        input('Press any key to continue . . . \n')
    elif (opcija == 2):
        print("2. Ispis podataka iz baze")
        Klasa1().prikaz_baze()
        input('Press any key to continue . . . \n')
        Klasa1().kategorije()
        input('Press any key to continue . . . \n')
    elif (opcija == 3):
        print("3. Analiza zavisnosti izmedju varijabli")
        Klasa2().kljucne_varijable()
        input('Press any key to continue . . . \n')
        Klasa2().univarijantna_raspodela()
        input('\nPress any key to continue . . . \n')
    elif (opcija == 4):
        print("\n4. Analiza pomocu Linearne Regresije\n")
        Klasa3().linearna_regresija()
        input('Press any key to continue . . . \n')
        Klasa3().interval_poverenja()
        input('\nPress any key to continue . . . \n')
    elif (opcija == 5):
        print("\n5. Poredjenje LR, Ridge, Lasso, ElasticNet regresije\n")
        Klasa3().ostale_metode()
        input('\nPress any key to continue . . . \n')
    elif (opcija == 6):
        print("\n******** Izasli ste iz programa - Dovidjenja! ***************************************************************")
        print("*************************************************************************************************************")
        sys.exit()
    else:
        print("Izabrali ste pogresan broj, pokusajte ponovo!")
        input('Press any key to continue . . . \n')

print("Kraj programa.")