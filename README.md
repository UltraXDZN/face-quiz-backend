# FaceQuiz backend

Ovo je backend za FIPU FaceQuiz aplikaciju

## Postavljanje Firebase-a:

Prije nego što se aplikacija upali potrebno je napraviti svoju `.env` datoteku te `serviceAccountKey.json` ključ.

`.env` datoteku možete ispuniti tako da:
1. napravite kopiju `.env_example` i preimenujete je u `.env`
2. ispunite novonapravljenu datoteku podacima iz *Firebase Console* > *Project Settings* > ***General***

`serviceAccountKey.json` je **Privatni ključ** koji omogućuje **APSOLUTNI** pristup firebase aplikaciji:
1. Odite na *Firebase Console* > *Project Settings* > ***Service Accounts***
2. Kliknite na ***Generate new private key***
3. Spremite ključ pod naziv `serviceAccountKey.json` (**UPOZORENJE**: Ovaj ključ nesmije ići na javni repozitorij!)

## Pokretanje projekta lokalno

Za lokalno izvođenje i pokretanje aplikacije, iduće naredbe pokrenite iz terminala/command prompta

Izradite virtualno okruženje (venv) pomoću sljedeće naredbe
```bash
python -m venv <naziv_okruženja>
```
Nakon što ste napravili virtualno okruženje, potrebno je ući u njega
```bash
source <naziv_okruženja>/bin/activate
```
Instalirajte sve pakete vezane uz backend iz requirements.txt-a
```bash
pip install -r requirements.txt
```
Pokrenite sljedeću naredbu kako biste podignuli lokalni server
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```