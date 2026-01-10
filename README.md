# FaceQuiz backend

Ovo je backend za FIPU FaceQuiz aplikaciju

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