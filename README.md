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

## Korištenje API-a

Svi implementirani API pozivi su testabilni na sljedećem *Postman* linku: https://www.postman.com/ultrax-6527639/workspace/facequizapi

### Users

| API poziv                                           | Opis                                |
| --------------------------------------------------- | ----------------------------------- |
| `GET /api/users/`                                   | Dohvati sve korisnike               |
| `POST /api/users/`                                  | Stvori korisnika                    |
| `GET /api/users/{email}`                            | Dohvati specifičnog korisnika       |
| `PATCH /api/users/{email}`                          | Ažuriraj korisnika                  |
| `DELETE /api/users/{email}`                         | Obriši korisnika                    |
| `GET /api/users/{email}/exams`                      | Dohvati korisnikove ispite          |
| `PUT /api/users/{email}/exams`                      | Ažuriraj pristup korisnika ispitu   |
| `DELETE /api/users/{email}/exams/{exam_id}`         | Obriši korisnikov pristup ispitu    |
| `GET /api/users/tracking/created-users`             | Dohvati praćenje stvorenih korisnika |
| `PUT /api/users/tracking/created-users/{email}`     | Ažuriraj praćenje stvorenog korisnika |

### Solutions

| API poziv                                                    | Opis                                  |
| ------------------------------------------------------------ | ------------------------------------- |
| `POST /api/solutions/`                                       | Upload rješenje                       |
| `GET /api/solutions/{exam_id}/{password}/users`              | Dohvati korisnike s bodovima          |
| `GET /api/solutions/{exam_id}/{password}/users/{email}/results` | Dohvati rezultate korisnika na ispitu |
| `GET /api/solutions/{exam_id}/{password}/results/all`        | Dohvati rezultate svih korisnika      |
| `POST /api/solutions/{exam_id}/{password}/users/{email}`     | Dohvati odgovore korisnika            |

### Exams

| API poziv                              | Opis                             |
| -------------------------------------- | -------------------------------- |
| `POST /api/exams/`                     | Stvori ispit                     |
| `GET /api/exams/data`                  | Dohvati podatke o svim ispitima  |
| `GET /api/exams/{exam_id}/metadata`    | Dohvati metapodatke ispita       |
| `GET /api/exams/{exam_id}/full`        | Dohvati cijeli ispit             |
| `GET /api/exams/{exam_id}/admin`       | Dohvati admin prikaz ispita      |
| `PUT /api/exams/{exam_id}`             | Ažuriraj ispit                   |
| `DELETE /api/exams/{exam_id}`          | Obriši ispit                     |

### Leaderboard

| API poziv                          | Opis                       |
| ---------------------------------- | -------------------------- |
| `GET /api/leaderboard/`            | Dohvati ljestvicu          |
| `PUT /api/leaderboard/`            | Ažuriraj ljestvicu         |
| `POST /api/leaderboard/generate`   | Generiraj ljestvicu        |

### Auth

| API poziv                    | Opis                      |
| ---------------------------- | ------------------------- |
| `GET /api/auth/google/login` | Google Login              |
| `GET /api/auth/google/callback` | Google Callback        |
| `POST /api/auth/verify`      | Verify User Token         |

### Default

| API poziv                    | Opis              |
| ---------------------------- | ----------------- |
| `GET /`                      | Root endpoint     |