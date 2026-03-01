# Rans0mWar3 — CyberSecurity

## Descrizione

Progetto didattico di **cybersecurity** incentrato sulla comprensione dei meccanismi crittografici utilizzati dai ransomware. Gli script implementano una simulazione di crittografia ibrida **RSA + Fernet (AES)** e il relativo processo di decifratura.

> **Disclaimer**: Questo progetto è stato realizzato esclusivamente a scopo educativo e di ricerca accademica. Non deve essere utilizzato per scopi malevoli.

---

## Architettura Crittografica

Il flusso simulato è il seguente:

```
┌─────────────────────────────────────────────────────┐
│  FASE DI CIFRATURA (simulata)                       │
│                                                     │
│  1. Generazione chiave simmetrica Fernet (AES-128)  │
│  2. Cifratura dei file con Fernet → .encrypt        │
│  3. Cifratura della chiave Fernet con RSA (pubblica)│
│  4. Salvataggio chiave Fernet cifrata su disco       │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│  FASE DI DECIFRATURA (Decryptor.py)                 │
│                                                     │
│  1. Lettura chiave Fernet cifrata da file           │
│  2. Decifratura con chiave RSA privata (PKCS1_OAEP) │
│  3. Scansione ricorsiva directory → file .encrypt   │
│  4. Decifratura file e ripristino nome originale    │
└─────────────────────────────────────────────────────┘
```

---

## Script Presenti

### `RSA_private_public_keys.py`

Genera una coppia di chiavi **RSA a 2048 bit** utilizzando `pycryptodome`:

- **Output**: `private.pem` (chiave privata) e `public.pem` (chiave pubblica).
- **Librerie**: `Crypto.PublicKey.RSA`, `Crypto.Random`.

### `Decryptor.py` — *Lazarus Decryptor*

Classe `Decryptor` che implementa il processo completo di decifratura:

| Metodo | Descrizione |
|---|---|
| `read_fernet_key(key_path)` | Legge la chiave Fernet cifrata e la decifra con la chiave RSA privata tramite `PKCS1_OAEP` |
| `find_encrypted_files()` | Scansione ricorsiva (`os.walk`) della directory target per trovare tutti i file `.encrypt` |
| `decrypt_files(file_path)` | Decifra un singolo file con Fernet, rimuove l'estensione `.encrypt` e ripristina il nome originale |
| `decrypt_system()` | Orchestratore: decifra tutti i file trovati nella directory |

**Uso da CLI**:
```bash
python Decryptor.py -k <chiave_fernet_cifrata> -d <directory_da_decifrare> -p <chiave_privata_RSA>
```

| Argomento | Descrizione |
|---|---|
| `-k` / `--key` | Percorso al file contenente la chiave Fernet cifrata |
| `-d` / `--directory` | Directory root contenente i file `.encrypt` |
| `-p` / `--private` | Percorso alla chiave privata RSA (`private.pem`) |

---

## Dipendenze

```
cryptography     # Fernet (AES simmetrico)
pycryptodome     # RSA, PKCS1_OAEP (asimmetrico)
```
