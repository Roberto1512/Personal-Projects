from flask import Flask, render_template, request, session, redirect
from flask_mysqldb import MySQL
from flask import flash
import hashlib
from urllib.parse import urlparse
import tempfile
#import os
#from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
#from tensorflow.keras.preprocessing import image

app = Flask(__name__)
app.secret_key = 'chiave_segreta'


app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'theverifiersdb'

mysql = MySQL(app)

# Carica il modello ResNet50 pre-addestrato
#model = ResNet50(weights='imagenet')


@app.route('/')
def homeUtente():
    if 'logged_in' in session:
        if 'user_type' in session:
            if session['user_type'] == 'admin':
                return render_template('viewsAdmin/homeAdmin.html')
            else:
                return render_template('viewsUtente/homeUtente.html')
    else:
        return render_template('viewsUtente/homeUtente.html')


@app.route('/homeAdmin')
def homeAdmin():
    return render_template('viewsAdmin/homeAdmin.html')


@app.route('/registration', methods=['POST', 'GET'])
def register():
    if request.method == 'GET':
        return render_template('viewsUtente/registration.html')

    if request.method == 'POST':
        nome = request.form['name']
        cognome = request.form['surname']
        email = request.form['email']
        password = request.form['password']
        hash_object = hashlib.sha256(password.encode())
        hashed_password = hash_object.hexdigest()

        phoneNumber = request.form['phone']

        cursor = mysql.connection.cursor()
        # Esegui la query per verificare se l'utente è già presente nel database
        cursor.execute("SELECT phone FROM credenzialiUtenti WHERE phone = %s", (phoneNumber,))
        result = cursor.fetchone()

        if result:
            # L'utente è già presente nel database, annulla l'operazione di registrazione
            cursor.close()
            return render_template('viewsUtente/registration.html', error="Registrazione non conclusa, numero di telefono già registrato.")

        # L'utente non è presente nel database, esegui l'operazione di registrazione
        cursor.execute('''INSERT INTO credenzialiUtenti VALUES(%s,%s,%s,%s,%s)''', (nome, cognome, email, hashed_password, phoneNumber))
        mysql.connection.commit()
        cursor.close()
        session['logged_in'] = True
        session['email'] = email
        session['user_type'] = 'utente'  # Imposta il tipo di utente come 'utente'
        return render_template('viewsUtente/subscription.html')


@app.route('/successo')
def success():
    return "Registrazione avvenuta con successo!"


# Funzione per verificare le credenziali nel database
@app.route('/log_in', methods=['POST', 'GET'])
def login():
    if request.method == 'GET':
        return render_template('login.html')

    if request.method == 'POST':
        email = request.form['email']
        password = request.form['pass']
        hash_object = hashlib.sha256(password.encode())
        hashed_password = hash_object.hexdigest()

        cursor = mysql.connection.cursor()
        cursor.execute("SELECT * FROM credenzialiUtenti WHERE email = %s AND password = %s", (email, hashed_password))
        mysql.connection.commit()
        result = cursor.fetchone()  # Fetch the first matching row

        if result:
            # Credenziali corrette, l'utente esiste nel database
            cursor.close()
            session['logged_in'] = True
            session['email'] = email
            session['user_type'] = 'utente'  # Imposta il tipo di utente come 'utente'

            return render_template('viewsUtente/homeUtente.html')
        else:
            # Credenziali errate o utente non trovato nel database
            cursor.close()
            return render_template('login.html', error="Credenziali errate. Riprova.")


@app.route('/recovery')
def recovery():
    return render_template('viewsUtente/recovery.html')


@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    session.pop('email', None)
    return render_template('viewsUtente/homeUtente.html')


@app.route('/log_inAdmin', methods=['POST', 'GET'])
def loginAdmin():
    if request.method == 'GET':
        return render_template('login.html')

    if request.method == 'POST':
        id = request.form['id']
        password = request.form['pass']

        cursor = mysql.connection.cursor()
        cursor.execute("SELECT * FROM credenzialiamministratori WHERE id = %s AND password = %s", (id, password))
        mysql.connection.commit()
        result = cursor.fetchone()  # Fetch the first matching row

        if result:
            # Credenziali corrette, l'utente esiste nel database
            cursor.close()
            session['logged_in'] = True
            session['user_type'] = 'admin'  # Imposta il tipo di utente come 'admin'
            return render_template('viewsAdmin/homeAdmin.html')
        else:
            # Credenziali errate o utente non trovato nel database
            cursor.close()
            return render_template('login.html', error="Credenziali errate. Riprova.")


@app.route('/gestioneProfilo')
def gestioneProfilo():
    email = session['email']
    cursor = mysql.connection.cursor()
    cursor.execute('SELECT nome, cognome, email, phone FROM credenzialiutenti WHERE email = %s', (email,))
    data = cursor.fetchone()

    return render_template('viewsUtente/gestioneProfilo.html', data=data)

@app.route('/modificaProfilo', methods=['GET', 'POST'])
def modificaProfilo():
    if request.method == 'GET':
        email = session['email']
        cursor = mysql.connection.cursor()
        cursor.execute('SELECT nome, cognome, email, phone FROM credenzialiUtenti WHERE email = %s', (email,))
        data = cursor.fetchone()
        cursor.close()
        return render_template('viewsUtente/modificaProfilo.html', data=data)

    if request.method == 'POST':
        email = session['email']
        old_password = request.form['old_password']
        new_password = request.form['new_password']

        cursor = mysql.connection.cursor()
        cursor.execute('SELECT password FROM credenzialiUtenti WHERE email = %s', (email,))
        result = cursor.fetchone()

        if result:
            stored_password = result[0]
            hash_object = hashlib.sha256(old_password.encode())
            hashed_password = hash_object.hexdigest()

            if hashed_password == stored_password:
                if new_password:
                    hash_object = hashlib.sha256(new_password.encode())
                    hashed_new_password = hash_object.hexdigest()
                    cursor.execute('UPDATE credenzialiUtenti SET password = %s WHERE email = %s', (hashed_new_password, email))
                    mysql.connection.commit()
                    cursor.close()
                    return redirect('/gestioneProfilo')
                else:
                    cursor.close()
                    error = "Inserisci una nuova password."
                    return render_template('viewsUtente/modificaProfilo.html', message=error)
            else:
                cursor.close()
                error = "Password vecchia non corretta."
                # Annullamento dell'interrogazione al database
                flash("Inserimento password errata")
                return redirect('/modificaProfilo')

        else:
            cursor.close()
            error = "Utente non trovato nel database."
            return render_template('viewsUtente/modificaProfilo.html', message=error)


@app.route('/gestioneNotizie', methods=['GET', 'POST'])
def gestioneNotizie():
    if request.method == 'GET':
        return render_template('viewsAdmin/gestioneNotizie.html')

    if request.method == 'POST':
        news_url = request.form['newsUrl']

        cursor = mysql.connection.cursor()
        cursor.execute('''INSERT INTO notizieaggiunte (URL) VALUES (%s)''', (news_url,))
        mysql.connection.commit()
        cursor.close()

        return redirect('/gestioneNotizie')


@app.route('/creazioneForm')
def creazioneForm():
    cursor = mysql.connection.cursor()
    cursor.execute('SELECT nome,oggetto, messaggio, risposta FROM form')
    message = cursor.fetchall()
    cursor.close()
    return render_template('viewsUtente/form.html', message=message)


@app.route('/aggiungiForm', methods=['GET', 'POST'])
def aggiungiForm():
    if request.method == 'GET':
        return render_template('viewsUtente/form.html')
    if request.method == 'POST':
        nome = request.form['Nome']
        email = request.form['Email']
        oggetto = request.form['Oggetto']
        messaggio = request.form['Messaggio']

        # creo un cursore per il db
        cursor = mysql.connection.cursor()
        cursor.execute('''INSERT INTO form (nome, email, oggetto, messaggio) VALUES (%s,%s,%s,%s)''',
                       (nome, email, oggetto, messaggio))
        mysql.connection.commit()
        cursor.close()
        return redirect('/creazioneForm')


@app.route('/rispondiAiForm', methods=['POST'])
def rispondiAIForm():
        form_id = request.form['form_id']
        risposta = request.form['risposta']
        cur = mysql.connection.cursor()
        cur.execute("UPDATE form SET risposta = %s WHERE id = %s", (risposta, form_id))
        mysql.connection.commit()
        cur.close()
        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM form WHERE risposta IS  NULL")
        form_senza_risposta = cur.fetchall()
        count = len(form_senza_risposta)
        if count > 0:
            return redirect('/senza_risposta')
        else:
            return redirect('/gestioneForm')


@app.route('/gestioneForm')
def gestioneForm():
    cursor = mysql.connection.cursor()
    cursor.execute('SELECT id, nome, oggetto, messaggio, risposta FROM form WHERE risposta IS NOT NULL ')
    form_risposta = cursor.fetchall()
    cursor.close()
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM form WHERE risposta IS  NULL")
    form_senza_risposta = cur.fetchall()
    count = len(form_senza_risposta)
    return render_template('viewsAdmin/gestioneForm.html', form_risposta=form_risposta, count=count)


@app.route('/senza_risposta')
def senzaRisposta():
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM form WHERE risposta IS  NULL")
    form_senza_risposta = cur.fetchall()
    cur.close()

    return render_template('viewsAdmin/rispondereAiForm.html', form_senza_risposta=form_senza_risposta)


@app.route('/verifica')
def verifica():
    return render_template('viewsUtente/verifica.html')


@app.route('/verificaDominio', methods=['POST'])
def verificaDominio():
    link = request.form['link']

    # Estrarre il dominio dal link utilizzando urlparse
    parsed_url = urlparse(link)
    domain = parsed_url.netloc
    domain = "https://"+domain+"/"
    # Connessione al database
    cursor = mysql.connection.cursor()

    # Esegui la query per controllare se il dominio è presente nel database
    cursor.execute("SELECT URL FROM dominiverificati WHERE URL = %s", (domain,))
    mysql.connection.commit()

    result = cursor.fetchone()  # Fetch the first matching row

    # Chiudi il cursore e la connessione al database
    cursor.close()

    if result:
        # Il dominio è presente nel database dei domini verificati
        return render_template('viewsUtente/verifica.html', message=f"Il dominio {domain} è attendibile.")
    else:
        # Il dominio non è presente nel database dei domini verificati
        return render_template('viewsUtente/verifica.html', message=f"Il dominio {domain} non è attendibile.")


@app.route('/verificaImmagine', methods=['POST', 'GET'])
def verificaImmagine():
    if request.method == 'GET':
        return render_template('viewsUtente/verifica.html')
    if request.method == 'POST':
        if 'image' in request.files:
            image = request.files['image']
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
                image.save(temp_file.name)  # Salva l'immagine nel file temporaneo

                img_path = temp_file.name  # Ottieni il percorso del file temporaneo
                import subprocess
                command = ['python', 'Image-Manipulation-Detection/analyze.py', '-p', img_path]
                checkout = subprocess.check_output(command).decode('utf-8').strip()
                return render_template('viewsUtente/verifica.html', checkout=checkout)
    return render_template('viewsUtente/verifica.html')


@app.route('/notizieVerificate', methods=['GET', 'POST'])
def notizieVerificate():
    if request.method == 'GET':
        # Fetch delle notizie verificate dalla tabella notizieaggiunte
        cursor = mysql.connection.cursor()
        cursor.execute("SELECT URL FROM notizieaggiunte")
        news_urls = [row[0] for row in cursor.fetchall()]
        cursor.close()

        return render_template('viewsUtente/notizieVerificate.html', news_urls=news_urls)

    if request.method == 'POST':
        news_url = request.form['newsUrl']

        cursor = mysql.connection.cursor()
        cursor.execute('''INSERT INTO notizieaggiunte (URL) VALUES (%s)''', (news_url,))
        mysql.connection.commit()
        cursor.close()

        return redirect('/notizieVerificate')


@app.route('/filtro')
def filtro():
    return render_template('viewsUtente/filtro.html')


@app.route('/segnalazioneNotizie', methods=['GET', 'POST'])
def segnalazioneNotizie():
    if request.method == 'GET':
        return render_template('viewsUtente/segnalazioneNotizie.html')

    if request.method == 'POST':
        news_url = request.form['newsUrl']

        cursor = mysql.connection.cursor()
        cursor.execute('''INSERT INTO notiziesegnalate (URL) VALUES (%s)''', (news_url,))
        mysql.connection.commit()
        cursor.close()

        return redirect('/segnalazioneNotizie')


@app.route('/gestioneSegnalazioni', methods=['GET', 'POST'])
def gestioneSegnalazioni():
    if request.method == 'POST':
        action = request.form['action']
        selected_urls = request.form.getlist('seleziona[]')

        if action == 'segna':
            # Esegui l'azione di segnalazione utilizzando gli URL selezionati
            cursor = mysql.connection.cursor()
            for url in selected_urls:
                # Aggiungi l'URL alla tabella blacklist
                cursor.execute("INSERT INTO blacklist (URL) VALUES (%s)", (url,))
                # Elimina l'URL dalla tabella notiziesegnalate
                cursor.execute("DELETE FROM notiziesegnalate WHERE URL = %s", (url,))
            mysql.connection.commit()
            cursor.close()

            return redirect('/gestioneSegnalazioni')

        elif action == 'elimina':
            # Elimina gli URL dalla tabella notiziesegnalate
            cursor = mysql.connection.cursor()
            for url in selected_urls:
                cursor.execute("DELETE FROM notiziesegnalate WHERE URL = %s", (url,))
            mysql.connection.commit()
            cursor.close()

            return redirect('/gestioneSegnalazioni')

    cursor = mysql.connection.cursor()
    cursor.execute("SELECT URL FROM notiziesegnalate")
    urls = [row[0] for row in cursor.fetchall()]
    cursor.close()
    return render_template('viewsAdmin/gestioneSegnalazioni.html', urls=urls)


@app.route('/dominiVerificati')
def dominiVerificati():
    # Connessione al database
    cursor = mysql.connection.cursor()

    # Eseguire la query per ottenere i domini verificati
    cursor.execute("SELECT URL FROM dominiverificati")
    mysql.connection.commit()
    # Ottenere i risultati della query
    domains = cursor.fetchall()

    # Chiudere il cursore e la connessione al database
    cursor.close()

    # Renderizzare il template HTML con i domini
    return render_template('viewsUtente/dominiVerificati.html', domains=domains)


@app.route('/gestioneDomini')
def gestioneDomini():
    # Connessione al database
    cursor = mysql.connection.cursor()

    # Eseguire la query per ottenere i domini verificati
    cursor.execute("SELECT URL FROM dominiverificati")
    mysql.connection.commit()
    # Ottenere i risultati della query
    domains = cursor.fetchall()

    # Chiudere il cursore e la connessione al database
    cursor.close()

    # Renderizzare il template HTML con i domini
    return render_template('viewsAdmin/gestioneDomini.html', domains=domains)




@app.route('/aggiungi_dominio', methods=['POST'])
def aggiungiDominio():
    domain = request.form['domain']

    # Connessione al database
    cursor = mysql.connection.cursor()

    # Esegui la query per verificare se il dominio è già presente nel database
    cursor.execute("SELECT URL FROM dominiverificati WHERE URL = %s", (domain,))
    result = cursor.fetchone()

    if result:
        # Il dominio è già presente nel database, annulla l'operazione di inserimento
        flash("Il dominio {} è già presente nel database.".format(domain))
    else:
        # Il dominio non è presente nel database, esegui l'operazione di inserimento
        cursor.execute("INSERT INTO dominiverificati (URL) VALUES (%s)", (domain,))
        mysql.connection.commit()

    # Chiudi il cursore e la connessione al database
    cursor.close()

    return redirect('/gestioneDomini')


@app.route('/rimuovi_dominio', methods=['POST'])
def rimuoviDominio():
    domain = request.form['domain']

    # Connessione al database
    cursor = mysql.connection.cursor()

    # Esegui la query per rimuovere il dominio dal database
    cursor.execute("DELETE FROM dominiverificati WHERE URL = %s", (domain,))
    mysql.connection.commit()

    # Chiudi il cursore e la connessione al database
    cursor.close()

    return redirect('/gestioneDomini')


@app.route('/elimina_domini_selezionati', methods=['POST'])
def eliminaDominiSelezionati():
    domains = request.form.getlist('domain')

    cursor = mysql.connection.cursor()

    for domain in domains:
        cursor.execute("DELETE FROM dominiverificati WHERE URL = %s", (domain,))

    mysql.connection.commit()
    cursor.close()

    return redirect('/gestioneDomini')


@app.route('/elimina_profilo', methods=['POST'])
def eliminaProfilo():
    email = session['email']


    cursor = mysql.connection.cursor()
    cursor.execute("DELETE FROM credenzialiUtenti WHERE email = %s", (email,))
    mysql.connection.commit()
    cursor.close()

    session.pop('logged_in', None)

    return render_template('viewsUtente/homeUtente.html')


@app.route('/subscription')
def subscription():
    return render_template('viewsUtente/subscription.html')


@app.route('/contatti')
def contatti():
    return render_template('viewsUtente/contatti.html')


@app.route('/faq')
def faq():
    return render_template('viewsUtente/faq.html')



if __name__ == "__main__":
    app.run(host='localhost', port=5000)
