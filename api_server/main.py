import signal
import sys
from http.client import BAD_REQUEST, FORBIDDEN, INTERNAL_SERVER_ERROR, METHOD_NOT_ALLOWED, NOT_FOUND, UNAUTHORIZED
from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
import secrets
from db import create_table_writekey, create_table_readkey, create_table_cfg, \
                create_table_output, get_db, delete_all_api, delete_all_cfg, delete_all_output, create_table_account
from cryptography.fernet import Fernet
import write
import read
import os

app = Flask(__name__)
app.secret_key = '9e7d2c88b4d94e8fb2c3dd1e9f2a7cd0'
CORS(app)

@app.route("/", methods=['GET', 'POST'])
def index():
    token = request.headers.get('Authorization')
    if token is not None and "Bearer " in token:
        token = token.split(" ")[1]

    if session.get('token') == token:
        if request.method == 'POST':
            if request.form.get('action1') == "READ":
                create_table_readkey(sys.argv[2])
                keyread = secrets.token_urlsafe(16)
                db = get_db(sys.argv[2])
                cursor = db.cursor()
                query = "INSERT INTO readkey_table(key) VALUES (?)"
                cursor.execute(query, [keyread])
                db.commit()
                return render_template("read.html", keyread=keyread)
            elif request.form.get('action2') == "WRITE":
                create_table_writekey(int(sys.argv[2]))
                keywrite = secrets.token_urlsafe(16)
                db = get_db(sys.argv[2])
                cursor = db.cursor()
                query = "INSERT INTO writekey_table(key) VALUES (?)"
                cursor.execute(query, [keywrite])
                db.commit()
                return render_template("write.html", keywrite=keywrite)
            else:
                return render_template("index.html")  # Add this line to handle cases where neither action1 nor action2 is triggered
        else:
            return render_template('index.html')

    else: return render_template('index.html')

@app.route("/register", methods=["POST"])
def register():
    username = request.form.get('username')
    password = request.form.get('password')
    
    with open("secret.key", "rb") as key_file:
        secret_key = key_file.read()
    
    registered = insert_account(username, password, secret_key)
    if registered:
        token = secrets.token_urlsafe(16)  # Generate session token
        session['token'] = token
        return jsonify({"message": "Register successful", "token": token})
    else:
        return jsonify({"message": "Account Has been Created"}), 401

@app.route("/login", methods=["POST"])
def login():
    username = request.form.get('username')
    password = request.form.get('password')

    with open("secret.key", "rb") as key_file:
        secret_key = key_file.read()

    if verifyaccount(username, secret_key) == password:
        token = secrets.token_urlsafe(16)  # Generate session token
        session['token'] = token
        return jsonify({"message": "Login successful", "token": token})
    else:
        return jsonify({"message": "Invalid username or password"}), 401

@app.route('/api/cfg', methods=['GET', 'POST', 'DELETE'])
def cfg():
    token = request.headers.get('x-session-token')
    if session.get('token') != token:
        return jsonify({'message': 'Unauthorized'}), 403

    if request.method == 'GET':
        keyread = request.headers.get('Authorization')
        try:
            if verifyKeyread(keyread):
                result = read.get_cfg(sys.argv[2])
                resp = jsonify(result)
                resp.status_code = 200
                return resp
            else:
                data = {
                    'status': 404,
                    'message': "APIKey Not Found"
                }
                resp = jsonify(data)
                resp.status_code = 404
                return resp

        except Exception as e:
            data = {
                'status': 500,
                'message': f"Server Error {e}"
            }
            resp = jsonify(data)
            resp.status_code = 500
            return resp

    if request.method == 'POST':
        keywrite = request.headers.get('Authorization')
        try:
            if verifyKeywrite(keywrite):
                data = request.json
                cfg = data
                result = write.insert_cfg(cfg, sys.argv[2])
                data = {
                    'status': 201,
                    'message': result        
                }
                resp = jsonify(data)
                resp.status_code = 201
                return resp
            else:
                data = {
                    'status': 404,
                    'message': "APIKey Not Found"
                }
                resp = jsonify(data)
                resp.status_code = 404
                return resp
        except Exception as e:
            data = {
                'status': 500,
                'message': f"Server Error {e}"
            }
            resp = jsonify(data)
            resp.status_code = 500
            return resp

    if request.method == 'DELETE':
        key = request.headers.get('Authorization')
        try:
            if verifyKeywrite(key):
                delete_all_cfg(sys.argv[2])
                data = {
                    'status': 201,
                    'message': "delete all cfg successful"        
                }
                resp = jsonify(data)
                resp.status_code = 201
                return resp
            else:
                data = {
                    'status': 404,
                    'message': "APIKey Not Found"
                }
                resp = jsonify(data)
                resp.status_code = 404
                return resp
        except Exception as e:
            data = {
                'status': 500,
                'message': f"Server Error {e}"
            }
            resp = jsonify(data)
            resp.status_code = 500
            return resp

@app.route('/api/output', methods=['GET', 'POST', 'DELETE'])
def output():
    token = request.headers.get('x-session-token')
    if session.get('token') != token:
        return jsonify({'message': 'Unauthorized'}), 403

    if request.method == 'GET':
        keyread = request.headers.get('Authorization')
        try:
            if verifyKeyread(keyread):
                result = read.get_output(sys.argv[2])
                resp = jsonify(result)
                resp.status_code = 200
                return resp
            else:
                data = {
                    'status': 404,
                    'message': "APIKey Not Found"
                }
                resp = jsonify(data)
                resp.status_code = 404
                return resp

        except Exception as e:
            data = {
                'status': 500,
                'message': f"Server Error {e}"
            }
            resp = jsonify(data)
            resp.status_code = 500
            return resp

    if request.method == 'POST':
        keywrite = request.headers.get('Authorization')
        try:
            if verifyKeywrite(keywrite):
                data = request.json
                cfg = data
                result = write.insert_output(cfg, sys.argv[2])
                data = {
                    'status': 201,
                    'message': result        
                }
                resp = jsonify(data)
                resp.status_code = 201
                return resp
            else:
                data = {
                    'status': 404,
                    'message': "APIKey Not Found"
                }
                resp = jsonify(data)
                resp.status_code = 404
                return resp
        except Exception as e:
            data = {
                'status': 500,
                'message': f"Server Error {e}"
            }
            resp = jsonify(data)
            resp.status_code = 500
            return resp

    if request.method == 'DELETE':
        key = request.headers.get('Authorization')
        try:
            if verifyKeywrite(key):
                delete_all_output(sys.argv[2])
                data = {
                    'status': 201,
                    'message': "delete all output successful"        
                }
                resp = jsonify(data)
                resp.status_code = 201
                return resp
            else:
                data = {
                    'status': 404,
                    'message': "APIKey Not Found"
                }
                resp = jsonify(data)
                resp.status_code = 404
                return resp
        except Exception as e:
            data = {
                'status': 500,
                'message': f"Server Error {e}"
            }
            resp = jsonify(data)
            resp.status_code = 500
            return resp

def verifyKeyread(key):
    db = get_db(sys.argv[2])
    cursor = db.cursor()
    query = "SELECT key FROM readkey_table WHERE key = ?"
    cursor.execute(query, [key])
    result = cursor.fetchone()
    if result is None:
        return False
    return True

def verifyKeywrite(key):
    db = get_db(sys.argv[2])
    cursor = db.cursor()
    query = "SELECT key FROM writekey_table WHERE key = ?"
    cursor.execute(query, [key])
    result = cursor.fetchone()
    if result is None:
        return False
    return True

def insert_account(username, password, key):
    if verifyaccount(username, key):
        return False
    fernet = Fernet(key)
    password = fernet.encrypt(password.encode()).decode()
    db = get_db(sys.argv[2])
    cursor = db.cursor()
    query = "INSERT INTO account(username, password) VALUES (?, ?)"
    cursor.execute(query, [username, password])
    db.commit()
    return True

def verifyaccount(username, key):
    fernet = Fernet(key)
    db = get_db(sys.argv[2])
    cursor = db.cursor()
    query = "SELECT password FROM account WHERE username = ?"
    cursor.execute(query, [username])
    result = cursor.fetchone()
    if result is None:
        return False
    password = fernet.decrypt(result[0].encode()).decode()
    return password

@app.errorhandler(404)
def not_found_handler(error):
    message = {
        'status': 404,
        'message': 'NOT_FOUND'
    }
    resp = jsonify(message)
    resp.status_code = 404
    return resp

@app.errorhandler(403)
def forbidden_handler(error):
    message = {
        'status': 403,
        'message': 'Forbidden Access'
    }
    resp = jsonify(message)
    resp.status_code = 403
    return resp

@app.errorhandler(405)
def method_not_allowed_handler(error):
    message = {
        'status': 405,
        'message': 'Method Not Allowed'
    }
    resp = jsonify(message)
    resp.status_code = 405
    return resp

@app.errorhandler(500)
def internal_server_error_handler(error):
    message = {
        'status': 500,
        'message': 'Internal Server Error'
    }
    resp = jsonify(message)
    resp.status_code = 500
    return resp

@app.errorhandler(400)
def bad_request_handler(error):
    message = {
        'status': 400,
        'message': 'Bad Request'
    }
    resp = jsonify(message)
    resp.status_code = 400
    return resp

@app.errorhandler(401)
def unauthorized_handler(error):
    message = {
        'status': 401,
        'message': 'Unauthorized Login'
    }
    resp = jsonify(message)
    resp.status_code = 401
    return resp

# Signal handler to catch Ctrl+C and other termination signals
def signal_handler(sig, frame):
    print("Signal received, cleaning up...")
    delete_all_api(sys.argv[2])
    os.remove(f"database-{sys.argv[2]}.db")
    sys.exit(0)

if __name__ == "__main__":
    # Register signal handlers for SIGINT (Ctrl+C) and SIGTERM
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        create_table_readkey(sys.argv[2])
        create_table_writekey(sys.argv[2])
        create_table_cfg(sys.argv[2])
        create_table_output(sys.argv[2])
        create_table_account(sys.argv[2])
        app.run(host=sys.argv[1], port=int(sys.argv[2]), debug=True)
    except Exception as e:
        print(f"Error: {e}")
