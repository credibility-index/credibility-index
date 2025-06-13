# auth.py
from flask import Blueprint, request, jsonify, current_app
from werkzeug.security import generate_password_hash, check_password_hash
import jwt
import datetime
import uuid
from functools import wraps
import sqlite3
from flask_cors import cross_origin

auth_bp = Blueprint('auth', __name__)

# Вспомогательная функция для работы с БД
def get_db_connection():
    conn = sqlite3.connect('news_analysis.db')
    conn.row_factory = sqlite3.Row
    return conn

# Декоратор для проверки токена
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None

        if 'Authorization' in request.headers:
            token = request.headers['Authorization'].split(" ")[1]

        if not token:
            return jsonify({'message': 'Токен отсутствует!'}), 401

        try:
            data = jwt.decode(token, current_app.config['SECRET_KEY'], algorithms=["HS256"])
            current_user = get_user_by_id(data['user_id'])
        except:
            return jsonify({'message': 'Неверный токен!'}), 401

        return f(current_user, *args, **kwargs)

    return decorated

# Вспомогательные функции для работы с пользователями
def get_user_by_id(user_id):
    conn = get_db_connection()
    user = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
    conn.close()
    return user

def get_user_by_email(email):
    conn = get_db_connection()
    user = conn.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()
    conn.close()
    return user

def create_user(email, password_hash, name=None):
    conn = get_db_connection()
    user_id = str(uuid.uuid4())
    conn.execute('INSERT INTO users (id, email, password_hash, name, verified, reputation) VALUES (?, ?, ?, ?, ?, ?)',
                 (user_id, email, password_hash, name, False, 0.1))
    conn.commit()
    conn.close()
    return get_user_by_id(user_id)

# Маршрут для регистрации
@auth_bp.route('/register', methods=['POST'])
@cross_origin()
def register():
    data = request.get_json()

    # Проверка данных
    if not data.get('email') or not data.get('password'):
        return jsonify({'error': 'Email и пароль обязательны'}), 400

    # Проверка, существует ли пользователь
    if get_user_by_email(data['email']):
        return jsonify({'error': 'Пользователь с таким email уже существует'}), 400

    # Хэширование пароля
    password_hash = generate_password_hash(data['password'])

    # Создание пользователя
    new_user = create_user(
        email=data['email'],
        password_hash=password_hash,
        name=data.get('name')
    )

    # Создание токена
    token = jwt.encode({
        'user_id': new_user['id'],
        'email': new_user['email'],
        'reputation': new_user['reputation'],
        'exp': datetime.datetime.utcnow() + datetime.timedelta(days=30)
    }, current_app.config['SECRET_KEY'], algorithm='HS256')

    return jsonify({
        'message': 'Регистрация успешна',
        'token': token,
        'user': {
            'id': new_user['id'],
            'email': new_user['email'],
            'name': new_user['name'],
            'reputation': new_user['reputation'],
            'verified': new_user['verified']
        }
    }), 201

# Маршрут для входа
@auth_bp.route('/login', methods=['POST'])
@cross_origin()
def login():
    data = request.get_json()

    # Поиск пользователя
    user = get_user_by_email(data['email'])

    if not user or not check_password_hash(user['password_hash'], data['password']):
        return jsonify({'error': 'Неверный email или пароль'}), 401

    # Создание токена
    token = jwt.encode({
        'user_id': user['id'],
        'email': user['email'],
        'reputation': user['reputation'],
        'exp': datetime.datetime.utcnow() + datetime.timedelta(days=30)
    }, current_app.config['SECRET_KEY'], algorithm='HS256')

    return jsonify({
        'token': token,
        'user': {
            'id': user['id'],
            'email': user['email'],
            'name': user['name'],
            'reputation': user['reputation'],
            'verified': user['verified']
        }
    })

# Маршрут для проверки токена
@auth_bp.route('/protected', methods=['GET'])
@token_required
@cross_origin()
def protected(current_user):
    return jsonify({
        'message': 'Это защищенный маршрут',
        'user': {
            'id': current_user['id'],
            'email': current_user['email'],
            'name': current_user['name'],
            'reputation': current_user['reputation']
        }
    })
