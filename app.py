from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from Model.Predict import malzeme_cikar

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

@app.route('/')
def index():
    return render_template('chat.html')

@socketio.on('user_message')
def handle_user_message(data):
    user_msg = data['message']

    ai_msg = malzeme_cikar(user_msg)
    emit('bot_message', {'message': ai_msg})

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=80)
