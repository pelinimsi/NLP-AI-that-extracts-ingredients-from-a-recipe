const socket = io();

function sendMessage() {
  const input = document.getElementById('message');
  const msg = input.value.trim();
  if (msg === '') return;

  appendMessage('Sen', msg, 'user');
  socket.emit('user_message', { message: msg });
  input.value = '';
}

function appendMessage(sender, text, type) {
  const chatBox = document.getElementById('chat');
  const msgDiv = document.createElement('div');
  const spaceDiv = document.getElementById('spaceDiv')
  msgDiv.classList.add('message', type);  
  msgDiv.innerHTML = `<strong>${sender}:</strong> ${text}`;
  chatBox.insertBefore(msgDiv, spaceDiv);
  chatBox.scrollTop = chatBox.scrollHeight;
}

socket.on('bot_message', function (data) {
  appendMessage('Yapay Zeka', data.message, 'bot');
});

document.getElementById('message').addEventListener('keydown', function (e) {
  if (e.key === 'Enter') {
    e.preventDefault();
    sendMessage();
  }
});