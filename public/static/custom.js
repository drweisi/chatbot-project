async function sendMessage() {
  const input = document.getElementById("message");
  const content = input.value.trim();
  if (!content) return;

  const chat = document.getElementById("chat-container");
  const userMsg = document.createElement("div");
  userMsg.className = "message user";
  userMsg.innerText = content;
  chat.appendChild(userMsg);
  input.value = "";

  chat.scrollTop = chat.scrollHeight;

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: content })
    });
    const data = await res.json();

    const botMsg = document.createElement("div");
    botMsg.className = "message gpt";
    botMsg.innerText = data.reply;
    chat.appendChild(botMsg);
    chat.scrollTop = chat.scrollHeight;
  } catch (error) {
    const botMsg = document.createElement("div");
    botMsg.className = "message gpt";
    botMsg.innerText = "网络错误：" + error.message;
    chat.appendChild(botMsg);
    chat.scrollTop = chat.scrollHeight;
  }
}

document.getElementById("message").addEventListener("keypress", function(e) {
  if (e.key === 'Enter') {
    sendMessage();
  }
});
