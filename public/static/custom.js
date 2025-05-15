// 自动聚焦
document.getElementById("message").focus();

async function sendMessage() {
  const input = document.getElementById("message");
  const content = input.value.trim();
  if (!content) return;

  const chatContainer = document.getElementById("chat-container");
  // 显示用户消息
  chatContainer.innerHTML += `<div class="message user">${content}</div>`;
  chatContainer.scrollTop = chatContainer.scrollHeight;
  // 清空输入框
  input.value = "";

  // 前端调试日志
  console.log("即将发送内容：", content);

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: content })
    });
    const data = await response.json();
    console.log("后端回复：", data);

    chatContainer.innerHTML += `<div class="message gpt">${data.reply}</div>`;
    chatContainer.scrollTop = chatContainer.scrollHeight;
  } catch (error) {
    chatContainer.innerHTML += `<div class="message gpt">发生错误：${error.message}</div>`;
    chatContainer.scrollTop = chatContainer.scrollHeight;
  }
}

// 支持回车发送
document.getElementById("message").addEventListener("keypress", function(e) {
  if (e.key === 'Enter') {
    sendMessage();
  }
});
