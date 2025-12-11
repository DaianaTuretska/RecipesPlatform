document.addEventListener("DOMContentLoaded", function () {
  // ---------------- OPEN/CLOSE UI LOGIC ----------------
  const chatButton = document.getElementById("assistantChatButton");
  const chatWindow = document.getElementById("assistantChatWindow");
  const closeBtn = document.getElementById("assistantCloseBtn");
  const apiUrl = chatWindow.dataset.apiUrl;

  if (!chatButton || !chatWindow || !closeBtn) {
    console.error("❌ Assistant elements not found. Check HTML IDs.");
    return;
  }

  // Initial UI state
  chatWindow.classList.remove("assistant-visible");
  chatButton.classList.remove("hidden");

  chatButton.addEventListener("click", function () {
    chatWindow.classList.add("assistant-visible");
    chatButton.classList.add("hidden");
  });

  closeBtn.addEventListener("click", function () {
    chatWindow.classList.remove("assistant-visible");
    setTimeout(() => chatButton.classList.remove("hidden"), 150);
  });

  // ---------------- SEND MESSAGE LOGIC ----------------
  const input = document.getElementById("assistantInput");
  const sendBtn = document.getElementById("assistantSendBtn");
  const messagesBox = document.getElementById("assistantMessages");

  if (!input || !sendBtn || !messagesBox) {
    console.error("❌ Message elements missing. Check widget.html.");
    return;
  }

  function addUserBubble(text) {
    const div = document.createElement("div");
    div.className = "assistant-msg-user";
    div.innerText = text;
    messagesBox.appendChild(div);
    messagesBox.scrollTop = messagesBox.scrollHeight;
  }

  function addBotBubble(html) {
    const div = document.createElement("div");
    div.className = "assistant-msg-bot formatted-response";
    div.innerHTML = html;
    messagesBox.appendChild(div);
    messagesBox.scrollTop = messagesBox.scrollHeight;
  }

  // Format recipe message into a pretty chat layout
  function renderRecipe(data) {
    const name = data.name || "Recipe";
    let ingredients = data.ingredients || "";
    const directions = data.directions || "";
    const time = data.total_time || null;
    const closePhrase = data.closing_phrase || null; // NEW FIELD

    // Convert ingredients string → array
    if (typeof ingredients === "string") {
      ingredients = ingredients
        .split(",")
        .map((i) => i.trim())
        .filter((i) => i.length > 0);
    }

    return `
        <div class="recipe-block">
            <div class="recipe-title">💡 ${name}</div>

            <div class="recipe-section">
                <strong>🧂 Ingredients:</strong>
                <ul>
                    ${ingredients.map((i) => `<li>${i}</li>`).join("")}
                </ul>
            </div>

            <div class="recipe-section">
                <strong>👨‍🍳 Directions:</strong>
                <p>${directions}</p>
            </div>

            ${
              time
                ? `
                <div class="recipe-section">
                    <strong>⏱️ Total time:</strong> ${time} minutes
                </div>
            `
                : ""
            }

            ${
              closePhrase
                ? `
                <div class="recipe-section recipe-close-phrase">
                    <em>${closePhrase}</em>
                </div>
            `
                : ""
            }
        </div>
    `;
  }

  function sendAssistantMessage() {
    const userText = input.value.trim();
    if (!userText) return;

    addUserBubble(userText);
    input.value = "";

    const loading = document.createElement("div");
    loading.className = "assistant-msg-bot";
    loading.innerText = "Typing...";
    messagesBox.appendChild(loading);
    messagesBox.scrollTop = messagesBox.scrollHeight;

    fetch(apiUrl, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-CSRFToken": getCookie("csrftoken"),
      },
      body: JSON.stringify({ message: userText }),
    })
      .then((res) => res.json())
      .then((data) => {
        loading.remove();

        if (data.error) {
          addBotBubble(`❌ ${data.error}`);
          return;
        }

        const html = renderRecipe(data);
        addBotBubble(html);
      })
      .catch((err) => {
        loading.remove();
        addBotBubble("❌ Error connecting to server.");
      });
  }

  sendBtn.addEventListener("click", sendAssistantMessage);

  input.addEventListener("keypress", function (e) {
    if (e.key === "Enter") sendAssistantMessage();
  });

  // ---------------- CSRF TOKEN HELP ----------------
  function getCookie(name) {
    let cookieValue = null;
    const cookies = document.cookie.split(";");
    for (const cookie of cookies) {
      const trimmed = cookie.trim();
      if (trimmed.startsWith(name + "=")) {
        cookieValue = trimmed.substring(name.length + 1);
        break;
      }
    }
    return cookieValue;
  }
});
